
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path
_REVISION_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REVISION_ROOT / "src" / "cnf"))
import json
import hashlib
import torch
import argparse
import random
import time
import numpy as np
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainer import train, valid, test
from fusion_model import PremiseSelectionModel
from dataset import FormulaGraphDataset
from utils import set_recorder, dump_pickle_file, load_pickle_file

NODE_IN_CHANNELS = 793  


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hyper_parameters():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--model_save", type=str,
                   default=str(_REVISION_ROOT / "results" / "runs" / "legacy_cnf"))
    p.add_argument("--root_dir", type=str,
                   default=str(_REVISION_ROOT / ".cache" / "datasets" / "cnf"))
    p.add_argument("--node_out_channels", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--engineering_test", type=int, choices=(0, 1), default=0,
                   help="must be 1 to permit a diagnostic run other than 30 epochs")
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--node_dict_name", default="node_dict.pkl")
    p.add_argument("--symbolic_cache_name", default="symbolic_baselines.jsonl")
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--diagnostic_train_predictions", type=int, choices=(0,1), default=0)
    
    p.add_argument("--use_lce_and", type=int, default=1)
    p.add_argument("--use_lce_or", type=int, default=1)
    p.add_argument("--use_lce_not", type=int, default=1)
    p.add_argument("--use_lce_contains", type=int, default=1)
    p.add_argument("--use_lce_equivalence", type=int, default=1)
    p.add_argument("--use_lce_forall", type=int, default=1)
    p.add_argument("--use_lce_exists", type=int, default=1)
    
    p.add_argument("--use_classifier_constraint", type=int, default=1)
    
    p.add_argument("--lce_scale", type=float, default=0.5,
                   help="LCE per-step scaling (default 0.5 matches paper)")
    p.add_argument("--rho_w", type=float, default=1.0,
                   help="Deprecated shared rule weight; rho_t/rho_s take precedence")
    p.add_argument("--rho_t", type=float, default=None,
                   help="Trigger-rule positive-logit weight")
    p.add_argument("--rho_s", type=float, default=None,
                   help="Weighted-similarity positive-logit weight")
    p.add_argument("--use_trigger_rule", type=int, default=1)
    p.add_argument("--use_similarity_rule", type=int, default=1)
    p.add_argument("--allow_legacy_rule_cache", type=int, default=0,
                   help="Diagnostics only; formal corrected runs require cache metadata")
    p.add_argument("--skip_test", type=int, default=0,
                   help="Validation-only tuning run; do not read the test loader")
    p.add_argument("--encoder_variant", choices=("lce", "tw", "operator_embedding",
                   "relation_messages", "parameter_matched_mlp"), default="lce")
    p.add_argument("--relation_bases", type=int, default=4)
    p.add_argument("--generic_mlp_width", type=int, default=None)
    p.add_argument("--use_term_walk", type=int, choices=(0, 1), default=1)
    p.add_argument("--quantifier_aggregation", choices=("top2", "all_child_mean", "binder_body"), default="top2")
    p.add_argument("--commutative_aggregation", choices=("ordered", "mean", "sum"), default="ordered")
    p.add_argument("--symbolic_baseline", choices=("lfr", "overlap", "idf", "sine", "trigger",
                   "weighted_similarity", "weighted_similarity_score"), default="lfr")
    p.add_argument("--warm_start_run", default=None)
    p.add_argument("--fusion_mode", choices=("binary","continuous","gated","neural_control"),default="binary")
    p.add_argument("--gate_hidden",type=int,default=16)
    return p.parse_args()


def acc_improved(candidate, incumbent):
    """Maximum validation ACC, strictly greater; earliest epoch on ties."""
    return incumbent is None or candidate > incumbent


def main():
    run_started = time.perf_counter()
    args = hyper_parameters()
    if args.epochs != 30 and not args.engineering_test:
        raise ValueError("This new protocol requires exactly 30 budget epochs; set --engineering_test 1 only for diagnostics")
    save_dir = os.path.join(args.model_save, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    recorder = set_recorder(args.exp_name, os.path.join(save_dir, "record.log"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device != "cpu":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)

    recorder.info("=" * 60)
    recorder.info(f"Experiment: {args.exp_name}")
    for k, v in vars(args).items():
        recorder.info(f"  {k}: {v}")
    recorder.info("=" * 60)

    node_dict_path = os.path.join(args.root_dir, args.node_dict_name)
    node_in_channels = len(load_pickle_file(node_dict_path))
    model = PremiseSelectionModel(
        node_in_channels, args.node_out_channels, args.layers,
        use_lce_and=bool(args.use_lce_and),
        use_lce_or=bool(args.use_lce_or),
        use_lce_not=bool(args.use_lce_not),
        use_lce_contains=bool(args.use_lce_contains),
        use_lce_equivalence=bool(args.use_lce_equivalence),
        use_lce_forall=bool(args.use_lce_forall),
        use_lce_exists=bool(args.use_lce_exists),
        use_classifier_constraint=bool(args.use_classifier_constraint),
        lce_scale=args.lce_scale,
        rho_w=args.rho_w,
        rho_t=args.rho_t,
        rho_s=args.rho_s,
        use_trigger_rule=bool(args.use_trigger_rule),
        use_similarity_rule=bool(args.use_similarity_rule),
        encoder_variant=args.encoder_variant,
        relation_bases=args.relation_bases,
        generic_mlp_width=args.generic_mlp_width,
        use_term_walk=bool(args.use_term_walk),
        quantifier_aggregation=args.quantifier_aggregation,
        commutative_aggregation=args.commutative_aggregation,
        symbolic_baseline=args.symbolic_baseline,
        fusion_mode=args.fusion_mode, gate_hidden=args.gate_hidden,
    ).to(device=device)

    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer)

    recorder.info("Loading data...")
    data_loading_started = time.perf_counter()
    metadata_path = os.path.join(args.root_dir, "rule_cache_metadata.json")
    if not args.allow_legacy_rule_cache:
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Corrected rule metadata missing: {metadata_path}; "
                "run scripts/generate_rule_cache.py first")
        with open(metadata_path) as stream:
            rule_metadata = json.load(stream)
        if not rule_metadata.get("corrected"):
            raise ValueError(f"Rule cache is not marked corrected: {metadata_path}")
        for filename in ("similarity2.json", "similarity4.json"):
            expected = rule_metadata["outputs"][filename.replace(".json", "_sha256")]
            actual = file_sha256(os.path.join(args.root_dir, filename))
            if actual != expected:
                raise ValueError(f"Rule cache hash mismatch: {filename}")
        if file_sha256(os.path.join(args.root_dir, "statements")) != rule_metadata["inputs"]["statements_sha256"]:
            raise ValueError("Statements hash does not match rule-cache metadata")
        for pair_file in rule_metadata["inputs"]["pair_files"]:
            if file_sha256(os.path.join(args.root_dir, pair_file["path"])) != pair_file["sha256"]:
                raise ValueError(f"Pair input hash mismatch: {pair_file['path']}")
    needs_symbolic_features = True  
    symbolic_features_file = (args.symbolic_cache_name if needs_symbolic_features
                              else "__disabled_symbolic_features__.jsonl")
    if needs_symbolic_features:
        symbolic_path = os.path.join(args.root_dir, args.symbolic_cache_name)
        symbolic_metadata_path = symbolic_path + ".metadata.json"
        if not os.path.isfile(symbolic_path) or not os.path.isfile(symbolic_metadata_path):
            raise FileNotFoundError(f"Symbolic cache or metadata missing: {symbolic_path}")
        with open(symbolic_metadata_path) as stream:
            symbolic_metadata = json.load(stream)
        if (symbolic_metadata.get("labels_ignored") is not True or
                file_sha256(symbolic_path) != symbolic_metadata.get("output_sha256")):
            raise ValueError(f"Invalid label-blind symbolic cache: {symbolic_path}")
    train_dataset = FormulaGraphDataset(
        os.path.join(args.root_dir, "train"), "train",
        os.path.join(args.root_dir, "statements"),
        node_dict_path,
        os.path.join(args.root_dir, "node_attr_dict.pkl"),
        os.path.join(args.root_dir, "edge_attr_dict.pkl"),
        symbolic_features_file=symbolic_features_file,
        rename=True)
    valid_dataset = FormulaGraphDataset(
        os.path.join(args.root_dir, "valid"), "valid",
        os.path.join(args.root_dir, "statements"),
        node_dict_path,
        os.path.join(args.root_dir, "node_attr_dict.pkl"),
        os.path.join(args.root_dir, "edge_attr_dict.pkl"),
        symbolic_features_file=symbolic_features_file,
        rename=True)
    test_dataset = None if args.skip_test else FormulaGraphDataset(
        os.path.join(args.root_dir, "test"), "test",
        os.path.join(args.root_dir, "statements"),
        node_dict_path,
        os.path.join(args.root_dir, "node_attr_dict.pkl"),
        os.path.join(args.root_dir, "edge_attr_dict.pkl"),
        symbolic_features_file=symbolic_features_file,
        rename=True)

    generator = torch.Generator().manual_seed(args.seed)
    loader_kw = dict(batch_size=args.batch_size, follow_batch=["x_s", "x_t"],
                     num_workers=args.num_workers, pin_memory=args.device != "cpu")
    if args.num_workers:
        loader_kw.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kw)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kw)
    test_loader = None if args.skip_test else DataLoader(test_dataset, shuffle=False, **loader_kw)
    recorder.info("Data loaded.")
    data_loading_seconds = time.perf_counter() - data_loading_started

    scaler = torch.cuda.amp.GradScaler() if args.device != "cpu" else None
    history = {f"{p}_{m}": [] for p in ("train", "valid") for m in ("loss", "acc", "f1", "recall", "precision")}
    history.update(train_seconds=[], valid_seconds=[])
    best_epoch = None
    start_budget = 0
    if args.warm_start_run:
        raise ValueError("Fusion study requires training from scratch")
    continuation = {"parent_run": args.warm_start_run, "prior_training_budget": start_budget,
                    "additional_epochs": args.epochs-start_budget, "total_training_budget": args.epochs,
                    "mode": "validation_best_weight_warm_start_with_optimizer_reset" if args.warm_start_run else "from_scratch",
                    "parent_checkpoint_epoch": best_epoch["epoch"] if best_epoch else None}
    (Path(save_dir)/"training_protocol.json").write_text(json.dumps(continuation,indent=2)+"\n")
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    if args.device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()

    for epoch in range(start_budget + 1, args.epochs + 1):
        recorder.info(f"--- Epoch {epoch}, lr={optimizer.param_groups[0]['lr']:.6f} ---")
        if args.device != "cpu": torch.cuda.synchronize(device)
        phase_started = time.perf_counter()
        tr_loss, tr_acc, tr_f1, tr_rec, tr_prec = train(epoch, train_loader, model, optimizer, device, recorder, scaler=scaler)
        if args.device != "cpu": torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        vl_loss, vl_acc, vl_f1, vl_rec, vl_prec = valid(epoch, valid_loader, model, device, recorder)
        if args.device != "cpu": torch.cuda.synchronize(device)
        valid_seconds = time.perf_counter() - phase_started

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc), ("train_f1", tr_f1),
                      ("train_recall", tr_rec), ("train_precision", tr_prec),
                      ("valid_loss", vl_loss), ("valid_acc", vl_acc), ("valid_f1", vl_f1),
                      ("valid_recall", vl_rec), ("valid_precision", vl_prec)]:
            history[k].append(v)
        history["train_seconds"].append(train_seconds)
        history["valid_seconds"].append(valid_seconds)

        if acc_improved(vl_acc, None if best_epoch is None else best_epoch["valid_acc"]):
            best_epoch = {
                "epoch": epoch,
                "state_dict": {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()},
                "valid_loss": vl_loss,
                "valid_acc": vl_acc,
            }
        lr_scheduler.step(vl_loss)
        
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": lr_scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                 "budget_epoch": epoch, "history": history, "best_epoch": best_epoch,
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": {"device":args.device,"state":torch.cuda.get_rng_state(device)} if args.device != "cpu" else None,
                 "loader_generator_rng": generator.get_state(), "args": vars(args), "training_protocol": continuation}
        temporary = Path(save_dir)/"last_training_state.pt.tmp"
        torch.save(state,temporary)
        temporary.replace(Path(save_dir)/"last_training_state.pt")

    training_seconds = time.perf_counter() - training_started
    peak_gpu_memory_bytes = (torch.cuda.max_memory_allocated(device)
                             if args.device != "cpu" else None)
    resource_record = {
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "training_seconds": training_seconds,
        "mean_train_epoch_seconds": sum(history["train_seconds"]) / len(history["train_seconds"]),
        "mean_validation_epoch_seconds": sum(history["valid_seconds"]) / len(history["valid_seconds"]),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "data_loading_seconds": data_loading_seconds,
    }

    torch.save(best_epoch["state_dict"], os.path.join(save_dir, "best_valid_acc.pt"))

    model.load_state_dict(best_epoch["state_dict"])
    
    prediction_splits = [("valid", valid_dataset, valid_loader)]
    if args.diagnostic_train_predictions:
        prediction_splits.append(("train", train_dataset,
            DataLoader(train_dataset, shuffle=False, **loader_kw)))
    for split, dataset_object, ordered_loader in prediction_splits:
        _, _, _, _, _, rows = test(ordered_loader, model, device, recorder,
                                   return_predictions=True)
        raw_path = Path(args.root_dir) / split / "raw" / (split + ".json")
        identifiers = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        if len(rows) != len(identifiers):
            raise RuntimeError("ACC prediction count differs from raw identifiers")
        output = Path(save_dir) / (split + "_predictions.jsonl")
        with output.open("w") as stream:
            for source, row in zip(identifiers, rows):
                stream.write(json.dumps({"conjecture": source[0], "premise": source[1],
                                         **row}, sort_keys=True) + "\n")

    if args.skip_test:
        resource_record["total_wall_seconds"] = time.perf_counter() - run_started
        results = {
            "experiment": args.exp_name,
            "dataset": "CNF",
            "config": {k: v for k, v in vars(args).items() if k != "model_save"},
            "rule_cache_metadata": metadata_path,
            "selection": "maximum validation accuracy; earliest epoch on ties",
            "validation_only": True,
            "best_validation": {"epoch": best_epoch["epoch"], "loss": round(best_epoch["valid_loss"], 6), "accuracy": best_epoch["valid_acc"]},
            "test": None,
            "resources": resource_record,
        }
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        dump_pickle_file(history, os.path.join(save_dir, "history.pkl"))
        recorder.info("Validation-only tuning run complete; test split was not evaluated")
        return

    recorder.info("=" * 60)
    recorder.info("TESTING THE PRE-SELECTED VALIDATION-ACC CHECKPOINT")
    results = {
        "experiment": args.exp_name,
        "dataset": "CNF",
        "config": {k: v for k, v in vars(args).items() if k != "model_save"},
        "rule_cache_metadata": metadata_path if os.path.isfile(metadata_path) else None,
        "diagnostic_legacy_rules": bool(args.allow_legacy_rule_cache),
        "best_validation": {"epoch": best_epoch["epoch"], "loss": best_epoch["valid_loss"], "accuracy": best_epoch["valid_acc"]},
        "selection": "maximum validation accuracy; earliest epoch on ties",
    }

    model.load_state_dict(best_epoch["state_dict"])
    if args.device != "cpu": torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    loss, acc, f1, rec, prec, predictions = test(
        test_loader, model, device, recorder, return_predictions=True)
    if args.device != "cpu": torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    raw_test_path = Path(args.root_dir) / "test" / "raw" / "test.json"
    identifiers = [json.loads(line) for line in raw_test_path.read_text().splitlines() if line.strip()]
    if len(identifiers) != len(predictions):
        raise RuntimeError("Prediction count does not match ordered test pairs")
    prediction_path = Path(save_dir) / "test_predictions.jsonl"
    with prediction_path.open("w") as stream:
        for source, prediction in zip(identifiers, predictions):
            stream.write(json.dumps({"conjecture": source[0], "premise": source[1], **prediction}, sort_keys=True) + "\n")
    resource_record.update(
        test_inference_seconds=inference_seconds,
        test_pairs_per_second=len(predictions) / inference_seconds,
    )
    entry = {
        "type": f"best_valid_acc_epoch{best_epoch['epoch']}",
        "valid_loss": round(best_epoch["valid_loss"], 6),
        "valid_acc": best_epoch["valid_acc"],
        "test_loss": round(loss, 6),
        "test_acc": round(acc, 6),
        "test_f1": round(f1, 6),
        "test_recall": round(rec, 6),
        "test_precision": round(prec, 6),
    }
    results["test"] = entry
    results["predictions"] = {
        "path": str(prediction_path), "pairs": len(predictions),
        "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
    }
    results["resources"] = resource_record
    resource_record["total_wall_seconds"] = time.perf_counter() - run_started
    recorder.info(f"  {entry['type']}: Acc={acc*100:.2f}% F1={f1*100:.2f}%")

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    dump_pickle_file(history, os.path.join(save_dir, "history.pkl"))

    recorder.info(f"All results saved to {save_dir}")
    recorder.info("DONE")


if __name__ == "__main__":
    main()
