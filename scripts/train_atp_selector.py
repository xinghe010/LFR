"""Train an ATP selector and write validation-selected provenance.

This script is the missing training counterpart of
``src/atp/select_premises_continuous_v2.py``.  It trains on the ATP train split,
selects the epoch with the highest validation ACC (earliest epoch on ties), and
writes:

  best_valid_acc.pt   checkpoint with model state, epoch, config hash, selection
  model_config.json   frozen effective config (purpose=atp_d0, epochs=30)
  results.json        provenance consumed by the selector

No checkpoint or training result is committed to the repository; they are
generated locally by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "mptp"))
sys.path.insert(0, str(ROOT / "src" / "atp"))
from dataset import FormulaGraphDataset
from fusion_model import PremiseSelectionModel
from trainer import train, valid
from utils import set_recorder

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_dir", type=Path, default=Path("data/atp"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--component", choices=("tw", "tw_lce", "tw_rules", "full"),
                        default="full")
    parser.add_argument("--node_out_channels", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate_manifest", type=Path, default=None)
    parser.add_argument("--rule_metadata", type=Path, default=None)
    parser.add_argument("--symbolic_metadata", type=Path, default=None)
    args = parser.parse_args()

    if args.epochs != 30:
        raise ValueError("The paper protocol requires exactly 30 epochs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    node_dict = pickle.load((args.root_dir / "node_dict.pkl").open("rb"))
    node_attr_dict = pickle.load((args.root_dir / "node_attr_dict.pkl").open("rb"))
    statements = args.root_dir / "statements"
    operator_names = {
        "exists": "?", "not": "~", "and": "&", "equivalence": "<=>",
        "forall": "!", "or": "|", "implication": "=>",
    }
    operator_type_indices = {
        name: node_attr_dict[symbol] for name, symbol in operator_names.items()}
    common = dict(
        statements_file=str(statements),
        node_dict_file=str(args.root_dir / "node_dict.pkl"),
        node_attr_dict_file=str(args.root_dir / "node_attr_dict.pkl"),
        edge_attr_dict_file=str(args.root_dir / "edge_attr_dict.pkl"),
        symbolic_features_file="symbolic_baselines.jsonl",
        rename=True,
    )
    train_dataset = FormulaGraphDataset(
        root=str(args.root_dir / "train"), data_class="train", **common)
    valid_dataset = FormulaGraphDataset(
        root=str(args.root_dir / "valid"), data_class="valid", **common)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, generator=generator,
                              follow_batch=["x_s", "x_t"])
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, follow_batch=["x_s", "x_t"])

    encoder, use_rules = {
        "tw": ("tw", False),
        "tw_lce": ("lce", False),
        "tw_rules": ("tw", True),
        "full": ("lce", True),
    }[args.component]
    model_options = {
        "use_lce_and": True, "use_lce_or": True, "use_lce_not": True,
        "use_lce_contains": True, "use_lce_equivalence": True,
        "use_lce_forall": True, "use_lce_exists": True,
        "use_classifier_constraint": use_rules,
        "lce_scale": 0.5, "rho_w": 1.0, "rho_t": 1.0, "rho_s": 0.3,
        "use_trigger_rule": use_rules, "use_similarity_rule": use_rules,
        "encoder_variant": encoder, "relation_bases": 4,
        "generic_mlp_width": None, "symbolic_baseline": "lfr",
        "use_term_walk": True, "quantifier_aggregation": "top2",
        "commutative_aggregation": "ordered",
        "operator_type_indices": operator_type_indices,
        "variable_type_index": node_attr_dict["var"],
    }
    model = PremiseSelectionModel(
        node_in_channels=len(node_dict),
        node_out_channels=args.node_out_channels,
        layers=args.layers,
        **model_options,
    ).to(device)

    recorder = set_recorder("atp_train", os.path.join(args.output_dir, "train.log"))
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer)

    best = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch, train_loader, model, optimizer, device, recorder)
        valid_loss, valid_acc = valid(epoch, valid_loader, model, device, recorder)
        scheduler.step(valid_loss)
        if best is None or valid_acc > best["valid_acc"]:
            best = {
                "epoch": epoch,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}")

    if best is None:
        raise RuntimeError("No validation epoch completed")

    base_config = {
        "purpose": "atp_d0",
        "epochs": args.epochs,
        "node_out_channels": args.node_out_channels,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "model": {**model_options, "fusion_mode": "continuous"},
    }
    config_sha = hashlib.sha256(
        json.dumps(base_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    model_config = {**base_config, "config_sha256": config_sha}
    config_path = args.output_dir / "model_config.json"
    config_path.write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")

    checkpoint_path = args.output_dir / "best_valid_acc.pt"
    torch.save({
        "model": best["state"],
        "epoch": best["epoch"],
        "config_sha256": config_sha,
        "selection": "max validation ACC; earliest epoch on tie",
    }, checkpoint_path)

    provenance = {}
    for key, path in (
        ("candidate_manifest_sha256", args.candidate_manifest),
        ("rule_metadata_sha256", args.rule_metadata),
        ("symbolic_metadata_sha256", args.symbolic_metadata),
    ):
        if path is not None:
            provenance[key] = sha256_file(path)

    result = {
        "schema": "twgnn-revised-selector-result-v2",
        "purpose": "atp_d0",
        "status": "PASS",
        "component": args.component,
        "epochs": args.epochs,
        "selection": {"epoch": best["epoch"], "valid_acc": best["valid_acc"],
                      "valid_loss": best["valid_loss"]},
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "effective_config_sha256": sha256_file(config_path),
        "provenance": provenance,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved: {checkpoint_path}")
    print(f"Saved: {config_path}")
    print(f"Saved: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
