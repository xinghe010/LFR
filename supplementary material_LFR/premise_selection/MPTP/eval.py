import os
import json
import torch
import argparse
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainer import train, valid, test
from model import PremiseSelectionModel
from dataset import FormulaGraphDataset
from utils import set_recorder, dump_pickle_file

NODE_IN_CHANNELS = 793

def hyper_parameters():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="default_run")
    p.add_argument("--model_save", type=str,
                   default="./results/")
    p.add_argument("--root_dir", type=str,
                   default="./dataset/")
    p.add_argument("--node_out_channels", type=int, default=32)
    p.add_argument("--edge_attr_out_channels", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=24)
    return p.parse_args()

def main():
    args = hyper_parameters()
    save_dir = os.path.join(args.model_save, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    recorder = set_recorder(args.exp_name, os.path.join(save_dir, "record.log"))

    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    recorder.info("=" * 60)
    recorder.info(f"Experiment: {args.exp_name}")
    for k, v in vars(args).items():
        recorder.info(f"  {k}: {v}")
    recorder.info("=" * 60)

    model = PremiseSelectionModel(
        NODE_IN_CHANNELS, args.node_out_channels, args.layers
    ).to(device=device)

    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer)

    recorder.info("Loading data...")
    train_dataset = FormulaGraphDataset(
        os.path.join(args.root_dir, "train"), "train",
        os.path.join(args.root_dir, "statements"),
        os.path.join(args.root_dir, "node_dict.pkl"),
        os.path.join(args.root_dir, "node_attr_dict.pkl"),
        os.path.join(args.root_dir, "edge_attr_dict.pkl"),
        rename=True)
    valid_dataset = FormulaGraphDataset(
        os.path.join(args.root_dir, "valid"), "valid",
        os.path.join(args.root_dir, "statements"),
        os.path.join(args.root_dir, "node_dict.pkl"),
        os.path.join(args.root_dir, "node_attr_dict.pkl"),
        os.path.join(args.root_dir, "edge_attr_dict.pkl"),
        rename=True)
    test_dataset = FormulaGraphDataset(
        os.path.join(args.root_dir, "test"), "test",
        os.path.join(args.root_dir, "statements"),
        os.path.join(args.root_dir, "node_dict.pkl"),
        os.path.join(args.root_dir, "node_attr_dict.pkl"),
        os.path.join(args.root_dir, "edge_attr_dict.pkl"),
        rename=True)

    loader_kw = dict(batch_size=args.batch_size, follow_batch=["x_s", "x_t"],
                     num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kw)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kw)
    recorder.info("Data loaded.")

    scaler = torch.cuda.amp.GradScaler() if args.device != "cpu" else None
    history = {f"{p}_{m}": [] for p in ("train", "valid") for m in ("loss", "acc", "f1", "recall", "precision")}
    best_epochs = []
    n_best = 5

    for epoch in range(1, args.epochs + 1):
        recorder.info(f"--- Epoch {epoch}, lr={optimizer.param_groups[0]['lr']:.6f} ---")
        tr_loss, tr_acc, tr_f1, tr_rec, tr_prec = train(epoch, train_loader, model, optimizer, device, recorder, scaler=scaler)
        vl_loss, vl_acc, vl_f1, vl_rec, vl_prec = valid(epoch, valid_loader, model, device, recorder)

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc), ("train_f1", tr_f1),
                      ("train_recall", tr_rec), ("train_precision", tr_prec),
                      ("valid_loss", vl_loss), ("valid_acc", vl_acc), ("valid_f1", vl_f1),
                      ("valid_recall", vl_rec), ("valid_precision", vl_prec)]:
            history[k].append(v)

        best_epochs.append({
            "epoch": epoch,
            "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            "valid_loss": vl_loss,
        })
        best_epochs.sort(key=lambda x: x["valid_loss"])
        best_epochs = best_epochs[:n_best]
        lr_scheduler.step(vl_loss)

    torch.save(best_epochs, os.path.join(save_dir, "top5_checkpoints.pt"))

    averaged_state = {}
    for key in best_epochs[0]["state_dict"]:
        params = [d["state_dict"][key] for d in best_epochs]
        if params[0].dtype in (torch.float32, torch.float64):
            averaged_state[key] = torch.stack(params).mean(0)
        else:
            averaged_state[key] = best_epochs[0]["state_dict"][key]
    torch.save(averaged_state, os.path.join(save_dir, "averaged_top5.pt"))

    recorder.info("=" * 60)
    recorder.info("TESTING 6 MODELS")
    results = {
        "experiment": args.exp_name,
        "dataset": "MPTP",
        "config": {k: v for k, v in vars(args).items() if k != "model_save"},
        "models": [],
    }

    for rank, ckpt in enumerate(best_epochs):
        model.load_state_dict(ckpt["state_dict"])
        loss, acc, f1, rec, prec = test(test_loader, model, device, recorder)
        entry = {
            "type": f"top{rank+1}_epoch{ckpt['epoch']}",
            "valid_loss": round(ckpt["valid_loss"], 6),
            "test_loss": round(loss, 6),
            "test_acc": round(acc, 6),
            "test_f1": round(f1, 6),
            "test_recall": round(rec, 6),
            "test_precision": round(prec, 6),
        }
        results["models"].append(entry)
        recorder.info(f"  {entry['type']}: Acc={acc*100:.2f}% F1={f1*100:.2f}%")

    model.load_state_dict(averaged_state)
    loss, acc, f1, rec, prec = test(test_loader, model, device, recorder)
    entry = {
        "type": "averaged_top5",
        "test_loss": round(loss, 6),
        "test_acc": round(acc, 6),
        "test_f1": round(f1, 6),
        "test_recall": round(rec, 6),
        "test_precision": round(prec, 6),
    }
    results["models"].append(entry)
    recorder.info(f"  averaged_top5: Acc={acc*100:.2f}% F1={f1*100:.2f}%")

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    dump_pickle_file(history, os.path.join(save_dir, "history.pkl"))

    recorder.info(f"All results saved to {save_dir}")
    recorder.info("DONE")

if __name__ == "__main__":
    main()
