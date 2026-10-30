import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import argparse
import random
import numpy as np
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainer import train
from model import PremiseSelectionModel
from dataset import FormulaGraphDataset
from utils import set_recorder, dump_pickle_file, py_plot

def hyper_parameters():
    params = argparse.ArgumentParser()
    params.add_argument(
        "--model_save",
        type=str,
        default="model_save",
        help="the directory to save models")
    params.add_argument(
        "--root_dir",
        type=str,
        default="dataset",
        help="the directory to save data")
    params.add_argument("--node_out_channels",
                        type=int,
                        default=128,
                        help="the dimension of node")
    params.add_argument("--edge_attr_out_channels",
                        type=int,
                        default=32,
                        help="the dimension of edge attr")
    params.add_argument("--layers",
                        type=int,
                        default=1,
                        help="the number of message passing steps")
    params.add_argument("--device",
                        type=str,

                        default='cuda:0',
                        help="device name {cpu, cuda:0, cuda:1}")
    params.add_argument("--epochs",
                        type=int,
                        default=30,
                        help='Number of training episodes')
    params.add_argument("--lr",
                        type=float,
                        default=0.0005,
                        help="Initial learning rate for Adam")
    params.add_argument("--weight_decay",
                        type=float,
                        default=1e-4,
                        help="L2 normalization penality")
    params.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch Size")
    params.add_argument("--seed",
                        type=int,
                        default=24,
                        help="Random seed")
    args = params.parse_args()
    return args

def main():
    args = hyper_parameters()
    device = torch.device(args.device)

    if not os.path.exists(args.model_save):
        os.makedirs(args.model_save)
    recorder = set_recorder("TWNN",
                            os.path.join(args.model_save, "record.log"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device != "cpu":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    params_info = ''
    for key, value in vars(args).items():
        params_info += '\n{}: {}'.format(key, value)
    recorder.info(params_info)

    model = PremiseSelectionModel(793,
                                  args.node_out_channels,
                                  args.layers).to(device)

    optimizer = Adam(params=[{'params': model.parameters()}], lr=args.lr,
                     weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer)

    recorder.info('------DATA LOADING------')

    train_dataset = FormulaGraphDataset(os.path.join(args.root_dir, "train"),
                                        "train",
                                        os.path.join(args.root_dir,
                                                     "statements"),
                                        os.path.join(args.root_dir,
                                                     "node_dict.pkl"),
                                        os.path.join(args.root_dir,
                                                     "node_attr_dict.pkl"),
                                        os.path.join(args.root_dir,
                                                     "edge_attr_dict.pkl"),
                                        rename=True)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              generator=generator,
                              follow_batch=["x_s", "x_t"])

    recorder.info(f'------DATA LOADED: {len(train_dataset)} samples------')

    recorder.info('------PROCESS START------')
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(1, args.epochs + 1):
        recorder.info('------learning rate is {}------'.format(
            optimizer.param_groups[0]["lr"]))

        train_loss, train_acc = train(epoch, train_loader, model, optimizer, device, recorder)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        lr_scheduler.step(train_loss)

    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "selection": "fixed final epoch; no validation split available"},
               os.path.join(args.model_save, "final_epoch.pt"))

    recorder.info('------saved fixed final epoch {}------'.format(args.epochs))

    dump_pickle_file(history, os.path.join(args.model_save, "history.pkl"))

    recorder.info('------PROCESS FINISH------')

if __name__ == "__main__":
    main()
