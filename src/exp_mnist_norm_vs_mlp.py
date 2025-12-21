"""
MNIST Norm-aggregation vs MLP baseline experiment

Run:
    python src/exp_mnist_norm_vs_mlp.py

Produces CSV and plots under `src/results/mnist_norm_vs_mlp/`.
"""
from __future__ import annotations

import argparse
import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from lp_norm_layer import LpNormLayer


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NormModel(nn.Module):
    """Norm-based classifier: Linear -> ReLU -> NormLayer -> Linear head."""
    def __init__(self, input_dim: int, h1: int, h2: int, learn_p: bool = True, p_init: float = 2.0):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, h1)
        self.layer2 = LpNormLayer(h1, h2, learn_p=learn_p, p_init=p_init)
        self.layer3 = nn.Linear(h2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten to (batch, 784)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)  # (batch, h2)
        logits = self.layer3(x)  # (batch, 10)
        return logits
    
    def p_mean(self):
        return self.layer2.p.mean().detach().cpu()
    
    def p_variance(self):
        return self.layer2.p.var(unbiased=False).detach().cpu()


class MLPModel(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten to (batch, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        logits = self.layer3(x)
        return logits


def train_epoch_gpu(model, opt, x_train, y_train, batch_size):
    """Train one epoch with data already on GPU."""
    model.train()
    n = x_train.size(0)
    total_loss = 0.0
    correct = 0

    # Shuffle on GPU
    perm = torch.randperm(n, device=x_train.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        xb = x_train[idx]
        yb = y_train[idx]

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()

        total_loss += float(loss.detach()) * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()

    return total_loss / n, correct / n


def eval_gpu(model, x, y, batch_size):
    """Evaluate with data already on GPU."""
    model.eval()
    n = x.size(0)
    correct = 0

    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size]
            yb = y[i:i+batch_size]
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()

    return correct / n


def load_mnist_to_gpu(data_root: str, device: torch.device, val_split: int = 5000, seed: int = 0):
    """Load MNIST and move entire dataset to GPU."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    os.makedirs(data_root, exist_ok=True)

    train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    # Load full training set to tensors
    x_train_full = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train_full = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    # Load test set
    x_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    # Split train/val with deterministic shuffle
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x_train_full), generator=g)

    train_idx = perm[:-val_split]
    val_idx = perm[-val_split:]

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    # Move to GPU
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    return x_train, y_train, x_val, y_val, x_test, y_test


def run_experiment(cmd_args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    out_dir = cmd_args.out_dir or os.path.join('results', 'mnist_norm_vs_mlp')
    os.makedirs(out_dir, exist_ok=True)

    seeds = cmd_args.seeds
    results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load data once to GPU
        x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_to_gpu(
            cmd_args.data_dir, device, val_split=cmd_args.val_split, seed=seed
        )
        print(f"Data loaded to {device}: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

        # build norm model
        norm_model = NormModel(784, cmd_args.h1, cmd_args.h2, learn_p=cmd_args.learn_p, p_init=cmd_args.p_init).to(device)
        opt_norm = torch.optim.Adam(norm_model.parameters(), lr=cmd_args.lr, weight_decay=cmd_args.weight_decay)

        # build mlp baseline
        mlp_model = MLPModel(784, cmd_args.h1, cmd_args.h2).to(device)
        opt_mlp = torch.optim.Adam(mlp_model.parameters(), lr=cmd_args.lr, weight_decay=cmd_args.weight_decay)

        print(f"Seed {seed}: Norm params={count_parameters(norm_model)}, MLP params={count_parameters(mlp_model)}")

        # train both for epochs, record curves
        norm_train_loss = []
        norm_train_acc = []
        norm_val_acc = []
        norm_p_history = []

        mlp_train_loss = []
        mlp_train_acc = []
        mlp_val_acc = []

        for ep in range(cmd_args.epochs):
            nl, na = train_epoch_gpu(norm_model, opt_norm, x_train, y_train, cmd_args.batch_size)
            mv = eval_gpu(norm_model, x_val, y_val, cmd_args.batch_size)
            norm_train_loss.append(nl)
            norm_train_acc.append(na)
            norm_val_acc.append(mv)

            p_mean = norm_model.p_mean()
            p_var = norm_model.p_variance()
            norm_p_history.append((p_mean, p_var))

            ml, ma = train_epoch_gpu(mlp_model, opt_mlp, x_train, y_train, cmd_args.batch_size)
            mv_mlp = eval_gpu(mlp_model, x_val, y_val, cmd_args.batch_size)
            mlp_train_loss.append(ml)
            mlp_train_acc.append(ma)
            mlp_val_acc.append(mv_mlp)

            print(f"seed={seed} epoch={ep+1}/{cmd_args.epochs} norm_loss={nl:.4f} norm_acc={na:.4f} p_mean={p_mean:.4f} p_var={p_var:.4f} | mlp_loss={ml:.4f} mlp_acc={ma:.4f}")

        # final test accuracy
        test_acc_norm = eval_gpu(norm_model, x_test, y_test, cmd_args.batch_size)
        test_acc_mlp = eval_gpu(mlp_model, x_test, y_test, cmd_args.batch_size)

        print(f"Seed {seed} TEST: norm_acc={test_acc_norm:.4f} mlp_acc={test_acc_mlp:.4f}")

        p_mean_final = norm_model.p_mean()
        p_var_final = norm_model.p_variance()

        results.append({
            'seed': seed,
            'norm_params': count_parameters(norm_model),
            'mlp_params': count_parameters(mlp_model),
            'test_acc_norm': float(test_acc_norm),
            'test_acc_mlp': float(test_acc_mlp),
            'p_mean_final': p_mean_final,
            'p_var_final': p_var_final,
            'norm_train_loss': norm_train_loss,
            'norm_train_acc': norm_train_acc,
            'mlp_train_loss': mlp_train_loss,
            'mlp_train_acc': mlp_train_acc,
        })

    # aggregate across seeds and write CSV and plots
    csv_path = os.path.join(out_dir, f"mnist_norm_vs_mlp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['seed','norm_params','mlp_params','test_acc_norm','test_acc_mlp','p_mean_final','p_var_final'])
        for r in results:
            w.writerow([r['seed'], r['norm_params'], r['mlp_params'], r['test_acc_norm'], r['test_acc_mlp'], r['p_mean_final'], r['p_var_final']])
    print(f"Saved summary CSV: {csv_path}")

    # plots: training curves averaged across seeds
    epochs = cmd_args.epochs
    all_norm_loss = np.array([r['norm_train_loss'] for r in results])
    all_norm_acc = np.array([r['norm_train_acc'] for r in results])
    all_mlp_loss = np.array([r['mlp_train_loss'] for r in results])
    all_mlp_acc = np.array([r['mlp_train_acc'] for r in results])

    plt.figure()
    plt.plot(np.arange(1, epochs+1), all_norm_loss.mean(axis=0), label='norm_loss')
    plt.fill_between(np.arange(1, epochs+1), all_norm_loss.mean(axis=0)-all_norm_loss.std(axis=0), all_norm_loss.mean(axis=0)+all_norm_loss.std(axis=0), alpha=0.2)
    plt.plot(np.arange(1, epochs+1), all_mlp_loss.mean(axis=0), label='mlp_loss')
    plt.fill_between(np.arange(1, epochs+1), all_mlp_loss.mean(axis=0)-all_mlp_loss.std(axis=0), all_mlp_loss.mean(axis=0)+all_mlp_loss.std(axis=0), alpha=0.2)
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.legend()
    plt.title('Train loss')
    plt.savefig(os.path.join(out_dir, 'train_loss.png'))

    plt.figure()
    plt.plot(np.arange(1, epochs+1), all_norm_acc.mean(axis=0), label='norm_acc')
    plt.fill_between(np.arange(1, epochs+1), all_norm_acc.mean(axis=0)-all_norm_acc.std(axis=0), all_norm_acc.mean(axis=0)+all_norm_acc.std(axis=0), alpha=0.2)
    plt.plot(np.arange(1, epochs+1), all_mlp_acc.mean(axis=0), label='mlp_acc')
    plt.fill_between(np.arange(1, epochs+1), all_mlp_acc.mean(axis=0)-all_mlp_acc.std(axis=0), all_mlp_acc.mean(axis=0)+all_mlp_acc.std(axis=0), alpha=0.2)
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    plt.legend()
    plt.title('Train accuracy')
    plt.savefig(os.path.join(out_dir, 'train_acc.png'))

    print('Done. Results and plots saved in', out_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--learn-p', action='store_true', default=True)
    p.add_argument('--no-learn-p', action='store_false', dest='learn_p')
    p.add_argument('--p-init', type=float, default=2.0)
    p.add_argument('--h1', type=int, default=256)
    p.add_argument('--h2', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--out-dir', type=str, default=None)
    p.add_argument('--data-dir', type=str, default=r'E:\ml_datasets',
                   help='directory to store/download datasets (default: E:\\ml_datasets)')
    p.add_argument('--val-split', type=int, default=5000)
    args = p.parse_args()
    run_experiment(args)
