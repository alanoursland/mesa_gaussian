"""
Experiment 2: Mesa Structure Verification

Does Linear → ReLU → Lp learn mesa structure?

This experiment verifies that the Norm architecture produces the expected
geometric structure before comparing to baselines:
    1. Internal scalar ≈ 0 inside the polytope (the mesa)
    2. Internal scalar > 0 outside, increasing with distance
    3. Decision boundary aligns with true polytope boundary

Run as:
    python exp_mesa_structure.py
    python exp_mesa_structure.py --seed 42
    python exp_mesa_structure.py --p-init 1.0
"""
from __future__ import annotations
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lp_norm_layer import LpNormLayer
from experiment_utils import train_model
from polytope_utils import (
    make_random_polytope,
    polytope_membership,
    min_face_slack,
    generate_polytope_dataset,
)


class PolytopeClassifier(nn.Module):
    """Norm-based binary classifier: Linear -> ReLU -> LpNorm -> Linear head."""
    def __init__(self, in_dim: int, hidden_dim: int, p_init: float = 2.0):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)
        self.l3 = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def internal_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """Return the Lp norm output (before final linear)."""
        x = F.relu(self.l1(x))
        return self.l2(x).squeeze(-1)


def evaluate_model(model, x, y, device):
    """Compute accuracy and loss."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).float().to(device)
        # CHANGED: Use .view(-1, 1) to ensure shape is (Batch, 1)
        y_t = torch.from_numpy(y).float().to(device).view(-1, 1)
        
        logits = model(x_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_t).item()
        
        preds = (logits > 0).float()
        acc = (preds == y_t).float().mean().item()
    return acc, loss

def compute_internal_scalar(model, x, device):
    """Get internal scalar values for all points."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).float().to(device)
        return model.internal_scalar(x_t).cpu().numpy()


def plot_decision_boundary(
    model, normals, radii, device, 
    xlim=(-2, 2), ylim=(-2, 2), resolution=200,
    save_path=None
):
    """Plot model decision boundary vs true polytope."""
    # Create grid
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], resolution),
        np.linspace(ylim[0], ylim[1], resolution)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    # Model predictions
    model.eval()
    with torch.no_grad():
        grid_t = torch.from_numpy(grid).to(device)
        logits = model(grid_t).cpu().numpy().reshape(xx.shape)
    
    # True membership
    true_labels = polytope_membership(grid, normals, radii).reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Model decision
    ax = axes[0]
    ax.contourf(xx, yy, (logits > 0).astype(float), levels=[-0.5, 0.5, 1.5], 
                colors=['#ffcccc', '#ccffcc'], alpha=0.7)
    ax.contour(xx, yy, logits, levels=[0], colors='red', linewidths=2)
    ax.contour(xx, yy, true_labels, levels=[0.5], colors='black', 
               linewidths=2, linestyles='--')
    ax.set_title('Decision Boundary\n(red=model, black dashed=true)')
    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    ax.set_aspect('equal')
    
    # Right: True polytope
    ax = axes[1]
    ax.contourf(xx, yy, true_labels, levels=[-0.5, 0.5, 1.5],
                colors=['#ffcccc', '#ccffcc'], alpha=0.7)
    ax.contour(xx, yy, true_labels, levels=[0.5], colors='black', linewidths=2)
    ax.set_title('True Polytope')
    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_internal_scalar_surface(
    model, normals, radii, device,
    xlim=(-2, 2), ylim=(-2, 2), resolution=200,
    save_path=None
):
    """Plot internal scalar as heatmap and compare to true distance."""
    # Create grid
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], resolution),
        np.linspace(ylim[0], ylim[1], resolution)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    # Internal scalar
    internal = compute_internal_scalar(model, grid, device).reshape(xx.shape)
    
    # True distance (negative inside, positive outside)
    true_dist = -min_face_slack(grid, normals, radii).reshape(xx.shape)
    true_dist = np.maximum(true_dist, 0)  # Clip to zero inside (like ReLU)
    
    # True polytope boundary
    true_labels = polytope_membership(grid, normals, radii).reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Left: Internal scalar
    ax = axes[0]
    im = ax.contourf(xx, yy, internal, levels=50, cmap='viridis')
    ax.contour(xx, yy, true_labels, levels=[0.5], colors='white', 
               linewidths=2, linestyles='--')
    plt.colorbar(im, ax=ax)
    ax.set_title('Internal Scalar (Lp output)')
    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    ax.set_aspect('equal')
    
    # Middle: True distance
    ax = axes[1]
    im = ax.contourf(xx, yy, true_dist, levels=50, cmap='viridis')
    ax.contour(xx, yy, true_labels, levels=[0.5], colors='white',
               linewidths=2, linestyles='--')
    plt.colorbar(im, ax=ax)
    ax.set_title('True Distance from Polytope')
    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    ax.set_aspect('equal')
    
    # Right: Difference / correlation
    ax = axes[2]
    # Scatter plot: true distance vs internal scalar
    mask = true_dist.ravel() < 2  # Focus on reasonable range
    ax.scatter(true_dist.ravel()[mask], internal.ravel()[mask], 
               alpha=0.1, s=1)
    ax.set_xlabel('True Distance')
    ax.set_ylabel('Internal Scalar')
    ax.set_title('Correlation')
    
    # Compute correlation
    corr = np.corrcoef(true_dist.ravel(), internal.ravel())[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_internal_vs_distance(
    model, x_test, normals, radii, device,
    save_path=None
):
    """Scatter plot of internal scalar vs true distance for test points."""
    # Get internal scalar
    internal = compute_internal_scalar(model, x_test, device)
    
    # True distance (positive outside, negative inside)
    slack = min_face_slack(x_test, normals, radii)
    true_dist = -slack  # Positive outside
    
    # Membership
    inside = slack >= 0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(true_dist[inside], internal[inside], 
               alpha=0.5, s=10, c='blue', label='Inside')
    ax.scatter(true_dist[~inside], internal[~inside],
               alpha=0.5, s=10, c='red', label='Outside')
    
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('True Signed Distance (negative=inside)')
    ax.set_ylabel('Internal Scalar')
    ax.set_title('Internal Scalar vs True Distance')
    ax.legend()
    
    # Stats
    inside_mean = internal[inside].mean()
    outside_mean = internal[~inside].mean()
    ax.text(0.95, 0.95, 
            f'Inside mean: {inside_mean:.3f}\nOutside mean: {outside_mean:.3f}',
            transform=ax.transAxes, verticalalignment='top', 
            horizontalalignment='right', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
    
    return inside_mean, outside_mean


def run_experiment(
    in_dim: int = 2,
    n_faces: int = 6,
    hidden_dim: int = 32,
    n_train: int = 2000,
    n_test: int = 500,
    epochs: int = 200,
    p_init: float = 2.0,
    seed: int = 0,
    out_dir: str = "results/exp_mesa_structure",
):
    # Setup
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("Experiment 2: Mesa Structure Verification")
    print("=" * 60)
    print(f"Seed: {seed}")
    print(f"p: {p_init}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Polytope faces: {n_faces}")
    print()
    
    # Generate polytope
    normals, radii = make_random_polytope(in_dim, n_faces, seed=seed)
    
    # Generate data
    x_train, y_train = generate_polytope_dataset(normals, radii, n_train, rng=rng)
    x_test, y_test = generate_polytope_dataset(normals, radii, n_test, rng=rng)
    
    print(f"Train: {n_train} points ({y_train.sum():.0f} inside)")
    print(f"Test:  {n_test} points ({y_test.sum():.0f} inside)")
    print()
    
    # Build and train model
    print("Training...")
    model = PolytopeClassifier(in_dim, hidden_dim, p_init=p_init)
    train_model(
        model, x_train, y_train, epochs,
        device=device, rng=rng, model_name="norm"
    )
    
    # Evaluate
    train_acc, train_loss = evaluate_model(model, x_train, y_train, device)
    test_acc, test_loss = evaluate_model(model, x_test, y_test, device)
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Train accuracy: {train_acc*100:.1f}%")
    print(f"Train loss:     {train_loss:.4f}")
    print(f"Test accuracy:  {test_acc*100:.1f}%")
    print(f"Test loss:      {test_loss:.4f}")
    print()
    
    # Internal scalar statistics
    internal_test = compute_internal_scalar(model, x_test, device)
    slack = min_face_slack(x_test, normals, radii)
    inside_mask = slack >= 0
    
    inside_mean = internal_test[inside_mask].mean()
    inside_std = internal_test[inside_mask].std()
    outside_mean = internal_test[~inside_mask].mean()
    outside_std = internal_test[~inside_mask].std()
    
    print("Internal Scalar Statistics:")
    print(f"  Inside polytope:  {inside_mean:.4f} ± {inside_std:.4f}")
    print(f"  Outside polytope: {outside_mean:.4f} ± {outside_std:.4f}")
    print(f"  Ratio (out/in):   {outside_mean/inside_mean:.2f}×" if inside_mean > 0 else "")
    print()
    
    # Mesa criterion
    inside_std = internal_test[inside_mask].std()
    separation = outside_mean - inside_mean
    
    print()
    print("Mesa Structure Assessment:")
    print(f"  Inside std:      {inside_std:.3f}")
    print(f"  Separation:      {separation:.2f}")
    print(f"  Flatness ratio:  {inside_std/separation:.3f} (lower = flatter)")
    print(f"  Correlation:     {corr:.3f}")
    print()
    
    if corr > 0.9:
        print("✓ STRONG DISTANCE STRUCTURE")
    elif corr > 0.7:
        print("✓ CLEAR DISTANCE STRUCTURE")
    elif corr > 0.5:
        print("~ WEAK DISTANCE STRUCTURE")
    else:
        print("✗ NO DISTANCE STRUCTURE")        
        
    # Visualizations (2D only)
    if in_dim == 2:
        print("Generating visualizations...")
        
        plot_decision_boundary(
            model, normals, radii, device,
            save_path=os.path.join(out_dir, f"decision_boundary_seed{seed}.png")
        )
        
        plot_internal_scalar_surface(
            model, normals, radii, device,
            save_path=os.path.join(out_dir, f"internal_surface_seed{seed}.png")
        )
        
        plot_internal_vs_distance(
            model, x_test, normals, radii, device,
            save_path=os.path.join(out_dir, f"internal_vs_distance_seed{seed}.png")
        )
    
    print("=" * 60)
    print("Done")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="Mesa structure verification")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-init", type=float, default=2.0)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-faces", type=int, default=6)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--out-dir", type=str, default="results/exp_mesa_structure")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        hidden_dim=args.hidden_dim,
        n_faces=args.n_faces,
        epochs=args.epochs,
        p_init=args.p_init,
        seed=args.seed,
        out_dir=args.out_dir,
    )