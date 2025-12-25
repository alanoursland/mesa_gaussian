"""
Experiment: Interior Gradient Comparison

Does the Lp architecture produce flatter interiors than Linear?

The mesa interpretation claims that Lp aggregation creates regions with
near-zero gradients inside the prototype. This experiment tests that
claim directly by measuring gradient norms.

Key metrics:
- |∇| inside the polytope (should be lower for Lp)
- |∇| at the boundary (should be similar)
- Ratio of interior gradients: Linear / Lp

Run as:
    python exp_gradient_comparison.py
    python exp_gradient_comparison.py --seed 42
    python exp_gradient_comparison.py --p-init 1.0
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lp_norm_layer import LpNormLayer
from experiment_utils import train_model
from polytope_utils import (
    make_random_polytope,
    polytope_membership,
    min_face_slack,
    generate_polytope_dataset,
)


# ============================================================================
# Model Definitions
# ============================================================================

class LpClassifier(nn.Module):
    """Lp-based classifier: Linear -> ReLU -> LpNorm -> Linear"""
    def __init__(self, in_dim: int, hidden_dim: int, p_init: float = 2.0):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = LpNormLayer(hidden_dim, out_dim=1, learn_p=False, p_init=p_init)
        self.l3 = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def internal_scalar(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        return self.l2(x).squeeze(-1)


class LinearClassifier(nn.Module):
    """Linear baseline: Linear -> ReLU -> Linear -> Linear"""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)
        self.l3 = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def internal_scalar(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        return self.l2(x).squeeze(-1)


# ============================================================================
# Gradient Computation
# ============================================================================

def compute_gradient_norms(model, x: np.ndarray, device) -> np.ndarray:
    """Compute |∇ internal_scalar| with respect to input for each point."""
    model.eval()
    x_t = torch.from_numpy(x).float().to(device).requires_grad_(True)
    
    internal = model.internal_scalar(x_t)
    
    # Compute gradients for each point
    grad_norms = []
    for i in range(len(x)):
        model.zero_grad()
        if x_t.grad is not None:
            x_t.grad.zero_()
        internal[i].backward(retain_graph=True)
        grad_norm = x_t.grad[i].norm().item()
        grad_norms.append(grad_norm)
    
    return np.array(grad_norms)


def compute_gradient_norms_batched(model, x: np.ndarray, device) -> np.ndarray:
    """Compute |∇ internal_scalar| using batched jacobian."""
    model.eval()
    x_t = torch.from_numpy(x).float().to(device)
    
    def internal_fn(x_in):
        return model.internal_scalar(x_in)
    
    # Compute jacobian
    jac = torch.autograd.functional.jacobian(internal_fn, x_t)
    # jac shape: (n_points, n_points, in_dim) - diagonal is what we want
    
    # Extract diagonal and compute norms
    grad_norms = []
    for i in range(len(x)):
        grad = jac[i, i, :]
        grad_norms.append(grad.norm().item())
    
    return np.array(grad_norms)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_accuracy(model, x: np.ndarray, y: np.ndarray, device) -> float:
    """Compute classification accuracy."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).float().to(device)
        y_t = torch.from_numpy(y).float().to(device).view(-1)
        logits = model(x_t).view(-1)
        preds = (logits > 0).float()
        acc = (preds == y_t).float().mean().item()
    return acc


# ============================================================================
# Modification
# ============================================================================

def convert_linear_to_positive_weights(model):
    """Flip negative weights in l2 by negating corresponding l1 hyperplanes."""
    with torch.no_grad():
        l2_weight = model.l2.weight.data  # (1, hidden_dim)
        l1_weight = model.l1.weight.data  # (hidden_dim, in_dim)
        l1_bias = model.l1.bias.data      # (hidden_dim,)
        
        negative_mask = l2_weight[0] < 0  # which weights are negative
        
        # Flip the sign of negative l2 weights
        l2_weight[0, negative_mask] = l2_weight[0, negative_mask].abs()
        
        # Negate corresponding l1 rows (flips the half-space)
        l1_weight[negative_mask] = -l1_weight[negative_mask]
        l1_bias[negative_mask] = -l1_bias[negative_mask]
        
        n_flipped = negative_mask.sum().item()
        print(f"Flipped {n_flipped}/{len(negative_mask)} hyperplanes")
    
    return model


# ============================================================================
# Visualization
# ============================================================================

def plot_gradient_comparison(
    lp_grads: np.ndarray,
    linear_grads: np.ndarray,
    distances: np.ndarray,
    save_path: str = None
):
    """Plot gradient norms vs distance for both models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    inside = distances <= 0
    
    # Left: Lp gradients
    ax = axes[0]
    ax.scatter(distances[inside], lp_grads[inside], alpha=0.5, s=10, c='blue', label='Inside')
    ax.scatter(distances[~inside], lp_grads[~inside], alpha=0.5, s=10, c='red', label='Outside')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance (negative = inside)')
    ax.set_ylabel('|∇|')
    ax.set_title('Lp Gradient Norms')
    ax.legend()
    
    # Middle: Linear gradients
    ax = axes[1]
    ax.scatter(distances[inside], linear_grads[inside], alpha=0.5, s=10, c='blue', label='Inside')
    ax.scatter(distances[~inside], linear_grads[~inside], alpha=0.5, s=10, c='red', label='Outside')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance (negative = inside)')
    ax.set_ylabel('|∇|')
    ax.set_title('Linear Gradient Norms')
    ax.legend()
    
    # Right: Direct comparison (inside only)
    ax = axes[2]
    max_val = max(lp_grads[inside].max(), linear_grads[inside].max()) * 1.1
    ax.scatter(lp_grads[inside], linear_grads[inside], alpha=0.5, s=10)
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Lp |∇|')
    ax.set_ylabel('Linear |∇|')
    ax.set_title('Interior Gradient Comparison')
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    # plt.show()


def plot_gradient_histograms(
    lp_grads: np.ndarray,
    linear_grads: np.ndarray,
    distances: np.ndarray,
    save_path: str = None
):
    """Histogram of interior gradient norms."""
    inside = distances <= 0
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = np.linspace(0, max(lp_grads[inside].max(), linear_grads[inside].max()), 30)
    
    ax.hist(lp_grads[inside], bins=bins, alpha=0.5, label=f'Lp (mean={lp_grads[inside].mean():.3f})')
    ax.hist(linear_grads[inside], bins=bins, alpha=0.5, label=f'Linear (mean={linear_grads[inside].mean():.3f})')
    
    ax.set_xlabel('|∇| (interior points)')
    ax.set_ylabel('Count')
    ax.set_title('Interior Gradient Norm Distribution')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    # plt.show()


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    in_dim: int = 2,
    n_faces: int = 6,
    hidden_dim: int = 32,
    n_train: int = 2000,
    n_test: int = 500,
    epochs: int = 200,
    p_init: float = 2.0,
    seed: int = 0,
    out_dir: str = "results/exp_gradient_comparison",
):
    # Setup
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Experiment: Interior Gradient Comparison")
    print("=" * 60)
    print(f"Seed: {seed}")
    print(f"Lp p: {p_init}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Polytope faces: {n_faces}")
    print()

    # Generate polytope and data
    normals, radii = make_random_polytope(in_dim, n_faces, seed=seed)
    x_train, y_train = generate_polytope_dataset(normals, radii, n_train, rng=rng)
    x_test, y_test = generate_polytope_dataset(normals, radii, n_test, rng=rng)
    
    # Compute distances
    distances = -min_face_slack(x_test, normals, radii)  # positive outside
    
    print(f"Train: {n_train} points ({y_train.sum():.0f} inside)")
    print(f"Test:  {n_test} points ({y_test.sum():.0f} inside)")
    print()

    # Train Lp model
    print("--- Training Lp model ---")
    lp_model = LpClassifier(in_dim, hidden_dim, p_init=p_init)
    train_model(lp_model, x_train, y_train, epochs, device=device, rng=rng, model_name="lp")
    
    # Train Linear model
    print("\n--- Training Linear model ---")
    linear_model = LinearClassifier(in_dim, hidden_dim)
    train_model(linear_model, x_train, y_train, epochs, device=device, rng=rng, model_name="linear")

    # Evaluate accuracy
    lp_acc = evaluate_accuracy(lp_model, x_test, y_test.squeeze(), device)
    linear_acc = evaluate_accuracy(linear_model, x_test, y_test.squeeze(), device)

    print()
    print("=" * 60)
    print("Classification Accuracy")
    print("=" * 60)
    print(f"Lp:     {lp_acc*100:.1f}%")
    print(f"Linear: {linear_acc*100:.1f}%")

    # Compute gradient norms
    print()
    print("Computing gradient norms...")
    lp_grads = compute_gradient_norms(lp_model, x_test, device)
    linear_grads = compute_gradient_norms(linear_model, x_test, device)

    # Partition by region
    inside = distances <= 0
    boundary = (distances > 0) & (distances < 0.2)
    outside = distances >= 0.2

    # Report statistics
    print()
    print("=" * 60)
    print("Gradient Norm Statistics")
    print("=" * 60)
    print()
    print(f"{'Region':<15} {'Lp |∇|':>12} {'Linear |∇|':>12} {'Ratio':>10}")
    print("-" * 50)
    
    lp_inside = lp_grads[inside].mean()
    linear_inside = linear_grads[inside].mean()
    ratio_inside = linear_inside / lp_inside if lp_inside > 0 else float('nan')
    print(f"{'Interior':<15} {lp_inside:>12.4f} {linear_inside:>12.4f} {ratio_inside:>10.2f}×")
    
    if boundary.sum() > 0:
        lp_boundary = lp_grads[boundary].mean()
        linear_boundary = linear_grads[boundary].mean()
        ratio_boundary = linear_boundary / lp_boundary if lp_boundary > 0 else float('nan')
        print(f"{'Boundary':<15} {lp_boundary:>12.4f} {linear_boundary:>12.4f} {ratio_boundary:>10.2f}×")
    
    if outside.sum() > 0:
        lp_outside = lp_grads[outside].mean()
        linear_outside = linear_grads[outside].mean()
        ratio_outside = linear_outside / lp_outside if lp_outside > 0 else float('nan')
        print(f"{'Exterior':<15} {lp_outside:>12.4f} {linear_outside:>12.4f} {ratio_outside:>10.2f}×")

    # Assessment
    print()
    print("=" * 60)
    print("Assessment")
    print("=" * 60)
    print()
    print(f"Interior gradient ratio (Linear/Lp): {ratio_inside:.2f}×")
    print()
    
    if ratio_inside > 2.0:
        print("✓ Lp produces significantly flatter interiors")
        print(f"  Linear gradients are {ratio_inside:.1f}× larger inside the polytope")
    elif ratio_inside > 1.2:
        print("~ Lp produces somewhat flatter interiors")
        print(f"  Linear gradients are {ratio_inside:.1f}× larger inside the polytope")
    else:
        print("✗ No significant difference in interior flatness")
        print(f"  Gradient ratio is only {ratio_inside:.2f}×")

    # Visualizations
    print()
    print("Generating visualizations...")
    
    plot_gradient_comparison(
        lp_grads, linear_grads, distances,
        save_path=os.path.join(out_dir, f"gradient_comparison_seed{seed}.png")
    )
    
    plot_gradient_histograms(
        lp_grads, linear_grads, distances,
        save_path=os.path.join(out_dir, f"gradient_histograms_seed{seed}.png")
    )

    # ========================================================================
    # Theory test: Does flipping negative weights create a mesa in Linear?
    # ========================================================================
    print()
    print("=" * 60)
    print("Theory Test: Flipping Negative Weights in Linear Model")
    print("=" * 60)
    print()
    
    # Check current weight distribution
    l2_weight = linear_model.l2.weight.data.cpu().numpy().flatten()
    n_positive = (l2_weight > 0).sum()
    n_negative = (l2_weight < 0).sum()
    print(f"Linear l2 weights: {n_positive} positive, {n_negative} negative")
    
    if n_negative > 0:
        # Make a copy of the linear model
        linear_flipped = LinearClassifier(in_dim, hidden_dim)
        linear_flipped.load_state_dict(linear_model.state_dict())
        linear_flipped.to(device)
        
        # Flip negative weights
        with torch.no_grad():
            l2_w = linear_flipped.l2.weight.data  # (1, hidden_dim)
            l1_w = linear_flipped.l1.weight.data  # (hidden_dim, in_dim)
            l1_b = linear_flipped.l1.bias.data    # (hidden_dim,)
            
            negative_mask = l2_w[0] < 0
            
            # Flip the sign of negative l2 weights
            l2_w[0, negative_mask] = l2_w[0, negative_mask].abs()
            
            # Negate corresponding l1 rows (flips the half-space direction)
            l1_w[negative_mask] = -l1_w[negative_mask]
            l1_b[negative_mask] = -l1_b[negative_mask]
            
            n_flipped = negative_mask.sum().item()
        
        print(f"Flipped {n_flipped} hyperplanes to make all weights positive")
        print()
        
        # Compute gradient norms for flipped model
        print("Computing gradient norms for flipped model...")
        flipped_grads = compute_gradient_norms(linear_flipped, x_test, device)
        
        # Compare statistics
        flipped_inside = flipped_grads[inside].mean()
        flipped_inside_std = flipped_grads[inside].std()
        
        print()
        print(f"{'Model':<20} {'Interior |∇| mean':>18} {'Interior |∇| std':>18}")
        print("-" * 58)
        print(f"{'Lp':<20} {lp_grads[inside].mean():>18.4f} {lp_grads[inside].std():>18.4f}")
        print(f"{'Linear (original)':<20} {linear_grads[inside].mean():>18.4f} {linear_grads[inside].std():>18.4f}")
        print(f"{'Linear (flipped)':<20} {flipped_inside:>18.4f} {flipped_inside_std:>18.4f}")
        
        # Count points with near-zero gradient
        threshold = 1.0
        lp_flat = (lp_grads[inside] < threshold).sum()
        linear_flat = (linear_grads[inside] < threshold).sum()
        flipped_flat = (flipped_grads[inside] < threshold).sum()
        n_inside = inside.sum()
        
        print()
        print(f"Points with |∇| < {threshold}:")
        print(f"  Lp:               {lp_flat}/{n_inside} ({100*lp_flat/n_inside:.1f}%)")
        print(f"  Linear (original): {linear_flat}/{n_inside} ({100*linear_flat/n_inside:.1f}%)")
        print(f"  Linear (flipped):  {flipped_flat}/{n_inside} ({100*flipped_flat/n_inside:.1f}%)")
        
        # Visualize flipped model
        print()
        print("Generating flipped model visualizations...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Histogram comparison
        ax = axes[0]
        bins = np.linspace(0, max(lp_grads[inside].max(), linear_grads[inside].max(), flipped_grads[inside].max()), 30)
        ax.hist(lp_grads[inside], bins=bins, alpha=0.5, label=f'Lp')
        ax.hist(flipped_grads[inside], bins=bins, alpha=0.5, label=f'Linear (flipped)')
        ax.set_xlabel('|∇| (interior points)')
        ax.set_ylabel('Count')
        ax.set_title('Lp vs Flipped Linear')
        ax.legend()
        
        # Original vs flipped linear
        ax = axes[1]
        ax.hist(linear_grads[inside], bins=bins, alpha=0.5, label='Original')
        ax.hist(flipped_grads[inside], bins=bins, alpha=0.5, label='Flipped')
        ax.set_xlabel('|∇| (interior points)')
        ax.set_ylabel('Count')
        ax.set_title('Linear: Original vs Flipped')
        ax.legend()
        
        # Scatter: flipped vs distance
        ax = axes[2]
        ax.scatter(distances[inside], flipped_grads[inside], alpha=0.5, s=10, c='blue', label='Inside')
        ax.scatter(distances[~inside], flipped_grads[~inside], alpha=0.5, s=10, c='red', label='Outside')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Distance (negative = inside)')
        ax.set_ylabel('|∇|')
        ax.set_title('Flipped Linear Gradient Norms')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"flipped_linear_seed{seed}.png"), dpi=150)
        print(f"Saved: {os.path.join(out_dir, f'flipped_linear_seed{seed}.png')}")
        # plt.show()
        
        # Assessment
        print()
        print("=" * 60)
        print("Theory Assessment")
        print("=" * 60)
        print()
        
        if flipped_flat > linear_flat * 2:
            print("✓ THEORY SUPPORTED: Flipping creates mesa structure")
            print(f"  Flat points increased from {linear_flat} to {flipped_flat}")
            print("  Negative weights were causing cancellation, destroying the mesa.")
        elif flipped_flat > linear_flat:
            print("~ PARTIAL SUPPORT: Flipping increases flatness somewhat")
            print(f"  Flat points increased from {linear_flat} to {flipped_flat}")
        else:
            print("✗ THEORY NOT SUPPORTED: Flipping doesn't create mesa")
            print(f"  Flat points: {linear_flat} → {flipped_flat}")
    else:
        print("No negative weights to flip—test not applicable.")

    # ========================================================================
    # Weight Analysis: Comparing weight distributions
    # ========================================================================
    print()
    print("=" * 60)
    print("Weight Analysis: Layer 2 Weight Distributions")
    print("=" * 60)
    print()
    
    # Get weights
    lp_weights = lp_model.l2._raw_w.data.cpu().numpy().flatten()
    linear_weights = linear_model.l2.weight.data.cpu().numpy().flatten()
    
    # Statistics
    print(f"{'Metric':<25} {'Lp':>12} {'Linear':>12}")
    print("-" * 50)
    print(f"{'Mean |w|':<25} {np.abs(lp_weights).mean():>12.4f} {np.abs(linear_weights).mean():>12.4f}")
    print(f"{'Std |w|':<25} {np.abs(lp_weights).std():>12.4f} {np.abs(linear_weights).std():>12.4f}")
    print(f"{'Max |w|':<25} {np.abs(lp_weights).max():>12.4f} {np.abs(linear_weights).max():>12.4f}")
    print(f"{'Min |w|':<25} {np.abs(lp_weights).min():>12.4f} {np.abs(linear_weights).min():>12.4f}")
    
    # Sparsity
    threshold = 0.01
    lp_sparse = (np.abs(lp_weights) < threshold).sum()
    linear_sparse = (np.abs(linear_weights) < threshold).sum()
    print(f"{'Weights |w| < 0.01':<25} {lp_sparse:>12} {linear_sparse:>12}")
    
    threshold = 0.1
    lp_small = (np.abs(lp_weights) < threshold).sum()
    linear_small = (np.abs(linear_weights) < threshold).sum()
    print(f"{'Weights |w| < 0.1':<25} {lp_small:>12} {linear_small:>12}")
    
    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.hist(lp_weights, bins=30, alpha=0.7, label='Lp')
    ax.hist(linear_weights, bins=30, alpha=0.7, label='Linear')
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Count')
    ax.set_title('Layer 2 Weight Distribution')
    ax.legend()
    
    ax = axes[1]
    ax.hist(np.abs(lp_weights), bins=30, alpha=0.7, label='Lp')
    ax.hist(np.abs(linear_weights), bins=30, alpha=0.7, label='Linear')
    ax.set_xlabel('|Weight|')
    ax.set_ylabel('Count')
    ax.set_title('Layer 2 Absolute Weight Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"weight_distribution_seed{seed}.png"), dpi=150)
    print(f"\nSaved: {os.path.join(out_dir, f'weight_distribution_seed{seed}.png')}")
    # plt.show()

    # ========================================================================
    # ReLU Activation Analysis: What's active inside vs outside?
    # ========================================================================
    print()
    print("=" * 60)
    print("ReLU Activation Analysis")
    print("=" * 60)
    print()
    
    with torch.no_grad():
        x_t = torch.from_numpy(x_test).float().to(device)
        
        lp_h = F.relu(lp_model.l1(x_t))
        linear_h = F.relu(linear_model.l1(x_t))
        
        # Fraction of ReLUs active per point
        lp_active = (lp_h > 0).float().mean(dim=1).cpu().numpy()
        linear_active = (linear_h > 0).float().mean(dim=1).cpu().numpy()
    
    print(f"{'Region':<15} {'Lp active %':>15} {'Linear active %':>15}")
    print("-" * 45)
    print(f"{'Inside':<15} {lp_active[inside].mean()*100:>15.1f} {linear_active[inside].mean()*100:>15.1f}")
    print(f"{'Outside':<15} {lp_active[~inside].mean()*100:>15.1f} {linear_active[~inside].mean()*100:>15.1f}")


    # ========================================================================
    # Center Analysis: What's happening at the polytope center?
    # ========================================================================
    print()
    print("=" * 60)
    print("Center Analysis: Neuron Activity at Polytope Center")
    print("=" * 60)
    print()
    
    # Get center of polytope (mean of inside points)
    inside_points = x_test[inside]
    center = inside_points.mean(axis=0, keepdims=True)
    print(f"Polytope center: {center.flatten()}")
    print()
    
    with torch.no_grad():
        center_t = torch.from_numpy(center).float().to(device)
        
        # Layer 1 outputs (pre-ReLU and post-ReLU)
        lp_pre = (lp_model.l1(center_t)).cpu().numpy().flatten()
        lp_post = F.relu(lp_model.l1(center_t)).cpu().numpy().flatten()
        
        linear_pre = (linear_model.l1(center_t)).cpu().numpy().flatten()
        linear_post = F.relu(linear_model.l1(center_t)).cpu().numpy().flatten()
    
    # Get layer 2 weights
    lp_w = lp_model.l2._raw_w.data.cpu().numpy().flatten()
    linear_w = linear_model.l2.weight.data.cpu().numpy().flatten()
    
    # Identify active/inactive neurons
    lp_active_idx = np.where(lp_post > 0)[0]
    lp_inactive_idx = np.where(lp_post == 0)[0]
    
    linear_active_idx = np.where(linear_post > 0)[0]
    linear_inactive_idx = np.where(linear_post == 0)[0]
    
    print("=" * 60)
    print("Lp Model")
    print("=" * 60)
    print(f"\nActive neurons at center: {len(lp_active_idx)}/{len(lp_post)}")
    print(f"{'Index':<8} {'Pre-ReLU':>12} {'Post-ReLU':>12} {'L2 Weight':>12}")
    print("-" * 46)
    for idx in lp_active_idx:
        print(f"{idx:<8} {lp_pre[idx]:>12.4f} {lp_post[idx]:>12.4f} {lp_w[idx]:>12.4f}")
    
    print(f"\nInactive neurons at center: {len(lp_inactive_idx)}/{len(lp_post)}")
    print(f"{'Index':<8} {'Pre-ReLU':>12} {'Post-ReLU':>12} {'L2 Weight':>12}")
    print("-" * 46)
    for idx in lp_inactive_idx[:10]:  # First 10 only
        print(f"{idx:<8} {lp_pre[idx]:>12.4f} {lp_post[idx]:>12.4f} {lp_w[idx]:>12.4f}")
    if len(lp_inactive_idx) > 10:
        print(f"... and {len(lp_inactive_idx) - 10} more")
    
    print()
    print("=" * 60)
    print("Linear Model")
    print("=" * 60)
    print(f"\nActive neurons at center: {len(linear_active_idx)}/{len(linear_post)}")
    print(f"{'Index':<8} {'Pre-ReLU':>12} {'Post-ReLU':>12} {'L2 Weight':>12}")
    print("-" * 46)
    for idx in linear_active_idx:
        print(f"{idx:<8} {linear_pre[idx]:>12.4f} {linear_post[idx]:>12.4f} {linear_w[idx]:>12.4f}")
    
    print(f"\nInactive neurons at center: {len(linear_inactive_idx)}/{len(linear_post)}")
    print(f"{'Index':<8} {'Pre-ReLU':>12} {'Post-ReLU':>12} {'L2 Weight':>12}")
    print("-" * 46)
    for idx in linear_inactive_idx[:10]:  # First 10 only
        print(f"{idx:<8} {linear_pre[idx]:>12.4f} {linear_post[idx]:>12.4f} {linear_w[idx]:>12.4f}")
    if len(linear_inactive_idx) > 10:
        print(f"... and {len(linear_inactive_idx) - 10} more")
    
    # Summary statistics
    print()
    print("=" * 60)
    print("Summary: Weights for Active vs Inactive Neurons")
    print("=" * 60)
    print()
    print(f"{'Model':<15} {'Active |w| mean':>18} {'Inactive |w| mean':>18}")
    print("-" * 55)
    
    lp_active_w_mean = np.abs(lp_w[lp_active_idx]).mean() if len(lp_active_idx) > 0 else 0
    lp_inactive_w_mean = np.abs(lp_w[lp_inactive_idx]).mean() if len(lp_inactive_idx) > 0 else 0
    print(f"{'Lp':<15} {lp_active_w_mean:>18.4f} {lp_inactive_w_mean:>18.4f}")
    
    linear_active_w_mean = np.abs(linear_w[linear_active_idx]).mean() if len(linear_active_idx) > 0 else 0
    linear_inactive_w_mean = np.abs(linear_w[linear_inactive_idx]).mean() if len(linear_inactive_idx) > 0 else 0
    print(f"{'Linear':<15} {linear_active_w_mean:>18.4f} {linear_inactive_w_mean:>18.4f}")

    print()
    print("=" * 60)
    print("Done")
    print("=" * 60)
    
    return {
        'lp_acc': lp_acc,
        'linear_acc': linear_acc,
        'lp_inside_grad': lp_inside,
        'linear_inside_grad': linear_inside,
        'ratio': ratio_inside,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Interior gradient comparison")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-init", type=float, default=2.0)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-faces", type=int, default=6)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--out-dir", type=str, default="results/exp_gradient_comparison")
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