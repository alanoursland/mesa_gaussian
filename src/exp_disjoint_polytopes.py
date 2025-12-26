"""
Experiment: Disjoint Polytope Composition

How do networks represent "inside A OR inside B" where A and B are disjoint polytopes?

This experiment tests whether:
1. The network learns separate internal representations for each polytope
2. Mesa structure emerges for each component
3. Lp vs Linear aggregation affects the learned decomposition

Architecture:
    Linear(2, hidden) → ReLU → Aggregation(hidden, 2) → combination → output

The middle layer has 2 outputs, allowing the network to potentially
learn one representation per polytope.

Run as:
    python exp_disjoint_polytopes.py
    python exp_disjoint_polytopes.py --seed 42
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
from polytope_utils import make_random_polytope, polytope_membership, min_face_slack


# ============================================================================
# Polytope Generation
# ============================================================================

def make_disjoint_polytopes(
    n_faces: int = 5,
    center1: tuple = (-0.7, 0.0),
    center2: tuple = (0.7, 0.0),
    scale: float = 0.5,
    seed: int = 0,
) -> tuple:
    """
    Generate two disjoint convex polytopes.
    
    Returns:
        normals1, radii1: First polytope
        normals2, radii2: Second polytope
    """
    rng = np.random.default_rng(seed)
    
    # Generate base polytopes centered at origin
    normals1, radii1 = make_random_polytope(2, n_faces, seed=seed)
    normals2, radii2 = make_random_polytope(2, n_faces, seed=seed + 1000)
    
    # Scale down radii
    radii1 = radii1 * scale
    radii2 = radii2 * scale
    
    # Shift polytopes by adjusting radii
    # For a polytope defined by n·x ≤ r, shifting center by c gives n·(x-c) ≤ r
    # which is n·x ≤ r + n·c
    center1 = np.array(center1)
    center2 = np.array(center2)
    
    radii1 = radii1 + normals1 @ center1
    radii2 = radii2 + normals2 @ center2
    
    return normals1, radii1, normals2, radii2


def disjoint_membership(x: np.ndarray, normals1, radii1, normals2, radii2) -> np.ndarray:
    """Return 1 if inside either polytope, 0 otherwise."""
    inside1 = polytope_membership(x, normals1, radii1)
    inside2 = polytope_membership(x, normals2, radii2)
    return (inside1 | inside2).astype(np.float32)


def disjoint_min_slack(x: np.ndarray, normals1, radii1, normals2, radii2) -> tuple:
    """Return min face slack for each polytope separately."""
    slack1 = min_face_slack(x, normals1, radii1)
    slack2 = min_face_slack(x, normals2, radii2)
    return slack1, slack2


def generate_disjoint_dataset(
    normals1, radii1, normals2, radii2,
    n_points: int,
    rng: np.random.Generator,
    bounds: float = 1.5,
) -> tuple:
    """Generate dataset for disjoint polytope classification."""
    x = rng.uniform(-bounds, bounds, size=(n_points, 2)).astype(np.float32)
    y = disjoint_membership(x, normals1, radii1, normals2, radii2)
    return x, y.reshape(-1, 1)


# ============================================================================
# Model Definitions
# ============================================================================

class DisjointLpClassifier(nn.Module):
    """
    Lp-based classifier with 2 internal outputs for disjoint polytopes.
    
    Architecture: Linear → ReLU → LpNorm(out=2) → ReLU → Linear → output
    
    The two Lp outputs can potentially specialize to each polytope.
    Final combination learns how to merge them for classification.
    """
    def __init__(self, in_dim: int, hidden_dim: int, p_init: float = 1.0):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = LpNormLayer(hidden_dim, out_dim=2, learn_p=False, p_init=p_init)
        self.l3 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def internal_scalars(self, x: torch.Tensor) -> torch.Tensor:
        """Return both Lp outputs (before final linear)."""
        x = F.relu(self.l1(x))
        return self.l2(x)  # Shape: (batch, 2)


class DisjointLinearClassifier(nn.Module):
    """
    Linear baseline with 2 internal outputs for disjoint polytopes.
    
    Architecture: Linear → ReLU → Linear(out=2) → ReLU → Linear → output
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 2)
        self.l3 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def internal_scalars(self, x: torch.Tensor) -> torch.Tensor:
        """Return both linear outputs (before ReLU and final linear)."""
        x = F.relu(self.l1(x))
        return self.l2(x)  # Shape: (batch, 2)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, x, y, device):
    """Compute accuracy and loss."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).float().to(device)
        y_t = torch.from_numpy(y).float().to(device).view(-1, 1)
        logits = model(x_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_t).item()
        preds = (logits > 0).float()
        acc = (preds == y_t).float().mean().item()
    return acc, loss


def compute_internal_scalars(model, x, device):
    """Get both internal scalar values for all points."""
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x).float().to(device)
        return model.internal_scalars(x_t).cpu().numpy()


# ============================================================================
# Visualization
# ============================================================================

def plot_decision_boundary(
    model, normals1, radii1, normals2, radii2, device,
    xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), resolution=200,
    save_path=None
):
    """Plot model decision boundary vs true disjoint polytopes."""
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
    true_labels = disjoint_membership(grid, normals1, radii1, normals2, radii2).reshape(xx.shape)
    
    # Individual polytope boundaries
    inside1 = polytope_membership(grid, normals1, radii1).reshape(xx.shape)
    inside2 = polytope_membership(grid, normals2, radii2).reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Model decision
    ax = axes[0]
    ax.contourf(xx, yy, (logits > 0).astype(float), levels=[-0.5, 0.5, 1.5], 
                colors=['#ffcccc', '#ccffcc'], alpha=0.7)
    ax.contour(xx, yy, logits, levels=[0], colors='red', linewidths=2)
    ax.contour(xx, yy, inside1, levels=[0.5], colors='blue', 
               linewidths=2, linestyles='--', label='Polytope 1')
    ax.contour(xx, yy, inside2, levels=[0.5], colors='orange', 
               linewidths=2, linestyles='--', label='Polytope 2')
    ax.set_title('Decision Boundary\n(red=model, dashed=true polytopes)')
    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    ax.set_aspect('equal')
    
    # Right: True polytopes
    ax = axes[1]
    ax.contourf(xx, yy, true_labels, levels=[-0.5, 0.5, 1.5],
                colors=['#ffcccc', '#ccffcc'], alpha=0.7)
    ax.contour(xx, yy, inside1, levels=[0.5], colors='blue', linewidths=2, label='Polytope 1')
    ax.contour(xx, yy, inside2, levels=[0.5], colors='orange', linewidths=2, label='Polytope 2')
    ax.set_title('True Disjoint Polytopes')
    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_internal_surfaces(
    model, normals1, radii1, normals2, radii2, device,
    xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), resolution=200,
    save_path=None
):
    """Plot both internal scalar surfaces."""
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], resolution),
        np.linspace(ylim[0], ylim[1], resolution)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    # Internal scalars
    internals = compute_internal_scalars(model, grid, device)
    internal1 = internals[:, 0].reshape(xx.shape)
    internal2 = internals[:, 1].reshape(xx.shape)
    
    # True polytope boundaries
    inside1 = polytope_membership(grid, normals1, radii1).reshape(xx.shape)
    inside2 = polytope_membership(grid, normals2, radii2).reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Left: Internal scalar 1
    ax = axes[0]
    im = ax.contourf(xx, yy, internal1, levels=50, cmap='viridis')
    ax.contour(xx, yy, inside1, levels=[0.5], colors='blue', linewidths=2, linestyles='--')
    ax.contour(xx, yy, inside2, levels=[0.5], colors='orange', linewidths=2, linestyles='--')
    plt.colorbar(im, ax=ax)
    ax.set_title('Internal Scalar 1')
    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    ax.set_aspect('equal')
    
    # Middle: Internal scalar 2
    ax = axes[1]
    im = ax.contourf(xx, yy, internal2, levels=50, cmap='viridis')
    ax.contour(xx, yy, inside1, levels=[0.5], colors='blue', linewidths=2, linestyles='--')
    ax.contour(xx, yy, inside2, levels=[0.5], colors='orange', linewidths=2, linestyles='--')
    plt.colorbar(im, ax=ax)
    ax.set_title('Internal Scalar 2')
    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    ax.set_aspect('equal')
    
    # Right: Minimum of the two (effective distance)
    ax = axes[2]
    internal_min = np.minimum(internal1, internal2)
    im = ax.contourf(xx, yy, internal_min, levels=50, cmap='viridis')
    ax.contour(xx, yy, inside1, levels=[0.5], colors='blue', linewidths=2, linestyles='--')
    ax.contour(xx, yy, inside2, levels=[0.5], colors='orange', linewidths=2, linestyles='--')
    plt.colorbar(im, ax=ax)
    ax.set_title('Min(Internal 1, Internal 2)')
    ax.set_xlabel('x₀')
    ax.set_ylabel('x₁')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_specialization_analysis(
    model, x_test, normals1, radii1, normals2, radii2, device,
    save_path=None
):
    """Analyze whether each internal scalar specializes to one polytope."""
    internals = compute_internal_scalars(model, x_test, device)
    
    # Compute which polytope each point belongs to
    slack1 = min_face_slack(x_test, normals1, radii1)
    slack2 = min_face_slack(x_test, normals2, radii2)
    
    inside1 = slack1 >= 0
    inside2 = slack2 >= 0
    outside_both = ~(inside1 | inside2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Left: Internal 1 vs distance to polytope 1
    ax = axes[0]
    dist1 = -slack1  # Positive outside
    colors = np.where(inside1, 'blue', np.where(inside2, 'orange', 'gray'))
    ax.scatter(dist1, internals[:, 0], c=colors, alpha=0.5, s=10)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance from Polytope 1')
    ax.set_ylabel('Internal Scalar 1')
    ax.set_title('Internal 1 vs Polytope 1 Distance')
    
    # Middle: Internal 2 vs distance to polytope 2
    ax = axes[1]
    dist2 = -slack2
    ax.scatter(dist2, internals[:, 1], c=colors, alpha=0.5, s=10)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance from Polytope 2')
    ax.set_ylabel('Internal Scalar 2')
    ax.set_title('Internal 2 vs Polytope 2 Distance')
    
    # Right: Internal 1 vs Internal 2 (colored by region)
    ax = axes[2]
    ax.scatter(internals[inside1, 0], internals[inside1, 1], 
               c='blue', alpha=0.5, s=10, label='Inside Polytope 1')
    ax.scatter(internals[inside2, 0], internals[inside2, 1], 
               c='orange', alpha=0.5, s=10, label='Inside Polytope 2')
    ax.scatter(internals[outside_both, 0], internals[outside_both, 1], 
               c='gray', alpha=0.3, s=10, label='Outside both')
    ax.set_xlabel('Internal Scalar 1')
    ax.set_ylabel('Internal Scalar 2')
    ax.set_title('Internal Space (colored by region)')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()
    
    # Return specialization statistics
    return {
        'corr_int1_dist1': np.corrcoef(dist1, internals[:, 0])[0, 1],
        'corr_int1_dist2': np.corrcoef(dist2, internals[:, 0])[0, 1],
        'corr_int2_dist1': np.corrcoef(dist1, internals[:, 1])[0, 1],
        'corr_int2_dist2': np.corrcoef(dist2, internals[:, 1])[0, 1],
    }


def plot_mesa_assessment(
    model, x_test, normals1, radii1, normals2, radii2, device,
    save_path=None
):
    """Assess mesa structure for each internal scalar within each polytope."""
    internals = compute_internal_scalars(model, x_test, device)
    
    slack1 = min_face_slack(x_test, normals1, radii1)
    slack2 = min_face_slack(x_test, normals2, radii2)
    
    inside1 = slack1 >= 0
    inside2 = slack2 >= 0
    outside_both = ~(inside1 | inside2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top left: Internal 1 distribution by region
    ax = axes[0, 0]
    ax.hist(internals[inside1, 0], bins=20, alpha=0.5, label='Inside P1', color='blue')
    ax.hist(internals[inside2, 0], bins=20, alpha=0.5, label='Inside P2', color='orange')
    ax.hist(internals[outside_both, 0], bins=20, alpha=0.3, label='Outside', color='gray')
    ax.set_xlabel('Internal Scalar 1')
    ax.set_ylabel('Count')
    ax.set_title('Internal 1 Distribution by Region')
    ax.legend()
    
    # Top right: Internal 2 distribution by region
    ax = axes[0, 1]
    ax.hist(internals[inside1, 1], bins=20, alpha=0.5, label='Inside P1', color='blue')
    ax.hist(internals[inside2, 1], bins=20, alpha=0.5, label='Inside P2', color='orange')
    ax.hist(internals[outside_both, 1], bins=20, alpha=0.3, label='Outside', color='gray')
    ax.set_xlabel('Internal Scalar 2')
    ax.set_ylabel('Count')
    ax.set_title('Internal 2 Distribution by Region')
    ax.legend()
    
    # Bottom: Statistics table as text
    ax = axes[1, 0]
    ax.axis('off')
    
    stats_lines = [
        "Mesa Statistics",
        "=" * 50,
        "",
        f"{'Region':<20} {'Int1 mean':>12} {'Int1 std':>12}",
        "-" * 50,
        f"{'Inside Polytope 1':<20} {internals[inside1, 0].mean():>12.3f} {internals[inside1, 0].std():>12.3f}",
        f"{'Inside Polytope 2':<20} {internals[inside2, 0].mean():>12.3f} {internals[inside2, 0].std():>12.3f}",
        f"{'Outside both':<20} {internals[outside_both, 0].mean():>12.3f} {internals[outside_both, 0].std():>12.3f}",
        "",
        f"{'Region':<20} {'Int2 mean':>12} {'Int2 std':>12}",
        "-" * 50,
        f"{'Inside Polytope 1':<20} {internals[inside1, 1].mean():>12.3f} {internals[inside1, 1].std():>12.3f}",
        f"{'Inside Polytope 2':<20} {internals[inside2, 1].mean():>12.3f} {internals[inside2, 1].std():>12.3f}",
        f"{'Outside both':<20} {internals[outside_both, 1].mean():>12.3f} {internals[outside_both, 1].std():>12.3f}",
    ]
    ax.text(0.05, 0.95, '\n'.join(stats_lines), transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top')
    
    # Bottom right: Specialization summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Determine specialization
    int1_p1_mean = internals[inside1, 0].mean()
    int1_p2_mean = internals[inside2, 0].mean()
    int2_p1_mean = internals[inside1, 1].mean()
    int2_p2_mean = internals[inside2, 1].mean()
    
    spec_lines = [
        "Specialization Analysis",
        "=" * 50,
        "",
        "If specialized, one internal should be low for one polytope",
        "and high for the other.",
        "",
        f"Internal 1: P1={int1_p1_mean:.2f}, P2={int1_p2_mean:.2f}",
        f"Internal 2: P1={int2_p1_mean:.2f}, P2={int2_p2_mean:.2f}",
        "",
    ]
    
    # Check for specialization pattern
    if int1_p1_mean < int1_p2_mean and int2_p2_mean < int2_p1_mean:
        spec_lines.append("✓ SPECIALIZED: Int1→P1, Int2→P2")
    elif int1_p2_mean < int1_p1_mean and int2_p1_mean < int2_p2_mean:
        spec_lines.append("✓ SPECIALIZED: Int1→P2, Int2→P1")
    elif int1_p1_mean < int1_p2_mean and int2_p1_mean < int2_p2_mean:
        spec_lines.append("~ BOTH LOW FOR P1 (partial specialization)")
    elif int1_p2_mean < int1_p1_mean and int2_p2_mean < int2_p1_mean:
        spec_lines.append("~ BOTH LOW FOR P2 (partial specialization)")
    else:
        spec_lines.append("✗ NO CLEAR SPECIALIZATION")
    
    ax.text(0.05, 0.95, '\n'.join(spec_lines), transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    n_faces: int = 5,
    hidden_dim: int = 32,
    n_train: int = 2000,
    n_test: int = 500,
    epochs: int = 300,
    p_init: float = 1.0,
    seed: int = 0,
    out_dir: str = "results/exp_disjoint_polytopes",
):
    # Setup
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("Experiment: Disjoint Polytope Composition")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"p: {p_init}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Polytope faces: {n_faces}")
    print()
    
    # Generate disjoint polytopes
    normals1, radii1, normals2, radii2 = make_disjoint_polytopes(
        n_faces=n_faces, seed=seed
    )
    
    # Generate data
    x_train, y_train = generate_disjoint_dataset(
        normals1, radii1, normals2, radii2, n_train, rng
    )
    x_test, y_test = generate_disjoint_dataset(
        normals1, radii1, normals2, radii2, n_test, rng
    )
    
    # Count points in each region
    inside1_train = polytope_membership(x_train, normals1, radii1).sum()
    inside2_train = polytope_membership(x_train, normals2, radii2).sum()
    inside1_test = polytope_membership(x_test, normals1, radii1).sum()
    inside2_test = polytope_membership(x_test, normals2, radii2).sum()
    
    print(f"Train: {n_train} points (P1: {inside1_train}, P2: {inside2_train}, total inside: {y_train.sum():.0f})")
    print(f"Test:  {n_test} points (P1: {inside1_test}, P2: {inside2_test}, total inside: {y_test.sum():.0f})")
    print()
    
    results = {}
    
    # ========================================================================
    # Train Lp model
    # ========================================================================
    print("=" * 70)
    print(f"Training Lp Model (p={p_init})")
    print("=" * 70)
    
    torch.manual_seed(seed)
    lp_model = DisjointLpClassifier(2, hidden_dim, p_init=p_init)
    train_model(lp_model, x_train, y_train, epochs, device=device, rng=rng, model_name="lp")
    
    lp_acc, lp_loss = evaluate_model(lp_model, x_test, y_test, device)
    print(f"\nLp Test Accuracy: {lp_acc*100:.1f}%")
    
    # Visualizations for Lp
    print("\nGenerating Lp visualizations...")
    
    plot_decision_boundary(
        lp_model, normals1, radii1, normals2, radii2, device,
        save_path=os.path.join(out_dir, f"lp_decision_boundary_seed{seed}.png")
    )
    
    plot_internal_surfaces(
        lp_model, normals1, radii1, normals2, radii2, device,
        save_path=os.path.join(out_dir, f"lp_internal_surfaces_seed{seed}.png")
    )
    
    lp_spec = plot_specialization_analysis(
        lp_model, x_test, normals1, radii1, normals2, radii2, device,
        save_path=os.path.join(out_dir, f"lp_specialization_seed{seed}.png")
    )
    
    plot_mesa_assessment(
        lp_model, x_test, normals1, radii1, normals2, radii2, device,
        save_path=os.path.join(out_dir, f"lp_mesa_assessment_seed{seed}.png")
    )
    
    results['Lp'] = {
        'acc': lp_acc,
        'specialization': lp_spec,
    }
    
    # ========================================================================
    # Train Linear model
    # ========================================================================
    print()
    print("=" * 70)
    print("Training Linear Model")
    print("=" * 70)
    
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    linear_model = DisjointLinearClassifier(2, hidden_dim)
    train_model(linear_model, x_train, y_train, epochs, device=device, rng=rng, model_name="linear")
    
    linear_acc, linear_loss = evaluate_model(linear_model, x_test, y_test, device)
    print(f"\nLinear Test Accuracy: {linear_acc*100:.1f}%")
    
    # Visualizations for Linear
    print("\nGenerating Linear visualizations...")
    
    plot_decision_boundary(
        linear_model, normals1, radii1, normals2, radii2, device,
        save_path=os.path.join(out_dir, f"linear_decision_boundary_seed{seed}.png")
    )
    
    plot_internal_surfaces(
        linear_model, normals1, radii1, normals2, radii2, device,
        save_path=os.path.join(out_dir, f"linear_internal_surfaces_seed{seed}.png")
    )
    
    linear_spec = plot_specialization_analysis(
        linear_model, x_test, normals1, radii1, normals2, radii2, device,
        save_path=os.path.join(out_dir, f"linear_specialization_seed{seed}.png")
    )
    
    plot_mesa_assessment(
        linear_model, x_test, normals1, radii1, normals2, radii2, device,
        save_path=os.path.join(out_dir, f"linear_mesa_assessment_seed{seed}.png")
    )
    
    results['Linear'] = {
        'acc': linear_acc,
        'specialization': linear_spec,
    }
    
    # ========================================================================
    # Summary
    # ========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"{'Model':<15} {'Accuracy':>10}")
    print("-" * 30)
    print(f"{'Lp':<15} {lp_acc*100:>9.1f}%")
    print(f"{'Linear':<15} {linear_acc*100:>9.1f}%")
    print()
    
    print("Specialization Correlations:")
    print("-" * 50)
    print(f"{'Model':<10} {'Int1-Dist1':>12} {'Int1-Dist2':>12} {'Int2-Dist1':>12} {'Int2-Dist2':>12}")
    print("-" * 50)
    for name, data in results.items():
        spec = data['specialization']
        print(f"{name:<10} {spec['corr_int1_dist1']:>12.3f} {spec['corr_int1_dist2']:>12.3f} "
              f"{spec['corr_int2_dist1']:>12.3f} {spec['corr_int2_dist2']:>12.3f}")
    
    print()
    print("Ideal specialization: Int1 correlates with Dist1, Int2 with Dist2")
    print("(or vice versa)")
    
    print()
    print("=" * 70)
    print("Done")
    print("=" * 70)
    
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Disjoint polytope composition experiment")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-init", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-faces", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out-dir", type=str, default="results/exp_disjoint_polytopes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        n_faces=args.n_faces,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        p_init=args.p_init,
        seed=args.seed,
        out_dir=args.out_dir,
    )