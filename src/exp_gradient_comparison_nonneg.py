"""
Experiment: Non-Negative Weight Constraints for Mesa Structure

Tests whether asymmetric regularization (penalizing only negative weights)
can induce mesa structure by enforcing parts-based representations.

Theoretical basis:
- Lee & Seung (1999) proved that non-negative constraints lead to "parts-based"
  learning where features only add, while signed weights enable "holistic"
  representations where features cancel out.
- Chorowski & Zurada (2015) showed that w >= 0 in neural networks transforms
  hidden neurons into interpretable, specific feature detectors.

The Lp architecture enforces non-cancellation via absolute value. This experiment
tests whether a "soft" version (asymmetric penalty on negative weights) can
achieve the same effect in standard Linear layers.

Penalty: loss += λ * Σ relu(-w)²
- Negative weights are penalized quadratically
- Positive weights are completely ignored
- This pushes negative weights toward zero without shrinking positive weights

Run as:
    python exp_gradient_comparison_nonneg.py
    python exp_gradient_comparison_nonneg.py --neg-lambda 0.1 0.5 1.0
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lp_norm_layer import LpNormLayer
from experiment_utils import train_model
from polytope_utils import (
    make_random_polytope,
    min_face_slack,
    generate_polytope_dataset,
)


# ============================================================================
# Model Definitions
# ============================================================================

class LpClassifier(nn.Module):
    """Lp-based classifier: Linear -> ReLU -> LpNorm -> Linear"""
    
    def __init__(self, in_dim: int, hidden_dim: int, p_init: float = 1.0):
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
    
    def l2_layer_weights(self) -> torch.Tensor:
        return self.l2._raw_w


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
    
    def l2_layer_weights(self) -> torch.Tensor:
        return self.l2.weight


# ============================================================================
# Non-Negative Weight Penalty
# ============================================================================

def negative_weight_penalty(weights: torch.Tensor) -> torch.Tensor:
    """
    Asymmetric L2 penalty on negative weights only.
    
    Following Lee & Seung (1999), non-negative constraints enforce "parts-based"
    representations where features can only add (no cancellation). This is the
    "soft" relaxation of the hard constraint w >= 0.
    
    Penalty = Σ relu(-w)²
    
    - Negative weights: penalized quadratically, pushing toward zero
    - Positive weights: zero penalty, free to grow large
    
    This differs from L1/L2 regularization which penalizes all weights equally,
    shrinking both positive and negative weights toward zero.
    
    References:
        Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by 
        non-negative matrix factorization. Nature, 401(6755), 788-791.
        
        Chorowski, J., & Zurada, J. M. (2015). Learning understandable neural 
        networks with nonnegative weight constraints. IEEE TNNLS, 26(1), 62-69.
    """
    # relu(-w) is positive when w is negative, zero when w is positive
    return torch.sum(F.relu(-weights) ** 2)


# ============================================================================
# Training with Non-Negative Penalty
# ============================================================================

def train_model_with_nonneg_penalty(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    device: torch.device,
    rng: np.random.Generator,
    neg_lambda: float,
    model_name: str = "model",
    batch_size: int = 256,
    lr: float = 1e-3,
    target_acc: float = 0.98,
    patience: int = 50,
) -> list[float]:
    """
    Train model with asymmetric penalty on negative l2 weights.
    
    This implements the "soft" version of non-negative constraints from
    Lee & Seung (1999). The penalty pushes negative weights toward zero
    while leaving positive weights unconstrained.
    """
    model.to(device)
    model.train()
    
    x_t = torch.from_numpy(x_train).float().to(device)
    y_t = torch.from_numpy(y_train).float().to(device).view(-1, 1)
    
    n_samples = len(x_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    epochs_above_target = 0
    
    for epoch in range(epochs):
        indices = rng.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            x_batch = x_t[batch_idx]
            y_batch = y_t[batch_idx]
            
            optimizer.zero_grad()
            logits = model(x_batch)
            bce_loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            
            # Asymmetric penalty: only penalize negative weights
            # This enforces "parts-based" representation (Lee & Seung, 1999)
            weights = model.l2_layer_weights()
            neg_penalty = neg_lambda * negative_weight_penalty(weights)
            loss = bce_loss + neg_penalty
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # Check accuracy
        model.eval()
        with torch.no_grad():
            logits_all = model(x_t).view(-1)
            preds = (logits_all > 0).float()
            acc = (preds == y_t.view(-1)).float().mean().item()
        model.train()
        
        if (epoch + 1) % 40 == 0 or epoch == epochs - 1:
            weights = model.l2_layer_weights()
            n_neg = (weights < 0).sum().item()
            neg_sum = F.relu(-weights).sum().item()
            print(f"{model_name} epoch {epoch+1}/{epochs} loss={avg_loss:.4f} acc={acc:.1%} "
                  f"(neg={n_neg}, neg_sum={neg_sum:.4f})")
        
        # Early stopping
        if acc >= target_acc:
            epochs_above_target += 1
            if epochs_above_target >= patience:
                print(f"{model_name} converged at epoch {epoch+1} (acc={acc:.1%} for {patience} epochs)")
                break
        else:
            epochs_above_target = 0
    
    print(f"Final train loss ({model_name})={avg_loss:.4f}, acc={acc:.1%}")
    return losses


# ============================================================================
# Gradient Computation
# ============================================================================

def compute_gradient_norms(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Compute |∇ internal_scalar| with respect to input for each point."""
    model.eval()
    x_t = torch.from_numpy(x).float().to(device).requires_grad_(True)
    internal = model.internal_scalar(x_t)
    
    grad_norms = []
    for i in range(len(x)):
        model.zero_grad()
        if x_t.grad is not None:
            x_t.grad.zero_()
        internal[i].backward(retain_graph=True)
        grad_norm = x_t.grad[i].norm().item()
        grad_norms.append(grad_norm)
    
    return np.array(grad_norms)


def evaluate_accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
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
# Visualization
# ============================================================================
def plot_comparison(
    results: dict,
    distances: np.ndarray,
    out_dir: str,
    seed: int,
):
    """Plot gradient histograms and weight distributions as separate images."""
    inside = distances <= 0
    
    # ========================================================================
    # Plot 1: Gradient histograms (interior only)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    all_grads = np.concatenate([r['grads'][inside] for r in results.values()])
    bins = np.linspace(0, np.percentile(all_grads, 95), 35)
    
    for name, data in results.items():
        ax.hist(data['grads'][inside], bins=bins, alpha=0.5, label=name)
    ax.set_xlabel('|∇| (interior points)')
    ax.set_ylabel('Count')
    ax.set_title('Interior Gradient Distribution')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"gradient_histograms_seed{seed}.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()
    
    # ========================================================================
    # Plot 2: Flat point counts bar chart
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    
    thresholds = [0.5, 1.0, 2.0]
    x_pos = np.arange(len(results))
    width = 0.25
    
    for i, thresh in enumerate(thresholds):
        counts = [((data['grads'][inside] < thresh).sum() / inside.sum() * 100) 
                  for data in results.values()]
        ax.bar(x_pos + i * width, counts, width, label=f'|∇| < {thresh}')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('% of interior points')
    ax.set_title('Flat Points by Threshold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(results.keys(), rotation=15, ha='right', fontsize=9)
    ax.legend()
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"flat_points_seed{seed}.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()
    
    # ========================================================================
    # Plot 3: Weight distribution (signed)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for name, data in results.items():
        w = data['weights']
        ax.hist(w, bins=30, alpha=0.5, label=name)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='w=0')
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Count')
    ax.set_title('L2 Layer Weight Distribution (signed)')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"weight_distribution_seed{seed}.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()
    
    # ========================================================================
    # Plot 4: Gradient vs negative weight summary scatter
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, data in results.items():
        flat_pct = (data['grads'][inside] < 1.0).sum() / inside.sum() * 100
        w = data['weights']
        neg_sum = np.abs(w[w < 0]).sum()
        ax.scatter(neg_sum, flat_pct, s=100, label=name)
        ax.annotate(name, (neg_sum, flat_pct), textcoords="offset points", 
                    xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('Sum of |negative weights|')
    ax.set_ylabel('% flat interior points (|∇| < 1.0)')
    ax.set_title('Mesa Flatness vs Negative Weight Magnitude')
    ax.legend(fontsize=8, loc='lower right')
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"flatness_vs_negweights_seed{seed}.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    in_dim: int = 2,
    n_faces: int = 6,
    hidden_dim: int = 32,
    n_train: int = 2000,
    n_test: int = 500,
    epochs: int = 500,
    p_init: float = 1.0,
    neg_lambdas: list[float] = None,
    seed: int = 0,
    out_dir: str = "results/exp_gradient_comparison_nonneg",
):
    """
    Test whether asymmetric non-negative penalty induces mesa structure.
    
    The penalty (Lee & Seung, 1999) enforces "parts-based" representation
    by pushing negative weights toward zero while leaving positive weights
    unconstrained. This is the "soft" version of the hard constraint w >= 0.
    
    Hypothesis: If cancellation (via negative weights) is what prevents
    mesa structure in Linear models, then penalizing negative weights
    should produce mesa structure similar to Lp.
    """
    if neg_lambdas is None:
        neg_lambdas = [0.1, 0.5, 1.0]
    
    # Setup
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print("Experiment: Non-Negative Weight Constraints for Mesa Structure")
    print("=" * 70)
    print()
    print("Theory: Lee & Seung (1999) - Parts-based learning")
    print("  - Signed weights enable cancellation → holistic representation")
    print("  - Non-negative weights prevent cancellation → parts-based representation")
    print()
    print("Penalty: loss += λ * Σ relu(-w)²")
    print("  - Negative weights: penalized, pushed toward zero")
    print("  - Positive weights: no penalty, free to grow")
    print()
    print(f"Seed: {seed}")
    print(f"p: {p_init}")
    print(f"Negative weight λ: {neg_lambdas}")
    print(f"Max epochs: {epochs}")
    print()
    
    # Generate polytope and data
    normals, radii = make_random_polytope(in_dim, n_faces, seed=seed)
    x_train, y_train = generate_polytope_dataset(normals, radii, n_train, rng=rng)
    x_test, y_test = generate_polytope_dataset(normals, radii, n_test, rng=rng)
    distances = -min_face_slack(x_test, normals, radii)
    inside = distances <= 0
    
    print(f"Train: {n_train} points ({y_train.sum():.0f} inside)")
    print(f"Test:  {n_test} points ({y_test.sum():.0f} inside)")
    print()
    
    results = {}
    
    # ========================================================================
    # Baselines (no regularization)
    # ========================================================================
    
    print(f"--- Training Lp model (p={p_init}, no reg) ---")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lp_model = LpClassifier(in_dim, hidden_dim, p_init=p_init)
    train_model(lp_model, x_train, y_train, epochs, device=device, rng=rng, model_name="lp")
    
    lp_grads = compute_gradient_norms(lp_model, x_test, device)
    lp_acc = evaluate_accuracy(lp_model, x_test, y_test.squeeze(), device)
    lp_weights = lp_model.l2._raw_w.data.cpu().numpy().flatten()
    
    results['Lp'] = {
        'grads': lp_grads,
        'acc': lp_acc,
        'weights': lp_weights,
    }
    
    print("\n--- Training Linear model (no reg) ---")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    linear_model = LinearClassifier(in_dim, hidden_dim)
    train_model(linear_model, x_train, y_train, epochs, device=device, rng=rng, model_name="linear")
    
    linear_grads = compute_gradient_norms(linear_model, x_test, device)
    linear_acc = evaluate_accuracy(linear_model, x_test, y_test.squeeze(), device)
    linear_weights = linear_model.l2.weight.data.cpu().numpy().flatten()
    
    results['Linear'] = {
        'grads': linear_grads,
        'acc': linear_acc,
        'weights': linear_weights,
    }
    
    # ========================================================================
    # Non-negative constrained models
    # ========================================================================
    
    for neg_lambda in neg_lambdas:
        # Linear with non-negative penalty
        print(f"\n--- Training Linear model (neg λ={neg_lambda}) ---")
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        
        linear_nn_model = LinearClassifier(in_dim, hidden_dim)
        train_model_with_nonneg_penalty(
            linear_nn_model, x_train, y_train, epochs, device, rng,
            neg_lambda=neg_lambda, model_name=f"linear-nn-{neg_lambda}"
        )
        
        linear_nn_grads = compute_gradient_norms(linear_nn_model, x_test, device)
        linear_nn_acc = evaluate_accuracy(linear_nn_model, x_test, y_test.squeeze(), device)
        linear_nn_weights = linear_nn_model.l2.weight.data.cpu().numpy().flatten()
        
        results[f'Linear nn={neg_lambda}'] = {
            'grads': linear_nn_grads,
            'acc': linear_nn_acc,
            'weights': linear_nn_weights,
        }
    
    # ========================================================================
    # Report results
    # ========================================================================
    
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()
    print(f"{'Model':<20} {'Accuracy':>10} {'|∇| mean':>12} {'Flat (<1.0)':>15} {'Neg w':>8}")
    print("-" * 70)
    
    for name, data in results.items():
        flat_count = (data['grads'][inside] < 1.0).sum()
        flat_pct = flat_count / inside.sum() * 100
        n_neg = (data['weights'] < 0).sum()
        print(f"{name:<20} {data['acc']*100:>9.1f}% {data['grads'][inside].mean():>12.2f} "
              f"{flat_count:>4}/{inside.sum()} ({flat_pct:>5.1f}%) {n_neg:>8}")
    
    print()
    print("=" * 70)
    print("Weight Statistics")
    print("=" * 70)
    print()
    print(f"{'Model':<20} {'w>0 mean':>10} {'w>0 cnt':>8} {'w<0 mean':>10} {'w<0 cnt':>8}")
    print("-" * 60)
    
    for name, data in results.items():
        w = data['weights']
        pos_w = w[w > 0]
        neg_w = w[w < 0]
        pos_mean = pos_w.mean() if len(pos_w) > 0 else 0
        neg_mean = neg_w.mean() if len(neg_w) > 0 else 0
        print(f"{name:<20} {pos_mean:>10.4f} {len(pos_w):>8} {neg_mean:>10.4f} {len(neg_w):>8}")
    
    # Visualization
    print()
    print("Generating visualization...")
    plot_comparison(results, distances, out_dir=out_dir, seed=seed)
    
    # ========================================================================
    # Assessment
    # ========================================================================
    
    print()
    print("=" * 70)
    print("Assessment")
    print("=" * 70)
    print()
    
    min_acc = 0.95
    valid_results = {k: v for k, v in results.items() if v['acc'] >= min_acc}
    
    if len(valid_results) < len(results):
        excluded = [k for k in results if k not in valid_results]
        print(f"Excluding models with <{min_acc:.0%} accuracy: {excluded}")
        print()
    
    if valid_results:
        lp_results = {k: v for k, v in valid_results.items() if k.startswith('Lp')}
        linear_results = {k: v for k, v in valid_results.items() if k.startswith('Linear')}
        
        def flat_count(data):
            return (data['grads'][inside] < 1.0).sum()
        
        if lp_results:
            best_lp = max(lp_results.items(), key=lambda x: flat_count(x[1]))
            print(f"Best Lp:     {best_lp[0]:<20} flat={flat_count(best_lp[1])}/{inside.sum()} "
                  f"({100*flat_count(best_lp[1])/inside.sum():.1f}%)")
        
        if linear_results:
            best_linear = max(linear_results.items(), key=lambda x: flat_count(x[1]))
            print(f"Best Linear: {best_linear[0]:<20} flat={flat_count(best_linear[1])}/{inside.sum()} "
                  f"({100*flat_count(best_linear[1])/inside.sum():.1f}%)")
        
        print()
        
        # Check if non-negative constraint helped
        baseline_linear = results.get('Linear')
        if baseline_linear and linear_results:
            baseline_flat = flat_count(baseline_linear)
            best_nn_flat = flat_count(best_linear[1])
            
            if best_nn_flat > baseline_flat * 2:
                print("✓ Non-negative penalty significantly increases mesa flatness")
                if lp_results and best_nn_flat >= flat_count(best_lp[1]) * 0.8:
                    print("  → Linear with non-negative constraint approaches Lp performance!")
                    print("  → Confirms: cancellation prevention is the key mechanism")
                else:
                    print("  → But still below Lp—soft penalty weaker than hard constraint")
            elif best_nn_flat > baseline_flat:
                print("~ Non-negative penalty somewhat increases mesa flatness")
            else:
                print("✗ Non-negative penalty does not increase mesa flatness")
    
    print()
    print("=" * 70)
    print("Done")
    print("=" * 70)
    
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Non-negative weight constraints for mesa structure (Lee & Seung, 1999)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-init", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-faces", type=int, default=6)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--neg-lambda", type=float, nargs='+', default=[0.1, 0.5, 1.0],
                   help="Penalty strength for negative weights")
    p.add_argument("--out-dir", type=str, default="results/exp_gradient_comparison_nonneg")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        hidden_dim=args.hidden_dim,
        n_faces=args.n_faces,
        epochs=args.epochs,
        p_init=args.p_init,
        neg_lambdas=args.neg_lambda,
        seed=args.seed,
        out_dir=args.out_dir,
    )