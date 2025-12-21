"""
Experiment: Polytope Boundary Sensitivity with Loss Variants

This experiment extends the original boundary sensitivity analysis to test
whether mesa-like geometry (flat interior, sharp boundary) arises from:

    (A) Architectural constraint (norm aggregation), or
    (B) Optimization incentive (loss function), or
    (C) Both / neither

Models compared:
    1. Norm model (CE loss) - baseline with architectural bias
    2. MLP baseline (CE loss) - baseline without architectural bias
    3. MLP + gradient penalty - test if loss can induce flatness
    4. Norm model + gradient penalty - control for interaction effects

Run as a script from the project root:
    python src/exp_polytope_boundary_sensitivity.py
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lp_norm_layer import LpNormLayer
from experiment_utils import (
    BandStats,
    EpsSummary,
    ModelProbeResults,
    train_model,
    compute_gradient_norms,
    probe_perturbations,
    compute_true_flip_rates,
    plot_flip_rate_vs_eps,
    plot_boundary_flip_vs_eps,
    plot_interior_flip_vs_eps,
    plot_boundary_vs_interior_bars,
    plot_flip_rate_vs_distance,
)
from polytope_utils import (
    make_random_polytope,
    polytope_membership,
    min_face_slack,
    generate_polytope_dataset,
)


# ============================================================================
# Model Definitions
# ============================================================================

class PolytopeClassifier(nn.Module):
    """Norm-based binary classifier: Linear -> ReLU -> NormLayer -> Linear head."""
    def __init__(self, in_dim: int, hidden_dim: int, learn_p: bool = True, p_init: float = 2.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.norm = LpNormLayer(hidden_dim, out_dim=1, learn_p=learn_p, p_init=p_init)
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any():
            raise ValueError("NaN in x") 
        v = F.relu(self.lin(x))
        if torch.isnan(v).any():
            print(f"self.lin.weight: {self.lin.weight}")
            print(f"self.lin.bias: {self.lin.bias}")
            raise ValueError("lin produced nan output.") 
        d = self.norm(v)  # (batch, 1)
        logits = self.head(d)
        return logits

    def internal_scalar(self, x: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.lin(x))
        return self.norm(v).squeeze(-1)
    
    def p_mean(self):
        return self.norm.p.mean().detach().cpu()
    
    def p_variance(self):
        return self.norm.p.var(unbiased=False).detach().cpu()


class MLPBaseline(nn.Module):
    """MLP baseline: Linear -> ReLU -> Linear -> Linear head."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.agg = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.lin(x))
        d = self.agg(v)  # (batch, 1)
        logits = self.head(d)
        return logits

    def internal_scalar(self, x: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.lin(x))
        return self.agg(v).squeeze(-1)


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model variant in the experiment."""
    name: str
    model_type: str  # "norm" or "mlp"
    loss_type: str  # "ce" or "grad_penalty"
    lambda_grad: float = 0.1

    def build_model(self, in_dim: int, hidden_dim: int) -> nn.Module:
        if self.model_type == "norm":
            return PolytopeClassifier(in_dim, hidden_dim, learn_p=True, p_init=2.0)
        elif self.model_type == "mlp":
            return MLPBaseline(in_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def get_internal_fn(self) -> Callable[[nn.Module, torch.Tensor], torch.Tensor]:
        if self.model_type == "norm":
            return get_internal_fn_norm
        else:
            return get_internal_fn_mlp


def get_internal_fn_norm(model, x):
    """Get internal scalar for norm model."""
    return model.internal_scalar(x)


def get_internal_fn_mlp(model, x):
    """Get internal scalar for MLP model."""
    return model.internal_scalar(x)


# ============================================================================
# Probing Infrastructure
# ============================================================================

@dataclass
class ModelState:
    """Holds a trained model and its cached probe data."""
    config: ModelConfig
    model: nn.Module
    logits_orig: np.ndarray
    internal_orig: np.ndarray
    grad_norms: np.ndarray
    results: ModelProbeResults


def prepare_model(
    config: ModelConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    probe_points: np.ndarray,
    epochs: int,
    device: torch.device,
    rng: np.random.Generator,
) -> ModelState:
    """Build, train, and prepare a model for probing."""
    in_dim = x_train.shape[1]
    hidden_dim = 64

    # Build and train
    model = config.build_model(in_dim, hidden_dim)
    train_model(
        model, x_train, y_train, epochs,
        device=device, rng=rng, model_name=config.name,
        loss_type=config.loss_type, lambda_grad=config.lambda_grad,
    )

    # Get original outputs
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(probe_points).to(device)
        logits_orig = model(x_t).cpu().numpy().reshape(-1)
        internal_orig = model.internal_scalar(x_t).cpu().numpy().reshape(-1)

    # Compute gradient norms
    internal_fn = config.get_internal_fn()
    grad_norms = compute_gradient_norms(model, probe_points, internal_fn, device)

    return ModelState(
        config=config,
        model=model,
        logits_orig=logits_orig,
        internal_orig=internal_orig,
        grad_norms=grad_norms,
        results=ModelProbeResults(name=config.name),
    )


def probe_model_at_eps(
    state: ModelState,
    probe_points: np.ndarray,
    probe_dists: np.ndarray,
    true_flip_rates: np.ndarray,
    eps_val: float,
    n_perturb: int,
    min_interior: int,
    rng: np.random.Generator,
    device: torch.device,
) -> EpsSummary:
    """Probe a model at a single epsilon value and return summary."""
    flip_rates, delta_logits, delta_internal = probe_perturbations(
        state.model, probe_points, state.logits_orig, state.internal_orig,
        eps_val, n_perturb, rng, device
    )

    # Partition by distance bands
    boundary_idx = (probe_dists >= 0.0) & (probe_dists <= eps_val)
    interior_idx = probe_dists >= (5.0 * eps_val)

    interior_count = int(np.sum(interior_idx))
    interior_ok = interior_count >= min_interior

    # Compute band statistics
    boundary_stats = BandStats.from_arrays(
        boundary_idx, flip_rates, true_flip_rates,
        delta_logits, delta_internal, state.grad_norms
    )
    interior_stats = BandStats.from_arrays(
        interior_idx, flip_rates, true_flip_rates,
        delta_logits, delta_internal, state.grad_norms
    ) if interior_ok else BandStats.nan()

    summary = EpsSummary(
        eps=eps_val,
        boundary_stats=boundary_stats,
        interior_stats=interior_stats,
        interior_ok=interior_ok,
    )

    # Store for final plotting
    state.results.probe_dists = probe_dists
    state.results.flip_rates = flip_rates
    state.results.true_flip_rates = true_flip_rates

    return summary


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    in_dim: int = 2,
    n_faces: int = 6,
    n_train: int = 2000,
    n_test: int = 1000,
    epochs: int = 200,
    eps: float = 0.1,
    n_perturb: int = 40,
    seed: int = 0,
    out_dir: str | None = None,
    tau: float = 1.0,
    eps_list: list[float] | None = None,
    min_interior: int = 20,
    lambda_grad: float = 0.1,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate polytope and datasets
    normals, radii = make_random_polytope(in_dim, n_faces, seed=seed)
    x_train, y_train = generate_polytope_dataset(normals, radii, n_train, rng=rng)
    x_test, _ = generate_polytope_dataset(normals, radii, n_test, rng=rng)

    # Select probe points (inside the polytope)
    dists = min_face_slack(x_test, normals, radii)
    inside_mask = dists >= 0.0
    inside_idx = np.where(inside_mask)[0]

    if len(inside_idx) == 0:
        raise RuntimeError("No inside test points — increase sampling region")

    # Subsample for probing
    rng_idx = rng.choice(inside_idx, size=min(400, len(inside_idx)), replace=False)
    probe_points_all = x_test[rng_idx]
    probe_dists_all = dists[rng_idx]

    # Define model configurations
    model_configs = [
        ModelConfig("norm", "norm", "ce"),
        ModelConfig("mlp", "mlp", "ce"),
        ModelConfig(f"mlp+gp(λ={lambda_grad})", "mlp", "grad_penalty", lambda_grad),
        ModelConfig(f"norm+gp(λ={lambda_grad})", "norm", "grad_penalty", lambda_grad),
    ]

    # Build and train all models
    print("=" * 60)
    print("Training models...")
    print("=" * 60)

    model_states: list[ModelState] = []
    for config in model_configs:
        print(f"\n--- Training {config.name} ---")
        # Use fresh RNG state for each model but deterministic based on seed
        model_rng = np.random.default_rng(seed)
        state = prepare_model(
            config, x_train, y_train, probe_points_all,
            epochs, device, model_rng
        )
        model_states.append(state)

    # Filter to confidently-correct points (using first model as reference)
    true_labels = polytope_membership(probe_points_all, normals, radii)
    reference_logits = model_states[0].logits_orig
    confident_mask = (true_labels == 1.0) & (reference_logits > float(tau))

    if confident_mask.sum() == 0:
        print("Warning: no confidently-correct probe points; relaxing threshold.")
        confident_mask = (true_labels == 1.0) & (reference_logits > 0.0)

    probe_points = probe_points_all[confident_mask]
    probe_dists = probe_dists_all[confident_mask]

    # Update cached arrays in model states
    for state in model_states:
        state.logits_orig = state.logits_orig[confident_mask]
        state.internal_orig = state.internal_orig[confident_mask]
        state.grad_norms = state.grad_norms[confident_mask]

    if probe_points.shape[0] == 0:
        print("No probe points available after confidence filtering. Exiting.")
        return

    print(f"\nProbing {probe_points.shape[0]} points")

    # Prepare output directories
    out_base = out_dir or "results/exp_polytope_boundary_sensitivity"
    os.makedirs(out_base, exist_ok=True)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Membership function for true flip computation
    membership_fn = partial(polytope_membership, normals=normals, radii=radii)

    # Eps sweep
    eps_values = eps_list if eps_list is not None else [eps]

    print("\n" + "=" * 60)
    print("Probing at different epsilon values...")
    print("=" * 60)

    for eps_val in eps_values:
        # Compute true flip rates (same for all models)
        true_flip_rates = compute_true_flip_rates(
            probe_points, eps_val, n_perturb, membership_fn, rng
        )

        # Probe each model
        for state in model_states:
            summary = probe_model_at_eps(
                state, probe_points, probe_dists, true_flip_rates,
                eps_val, n_perturb, min_interior, rng, device
            )
            state.results.eps_summaries.append(summary)

        # Print summary
        interior_ok = model_states[0].results.eps_summaries[-1].interior_ok
        interior_count = int(np.sum(probe_dists >= 5.0 * eps_val))

        if not interior_ok:
            print(f"\nEps={eps_val:.4f}: NOT ENOUGH interior probes "
                  f"(have={interior_count}, need={min_interior})")
        else:
            print(f"\nEps={eps_val:.4f} Summary:")
            print("-" * 60)
            print(f"{'Model':<20} {'|Δlogit|':>10} {'|Δint|':>10} {'|∇|':>10}")
            print("-" * 60)
            for state in model_states:
                s = state.results.eps_summaries[-1]
                print(f"{state.config.name:<20} "
                      f"{s.interior_stats.delta_logit:>10.4f} "
                      f"{s.interior_stats.delta_internal:>10.4f} "
                      f"{s.interior_stats.grad_norm:>10.4f}")

        # Save per-eps CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"probe_results_seed{seed}_eps{eps_val:.3f}_tau{tau:.2f}_{ts}.csv"
        csv_path = os.path.join(results_dir, csv_name)

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            # Header
            header = ["idx", "min_face_slack"]
            for state in model_states:
                name = state.config.name.replace(" ", "_")
                header.extend([
                    f"logit_{name}", f"internal_{name}", f"grad_{name}",
                    f"flip_rate_{name}", f"delta_logit_{name}", f"delta_internal_{name}"
                ])
            header.extend(["true_flip_rate", "interior_ok"])
            writer.writerow(header)

            # Data
            for i in range(probe_points.shape[0]):
                row = [i, float(probe_dists[i])]
                for state in model_states:
                    row.extend([
                        float(state.logits_orig[i]),
                        float(state.internal_orig[i]),
                        float(state.grad_norms[i]),
                        float(state.results.flip_rates[i]) if state.results.flip_rates is not None else float('nan'),
                        float('nan'),  # delta_logit per-point not stored
                        float('nan'),  # delta_internal per-point not stored
                    ])
                row.extend([float(true_flip_rates[i]), interior_ok])
                writer.writerow(row)
        print(f"Saved CSV: {csv_path}")

    # Aggregate summary
    model_results = [state.results for state in model_states]

    if eps_list is not None:
        print("\n" + "=" * 60)
        print("Aggregate Summary (mean over eps with valid interior)")
        print("=" * 60)

        # Compute ratios relative to norm model
        norm_state = model_states[0]
        valid_summaries = [s for s in norm_state.results.eps_summaries if s.interior_ok]

        if valid_summaries:
            print(f"\nInterior metrics (mean over {len(valid_summaries)} eps values):")
            print("-" * 70)
            print(f"{'Model':<20} {'|Δlogit|':>10} {'|∇|':>10} {'Ratio vs norm':>15}")
            print("-" * 70)

            # Get norm baseline mean gradient
            norm_grad = np.mean([s.interior_stats.grad_norm for s in valid_summaries])

            for state in model_states:
                valid = [s for s in state.results.eps_summaries if s.interior_ok]
                if valid:
                    delta = np.mean([s.interior_stats.delta_logit for s in valid])
                    grad = np.mean([s.interior_stats.grad_norm for s in valid])
                    ratio = grad / norm_grad if norm_grad > 0 else float('nan')
                    print(f"{state.config.name:<20} {delta:>10.4f} {grad:>10.4f} {ratio:>15.2f}×")
        else:
            print("\nNo eps values had sufficient interior probes")

    # Generate plots
    if eps_list is not None and len(eps_values) > 1:
        eps_vals = [s.eps for s in model_states[0].results.eps_summaries]
        interior_ok_list = [s.interior_ok for s in model_states[0].results.eps_summaries]
        true_flip_boundary = [s.boundary_stats.true_flip for s in model_states[0].results.eps_summaries]
        true_flip_interior = [s.interior_stats.true_flip for s in model_states[0].results.eps_summaries]

        plot_flip_rate_vs_eps(
            eps_vals, model_results, true_flip_boundary, true_flip_interior,
            interior_ok_list, save_path=os.path.join(out_base, 'flip_rate_vs_eps.png')
        )
        plot_boundary_flip_vs_eps(
            eps_vals, model_results, true_flip_boundary, interior_ok_list,
            save_path=os.path.join(out_base, 'boundary_flip_rate_vs_eps.png')
        )
        plot_interior_flip_vs_eps(
            eps_vals, model_results, true_flip_interior, interior_ok_list,
            save_path=os.path.join(out_base, 'interior_flip_rate_vs_eps.png')
        )

    plot_boundary_vs_interior_bars(
        model_results, save_path=os.path.join(out_base, 'boundary_vs_interior.png')
    )
    plot_flip_rate_vs_distance(
        model_results, eps_values[-1],
        save_path=os.path.join(out_base, 'flip_rate_vs_distance.png')
    )

    # Print hypothesis evaluation
    print("\n" + "=" * 60)
    print("Hypothesis Evaluation")
    print("=" * 60)

    if eps_list is not None:
        valid_norm = [s for s in model_states[0].results.eps_summaries if s.interior_ok]
        valid_mlp = [s for s in model_states[1].results.eps_summaries if s.interior_ok]
        valid_mlp_gp = [s for s in model_states[2].results.eps_summaries if s.interior_ok]

        if valid_norm and valid_mlp and valid_mlp_gp:
            norm_grad = np.mean([s.interior_stats.grad_norm for s in valid_norm])
            mlp_grad = np.mean([s.interior_stats.grad_norm for s in valid_mlp])
            mlp_gp_grad = np.mean([s.interior_stats.grad_norm for s in valid_mlp_gp])

            ratio_mlp = mlp_grad / norm_grad if norm_grad > 0 else float('nan')
            ratio_mlp_gp = mlp_gp_grad / norm_grad if norm_grad > 0 else float('nan')

            print(f"\nInterior |∇| ratios:")
            print(f"  MLP / Norm:      {ratio_mlp:.2f}×")
            print(f"  MLP+GP / Norm:   {ratio_mlp_gp:.2f}×")

            if ratio_mlp_gp < 2.0:
                print("\n→ H1 SUPPORTED: Gradient penalty closes the gap.")
                print("  Mesa geometry is loss-inducible; norm aggregation is one path among several.")
            elif ratio_mlp_gp < 5.0:
                print("\n→ H2 SUPPORTED: Gradient penalty partially closes the gap.")
                print("  GP helps but doesn't fully replicate architectural constraint.")
            else:
                print("\n→ H3 SUPPORTED: Gradient penalty doesn't significantly help.")
                print("  Gradient magnitude ≠ mesa geometry; norm aggregation does something else.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Polytope boundary sensitivity with loss variants"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--eps-list", type=str, default="0.02,0.05,0.1,0.2",
                   help="comma-separated eps values to sweep")
    p.add_argument("--n-perturb", type=int, default=40)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--tau", type=float, default=1.0,
                   help="confidence margin threshold for probe selection")
    p.add_argument("--min-interior", type=int, default=20,
                   help="minimum interior probes required per-eps")
    p.add_argument("--lambda-grad", type=float, default=0.1,
                   help="gradient penalty weight for GP variants")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eps_list = None
    if args.eps_list:
        eps_list = [float(s) for s in args.eps_list.split(",") if s.strip()]
    run_experiment(
        epochs=args.epochs,
        eps=args.eps,
        n_perturb=args.n_perturb,
        seed=args.seed,
        out_dir=args.out_dir,
        tau=args.tau,
        eps_list=eps_list,
        min_interior=args.min_interior,
        lambda_grad=args.lambda_grad,
    )
