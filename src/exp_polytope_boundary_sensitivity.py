"""
This experiment tests whether norm-aggregated ReLU networks exhibit robustness
inside learned polytope plateaus and sensitivity near polytope boundaries,
as predicted by the Polyhedral Mesa Gaussian interpretation.

Run as a script from the project root:
    python src/exp_polytope_boundary_sensitivity.py
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from functools import partial

import numpy as np
import torch

from blocks import PolytopeClassifier, MLPBaseline
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


def get_internal_fn_norm(model, x):
    """Get internal scalar for norm model."""
    import torch.nn.functional as F
    v = F.relu(model.block.lin(x))
    return model.block.norm(v)


def get_internal_fn_mlp(model, x):
    """Get internal scalar for MLP model."""
    import torch.nn.functional as F
    v = F.relu(model.lin(x))
    return model.agg(v).squeeze(-1)


def run_experiment(
    in_dim: int = 2,
    n_faces: int = 6,
    n_train: int = 2000,
    n_test: int = 1000,
    hidden_dim: int = 64,
    epochs: int = 200,
    eps: float = 0.1,
    n_perturb: int = 40,
    seed: int = 0,
    out_dir: str | None = None,
    tau: float = 1.0,
    eps_list: list[float] | None = None,
    min_interior: int = 20,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate polytope and datasets
    normals, radii = make_random_polytope(in_dim, n_faces, seed=seed)
    x_train, y_train = generate_polytope_dataset(normals, radii, n_train, rng=rng)
    x_test, _ = generate_polytope_dataset(normals, radii, n_test, rng=rng)

    # Build and train norm model
    norm_model = PolytopeClassifier(in_dim, hidden_dim, learn_p=True, p_init=2.0)
    train_model(
        norm_model, x_train, y_train, epochs,
        device=device, rng=rng, model_name="norm"
    )

    # Build and train MLP baseline
    mlp_model = MLPBaseline(in_dim, hidden_dim)
    train_model(
        mlp_model, x_train, y_train, epochs,
        device=device, rng=rng, model_name="mlp"
    )

    # Select probe points (inside the polytope)
    dists = min_face_slack(x_test, normals, radii)
    inside_mask = dists >= 0.0
    inside_idx = np.where(inside_mask)[0]

    if len(inside_idx) == 0:
        raise RuntimeError("No inside test points — increase sampling region or faces configuration")

    # Subsample for probing
    rng_idx = rng.choice(inside_idx, size=min(400, len(inside_idx)), replace=False)
    probe_points_all = x_test[rng_idx]
    probe_dists_all = dists[rng_idx]

    # Get original outputs for both models
    norm_model.eval()
    mlp_model.eval()

    with torch.no_grad():
        x_t = torch.from_numpy(probe_points_all).to(device)
        logits_orig_norm = norm_model(x_t).cpu().numpy().reshape(-1)
        internal_orig_norm = norm_model.internal_scalar(x_t).cpu().numpy().reshape(-1)
        logits_orig_mlp = mlp_model(x_t).cpu().numpy().reshape(-1)
        internal_orig_mlp = mlp_model.internal_scalar(x_t).cpu().numpy().reshape(-1)

    # Filter to confidently-correct points
    true_labels = polytope_membership(probe_points_all, normals, radii)
    confident_mask = (true_labels == 1.0) & (logits_orig_norm > float(tau))

    if confident_mask.sum() == 0:
        print("Warning: no confidently-correct probe points; relaxing threshold.")
        confident_mask = (true_labels == 1.0) & (logits_orig_norm > 0.0)

    probe_points = probe_points_all[confident_mask]
    probe_dists = probe_dists_all[confident_mask]
    logits_orig_norm = logits_orig_norm[confident_mask]
    internal_orig_norm = internal_orig_norm[confident_mask]
    logits_orig_mlp = logits_orig_mlp[confident_mask]
    internal_orig_mlp = internal_orig_mlp[confident_mask]

    if probe_points.shape[0] == 0:
        print("No probe points available after confidence filtering. Exiting.")
        return

    # Compute gradient norms
    grad_norms_norm = compute_gradient_norms(
        norm_model, probe_points, get_internal_fn_norm, device
    )
    grad_norms_mlp = compute_gradient_norms(
        mlp_model, probe_points, get_internal_fn_mlp, device
    )

    # Prepare output directories
    out_base = out_dir or "results/exp_polytope_boundary_sensitivity"
    os.makedirs(out_base, exist_ok=True)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Membership function for true flip computation
    membership_fn = partial(polytope_membership, normals=normals, radii=radii)

    # Eps sweep
    eps_values = eps_list if eps_list is not None else [eps]

    norm_results = ModelProbeResults(name="norm")
    mlp_results = ModelProbeResults(name="mlp")

    for eps_val in eps_values:
        # Probe perturbations for both models
        flip_rates_norm, delta_logits_norm, delta_internal_norm = probe_perturbations(
            norm_model, probe_points, logits_orig_norm, internal_orig_norm,
            eps_val, n_perturb, rng, device
        )
        flip_rates_mlp, delta_logits_mlp, delta_internal_mlp = probe_perturbations(
            mlp_model, probe_points, logits_orig_mlp, internal_orig_mlp,
            eps_val, n_perturb, rng, device
        )

        # True flip rates
        true_flip_rates = compute_true_flip_rates(
            probe_points, eps_val, n_perturb, membership_fn, rng
        )

        # Partition by distance bands
        boundary_idx = (probe_dists >= 0.0) & (probe_dists <= eps_val)
        interior_idx = probe_dists >= (5.0 * eps_val)

        interior_count = int(np.sum(interior_idx))
        interior_ok = interior_count >= min_interior

        # Compute band statistics
        boundary_stats_norm = BandStats.from_arrays(
            boundary_idx, flip_rates_norm, true_flip_rates,
            delta_logits_norm, delta_internal_norm, grad_norms_norm
        )
        interior_stats_norm = BandStats.from_arrays(
            interior_idx, flip_rates_norm, true_flip_rates,
            delta_logits_norm, delta_internal_norm, grad_norms_norm
        ) if interior_ok else BandStats.nan()

        boundary_stats_mlp = BandStats.from_arrays(
            boundary_idx, flip_rates_mlp, true_flip_rates,
            delta_logits_mlp, delta_internal_mlp, grad_norms_mlp
        )
        interior_stats_mlp = BandStats.from_arrays(
            interior_idx, flip_rates_mlp, true_flip_rates,
            delta_logits_mlp, delta_internal_mlp, grad_norms_mlp
        ) if interior_ok else BandStats.nan()

        norm_results.eps_summaries.append(EpsSummary(
            eps=eps_val,
            boundary_stats=boundary_stats_norm,
            interior_stats=interior_stats_norm,
            interior_ok=interior_ok,
        ))
        mlp_results.eps_summaries.append(EpsSummary(
            eps=eps_val,
            boundary_stats=boundary_stats_mlp,
            interior_stats=interior_stats_mlp,
            interior_ok=interior_ok,
        ))

        # Print summary
        if not interior_ok:
            print(f"\nEps={eps_val:.4f}: NOT ENOUGH interior probes (have={interior_count}, need={min_interior})")
        else:
            print(f"\nEps={eps_val:.4f} Summary:")
            print("model_flip, true_flip, mean|Δlogit|, mean|Δinternal|, mean|grad|")
            print(f"boundary (norm): {np.array(boundary_stats_norm.as_tuple())}")
            print(f"boundary (mlp):  {np.array(boundary_stats_mlp.as_tuple())}")
            print(f"interior (norm): {np.array(interior_stats_norm.as_tuple())}")
            print(f"interior (mlp):  {np.array(interior_stats_mlp.as_tuple())}")

        # Save per-eps CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"probe_results_seed{seed}_eps{eps_val:.3f}_tau{tau:.2f}_{ts}.csv"
        csv_path = os.path.join(results_dir, csv_name)

        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "idx", "min_face_slack",
                "logit_orig_norm", "internal_orig_norm", "grad_norm_norm",
                "logit_orig_mlp", "internal_orig_mlp", "grad_norm_mlp",
                "flip_rate_norm", "flip_rate_mlp", "true_flip_rate",
                "delta_logit_norm", "delta_logit_mlp",
                "delta_internal_norm", "delta_internal_mlp",
                "interior_ok",
            ])
            for i in range(probe_points.shape[0]):
                writer.writerow([
                    i, float(probe_dists[i]),
                    float(logits_orig_norm[i]), float(internal_orig_norm[i]), float(grad_norms_norm[i]),
                    float(logits_orig_mlp[i]), float(internal_orig_mlp[i]), float(grad_norms_mlp[i]),
                    float(flip_rates_norm[i]), float(flip_rates_mlp[i]), float(true_flip_rates[i]),
                    float(delta_logits_norm[i]), float(delta_logits_mlp[i]),
                    float(delta_internal_norm[i]), float(delta_internal_mlp[i]),
                    interior_ok,
                ])
        print(f"Saved CSV: {csv_path}")

    # Store final probe data for plotting
    norm_results.probe_dists = probe_dists
    norm_results.flip_rates = flip_rates_norm
    norm_results.true_flip_rates = true_flip_rates
    mlp_results.probe_dists = probe_dists
    mlp_results.flip_rates = flip_rates_mlp
    mlp_results.true_flip_rates = true_flip_rates

    # Aggregate summary
    model_results = [norm_results, mlp_results]

    if eps_list is not None:
        valid_norm = [s for s in norm_results.eps_summaries if s.interior_ok]
        if valid_norm:
            print(f"\nAggregate over eps sweep (mean over {len(valid_norm)} eps with interior_ok=True):")
            b_mean = np.mean([s.boundary_stats.as_tuple() for s in valid_norm], axis=0)
            i_mean = np.mean([s.interior_stats.as_tuple() for s in valid_norm], axis=0)
            valid_mlp = [s for s in mlp_results.eps_summaries if s.interior_ok]
            bm_mean = np.mean([s.boundary_stats.as_tuple() for s in valid_mlp], axis=0)
            im_mean = np.mean([s.interior_stats.as_tuple() for s in valid_mlp], axis=0)
            print("model_flip, true_flip, mean|Δlogit|, mean|Δinternal|, mean|grad|")
            print(f"boundary (norm mean): {b_mean}")
            print(f"interior (norm mean): {i_mean}")
            print(f"boundary (mlp mean):  {bm_mean}")
            print(f"interior (mlp mean):  {im_mean}")
        else:
            print("\nNo eps values had sufficient interior probes")

    # Generate plots
    if eps_list is not None and len(eps_values) > 1:
        eps_vals = [s.eps for s in norm_results.eps_summaries]
        interior_ok_list = [s.interior_ok for s in norm_results.eps_summaries]
        true_flip_boundary = [s.boundary_stats.true_flip for s in norm_results.eps_summaries]
        true_flip_interior = [s.interior_stats.true_flip for s in norm_results.eps_summaries]

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


def parse_args():
    p = argparse.ArgumentParser()
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
    )
