"""
This experiment tests whether norm-aggregated ReLU networks exhibit robustness
inside learned polytope plateaus and sensitivity near polytope boundaries,
as predicted by the Polyhedral Mesa Gaussian interpretation.

Run as a script from the project root:
    python src/exp_polytope_boundary_sensitivity.py
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import csv
from datetime import datetime

from blocks import PolytopeClassifier
from train_utils import train_step


def make_random_polytope(in_dim: int, n_faces: int, radius_scale: float = 1.0, seed: int | None = None):
    rng = np.random.default_rng(seed)
    # In 2D, make normals approximately evenly spread to avoid unbounded/skinny regions
    if in_dim == 2:
        angles = np.linspace(0, 2 * math.pi, n_faces, endpoint=False)
        angles += rng.uniform(-0.2, 0.2, size=angles.shape)  # small jitter
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    else:
        normals = rng.normal(size=(n_faces, in_dim))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
    # choose positive offsets so the origin is inside
    radii = rng.uniform(0.4, 1.0, size=(n_faces,)) * radius_scale
    return normals.astype(np.float32), radii.astype(np.float32)

def polytope_membership(x: np.ndarray, normals: np.ndarray, radii: np.ndarray) -> np.ndarray:
    # x: (N, D); normals: (M, D); radii: (M,)
    proj = x @ normals.T  # (N, M)
    slack = radii[None, :] - proj  # positive if inside that face
    # inside if all slack >= 0
    inside = np.all(slack >= 0.0, axis=1)
    return inside.astype(np.float32)


def min_face_slack(x: np.ndarray, normals: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Return signed minimum face slack: r_i - n_i·x.

    Positive values mean the point is inside that face (distance along normal),
    negative values mean outside. This is NOT Euclidean distance to the polytope.
    """
    proj = x @ normals.T
    slack = radii[None, :] - proj
    return np.min(slack, axis=1)


def sample_unit_directions(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, dim))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    v = v / norms
    return v


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

    normals, radii = make_random_polytope(in_dim, n_faces, seed=seed)

    # sample training data uniformly from a box that likely contains the polytope
    box = 1.5
    x_train = rng.uniform(-box, box, size=(n_train, in_dim)).astype(np.float32)
    y_train = polytope_membership(x_train, normals, radii).reshape(-1, 1)

    # test set
    x_test = rng.uniform(-box, box, size=(n_test, in_dim)).astype(np.float32)
    y_test = polytope_membership(x_test, normals, radii).reshape(-1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolytopeClassifier(in_dim, hidden_dim, learn_p=True, p_init=2.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train loop (mini-batches)
    batch_size = 128
    n_batches = max(1, n_train // batch_size)
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        ep_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            xb = x_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            loss = train_step(model, opt, xb, yb)
            ep_loss += loss
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {ep+1}/{epochs} loss={ep_loss / n_batches:.4f}")

    # Evaluate on test set; focus on points that are inside the true polytope
    x_test_np = x_test
    y_test_np = y_test
    dists = min_face_slack(x_test_np, normals, radii)
    inside_mask = (dists >= 0.0)
    inside_idx = np.where(inside_mask)[0]
    if len(inside_idx) == 0:
        raise RuntimeError("No inside test points — increase sampling region or faces configuration")

    # select subset for probing to keep compute small
    rng_idx = rng.choice(inside_idx, size=min(400, len(inside_idx)), replace=False)
    probe_points_all = x_test_np[rng_idx]
    probe_dists_all = dists[rng_idx]

    # compute original model outputs and gradients
    model.eval()
    with torch.no_grad():
        x_t_all = torch.from_numpy(probe_points_all).to(device)
        logits_orig_all = model(x_t_all).cpu().numpy().reshape(-1)
        # internal scalar (distance-like) — compute pre-bias value to be closer to a pure distance
        v_all = F.relu(model.block.lin(x_t_all))
        internal_orig_all = model.block.norm(v_all).cpu().numpy().reshape(-1)

    # filter probe points to confidently-correct points: true inside AND model predicts inside with margin > tau
    true_labels_all = polytope_membership(probe_points_all, normals, radii)
    confident_mask = (true_labels_all == 1.0) & (logits_orig_all > float(tau))
    if confident_mask.sum() == 0:
        print("Warning: no confidently-correct probe points found with the given tau; relaxing threshold.")
        # relax threshold
        confident_mask = (true_labels_all == 1.0) & (logits_orig_all > 0.0)

    probe_points = probe_points_all[confident_mask]
    probe_dists = probe_dists_all[confident_mask]
    logits_orig = logits_orig_all[confident_mask]
    internal_orig = internal_orig_all[confident_mask]

    if probe_points.shape[0] == 0:
        print("No probe points available after confidence filtering. Exiting experiment.")
        return

    # prepare output directories
    out_base = out_dir or "results/plots"
    os.makedirs(out_base, exist_ok=True)
    results_dir_root = os.path.join("results")
    os.makedirs(results_dir_root, exist_ok=True)

    # gradient norms of the internal scalar (pre-bias) w.r.t input
    grad_norms = []
    for x_vec in probe_points:
        x_var = torch.from_numpy(x_vec.reshape(1, -1)).to(device).requires_grad_(True)
        v = F.relu(model.block.lin(x_var))
        internal = model.block.norm(v)
        internal_scalar = internal.sum()
        model.zero_grad()
        if x_var.grad is not None:
            x_var.grad.zero_()
        internal_scalar.backward()
        gn = x_var.grad.detach().cpu().norm().item()
        grad_norms.append(gn)

    # probing: support sweeping eps values. If eps_list provided, loop over it; otherwise use single eps
    eps_values = eps_list if eps_list is not None else [eps]

    # arrays to collect summary stats per-eps
    eps_summary = []

    for eps_val in eps_values:
        flip_rates = []
        true_flip_rates = []
        delta_logits = []
        delta_internal = []
        for i, x0 in enumerate(probe_points):
            directions = sample_unit_directions(n_perturb, in_dim, rng)
            x_pert = x0[None, :] + eps_val * directions
            xt = torch.from_numpy(x_pert.astype(np.float32)).to(device)
            with torch.no_grad():
                logits_p = model(xt).cpu().numpy().reshape(-1)
                v_p = F.relu(model.block.lin(xt))
                internal_p = model.block.norm(v_p).cpu().numpy().reshape(-1)

            orig_logit = logits_orig[i]
            orig_internal = internal_orig[i]

            # model flips
            pred_orig = (orig_logit > 0.0).astype(np.float32)
            pred_p = (logits_p > 0.0).astype(np.float32)
            flips = (pred_p != pred_orig).astype(np.float32)
            flip_rate = flips.mean()

            # true label flips under the same perturbations
            true_labels_p = polytope_membership(x_pert, normals, radii)
            true_orig = polytope_membership(x0[None, :], normals, radii)[0]
            true_flips = (true_labels_p != true_orig).astype(np.float32)
            true_flip_rate = true_flips.mean()

            delta_l = np.mean(np.abs(logits_p - orig_logit))
            delta_int = np.mean(np.abs(internal_p - orig_internal))

            flip_rates.append(flip_rate)
            true_flip_rates.append(true_flip_rate)
            delta_logits.append(delta_l)
            delta_internal.append(delta_int)

        flip_rates = np.array(flip_rates)
        delta_logits = np.array(delta_logits)
        delta_internal = np.array(delta_internal)
        grad_norms = np.array(grad_norms)
        true_flip_rates = np.array(true_flip_rates)

        # partition probe points by geometric bands tied to the perturbation magnitude
        # boundary band: slack in [0, eps_val]
        # interior band: slack >= 5*eps_val
        boundary_idx = (probe_dists >= 0.0) & (probe_dists <= eps_val)
        interior_idx = probe_dists >= (5.0 * eps_val)

        def summarize_eps(idx_mask: np.ndarray) -> Tuple[float, float, float, float, float]:
            return (
                float(flip_rates[idx_mask].mean()) if idx_mask.any() else float('nan'),
                float(true_flip_rates[idx_mask].mean()) if idx_mask.any() else float('nan'),
                float(delta_logits[idx_mask].mean()) if idx_mask.any() else float('nan'),
                float(delta_internal[idx_mask].mean()) if idx_mask.any() else float('nan'),
                float(grad_norms[idx_mask].mean()) if idx_mask.any() else float('nan'),
            )

        bound_stats = summarize_eps(boundary_idx)

        interior_count = int(np.sum(interior_idx))
        if interior_count < int(min_interior):
            # Not enough interior probes remain for this eps; record NaNs and report.
            nan_stats = (float('nan'),) * 5
            interior_stats = nan_stats
            print(f"\nEps={eps_val:.4f}: NOT ENOUGH interior probes (have={interior_count}, need={min_interior}) — setting interior stats to NaN")
        else:
            interior_stats = summarize_eps(interior_idx)
            print(f"\nEps={eps_val:.4f} Summary (boundary vs interior):")
            print("model_flip, true_flip, mean|Δlogit|, mean|Δinternal|, mean|grad|")
            print("boundary:", np.array(bound_stats))
            print("interior:", np.array(interior_stats))

        # save per-eps CSV
        results_dir = results_dir_root
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"probe_results_seed{seed}_eps{eps_val:.3f}_tau{tau:.2f}_{ts}.csv"
        csv_path = os.path.join(results_dir, csv_name)
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "idx",
                "min_face_slack",
                "logit_orig",
                "internal_orig",
                "grad_norm",
                "model_flip_rate",
                "true_flip_rate",
                "delta_logit",
                "delta_internal",
                "interior_ok",
            ])
            for i in range(probe_points.shape[0]):
                writer.writerow([
                    i,
                    float(probe_dists[i]),
                    float(logits_orig[i]),
                    float(internal_orig[i]),
                    float(grad_norms[i]),
                    float(flip_rates[i]),
                    float(true_flip_rates[i]),
                    float(delta_logits[i]),
                    float(delta_internal[i]),
                    bool(interior_count >= int(min_interior)),
                ])
        print(f"Saved CSV results: {csv_path}")

        interior_ok = not any([isinstance(v, float) and np.isnan(v) for v in interior_stats])
        eps_summary.append((eps_val, bound_stats, interior_stats, interior_ok))

    # If sweeping eps, plot flip-rate curves vs eps
    if eps_list is not None and len(eps_summary) > 0:
        eps_vals = [e for e, _, _, _ in eps_summary]
        b_model = [s[0] for _, s, _, _ in eps_summary]
        b_true = [s[1] for _, s, _, _ in eps_summary]
        i_model = [t[0] for _, _, t, _ in eps_summary]
        i_true = [t[1] for _, _, t, _ in eps_summary]
        interior_ok = [ok for _, _, _, ok in eps_summary]

        # combined plot (boundary + interior)
        plt.figure(figsize=(6, 4))
        plt.plot(eps_vals, b_model, '-o', label='boundary model flip')
        plt.plot(eps_vals, b_true, '-x', label='boundary true flip')
        plt.plot(eps_vals, i_model, '-o', label='interior model flip')
        plt.plot(eps_vals, i_true, '-x', label='interior true flip')
        plt.xlabel('eps')
        plt.ylabel('flip rate')
        plt.title('Flip rate vs eps (boundary vs interior)')
        plt.legend()
        sweep_path = os.path.join(out_base, 'flip_rate_vs_eps.png')
        plt.tight_layout()
        plt.savefig(sweep_path)
        print(f"Saved sweep plot: {sweep_path}")

        # annotate points where interior_ok is False
        for e_val, ok in zip(eps_vals, interior_ok):
            if not ok:
                plt.figure(figsize=(1, 0.5))
                # small marker file to note missing interior; also mark on combined plot with text
                plt.close()
        # also mark on combined plot with a vertical dashed line for each missing interior
        for e_val, ok in zip(eps_vals, interior_ok):
            if not ok:
                plt.figure(1)
                plt.axvline(e_val, color='gray', linestyle=':', alpha=0.6)

        # separate plots: boundary (model vs true)
        plt.figure(figsize=(6, 4))
        plt.plot(eps_vals, b_model, '-o', label='model_flip')
        plt.plot(eps_vals, b_true, '-x', label='true_flip')
        # mark missing interior points on boundary plot
        for e_val, ok in zip(eps_vals, interior_ok):
            if not ok:
                plt.axvline(e_val, color='gray', linestyle=':', alpha=0.6)
        plt.xlabel('eps')
        plt.ylabel('flip rate')
        plt.title('Boundary flip rate vs eps')
        plt.legend()
        bpath = os.path.join(out_base, 'boundary_flip_rate_vs_eps.png')
        plt.tight_layout()
        plt.savefig(bpath)
        print(f"Saved boundary sweep plot: {bpath}")

        # separate plots: interior (model vs true)
        plt.figure(figsize=(6, 4))
        plt.plot(eps_vals, i_model, '-o', label='model_flip')
        plt.plot(eps_vals, i_true, '-x', label='true_flip')
        # highlight eps values where interior_ok is False by marking points as NaN (they will not plot)
        for e_val, ok in zip(eps_vals, interior_ok):
            if not ok:
                plt.axvline(e_val, color='gray', linestyle=':', alpha=0.6)
        plt.xlabel('eps')
        plt.ylabel('flip rate')
        plt.title('Interior flip rate vs eps')
        plt.legend()
        ipath = os.path.join(out_base, 'interior_flip_rate_vs_eps.png')
        plt.tight_layout()
        plt.savefig(ipath)
        print(f"Saved interior sweep plot: {ipath}")

    # Final summary: either a labeled single-eps summary or an aggregate over the sweep
    if eps_list is not None:
        # aggregate over eps values that had a valid interior (interior_ok True)
        valid = [entry for entry in eps_summary if entry[3]]
        if len(valid) > 0:
            # valid is list of (eps_val, bound_stats, interior_stats, interior_ok)
            b_vals = np.array([v[1] for v in valid])  # shape (k,5)
            i_vals = np.array([v[2] for v in valid])
            b_mean = np.nanmean(b_vals, axis=0)
            i_mean = np.nanmean(i_vals, axis=0)
            print(f"\nAggregate over eps sweep (mean over {len(valid)} eps with interior_ok=True):")
            print("model_flip, true_flip, mean|Δlogit|, mean|Δinternal|, mean|grad|")
            print("boundary (mean):", b_mean)
            print("interior (mean):", i_mean)
        else:
            print("\nAggregate over eps sweep: no eps values had sufficient interior probes (all interior stats NaN)")
    else:
        # single-eps summary tied to the provided `eps` argument
        flip_rates = np.array(flip_rates)
        delta_logits = np.array(delta_logits)
        delta_internal = np.array(delta_internal)
        grad_norms = np.array(grad_norms)
        true_flip_rates = np.array(true_flip_rates)

        # partition probe points by geometric bands tied to the perturbation magnitude
        # boundary band: slack in [0, eps]
        # interior band: slack >= 5*eps
        boundary_idx = (probe_dists >= 0.0) & (probe_dists <= eps)
        interior_idx = probe_dists >= (5.0 * eps)

        def summarize(idx_mask: np.ndarray) -> Tuple[float, float, float, float, float]:
            return (
                float(flip_rates[idx_mask].mean()) if idx_mask.any() else float('nan'),
                float(true_flip_rates[idx_mask].mean()) if idx_mask.any() else float('nan'),
                float(delta_logits[idx_mask].mean()) if idx_mask.any() else float('nan'),
                float(delta_internal[idx_mask].mean()) if idx_mask.any() else float('nan'),
                float(grad_norms[idx_mask].mean()) if idx_mask.any() else float('nan'),
            )

        bound_stats = summarize(boundary_idx)
        interior_stats = summarize(interior_idx)

        print(f"\nFinal single-eps summary (eps={eps}):")
        print("model_flip, true_flip, mean|Δlogit|, mean|Δinternal|, mean|grad|")
        print("boundary:", np.array(bound_stats))
        print("interior:", np.array(interior_stats))

    # plotting
    os.makedirs(out_dir or "results/plots", exist_ok=True)
    out_base = out_dir or "results/plots"

    # bar plot comparing the four metrics
    labels = ["model_flip", "true_flip", "|Δlogit|", "|Δinternal|", "|grad|"]
    bvals = bound_stats
    ivals = interior_stats
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, bvals, width, label="boundary")
    plt.bar(x + width / 2, ivals, width, label="interior")
    plt.xticks(x, labels)
    plt.ylabel("value")
    plt.title("Boundary vs Interior sensitivity (higher = more sensitive)")
    plt.legend()
    ppath = os.path.join(out_base, "boundary_vs_interior.png")
    plt.tight_layout()
    plt.savefig(ppath)
    print(f"Saved plot: {ppath}")

    # scatter: flip rate vs distance
    plt.figure(figsize=(6, 4))
    plt.scatter(probe_dists, flip_rates, s=20, alpha=0.7, label='model_flip')
    plt.scatter(probe_dists, true_flip_rates, s=20, alpha=0.7, label='true_flip')
    plt.xlabel("true min face slack")
    plt.ylabel("flip rate (eps={:.3f})".format(eps))
    plt.title("Flip rate vs true min-face slack")
    plt.legend()
    s2 = os.path.join(out_base, "flip_rate_vs_distance.png")
    plt.tight_layout()
    plt.savefig(s2)
    print(f"Saved plot: {s2}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--eps-list", type=str, default="0.02,0.05,0.1,0.2", help="comma-separated eps values to sweep, e.g. 0.02,0.05,0.1,0.2")
    p.add_argument("--n-perturb", type=int, default=40)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--tau", type=float, default=1.0, help="confidence margin threshold for probe selection")
    p.add_argument("--min-interior", type=int, default=20, help="minimum number of interior probe points required per-eps; otherwise interior stats become NaN")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eps_list = None
    if args.eps_list:
        eps_list = [float(s) for s in args.eps_list.split(",") if s.strip()]
    run_experiment(epochs=args.epochs, eps=args.eps, n_perturb=args.n_perturb, seed=args.seed, out_dir=args.out_dir, tau=args.tau, eps_list=eps_list, min_interior=args.min_interior)
