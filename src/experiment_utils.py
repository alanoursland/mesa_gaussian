"""
Utility functions for running experiments: training, probing, and plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from train_utils import train_step


@dataclass
class ProbeResult:
    """Results from probing a single model at a single epsilon value."""
    flip_rates: np.ndarray  # (n_probes,)
    delta_logits: np.ndarray  # (n_probes,)
    delta_internal: np.ndarray  # (n_probes,)
    grad_norms: np.ndarray  # (n_probes,)


@dataclass
class BandStats:
    """Summary statistics for a band (boundary or interior)."""
    model_flip: float
    true_flip: float
    delta_logit: float
    delta_internal: float
    grad_norm: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.model_flip, self.true_flip, self.delta_logit,
                self.delta_internal, self.grad_norm)

    @classmethod
    def nan(cls) -> "BandStats":
        return cls(float('nan'), float('nan'), float('nan'),
                   float('nan'), float('nan'))

    @classmethod
    def from_arrays(
        cls,
        mask: np.ndarray,
        flip_rates: np.ndarray,
        true_flip_rates: np.ndarray,
        delta_logits: np.ndarray,
        delta_internal: np.ndarray,
        grad_norms: np.ndarray
    ) -> "BandStats":
        """Compute mean statistics over masked indices."""
        if not mask.any():
            return cls.nan()
        return cls(
            model_flip=float(flip_rates[mask].mean()),
            true_flip=float(true_flip_rates[mask].mean()),
            delta_logit=float(delta_logits[mask].mean()),
            delta_internal=float(delta_internal[mask].mean()),
            grad_norm=float(grad_norms[mask].mean()),
        )


@dataclass
class EpsSummary:
    """Summary of probing results at a single epsilon value."""
    eps: float
    boundary_stats: BandStats
    interior_stats: BandStats
    interior_ok: bool  # True if enough interior probes existed


@dataclass
class ModelProbeResults:
    """Complete probing results for a single model across all epsilon values."""
    name: str
    eps_summaries: list[EpsSummary] = field(default_factory=list)

    # Per-probe arrays (from last eps, for plotting flip vs distance)
    probe_dists: np.ndarray | None = None
    flip_rates: np.ndarray | None = None
    true_flip_rates: np.ndarray | None = None


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
    model_name: str = "model",
    log_interval: int | None = None,
    loss_type: str = "ce",
    lambda_grad: float = 0.1,
) -> float:
    """Train a model with the standard training loop.

    Args:
        model: PyTorch model to train
        x_train: (N, D) training features
        y_train: (N, 1) training labels
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        device: Device to train on
        rng: NumPy random generator for shuffling
        verbose: Whether to print progress
        model_name: Name for logging
        log_interval: How often to log (default: epochs // 5)
        loss_type: Loss function type ("ce", "grad_penalty", "confidence")
        lambda_grad: Weight for gradient penalty (only used if loss_type="grad_penalty")

    Returns:
        Final training loss
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rng is None:
        rng = np.random.default_rng()
    if log_interval is None:
        log_interval = max(1, epochs // 5)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = x_train.shape[0]
    n_batches = max(1, n_train // batch_size)
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)

    ep_loss = 0.0
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        ep_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            xb = x_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            loss = train_step(model, opt, xb, yb, loss_type=loss_type, lambda_grad=lambda_grad)
            ep_loss += loss

        if verbose and (ep + 1) % log_interval == 0:
            avg_loss = ep_loss / n_batches
            # Try to get p value for norm models
            p_str = ""
            if hasattr(model, 'p_mean'):
                pval = model.p_mean()
                pvar = model.p_variance()
                p_str = f" p_mean={pval:.4f} p_var={pvar:.4f}"
            # try:
            # except (AttributeError, Exception) as e:
            #     print(e)
            #     pass
            print(f"{model_name} epoch {ep+1}/{epochs} loss={avg_loss:.4f}{p_str}")

    final_loss = ep_loss / n_batches
    if verbose:
        print(f"Final train loss ({model_name})={final_loss:.4f}")

    return final_loss


def compute_gradient_norms(
    model: nn.Module,
    points: np.ndarray,
    internal_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute gradient norms of internal scalar w.r.t. input for each point.

    Args:
        model: The model to probe
        points: (N, D) array of input points
        internal_fn: Function that takes (model, x) and returns internal scalar
        device: Device to use

    Returns:
        (N,) array of gradient norms
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grad_norms = []
    for x_vec in points:
        x_var = torch.from_numpy(x_vec.reshape(1, -1)).to(device).requires_grad_(True)
        internal = internal_fn(model, x_var)
        internal_scalar = internal.sum()

        model.zero_grad()
        if x_var.grad is not None:
            x_var.grad.zero_()
        internal_scalar.backward()

        gn = x_var.grad.detach().cpu().norm().item()
        grad_norms.append(gn)

    return np.array(grad_norms)


def probe_perturbations(
    model: nn.Module,
    points: np.ndarray,
    logits_orig: np.ndarray,
    internal_orig: np.ndarray,
    eps: float,
    n_perturb: int,
    rng: np.random.Generator,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probe model sensitivity to perturbations.

    Args:
        model: Model to probe (must have internal_scalar method)
        points: (N, D) probe points
        logits_orig: (N,) original logits at probe points
        internal_orig: (N,) original internal values at probe points
        eps: Perturbation magnitude
        n_perturb: Number of perturbations per point
        rng: NumPy random generator
        device: Device to use

    Returns:
        flip_rates: (N,) fraction of perturbations that flip prediction
        delta_logits: (N,) mean absolute change in logits
        delta_internal: (N,) mean absolute change in internal value
    """
    from polytope_utils import sample_unit_directions

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = points.shape[1]
    flip_rates = []
    delta_logits = []
    delta_internal = []

    model.eval()
    for i, x0 in enumerate(points):
        directions = sample_unit_directions(n_perturb, in_dim, rng)
        x_pert = x0[None, :] + eps * directions
        xt = torch.from_numpy(x_pert.astype(np.float32)).to(device)

        with torch.no_grad():
            logits_p = model(xt).cpu().numpy().reshape(-1)
            internal_p = model.internal_scalar(xt).cpu().numpy().reshape(-1)

        orig_logit = logits_orig[i]
        orig_internal = internal_orig[i]

        # Compute flips
        pred_orig = (orig_logit > 0.0).astype(np.float32)
        pred_p = (logits_p > 0.0).astype(np.float32)
        flips = (pred_p != pred_orig).astype(np.float32)
        flip_rate = flips.mean()

        delta_l = np.mean(np.abs(logits_p - orig_logit))
        delta_int = np.mean(np.abs(internal_p - orig_internal))

        flip_rates.append(flip_rate)
        delta_logits.append(delta_l)
        delta_internal.append(delta_int)

    return np.array(flip_rates), np.array(delta_logits), np.array(delta_internal)


def compute_true_flip_rates(
    points: np.ndarray,
    eps: float,
    n_perturb: int,
    membership_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute true label flip rates under perturbations.

    Args:
        points: (N, D) probe points
        eps: Perturbation magnitude
        n_perturb: Number of perturbations per point
        membership_fn: Function that returns membership labels for points
        rng: NumPy random generator

    Returns:
        (N,) array of true flip rates
    """
    from polytope_utils import sample_unit_directions

    in_dim = points.shape[1]
    true_flip_rates = []

    for x0 in points:
        directions = sample_unit_directions(n_perturb, in_dim, rng)
        x_pert = x0[None, :] + eps * directions

        true_labels_p = membership_fn(x_pert)
        true_orig = membership_fn(x0[None, :])[0]
        true_flips = (true_labels_p != true_orig).astype(np.float32)
        true_flip_rates.append(true_flips.mean())

    return np.array(true_flip_rates)


# ============================================================================
# Plotting utilities
# ============================================================================

def plot_flip_rate_vs_eps(
    eps_vals: list[float],
    model_results: list[ModelProbeResults],
    true_flip_boundary: list[float],
    true_flip_interior: list[float],
    interior_ok: list[bool],
    save_path: str | None = None,
    title: str = "Flip rate vs eps",
) -> None:
    """Plot flip rates vs epsilon for multiple models."""
    plt.figure(figsize=(6, 4))

    markers = ['o', 's', '^', 'D']
    for i, results in enumerate(model_results):
        marker = markers[i % len(markers)]
        b_flip = [s.boundary_stats.model_flip for s in results.eps_summaries]
        i_flip = [s.interior_stats.model_flip for s in results.eps_summaries]

        plt.plot(eps_vals, b_flip, f'-{marker}', label=f'boundary ({results.name})')
        plt.plot(eps_vals, i_flip, f'--{marker}', label=f'interior ({results.name})')

    plt.plot(eps_vals, true_flip_boundary, '-x', label='boundary true flip')
    plt.plot(eps_vals, true_flip_interior, '-x', label='interior true flip')

    # Mark eps values where interior_ok is False
    for e_val, ok in zip(eps_vals, interior_ok):
        if not ok:
            plt.axvline(e_val, color='gray', linestyle=':', alpha=0.6)

    plt.xlabel('eps')
    plt.ylabel('flip rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")


def plot_boundary_flip_vs_eps(
    eps_vals: list[float],
    model_results: list[ModelProbeResults],
    true_flip: list[float],
    interior_ok: list[bool],
    save_path: str | None = None,
) -> None:
    """Plot boundary flip rates vs epsilon."""
    plt.figure(figsize=(6, 4))

    markers = ['o', 's', '^', 'D']
    for i, results in enumerate(model_results):
        marker = markers[i % len(markers)]
        b_flip = [s.boundary_stats.model_flip for s in results.eps_summaries]
        plt.plot(eps_vals, b_flip, f'-{marker}', label=f'model_flip ({results.name})')

    plt.plot(eps_vals, true_flip, '-x', label='true_flip')

    for e_val, ok in zip(eps_vals, interior_ok):
        if not ok:
            plt.axvline(e_val, color='gray', linestyle=':', alpha=0.6)

    plt.xlabel('eps')
    plt.ylabel('flip rate')
    plt.title('Boundary flip rate vs eps')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")


def plot_interior_flip_vs_eps(
    eps_vals: list[float],
    model_results: list[ModelProbeResults],
    true_flip: list[float],
    interior_ok: list[bool],
    save_path: str | None = None,
) -> None:
    """Plot interior flip rates vs epsilon."""
    plt.figure(figsize=(6, 4))

    markers = ['o', 's', '^', 'D']
    for i, results in enumerate(model_results):
        marker = markers[i % len(markers)]
        i_flip = [s.interior_stats.model_flip for s in results.eps_summaries]
        plt.plot(eps_vals, i_flip, f'-{marker}', label=f'model_flip ({results.name})')

    plt.plot(eps_vals, true_flip, '-x', label='true_flip')

    for e_val, ok in zip(eps_vals, interior_ok):
        if not ok:
            plt.axvline(e_val, color='gray', linestyle=':', alpha=0.6)

    plt.xlabel('eps')
    plt.ylabel('flip rate')
    plt.title('Interior flip rate vs eps')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")


def plot_boundary_vs_interior_bars(
    model_results: list[ModelProbeResults],
    save_path: str | None = None,
) -> None:
    """Plot bar chart comparing boundary vs interior metrics for all models."""
    labels = ["model_flip", "true_flip", "|Δlogit|", "|Δinternal|", "|grad|"]
    x = np.arange(len(labels))

    n_models = len(model_results)
    width = 0.8 / (2 * n_models)  # 2 bars per model (boundary + interior)

    plt.figure(figsize=(10, 4))

    for i, results in enumerate(model_results):
        # Aggregate stats (mean over eps values with valid interior)
        valid = [s for s in results.eps_summaries if s.interior_ok]
        if valid:
            b_stats = np.mean([s.boundary_stats.as_tuple() for s in valid], axis=0)
            i_stats = np.mean([s.interior_stats.as_tuple() for s in valid], axis=0)
        else:
            b_stats = np.full(5, np.nan)
            i_stats = np.full(5, np.nan)

        offset = (i - n_models / 2 + 0.5) * 2 * width
        plt.bar(x + offset - width/2, b_stats, width, label=f"boundary ({results.name})")
        plt.bar(x + offset + width/2, i_stats, width, label=f"interior ({results.name})")

    plt.xticks(x, labels)
    plt.ylabel("value")
    plt.title("Boundary vs Interior sensitivity (higher = more sensitive)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")


def plot_flip_rate_vs_distance(
    model_results: list[ModelProbeResults],
    eps: float,
    save_path: str | None = None,
) -> None:
    """Plot flip rate vs true min-face slack for all models."""
    plt.figure(figsize=(6, 4))

    markers = ['o', 'x', 's', '^']
    for i, results in enumerate(model_results):
        if results.probe_dists is not None and results.flip_rates is not None:
            marker = markers[i % len(markers)]
            plt.scatter(results.probe_dists, results.flip_rates,
                       s=20, alpha=0.7, marker=marker, label=f'model_flip ({results.name})')

    # Plot true flip rates from the first model that has them
    for results in model_results:
        if results.probe_dists is not None and results.true_flip_rates is not None:
            plt.scatter(results.probe_dists, results.true_flip_rates,
                       s=20, alpha=0.7, marker='+', label='true_flip')
            break

    plt.xlabel("true min face slack")
    plt.ylabel(f"flip rate (eps={eps:.3f})")
    plt.title("Flip rate vs true min-face slack")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
