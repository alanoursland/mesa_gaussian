"""
Utility functions for generating and working with convex polytopes.
"""
from __future__ import annotations

import math
import numpy as np


def make_random_polytope(
    in_dim: int,
    n_faces: int,
    radius_scale: float = 1.0,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random convex polytope defined by half-space constraints.

    The polytope is defined as: {x : n_i^T x <= r_i for all i}

    Args:
        in_dim: Input dimension
        n_faces: Number of faces (half-space constraints)
        radius_scale: Scale factor for the radii
        seed: Random seed for reproducibility

    Returns:
        normals: (n_faces, in_dim) array of unit normal vectors
        radii: (n_faces,) array of offsets (all positive, so origin is inside)
    """
    rng = np.random.default_rng(seed)

    if in_dim == 2:
        # In 2D, make normals approximately evenly spread to avoid unbounded/skinny regions
        angles = np.linspace(0, 2 * math.pi, n_faces, endpoint=False)
        angles += rng.uniform(-0.2, 0.2, size=angles.shape)  # small jitter
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    else:
        normals = rng.normal(size=(n_faces, in_dim))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms

    # Choose positive offsets so the origin is inside
    radii = rng.uniform(0.4, 1.0, size=(n_faces,)) * radius_scale

    return normals.astype(np.float32), radii.astype(np.float32)


def polytope_membership(
    x: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray
) -> np.ndarray:
    """Check if points are inside the polytope.

    Args:
        x: (N, D) array of points
        normals: (M, D) array of face normals
        radii: (M,) array of face offsets

    Returns:
        (N,) array of floats (1.0 if inside, 0.0 if outside)
    """
    proj = x @ normals.T  # (N, M)
    slack = radii[None, :] - proj  # positive if inside that face
    inside = np.all(slack >= 0.0, axis=1)
    return inside.astype(np.float32)


def min_face_slack(
    x: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray
) -> np.ndarray:
    """Return signed minimum face slack: min_i (r_i - n_i·x).

    Positive values mean the point is inside (distance to nearest face along normal).
    Negative values mean outside.

    Note: This is NOT Euclidean distance to the polytope boundary.

    Args:
        x: (N, D) array of points
        normals: (M, D) array of face normals
        radii: (M,) array of face offsets

    Returns:
        (N,) array of minimum slack values
    """
    proj = x @ normals.T
    slack = radii[None, :] - proj
    return np.min(slack, axis=1)


def sample_unit_directions(
    n: int,
    dim: int,
    rng: np.random.Generator
) -> np.ndarray:
    """Sample n random unit vectors in the given dimension.

    Args:
        n: Number of vectors to sample
        dim: Dimension of each vector
        rng: NumPy random generator

    Returns:
        (n, dim) array of unit vectors
    """
    v = rng.normal(size=(n, dim))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


def generate_polytope_dataset(
    normals: np.ndarray,
    radii: np.ndarray,
    n_samples: int,
    box_size: float = 1.5,
    rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of points with polytope membership labels.

    Args:
        normals: (M, D) array of face normals
        radii: (M,) array of face offsets
        n_samples: Number of samples to generate
        box_size: Half-size of the sampling box
        rng: NumPy random generator

    Returns:
        x: (n_samples, D) array of points
        y: (n_samples, 1) array of labels (1.0 inside, 0.0 outside)
    """
    if rng is None:
        rng = np.random.default_rng()

    in_dim = normals.shape[1]
    x = rng.uniform(-box_size, box_size, size=(n_samples, in_dim)).astype(np.float32)
    y = polytope_membership(x, normals, radii).reshape(-1, 1)

    return x, y
