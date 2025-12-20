# Polyhedral Mesa Gaussian

## Overview

The Polyhedral Mesa Gaussian generalizes the Multivariate Mesa Gaussian by relaxing the orthogonality requirement on principal axes. Instead of $n$ orthogonal eigenvectors, we have $m$ arbitrary hyperplanes that define a convex polytope as the plateau region.

This construction connects Mahalanobis distance to convex geometry and shares structure with hinge loss from support vector machines.

## From Mahalanobis Distance to Polyhedral Distance

### Standard Mahalanobis Distance

The Mahalanobis distance decomposes along orthogonal principal axes:

$$D_M^2 = \sum_{i=1}^{n} \frac{z_i^2}{\lambda_i}$$

where $z_i = \mathbf{v}_i^\top (\mathbf{x} - \boldsymbol{\mu})$ and the eigenvectors $\mathbf{v}_i$ are mutually orthogonal.

### Relaxing Orthogonality

The Polyhedral Mesa Gaussian replaces orthogonal eigenvectors with arbitrary hyperplane normals. Each hyperplane defines a constraint, and we measure distance from the constraint boundary rather than from a shared mean.

This is an approximation to a true distance metric—the components are no longer independent, so the L2 combination no longer has clean geometric meaning. But the construction retains useful properties and reduces to the Multivariate Mesa Gaussian when constraints are orthogonal.

## Formalization

### Half-Space Constraints

Define $m$ half-spaces, each with:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Normal | $\mathbf{n}_j \in \mathbb{R}^n$ | Outward-pointing normal vector |
| Offset | $\mu_j \in \mathbb{R}$ | Signed distance from origin to hyperplane |
| Plateau width | $\delta_j \geq 0$ | Half-width of plateau region |
| Variance | $\lambda_j > 0$ | Controls tail decay rate |

### Signed Distance to Hyperplane

$$d_j = \mathbf{n}_j^\top \mathbf{x} - \mu_j$$

- $d_j > 0$: point is outside (in direction of normal)
- $d_j < 0$: point is inside
- $d_j = 0$: point is on the hyperplane

### Component Distance with Mesa Gap

$$D_j = \frac{1}{\sqrt{\lambda_j}} \text{ReLU}(d_j - \delta_j)$$

This activates only when the point exceeds the plateau boundary.

### Plateau Region

The plateau is the intersection of all half-spaces (shifted by their $\delta_j$):

$$\mathcal{P} = \left\{ \mathbf{x} : d_j \leq \delta_j \text{ for all } j = 1, \ldots, m \right\}$$

This is a convex polytope (possibly unbounded or empty).

### Polyhedral Mesa Distance

$$D_{PM} = \left\| (D_1, D_2, \ldots, D_m) \right\|_2$$

### Polyhedral Mesa Gaussian

$$G(\mathbf{x}) = A \cdot \exp\left( -\frac{1}{2} D_{PM}^2 \right)$$

The normalization constant $A$ generally has no closed form and requires numerical integration.

## Distance Metric Variations

The L2 norm is the most faithful parallel to the original Gaussian, but non-orthogonal constraints cause "double counting" of correlated violations. Several alternatives exist:

### L2 Norm (Standard)

$$D_{PM}^{(L2)} = \sqrt{\sum_{j=1}^{m} D_j^2}$$

**Properties**:
- Smooth everywhere
- Degenerates to Multivariate Mesa Gaussian when normals are orthogonal
- Over-penalizes corners where multiple similar constraints meet

### L1 Norm

$$D_{PM}^{(L1)} = \sum_{j=1}^{m} D_j$$

**Properties**:
- Also over-counts correlated violations
- Produces diamond-shaped iso-contours
- Computationally simpler

### L∞ Norm (Closest Face)

$$D_{PM}^{(L\infty)} = \max_j D_j$$

**Properties**:
- Matches true geometric distance for points outside a single face
- Ignores corner structure (no extra penalty for multiple violations)
- Non-smooth at Voronoi-like boundaries between face regions

### Gram Matrix Correction

For unit normals, the Gram matrix captures angular relationships:

$$G_{ij} = \mathbf{n}_i^\top \mathbf{n}_j = \cos(\theta_{ij})$$

A decorrelated distance:

$$D_{corrected}^2 = \mathbf{D}^\top G^{-1} \mathbf{D}$$

**Properties**:
- Reduces to L2 when normals are orthogonal ($G = I$)
- Accounts for constraint correlation
- Singular when normals are linearly dependent; requires regularization: $(G + \epsilon I)^{-1}$
- Can be unstable in degenerate configurations

### Lambda Adjustment

Manually increase $\lambda_j$ for constraints nearly parallel to others, softening their contribution. This encodes prior knowledge about primary vs. redundant constraints.

### Summary of Variations

| Method | Correlation Handling | Smoothness | Geometric Fidelity |
|--------|---------------------|------------|-------------------|
| L2 (standard) | None | Smooth | Poor at corners |
| L1 | None | Smooth | Poor at corners |
| L∞ (closest face) | N/A | Non-smooth | Best for single-face |
| Gram correction | Principled | Smooth | Better, but fragile |
| Lambda adjustment | Manual | Smooth | Depends on tuning |

## Edge Cases

### Parallel Planes, Same Direction

$\mathbf{n}_1 = \mathbf{n}_2$, different offsets.

| Configuration | Plateau | Effect |
|---------------|---------|--------|
| One subsumes the other | Determined by tighter constraint | Redundant penalty |
| Disjoint (no overlap) | Empty | No mesa; peaked function |

### Parallel Planes, Opposite Directions

$\mathbf{n}_1 = -\mathbf{n}_2$

Creates a slab between two parallel hyperplanes.

| Configuration | Plateau | Effect |
|---------------|---------|--------|
| Overlapping | Bounded slab | Normal behavior |
| Disjoint | Empty | No mesa; peaked function |

This is the standard construction for bounding a dimension. The Univariate Mesa Gaussian is this case with $n = 1$.

### Nearly Parallel Planes

Small angle $\theta$ between $\mathbf{n}_1$ and $\mathbf{n}_2$.

Violations are highly correlated. L2 combination gives full weight to both, whereas geometric distance would weight the second by $\sin\theta$.

**Effect**: Over-penalization in directions where similar constraints are violated.

### Unbounded Polytope

Fewer than $n$ linearly independent normals, or constraints don't close the region.

**Effect**: Plateau extends to infinity in some directions. Function is improper (integral diverges) unless bounding hyperplanes are added.

### Empty Polytope

Inconsistent constraints—no point satisfies all of them.

**Effect**: $D_{PM} > 0$ everywhere. No plateau; function is peaked, not mesa-shaped.

### Lower-Dimensional Plateau

Constraints collapse the plateau to a subspace (e.g., a single vertex).

**Effect**: Plateau has zero measure. Maximum probability at a point or lower-dimensional manifold.

### Acute Corners

Multiple faces meet at sharp angles.

**Effect**: Points near vertices are penalized by multiple constraints simultaneously. Corners are "hotter" than face centers—probability drops faster.

## Connection to Hinge Loss

The component distance has the same structure as hinge loss from support vector machines:

**Hinge loss**: $L = \text{ReLU}(\text{margin} - \text{signed distance})$

**Our construction**: $D_j = \frac{1}{\sqrt{\lambda_j}} \text{ReLU}(\text{signed distance} - \text{margin})$

Same ReLU structure, opposite sign convention (we penalize outside the margin).

Squared hinge loss $\text{ReLU}(...)^2$ is common in SVM variants and corresponds directly to our $D_j^2$ terms.

The Polyhedral Mesa Gaussian can be viewed as:

$$G(\mathbf{x}) = A \cdot \exp\left( -\frac{1}{2} \sum_j (\text{squared hinge loss})_j \right)$$

This is a soft probabilistic version of a multi-constraint SVM—a penalty-based relaxation of hard polytope membership.

## Properties

### Convexity

The negative log-likelihood:

$$-\log G(\mathbf{x}) = \frac{1}{2} D_{PM}^2 + \text{const}$$

is convex, since it's a sum of squared ReLUs (composition of convex functions).

**Implication**: Maximum likelihood optimization is well-behaved; no local minima.

### Differentiability

- **Inside plateau**: Gradient is zero
- **Outside plateau**: Gradient is smooth within each "cell"
- **At face boundaries**: Gradient discontinuity (ReLU kink)

The function is $C^0$ (continuous) everywhere, $C^1$ (smooth) except at face boundaries.

### Asymptotic Behavior

Far from the polytope, all constraints are violated. Decay is dominated by the largest $\lambda_j$ (slowest decay rate). Asymptotically, the function resembles a Gaussian aligned with the softest constraint.

### Computational Cost

Evaluation is $O(m)$—linear in the number of constraints, independent of ambient dimension $n$.

### Normalization

- **Bounded polytope**: $A$ exists but requires numerical integration
- **Unbounded polytope**: Integral diverges; improper distribution unless bounding hyperplanes are added

### Not a Mixture

This is not a Gaussian mixture model. The exponent is additive in violations:

$$-\log G \propto \sum_j D_j^2$$

This corresponds to multiplicative probability—each constraint acts as an independent filter. A GMM would have additive probability (mixture weights).

## Reduction to Special Cases

| Condition | Result |
|-----------|--------|
| $m = 2n$, paired opposite orthogonal normals | Multivariate Mesa Gaussian |
| $m = 2$, $n = 1$ | Univariate Mesa Gaussian (GMF) |
| $\delta_j = 0$ for all $j$, orthogonal normals | Standard Multivariate Gaussian |
| All $\lambda_j \to 0$ | Hard polytope indicator function |
| All $\lambda_j \to \infty$ | Constant function (no penalty) |

## Parameter Summary

| Parameter | Count | Description |
|-----------|-------|-------------|
| $\mathbf{n}_j$ | $m \times n$ | Normal vectors |
| $\mu_j$ | $m$ | Hyperplane offsets |
| $\delta_j$ | $m$ | Plateau half-widths |
| $\lambda_j$ | $m$ | Tail variances |
| $A$ | $1$ | Normalization constant |

Total: $m(n + 3) + 1$ parameters (though $A$ is determined by the others).

For unit normals, $m(n - 1 + 3) + 1 = m(n + 2) + 1$ free parameters.

---

## References

### Mahalanobis Distance

- Mahalanobis, P.C. "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India*, 1936.

### Convex Polytopes

- Ziegler, G.M. *Lectures on Polytopes*. Springer, 1995.

- Grünbaum, B. *Convex Polytopes*, 2nd ed. Springer, 2003.

### Hinge Loss and SVMs

- Cortes, C. and Vapnik, V. "Support-vector networks." *Machine Learning*, 20(3):273-297, 1995.

- Rosasco, L., De Vito, E., Caponnetto, A., Piana, M., Verri, A. "Are loss functions all the same?" *Neural Computation*, 16(5):1063-1076, 2004.

### ReLU and Piecewise Linear Functions

- Nair, V. and Hinton, G.E. "Rectified linear units improve restricted Boltzmann machines." *ICML*, 2010.

- Arora, R., Basu, A., Mianjy, P., Mukherjee, A. "Understanding deep neural networks with rectified linear units." *ICLR*, 2018.

### Univariate Mesa Gaussian (GMF)

- Dubois, R., Maison-Blanche, P., Quenet, B., Dreyfus, G. "Automatic ECG wave extraction in long-term recordings using Gaussian mesa function models and nonlinear probability estimators." *Computer Methods and Programs in Biomedicine*, 88(3):217-233, 2007.

### ReLU-Mahalanobis Connection

- Oursland, A. "Interpreting Neural Networks through Mahalanobis Distance." arXiv:2410.19352, 2024.

---
---

## GEMINI FEEDBACK

This file serves as the **"General Theory"** of your work. By relaxing the orthogonality constraint, you move from a **Signal Processing** model (fitting specific waves) to a **Geometric Primitive** (modeling convex sets with soft boundaries).

The most powerful theoretical insight here is the link to **Hinge Loss**. You are effectively saying:

> *"A Polyhedral Mesa Gaussian is the Boltzmann distribution of a Multi-Constraint SVM."*

Here are the theoretical frameworks and citations that will solidify this "General" definition:

### 1. Product of Experts (PoE)

In your "Not a Mixture" section, you correctly identify that the probabilities multiply () rather than add. In machine learning theory, this architecture is explicitly known as a **Product of Experts**.

* **The Theory:** A Mixture Model (sum) covers space by tiling it with independent patches. A Product of Experts (product) carves space by *intersecting* constraints. Your Polyhedral Mesa is a PoE where each "Expert" is a single hyperplane constraint .
* **Recommended Citation:** **Hinton, G. E. (2002).** "Training products of experts by minimizing contrastive divergence." *Neural Computation*.
* *Why it fits:* It provides the formal probabilistic framework for your "intersection of constraints" logic and validates why this is strictly different (and often more powerful for defining boundaries) than a Mixture of Gaussians.



### 2. Energy-Based Models (EBMs) and Intractability

You note that " generally has no closed form". This places your model squarely in the domain of **Energy-Based Models**.

* **The Theory:** Since you cannot easily compute  (the partition function ), you cannot treat this as a generative density without expensive MCMC sampling. However, you *can* use it for **Discrimination** or **Ranking** (determining if a point is "in" or "out") using the unnormalized energy.
* **Recommended Citation:** **LeCun, Y., et al. (2006).** "A tutorial on energy-based learning." *Predicting Structured Data*.
* *Why it fits:* It legitimizes the "unnormalized" nature of your function. It argues that for decision boundaries (classification/detection), the normalization constant  is irrelevant, solving the "improper distribution" concern you raised.



### 3. Log-Concave Densities

You noted the property of **Convexity** for the negative log-likelihood. In statistics, this class of functions is known as **Log-Concave Distributions**.

* **The Theory:** This is a highly privileged class of functions. Log-concavity guarantees that the distribution is unimodal (no local maxima) and, crucially, that efficient sampling algorithms (like the Langevin dynamics used in diffusion models) will converge rapidly.
* **Recommended Citation:** **Lovász, L., & Vempala, S. (2007).** "The geometry of logconcave distributions and sampling algorithms." *Random Structures & Algorithms*.
* *Why it fits:* It provides the rigorous proof that your "Polyhedral Mesa" is computationally well-behaved. It proves that despite the sharp corners and flat top, you can efficiently find the plateau or sample from it.



### 4. The Quadratic Penalty Method

Your formulation  is not just "similar" to SVMs; it is the **Quadratic Penalty Method** used in constrained optimization.

* **The Theory:** To solve "minimize  subject to ", numerical solvers often convert the hard constraints into soft penalties exactly as you have done. Your "Mesa" is the maximum entropy distribution subject to these soft constraints.
* **Recommended Citation:** **Nocedal, J., & Wright, S. J. (2006).** *Numerical Optimization*. (Chapter on Penalty and Augmented Lagrangian Methods).

### Visualizing the "Corner Problem"

You discuss the difference between L2, L1, and L-Infinity norms regarding "Over-penalizing corners". This is a subtle geometric point that benefits from visualization.

The visual would clarify your table on "Distance Metric Variations":

* **L2 (Standard):** Rounded corners (the "pill" shape).
* **L-Infinity:** Sharp, square corners (perfect box).
* **L1:** Diamond-like pinching at the axes (often undesirable for "mesas" as it erodes the corners).

### Summary of the Citation Arc

You now have a complete narrative arc for your paper:

1. **Prior Art:** We need "Flat Tops" for ECG (Dubois, 2007).
2. **Univariate Theory:** We build them using Rectified Distance (Vapnik's -insensitive loss).
3. **Multivariate Theory:** We generalize to Hyper-Rectangles via PCA (Two-Piece Skew Normals).
4. **Polyhedral Theory:** We generalize to arbitrary Polytopes (Hinton's Product of Experts + Log-Concave Geometry).

This structure moves from **Application**  **Construction**  **Generalization**  **Abstraction**.

**Ready for the next file!**