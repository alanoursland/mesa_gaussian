# Multivariate Mesa Gaussian via Mesa Mahalanobis Distance

## Standard Multivariate Gaussian

The multivariate Gaussian in $\mathbb{R}^n$ is:

$$G(\mathbf{x}) = A \cdot \exp\left( -\frac{1}{2} D_M^2 \right)$$

where $A$ is the normalization constant and the Mahalanobis distance is:

$$D_M = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

with:
- $\mathbf{x} \in \mathbb{R}^n$ — input vector
- $\boldsymbol{\mu} \in \mathbb{R}^n$ — mean vector
- $\Sigma \in \mathbb{R}^{n \times n}$ — positive definite covariance matrix

## PCA Decomposition

The covariance matrix admits eigendecomposition:

$$\Sigma = V \Lambda V^\top$$

where:
- $V = [\mathbf{v}_1 | \mathbf{v}_2 | \cdots | \mathbf{v}_n]$ — orthogonal matrix of eigenvectors (principal axes)
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ — eigenvalues (variances along principal axes)

Define the scalar projection onto the $i$-th principal axis:

$$z_i = \mathbf{v}_i^\top (\mathbf{x} - \boldsymbol{\mu})$$

## Component-wise Mahalanobis Distance

The squared Mahalanobis distance decomposes as:

$$D_M^2 = \sum_{i=1}^{n} \frac{z_i^2}{\lambda_i}$$

Define the component-wise Mahalanobis distance:

$$D_{M,i} = \frac{|z_i|}{\sqrt{\lambda_i}}$$

Then:

$$D_M = \left\| (D_{M,1}, D_{M,2}, \ldots, D_{M,n}) \right\|_2$$

## ReLU Decomposition

Using the identity $|z| = \text{ReLU}(z) + \text{ReLU}(-z)$, each component-wise distance splits into two half-components:

$$D_{M,i,0} = \frac{1}{\sqrt{\lambda_i}} \text{ReLU}(z_i)$$

$$D_{M,i,1} = \frac{1}{\sqrt{\lambda_i}} \text{ReLU}(-z_i)$$

The full Mahalanobis distance is the L2-norm over all $2n$ half-components:

$$D_M = \left\| (D_{M,1,0}, D_{M,1,1}, \ldots, D_{M,n,0}, D_{M,n,1}) \right\|_2$$

Since the half-components have disjoint support (at most one of $D_{M,i,0}$ or $D_{M,i,1}$ is nonzero for each $i$), this is equivalent to the original formulation.

## Introducing the Mesa Gap

To create a plateau region along each principal axis, shift each ReLU activation outward by $\delta_i$:

$$D_{M,i,0} = \frac{1}{\sqrt{\lambda_i}} \text{ReLU}\left( \mathbf{v}_i^\top (\mathbf{x} - \boldsymbol{\mu} - \delta_i \mathbf{v}_i) \right)$$

$$D_{M,i,1} = \frac{1}{\sqrt{\lambda_i}} \text{ReLU}\left( \mathbf{v}_i^\top (-\mathbf{x} + \boldsymbol{\mu} - \delta_i \mathbf{v}_i) \right)$$

Since $\mathbf{v}_i^\top \mathbf{v}_i = 1$, these simplify to:

$$D_{M,i,0} = \frac{1}{\sqrt{\lambda_i}} \text{ReLU}(z_i - \delta_i)$$

$$D_{M,i,1} = \frac{1}{\sqrt{\lambda_i}} \text{ReLU}(-z_i - \delta_i)$$

This yields three regions per axis:
- **Positive tail** ($z_i > \delta_i$): $D_{M,i,0} > 0$, $D_{M,i,1} = 0$
- **Plateau** ($-\delta_i \leq z_i \leq \delta_i$): $D_{M,i,0} = 0$, $D_{M,i,1} = 0$
- **Negative tail** ($z_i < -\delta_i$): $D_{M,i,0} = 0$, $D_{M,i,1} > 0$

## Asymmetric Scale Parameters

For full generality matching the univariate GMF, allow independent scale parameters for each half-axis:

$$D_{M,i,0} = \frac{1}{\sqrt{\lambda_{i,0}}} \text{ReLU}(z_i - \delta_i)$$

$$D_{M,i,1} = \frac{1}{\sqrt{\lambda_{i,1}}} \text{ReLU}(-z_i - \delta_i)$$

where:
- $\lambda_{i,0}$ — variance of the positive tail along axis $i$
- $\lambda_{i,1}$ — variance of the negative tail along axis $i$

## Multivariate Mesa Mahalanobis Distance

The full distance is the L2-norm over all $2n$ half-components:

$$D_{MM} = \left\| (D_{M,1,0}, D_{M,1,1}, D_{M,2,0}, D_{M,2,1}, \ldots, D_{M,n,0}, D_{M,n,1}) \right\|_2$$

The disjoint support property still holds: at most one of $D_{M,i,0}$ or $D_{M,i,1}$ is nonzero for each $i$.

## Multivariate Mesa Gaussian

$$G(\mathbf{x}) = A \cdot \exp\left( -\frac{1}{2} D_{MM}^2 \right)$$

## Plateau Geometry

The plateau region ($D_{MM} = 0$) occurs when all components are simultaneously in their plateaus:

$$\mathcal{P} = \left\{ \mathbf{x} : |z_i| \leq \delta_i \text{ for all } i \right\}$$

In principal component space, this is a **hyperrectangle** with half-widths $\delta_1, \delta_2, \ldots, \delta_n$.

In the original space, this hyperrectangle is rotated by $V$, yielding an oriented box centered at $\boldsymbol{\mu}$.

## L1-Norm Approximation

The L1-norm alternative:

$$D_{MM}^{(L1)} = \sum_{i=1}^{n} \left( D_{M,i,0} + D_{M,i,1} \right)$$

### Norm Equivalence Bounds

For any vector $\mathbf{d} \in \mathbb{R}^{2n}$:

$$\|\mathbf{d}\|_2 \leq \|\mathbf{d}\|_1 \leq \sqrt{2n} \|\mathbf{d}\|_2$$

However, due to disjoint support, if $k$ components are outside their plateaus:

$$\|\mathbf{d}\|_2 \leq \|\mathbf{d}\|_1 \leq \sqrt{k} \|\mathbf{d}\|_2$$

When $k$ is small (most dimensions in plateau), L1 and L2 are close regardless of ambient dimension $n$.

### Geometric Interpretation

The L1-norm produces diamond-shaped (cross-polytope) iso-contours rather than ellipsoidal, with gradient discontinuities at coordinate hyperplanes. The decay structure $\exp(-\frac{1}{2}d^2)$ is preserved.

## Reduction to Special Cases

| Condition | Result |
|-----------|--------|
| $n = 1$ | Univariate Mesa Gaussian (GMF) |
| $\delta_i = 0$ for all $i$ | Standard multivariate Gaussian |
| $\lambda_{i,0} = \lambda_{i,1}$ for all $i$ | Symmetric multivariate mesa Gaussian |
| $\Sigma = \sigma^2 I$, equal $\delta_i$ | Isotropic mesa Gaussian with hypercube plateau |

## Parameter Summary

| Parameter | Count | Description |
|-----------|-------|-------------|
| $\boldsymbol{\mu}$ | $n$ | Center of plateau |
| $V$ | $n(n-1)/2$ | Principal axis orientations (orthogonal matrix, $n(n-1)/2$ free parameters) |
| $\lambda_{i,0}$ | $n$ | Positive-side variances |
| $\lambda_{i,1}$ | $n$ | Negative-side variances |
| $\delta_i$ | $n$ | Half-widths of plateau per axis |
| $A$ | $1$ | Amplitude |

Total free parameters: $\frac{n(n-1)}{2} + 4n + 1$

For comparison, a standard multivariate Gaussian has $n + \frac{n(n+1)}{2}$ parameters (mean plus symmetric covariance).

---

## References

### Background

- Mahalanobis, P.C. "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India*, 1936.

- Jolliffe, I.T. *Principal Component Analysis*, 2nd ed. Springer, 2002.

- Horn, R.A. and Johnson, C.R. *Matrix Analysis*, 2nd ed. Cambridge University Press, 2012.

### Norm Equivalence and High-Dimensional Geometry

- Vershynin, R. *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge University Press, 2018.

- Ledoux, M. *The Concentration of Measure Phenomenon*. American Mathematical Society, 2001.

### Univariate GMF (Prior Art)

- Dubois, R., Maison-Blanche, P., Quenet, B., Dreyfus, G. "Automatic ECG wave extraction in long-term recordings using Gaussian mesa function models and nonlinear probability estimators." *Computer Methods and Programs in Biomedicine*, 88(3):217-233, 2007.

- Badilini, F., et al. "Automatic analysis of cardiac repolarization morphology using Gaussian mesa function modeling." *Journal of Electrocardiology*, 41(6):588-594, 2008.

### ReLU-Mahalanobis Connection

- Oursland, A. "Interpreting Neural Networks through Mahalanobis Distance." arXiv:2410.19352, 2024.