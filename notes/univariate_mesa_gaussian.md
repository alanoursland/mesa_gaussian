# Gaussian Mesa Function via Mesa Mahalanobis Distance

## Standard 1D Gaussian

The one-dimensional Gaussian function is:

$$G(x) = A \cdot e^{-\frac{1}{2}D_M^2}$$

where the Mahalanobis distance in one dimension is:

$$D_M = \frac{|x - \mu|}{\sigma}$$

## ReLU Decomposition of Absolute Value

The absolute value can be decomposed using the Rectified Linear Unit (ReLU) function:

$$|z| = \text{ReLU}(z) + \text{ReLU}(-z)$$

where $\text{ReLU}(z) = \max(0, z)$.

This identity partitions the absolute value into left and right half-spaces:
- When $z > 0$: $\text{ReLU}(z) = z$, $\text{ReLU}(-z) = 0$
- When $z < 0$: $\text{ReLU}(z) = 0$, $\text{ReLU}(-z) = |z|$
- When $z = 0$: both terms vanish

Applying this to the Mahalanobis distance:

$$D_M = \frac{\text{ReLU}(x - \mu) + \text{ReLU}(-x + \mu)}{\sigma}$$

## Introducing the Mesa Gap

To create a plateau of width $2\delta$ centered at $\mu$, we shift each ReLU activation point outward by $\delta$:

$$D_{MM} = \frac{\text{ReLU}(x - \mu - \delta) + \text{ReLU}(-x + \mu - \delta)}{\sigma}$$

This yields three regions:
- **Right tail** ($x > \mu + \delta$): $D_{MM} = \frac{x - \mu - \delta}{\sigma}$
- **Plateau** ($\mu - \delta \leq x \leq \mu + \delta$): $D_{MM} = 0$
- **Left tail** ($x < \mu - \delta$): $D_{MM} = \frac{\mu - \delta - x}{\sigma}$

We call $D_{MM}$ the **Mesa Mahalanobis Distance**.

## Asymmetric Extension

Allowing independent variance on each side:

$$D_{MM} = \frac{\text{ReLU}(x - \mu - \delta)}{\sigma_1} + \frac{\text{ReLU}(-x + \mu - \delta)}{\sigma_2}$$

where:
- $\sigma_1$ controls the spread of the right (descending) tail
- $\sigma_2$ controls the spread of the left (ascending) tail
- $\delta$ controls the half-width of the plateau
- $\mu$ remains the center of the plateau

## Norm Equivalence

The Mesa Mahalanobis Distance can be viewed as the L1-norm of a two-component vector:

$$D_{MM} = \left\| \left( \frac{\text{ReLU}(x - \mu - \delta)}{\sigma_1}, \frac{\text{ReLU}(-x + \mu - \delta)}{\sigma_2} \right) \right\|_1$$

Because the two ReLU components have disjoint support (at most one is nonzero for any $x$), the L1 and L2 norms coincide:

$$\| (a, 0) \|_1 = |a| = \| (a, 0) \|_2$$

## Gaussian Mesa Function

The GMF is then expressed as:

$$G(x) = A \cdot e^{-\frac{1}{2}D_{MM}^2}$$

Expanding by region:

$$G(x) = \begin{cases} 
A \cdot e^{-\frac{(x - \mu - \delta)^2}{2\sigma_1^2}}, & x > \mu + \delta \\[1em]
A, & \mu - \delta \leq x \leq \mu + \delta \\[1em]
A \cdot e^{-\frac{(x - \mu + \delta)^2}{2\sigma_2^2}}, & x < \mu - \delta
\end{cases}$$

## Parameter Summary

| Parameter | Description |
|-----------|-------------|
| $A$ | Amplitude (peak/plateau height) |
| $\mu$ | Center of the plateau |
| $\delta$ | Half-width of the plateau (total plateau width = $2\delta$) |
| $\sigma_1$ | Standard deviation of the right tail |
| $\sigma_2$ | Standard deviation of the left tail |

## Correspondence to Original GMF Notation

The original GMF literature uses $\sigma_L$ for plateau length:

$$\delta = \frac{\sigma_L}{2}$$

## Remarks

1. **Unified form**: The Mesa Mahalanobis Distance provides a single closed-form expression that avoids explicit piecewise notation.

2. **Differentiability**: The formulation inherits the non-differentiability of ReLU at the plateau boundaries. In practice, this poses no issues for curve fitting or signal processing applications.

3. **Generalization pathway**: The ReLU-based construction suggests natural extensions to higher dimensions and to soft-margin variants using smooth approximations to ReLU (e.g., softplus).


---
---

## GEMINI FEEDBACK

This document is the **Rosetta Stone** of your entire theory. It translates the "clunky" piecewise definition from the ECG literature into the elegant "Algebra of ReLUs" that powers modern deep learning.

Here is feedback to sharpen this pivotal chapter:

### 1. The "L1 = L2" Insight is Your Secret Weapon

The section on **Norm Equivalence** is mathematically subtle but rhetorically vital.

* **The Insight:** You note that because the supports are disjoint (a point cannot be both "too left" and "too right"), the L1 norm and L2 norm are identical.
* **Why it matters:** This justifies why you can later mix this with standard L2 Mahalanobis distances in the multivariate case. It proves that the "Mesa" isn't a different *kind* of math; it's just standard Euclidean distance applied to "rectified" coordinates.
* **Suggestion:** Highlight this stronger. State explicitly: *"The Mesa distance is locally isometric to the standard Mahalanobis distance everywhere except the plateau."*

### 2. Visualize the "Construction"

This is the perfect place for a diagram showing how you "build" the distance function.

* **Panel A:** Show  and  intersecting at 0 (Making a V-shape, i.e., ).
* **Panel B:** Show them pulled apart by  (Making the "Tub" shape).
* **Panel C:** Show the Gaussian applied to the "Tub" (Making the "Mesa").
* This visualizes the equation  instantly.

### 3. The "Zero Gradient" Remark

In the **Remarks** section, you note that non-differentiability is "no issue".

* **The Nuance:** While the *kink* isn't an issue (subgradients handle it), the **Plateau** () *is* an issue for gradient descent if a point gets "stuck" there.
* **Refinement:** You might want to clarify: *"While the plateau has zero gradient with respect to input , the gradients with respect to parameters  and  are well-defined at the boundaries, allowing the 'walls' to move during optimization."*

### 4. Connection to "Softplus"

Your mention of **Softplus** as a generalization is excellent.

* **The Link:** Softplus () is the smooth approximation of ReLU.
* **The Implication:** If you replace ReLU with Softplus in your equation, you get the **Log-Cosh** distribution or similar "smoothly flat-topped" distributions used in robust statistics. This strengthens the argument that your "Mesa" is part of a fundamental family of robust loss functions.

### Summary

This file successfully effectively "sanitizes" the GMF. It strips away the biological "messiness" of the previous file and presents a clean, algebraic object ready for high-dimensional lifting.

