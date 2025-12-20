# Polyhedral Mesa Gaussians as Foundational Representation in ReLU Networks

## Introduction

This document presents a theoretical interpretation of standard feedforward neural networks with ReLU activations. We propose that such networks implicitly compute and compose Polyhedral Mesa Gaussian structures.

**This is purely an interpretive framework.** We do not propose any new architecture, training procedure, or modification to existing networks. The goal is to provide a geometric and probabilistic lens through which to understand what ReLU networks already compute.

## The Core Claim

> Standard feedforward ReLU networks compute hierarchical compositions of Polyhedral Mesa Gaussians. Each layer defines a set of hyperplane constraints, and the network's representation at each stage can be understood as a distance profile relative to learned polytopes.

## Single ReLU Neuron as Half-Space Distance

A single ReLU neuron computes:

$$h = \text{ReLU}(\mathbf{w}^\top \mathbf{x} + b)$$

This is equivalent to a one-sided distance from a hyperplane:

$$h = \text{ReLU}\left( \frac{1}{\sqrt{\lambda}} \mathbf{v}^\top (\mathbf{x} - \boldsymbol{\mu}) \right)$$

where:
- $\mathbf{v} = \frac{\mathbf{w}}{\|\mathbf{w}\|}$ is the unit normal
- $\sqrt{\lambda} = \frac{1}{\|\mathbf{w}\|}$ defines the scale
- $\boldsymbol{\mu}$ satisfies $\mathbf{v}^\top \boldsymbol{\mu} = -\frac{b}{\|\mathbf{w}\|}$

The neuron outputs zero when $\mathbf{x}$ is on the "inside" of the hyperplane (the plateau side) and outputs positive values proportional to the signed distance when $\mathbf{x}$ is on the "outside."

This corresponds exactly to the half-component distance $D_{M,i,j}$ from the Mesa Mahalanobis framework.

## Two-Layer Network

Consider:

$$\mathbf{h} = \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{y} = W_2 \mathbf{h} + \mathbf{b}_2$$

### Layer 1: Defining Hyperplane Constraints

Each row of $W_1$ defines a hyperplane in input space. The ReLU activations compute the vector of half-component distances:

$$\mathbf{h} = (D_1, D_2, \ldots, D_m)$$

where $D_j = 0$ when inside the $j$-th constraint and $D_j > 0$ when outside.

### Layer 2: Combining Distances

With positive weights, the second layer computes weighted sums of distances:

$$y_k = \sum_j w_{kj} D_j + b_k$$

This is an L1-norm-like combination of constraint violations. Each output can be interpreted as an approximate Mesa Mahalanobis distance for a different polytope, where:
- The weights $w_{kj}$ adjust the relative importance (effective $\lambda_j$) of each constraint
- Zero weights deselect constraints from participating in that polytope
- The bias $b_k$ shifts the overall threshold

## Positive Weights as Intersection

When all weights are positive, the second layer computes a weighted combination of violations. The region where the output is minimal (near zero) corresponds to the **intersection** of the selected half-spaces.

Adding more constraints (more positive weights) can only shrink or maintain the mesa—never grow it. This follows from convex geometry: intersection of half-spaces produces a convex polytope, and adding constraints can only tighten the intersection.

**Key property**: The mesa (region of minimal output) is always convex.

## Negative Weights

Negative weights introduce complexity that is not fully addressed by this framework.

Intuitively, negative weights might implement:
- **Relative distance**: comparing distance to one set of constraints versus another
- **Union-like operations**: relaxing the intersection by excluding or inverting constraints
- **Competing polytopes**: evidence for one class is evidence against another

However, union of convex sets is not generally convex, and negative weights can create non-convex decision regions. A complete treatment of negative weights and their geometric interpretation is outside the scope of this paper.

For the present analysis, we focus on the positive-weight case, which corresponds cleanly to polytope intersection and the Mesa Gaussian interpretation.

## Shared Hyperplanes

A network with $m$ neurons in the first layer defines $m$ hyperplanes. These hyperplanes are **shared vocabulary**—the second layer can combine them into multiple different polytopes.

Each output neuron in layer 2 selects a subset of hyperplanes (via non-zero weights) and weights their contributions. The same hyperplane can participate in multiple mesas, just as the same constraint can be relevant to multiple classes.

This provides parameter efficiency: $m$ hyperplanes can define exponentially many possible polytopes through different combinations.

## Deeper Networks

Consider a three-layer network:

$$\mathbf{h}_1 = \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \text{ReLU}(W_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$\mathbf{y} = W_3 \mathbf{h}_2 + \mathbf{b}_3$$

### Dual Interpretation of Middle Layers

The second linear layer $W_2$ has two simultaneous roles:

**Looking backward (aggregation)**: It combines the distances from $\mathbf{h}_1$, computing L1-norm-like summaries of first-layer constraint violations.

**Looking forward (new constraints)**: It defines new hyperplanes, but in **distance space** rather than input space. Each row of $W_2$ defines a linear constraint on the distance profile.

### Hierarchy of Polytopes

Each layer builds a Polyhedral Mesa Gaussian over the representation space of the previous layer:

| Layer | Space | Interpretation |
|-------|-------|----------------|
| 1 | Input space $\mathbb{R}^n$ | Polytope in feature space |
| 2 | Distance space $\mathbb{R}^{m_1}_{\geq 0}$ | Polytope in violation-profile space |
| 3 | Distance-of-distance space $\mathbb{R}^{m_2}_{\geq 0}$ | Higher-order constraint combinations |

The network progressively refines "inside vs. outside" distinctions using the vocabulary of previous layers' violations.

### Preimage Complexity

A simple polytope in distance space corresponds to a potentially complex region in input space. The preimage of a convex set under ReLU composition is a union of convex regions (the cells of the piecewise-linear partition). This explains how deep networks can represent non-convex decision boundaries while each layer operates with convex primitives.

## The Interpretive Inversion

The Mesa Gaussian framework inverts the standard interpretation of neural network activations:

| Standard Interpretation | Mesa Interpretation |
|------------------------|---------------------|
| High activation = feature detected | High activation = deviation from prototype |
| Neuron fires when input matches | Neuron fires when input violates constraint |
| Magnitude indicates confidence | Magnitude indicates anomaly |
| Prototype is peak activation | Prototype is zero activation (plateau) |

### Why This Matters

The standard view treats neurons as **template matchers**: "this input looks like what I'm looking for."

The Mesa view treats neurons as **deviation detectors**: "this input is far from the normal region."

In this interpretation:
- The mesa (plateau region) represents the **prototype space**—the set of inputs that are maximally "normal" or "ideal" for the learned representation
- Positive ReLU outputs indicate **deviation** from this prototype space
- The network's output summarizes cumulative deviation across a hierarchy of learned constraints

This aligns with:
- **Energy-based models**: low energy (low activation) = high probability
- **Anomaly detection**: distance from prototype = anomaly score
- **Mahalanobis distance**: measuring deviation from a learned distribution

## What This Framework Provides

### Geometric Interpretation

Every ReLU network has an implicit polyhedral geometry. The Mesa framework makes this geometry explicit and connects it to probability theory via the Mahalanobis distance.

### Prototype Regions

Rather than asking "what single input maximizes this neuron?", we can ask "what region of inputs keeps this layer quiescent?" The mesa is an extended prototype—a volume, not a point.

### Compositional Structure

The layer-by-layer polytope construction provides a compositional account of representation: each layer refines the partition of space using increasingly abstract constraint combinations.

## What This Framework Does Not Provide

### Training Procedures

This is purely interpretive. We make no claims about how networks should be trained, initialized, or regularized.

### Architecture Modifications

We do not propose any changes to standard architectures. The framework applies to existing ReLU networks as-is.

### Complete Treatment of Negative Weights

The role of negative weights in the second layer and beyond remains an open question. The current framework handles positive-weight (intersection) operations cleanly but does not fully account for the geometric effects of negative weights.

### Quantitative Predictions

This framework provides qualitative interpretation, not quantitative predictions about network performance. Empirical validation of interpretive claims requires further research.

## Open Questions

### On Negative Weights

1. Do negative weights implement union, relative distance, or something else?
2. Is there a canonical decomposition that separates intersection from union operations?
3. How do negative weights affect the convexity of learned regions?

### On Depth

4. What can layer $k+1$ represent that layer $k$ cannot, in Mesa terms?
5. How does the L1 approximation error propagate across layers?
6. Is there a Mesa-Gaussian analog of depth separation theorems?

### On Learning

7. Do trained networks naturally learn interpretable polytope structures?
8. Does regularization encourage simpler or more complex mesa geometry?
9. Can we characterize what makes a mesa "good" for a given task?

### On Probability

10. What loss function corresponds to maximum likelihood under the Mesa Gaussian model?
11. Is there a formal connection between cross-entropy loss and Mesa Gaussian fitting?
12. Can the normalization constant $A$ be estimated or bounded for learned networks?

## Conclusion

The Polyhedral Mesa Gaussian framework offers a geometric and probabilistic interpretation of standard ReLU networks. It recasts neurons as constraint-violation detectors rather than template matchers, and layers as polytope constructors operating on increasingly abstract distance spaces.

This interpretation does not change what networks compute—it changes how we understand what they compute. The goal is to provide a foundation for further research into neural network interpretability, with the hope that understanding the implicit Mesa Gaussian structure can lead to new insights about representation, generalization, and robustness.

---

## References

### Neural Network Interpretability

- Montúfar, G., Pascanu, R., Cho, K., Bengio, Y. "On the number of linear regions of deep neural networks." *NIPS*, 2014.

- Arora, R., Basu, A., Mianjy, P., Mukherjee, A. "Understanding deep neural networks with rectified linear units." *ICLR*, 2018.

### ReLU Networks and Piecewise Linear Functions

- Nair, V. and Hinton, G.E. "Rectified linear units improve restricted Boltzmann machines." *ICML*, 2010.

- Hanin, B. and Rolnick, D. "Deep ReLU networks have surprisingly few activation patterns." *NeurIPS*, 2019.

### Mahalanobis Distance and Neural Networks

- Oursland, A. "Interpreting Neural Networks through Mahalanobis Distance." arXiv:2410.19352, 2024.

### Convex Geometry

- Ziegler, G.M. *Lectures on Polytopes*. Springer, 1995.

### Energy-Based Models

- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., Huang, F.J. "A tutorial on energy-based learning." *Predicting Structured Data*, MIT Press, 2006.

### Gaussian Mesa Functions (Original)

- Dubois, R., Maison-Blanche, P., Quenet, B., Dreyfus, G. "Automatic ECG wave extraction in long-term recordings using Gaussian mesa function models and nonlinear probability estimators." *Computer Methods and Programs in Biomedicine*, 88(3):217-233, 2007.