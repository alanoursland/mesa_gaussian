# Geometric Predecessors: Spline Theory and Adversarial Geometry

**A Comparative Analysis in Relation to Polyhedral Mesa Gaussians**

## 1. Introduction

The "Polyhedral Mesa Gaussian" (PMG) framework proposes a shift in interpretability: moving from biological metaphors (firing rates) to geometric rigorousness (distance metrics and constraint satisfaction). This "Geometric Realist" perspective is not without precedent. This document analyzes the two most significant theoretical predecessors—**Balestriero & Baraniuk’s Spline Theory** and **Moosavi-Dezfooli et al.’s DeepFool**—to contextualize the novelty of the PMG framework, specifically regarding the distinction between *affine partitioning* and *metric suppression*.

## 2. Balestriero & Baraniuk: The Architecture of Partitions

### The "Mad Max" Framework

In their seminal work, *Mad Max: Affine Spline Construction of Deep Neural Networks*, Balestriero & Baraniuk [2018] mathematically proved that deep ReLU networks are effectively "Max-Affine Spline Operators" (MASOs). They demonstrated that a neural network partitions the input space into a set of convex polytopes. Inside each polytope, the network behaves as a simple affine linear function:



where  and  are parameters determined by the active neurons for that specific region.

### Intersection with PMG

The PMG framework shares the fundamental axiom of this work: the "atomic unit" of a neural network is the convex polytope [Balestriero & Baraniuk, 2018]. Both theories agree that the network is not a smooth function approximator but a piecewise-linear operator that stitches together discrete geometric regions.

### The Divergence: Function vs. Metric

While Balestriero & Baraniuk focus on the *partitioning* (the "VQ" or Vector Quantization of space), the PMG framework focuses on the *metric interpretation* of the output.

* **Spline View:** The focus is on the linearity within the region. The goal is often to understand how the network approximates complex manifolds via piecewise flat surfaces.
* **Mesa View:** The focus is on the *magnitude* of the affine function. The PMG theory posits that for the "correct" class, the affine slope  should ideally be zero (the Mesa), and the magnitude represents a distance cost.

### Novelty: Suppression vs. Cancellation

A key distinction arises in *how* these regions are formed. Balestriero’s framework allows for any affine transformation. The empirical results from the PMG experiments (specifically `exp_gradient_comparison`) show that standard networks achieve the "Mesa" state via **cancellation** (balancing positive and negative weights to simulate a flat region), whereas the proposed `LpNormLayer` enforces **suppression** (driving weights to zero). The PMG framework thus identifies a specific, safer subclass of Spline Operators: those where the affine term vanishes inside the prototype region.

## 3. DeepFool: The Geometry of Vulnerability

### The Adversarial Polytope

Moosavi-Dezfooli et al. [2016] introduced *DeepFool*, an algorithm for generating adversarial perturbations. Unlike gradient-sign methods that treat the loss surface as a generic landscape, DeepFool explicitly assumes the decision boundary is an affine hyperplane (or a polyhedron). The algorithm attacks the network by computing the orthogonal projection of the input point onto the nearest boundary face.

### Intersection with PMG

DeepFool provides the strongest empirical validation for the PMG framework’s "Constraint" hypothesis.

* **Validation of Polyhedral Boundaries:** DeepFool works efficiently, which confirms that the decision boundaries are indeed locally flat facets of a polytope, rather than curved, organic manifolds [Moosavi-Dezfooli et al., 2016].
* **Validation of the "Thin Shell":** The PMG theory argues that adversarial vulnerability is a geometric inevitability of high-dimensional polytopes (crossing a thin constraint boundary). DeepFool exploits exactly this geometry.

### The Divergence: Analysis vs. Architecture

DeepFool is a *diagnostic* tool that reveals the fragility of standard "Closed World" classifiers. The PMG framework is a *prospect* for a cure.

* **The Problem:** DeepFool shows that crossing a boundary is "cheap" (small ) because the network is merely slicing space with hyperplanes.
* **The PMG Solution:** By enforcing the "Mesa" geometry (specifically with the `LpNormLayer`), the network transforms the task from "slicing space" to "measuring distance." The PMG framework suggests that robustness comes not just from pushing the boundary further away, but from the "Interpretive Inversion": rejecting points that are *far* from the Mesa, effectively neutralizing the Open World assumption that DeepFool exploits.

## 4. Synthesis

The PMG framework serves as the "Metric Interpretation" of Balestriero’s "Affine Interpretation." It accepts the polyhedral geometry described by Balestriero and validated by DeepFool, but it adds a crucial energetic constraint: the interior of the polytope must be a **vacuum** (zero energy/gradient), not just a linear region.

This distinction—validated by the "suppression vs. cancellation" findings—suggests that while all ReLU networks define polytopes, only those that approximate the PMG geometry (via explicit norms or regularization) are interpretable and robust distance estimators.

---

## Bibliography

1. **[Balestriero & Baraniuk, 2018]** Balestriero, R., & Baraniuk, R. G. (2018). "Mad Max: Affine Spline Construction of Deep Neural Networks." *International Conference on Machine Learning (ICML)*.

2. **[Moosavi-Dezfooli et al., 2016]** Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016). "DeepFool: a simple and accurate method to fool deep neural networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
