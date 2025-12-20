# Distance from Prototype: A Reinterpretation of ReLU Activations

## Introduction

This document presents a reinterpretation of neural network activations in ReLU networks. Rather than viewing activation magnitude as confidence in feature presence, we propose that activations represent **distance from a prototype region**.

In this view:
- **Zero activation** means the input is within the prototype region
- **Positive activation** means the input is outside the prototype region, with magnitude indicating how far

The prototype is not a single point but a **surface** or **area**—a region in input space where the feature is maximally present. This interpretation follows directly from the algebraic structure of linear layers with ReLU activations and their equivalence to Mahalanobis distance computations.

## Two Competing Interpretations

### The Confidence Interpretation

The prevailing view treats neurons as template matchers:

$$h = \text{ReLU}(w^\top x + b)$$

Under this interpretation:
- $w$ is a learned template
- $w^\top x$ measures similarity between input and template
- Large positive values indicate high confidence that the feature is present
- The neuron "fires" when it detects its preferred stimulus

This view is motivated by analogy to cosine similarity:

$$\cos(\theta) = \frac{w^\top x}{\|w\| \|x\|}$$

where $\theta$ is the angle between the weight vector and input vector.

### The Distance Interpretation

We propose an alternative:

$$h = \text{ReLU}(w^\top x + b)$$

Under this interpretation:
- The equation $w^\top x + b = 0$ defines a hyperplane
- $w^\top x + b$ is the signed distance from that hyperplane (scaled by $\|w\|$)
- ReLU clips to positive values, giving one-sided distance
- Zero means the input is on the "inside" of the hyperplane
- Positive values indicate the input is "outside," with magnitude proportional to distance

The neuron outputs how far the input deviates from an acceptable region, not how much it resembles a template.

## The Normalization Problem

The confidence interpretation borrows intuition from cosine similarity, but the network computes $w^\top x + b$, not normalized cosine.

### What the Raw Dot Product Actually Measures

The value $w^\top x$ conflates multiple factors:

| Factor | Effect |
|--------|--------|
| Angular similarity | What we'd want for template matching |
| Input magnitude $\|x\|$ | Large inputs give large outputs regardless of angle |
| Weight magnitude $\|w\|$ | Amplifies everything uniformly |
| Bias $b$ | Shifts the output arbitrarily |

Without normalization, a vector pointing 45° away from $w$ can produce a larger dot product than a vector pointing directly at $w$, if $\|x\|$ is large enough.

**The raw dot product does not measure similarity without normalization.**

### The Distance Interpretation Requires No Normalization

The distance interpretation is not an analogy—it's what the equation literally computes:

$$\text{signed distance} = \frac{w^\top x + b}{\|w\|}$$

- The sign indicates which side of the hyperplane
- The magnitude indicates how far
- The bias $b$ shifts the hyperplane location
- The weight magnitude $\|w\|$ scales the distance (equivalent to $1/\sqrt{\lambda}$ in Mahalanobis terms)

This is the algebraic fact of what the linear equation computes. No normalization is required for this interpretation to hold.

### Summary

| Interpretation | Mathematical Basis | Normalization Required |
|----------------|-------------------|------------------------|
| Confidence (cosine similarity) | $\frac{w^\top x}{\|w\|\|x\|}$ | Yes—both $w$ and $x$ |
| Distance (hyperplane) | $w^\top x + b$ | No—inherent in the equation |

The confidence interpretation imposes requirements the architecture doesn't satisfy. The distance interpretation describes what the math already does.

## Normalization Techniques and Both Interpretations

Modern networks use normalization extensively: Batch Normalization, Layer Normalization, Weight Normalization, etc. These techniques improve training stability and generalization.

### Effect on Confidence Interpretation

Normalization makes the confidence interpretation more plausible by controlling magnitudes. With normalized activations, the dot product more closely approximates angular similarity.

### Effect on Distance Interpretation

Normalization doesn't invalidate the distance interpretation—it changes the space in which distances are computed. Distances are now measured in a normalized coordinate system, but they're still distances from hyperplanes.

### The Key Difference

The confidence interpretation **requires** normalization to make sense. Without it, magnitudes confound the similarity signal.

The distance interpretation **works regardless**. Normalize or don't—the geometry holds either way. Normalization changes the metric space; it doesn't change the fact that you're measuring distances.

The Polyhedral Mesa Gaussian framework absorbs normalization naturally. It's agnostic to whether the distance computation happens in raw or normalized space.

## The Prototype as a Region

### Points vs. Regions

The confidence interpretation implies a point prototype: the single input that maximally activates the neuron.

The distance interpretation implies a region prototype: the set of all inputs that produce zero activation.

### The Mesa as Prototype

In the Polyhedral Mesa Gaussian framework:
- The **mesa** (plateau region) is where the distance is zero
- The **prototype surface** is the boundary of this region
- **Activation** measures distance from the nearest prototype surface

For a single neuron, the prototype region is a half-space. For multiple neurons combined, it's a polytope—the intersection of half-spaces.

### Why Regions Are More Natural

Consider the concept "cat." Is there a single optimal cat that all other cats approximate? Or is there a region of cat-space, where many different cats are equally valid?

The mesa interpretation aligns with the latter. The prototype is:
- Not a point but a volume
- Not a template but a region of acceptability
- Defined by its boundaries, not its center

Inputs inside the mesa are equally "cat-like." Deviations from the mesa indicate how far from cat-ness the input lies.

## Formal Structure: Polyhedral Mesa Gaussians

The distance-from-prototype interpretation is formalized by the Polyhedral Mesa Gaussian:

$$G(x) = A \cdot \exp\left( -\frac{1}{2} D_{PM}^2 \right)$$

where $D_{PM}$ is the Polyhedral Mesa distance—a combination of one-sided distances from hyperplane constraints.

### Key Properties

- **Plateau**: Region where $D_{PM} = 0$ (the prototype area)
- **Decay**: Gaussian-like falloff with distance from the plateau
- **Convexity**: The prototype region is always convex (intersection of half-spaces)
- **Compositionality**: Multiple layers build hierarchies of prototype regions

### Connection to Mahalanobis Distance

The construction derives from Mahalanobis distance via:
1. Decomposing along principal axes
2. Using $|z| = \text{ReLU}(z) + \text{ReLU}(-z)$ to split each axis
3. Introducing a gap $\delta$ to create the plateau
4. Relaxing orthogonality for the polyhedral case

This provides a principled foundation: the distance interpretation isn't imposed on the network—it emerges from the algebraic equivalence between ReLU layers and distance computations.

## Implications

### Zero as Natural Representation

Zero activation is not "failure to detect." It's the natural state for inputs within the prototype region.

This reframes sparsity: a sparse activation pattern means the input lies within many prototype regions simultaneously. Most constraints are satisfied; only a few are violated.

### What Training Learns

Under this interpretation, training doesn't learn "what features look like." It learns the boundaries of prototype regions—the surfaces that separate acceptable from unacceptable.

The feature emerges as the region bounded by learned constraints. The network learns to carve space, not to match templates.

### Recognition by Exclusion

If features are defined by boundaries rather than templates, recognition becomes a process of exclusion:

An input is classified as "dog" not because it matches a dog template, but because it falls within the dog prototype region—the space that remains after excluding cats, birds, fish, etc.

The network learns what each class is **not**, and the class emerges as what survives the exclusion.

## Forward Pointers

Several implications of this framework warrant further investigation:

### Adversarial Examples

If recognition is boundary-based, adversarial perturbations need only cross boundaries—not match templates. Small perturbations might escape all prototype regions, falling into unintended classifications.

### Negative Weights

Positive weights in aggregation layers implement intersection of constraints. Negative weights may implement union or relative distance comparisons. A complete geometric account of negative weights remains open.

### Sparsity

The mesa interpretation provides a different motivation for sparse representations: sparsity reflects that inputs naturally satisfy most constraints, violating only a few. This is expected, not surprising.

### Relationship to Prototype Theory

The prototype-as-region interpretation resonates with prototype theory in cognitive science (Rosch, 1970s), where categories have graded structure around prototypes rather than sharp definitional boundaries. The mesa formalizes this intuition.

---

## References

### Mahalanobis Distance

- Mahalanobis, P.C. "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India*, 1936.

### Neural Network Interpretability

- Oursland, A. "Interpreting Neural Networks through Mahalanobis Distance." arXiv:2410.19352, 2024.

- Oursland, A. "Neural Networks Use Distance Metrics." arXiv:2411.17932, 2024.

- Oursland, A. "Neural Networks Learn Distance Metrics." arXiv:2502.02103, 2025.

### Normalization Techniques

- Ioffe, S. and Szegedy, C. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML*, 2015.

- Ba, J.L., Kiros, J.R., Hinton, G.E. "Layer Normalization." arXiv:1607.06450, 2016.

### Prototype Theory (Cognitive Science)

- Rosch, E. "Cognitive representations of semantic categories." *Journal of Experimental Psychology: General*, 104(3):192-233, 1975.

### Gaussian Mesa Functions

- Dubois, R., Maison-Blanche, P., Quenet, B., Dreyfus, G. "Automatic ECG wave extraction in long-term recordings using Gaussian mesa function models and nonlinear probability estimators." *Computer Methods and Programs in Biomedicine*, 88(3):217-233, 2007.