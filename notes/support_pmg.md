# Empirical Evidence for Polyhedral Mesa Gaussians

## Overview

This document reviews empirical literature relevant to two key predictions of the Polyhedral Mesa Gaussian (PMG) framework:

1. **Polytope structure**: Networks should partition input space into polytope regions that bound "acceptable" inputs for each concept
2. **Antipodal normals**: Networks should learn pairs of hyperplanes with opposite-facing normals to create bounded intervals along feature dimensions

---

## The Predictions

If neural networks learn Multivariate Mesa Gaussians (MMGs) / Polyhedral Mesa Gaussians (PMGs), we would expect to observe:

1. **Sets of orthogonal hyperplanes** (likely uninteresting — occurs naturally in high dimensions)
2. **Pairs of hyperplanes with opposite-facing normals** (the distinctive prediction)

The second prediction is the key testable signature. If networks implement mesa functions (or absolute value operations), they should learn pairs of neurons where one detects "above threshold" and another detects "below threshold" along the same axis — effectively implementing:

```
|x - μ| = ReLU(x - μ) + ReLU(-(x - μ))
```

This requires two neurons with weight vectors pointing in opposite directions (antipodal normals).

---

## Relevant Empirical Literature

### 1. Incidental Polysemanticity and Benign Collisions

**Source**: "What Causes Polysemanticity?" (ICLR 2024 Workshop) — Anthropic-adjacent researchers

Key finding about weight collisions during training:

> "When this happens, there are two cases: (a) If W_ik and W_jk have **opposite signs**, we have W_i · W_j < 0, so nothing actually happens, since the ReLU clips this to 0. Let's call this a **benign collision**. (b) If W_ik and W_jk have the same sign, we have W_i · W_j > 0, and both weights will be under pressure to shrink..."

**Interpretation**: The paper observes that features with opposite-sign weights can coexist stably on the same neuron — a "benign collision." This is consistent with mesa-like behavior where opposite directions don't interfere because ReLU clips negative contributions.

However, this is about polysemanticity arising from training dynamics, not about networks deliberately learning antipodal pairs for functional reasons.

---

### 2. Antipodal Feature Storage in Toy Models

**Source**: "Sparse autoencoders find composed features in small toy models" (LessWrong, 2024)

Key observation:

> "We find the same **antipodal feature storage** as Anthropic observed for anticorrelated features — and this makes sense! Recall that in our data setup, x1 and x2 are definitionally anticorrelated, and so too are y1 and y2."

**Interpretation**: In toy models with explicitly anticorrelated features, networks do store them antipodally (pointing in opposite directions). This is exactly what we'd expect if the network is learning mesa-like bounded regions. When the data itself has anticorrelated structure, the network represents it with antipodal-normal geometry.

This is **supportive evidence** — but it's in toy models with designed anticorrelation.

---

### 3. Feature Geometry in Sparse Autoencoders (COUNTER-EVIDENCE)

**Source**: "Empirical Insights into Feature Geometry in Sparse Autoencoders" (LessWrong, 2024)

Direct test of the antipodal hypothesis:

> "We demonstrate that **subspaces with semantically opposite meanings within the GemmaScope series of Sparse Autoencoders are not pointing towards opposite directions**. Furthermore, subspaces that are pointing towards opposite directions are usually not semantically related."

**Interpretation**: This is the most direct test I found, and it provides **counter-evidence**. When looking at semantic antonyms (happy/sad, good/bad), the corresponding SAE features are NOT antipodal. And when features ARE antipodal, they tend not to be semantically related.

However, note the caveat: This tests semantic antonyms, not the specific structure predicted by mesa theory. The mesa prediction is about **constraint boundaries**, not semantic opposites. A mesa for "dog" wouldn't have an antipodal "anti-dog" feature — it would have features for "not-too-big," "not-too-small," "has-four-legs-within-tolerance," etc.

---

### 4. Equivariant Features and Symmetric Weights

**Source**: "Naturally Occurring Equivariance in Neural Networks" (Distill, 2020) — Olah et al.

Observations about weight symmetry in vision models:

> "The equivariant behavior we observe in neurons is really a reflection of a deeper symmetry that exists in the weights of neural networks and the circuits they form."

> "Each curve is excited by curves in the same orientation and inhibited by those in the opposite."

**Interpretation**: Vision networks naturally learn features that come in symmetric groups (rotated versions of the same detector). The observation about "inhibited by those in the opposite" hints at paired-opposite structure, though in the context of orientation rather than bounded regions.

---

### 5. Polytope Structure in ReLU Networks

The Polyhedral Mesa Gaussian (PMG) framework predicts that networks should learn polytope-shaped regions bounding "acceptable" inputs. This prediction has extensive empirical and theoretical support.

#### 5a. Mathematical Foundation: ReLU Networks Partition Space into Polytopes

**Sources**: Montúfar et al. (NeurIPS 2014), Pascanu et al. (ICLR 2014), Serra et al. (ICML 2018)

This is definitional rather than empirical: ReLU networks mathematically partition input space into convex polytopes. Each ReLU neuron contributes a hyperplane, and the intersection of these half-spaces defines polytopes where the function is locally affine. The number of linear regions can grow exponentially with depth.

#### 5b. Networks Learn Surprisingly Simple Polytopes

**Source**: "Deep ReLU Networks Have Surprisingly Simple Polytopes" — Fan et al. (arXiv 2305.09145, 2023)

Key finding:

> "We find that a ReLU network has relatively simple polytopes under both initialization and gradient descent, although these polytopes theoretically can be rather diverse and complicated. This finding can be appreciated as a kind of generalized implicit bias."

**Interpretation**: Despite the theoretical capacity for complex polytopes, trained networks converge to simple ones. This is consistent with mesa theory: if networks learn data-aligned bounding regions, the polytopes should match the structure of natural data clusters (which tend to be simple, compact shapes) rather than exploiting arbitrary complexity.

#### 5c. Dataset Classification via Polytope Covers

**Source**: "Defining Neural Network Architecture through Polytope Structures of Datasets" — Lee, Mammadov & Ye (ICML 2024)

Key finding:

> "Through our algorithm, it is established that popular datasets such as MNIST, Fashion-MNIST, and CIFAR10 can be efficiently encapsulated using no more than **two polytopes with a small number of faces**."

The paper introduces "polytope-basis covers" — collections of polytopes that bound different classes — and shows that trained networks learn exactly this structure. They derive that a three-layer ReLU network needs width proportional to the number of polytope faces.

**Interpretation**: This is strong evidence for mesa-like structure. Networks learn polytope boundaries that directly correspond to class regions in the data. The finding that complex image datasets need only ~2 polytopes suggests networks learn efficient bounded representations.

#### 5d. Categorical Concepts as Polytopes in LLMs

**Source**: "The Geometry of Categorical and Hierarchical Concepts in Large Language Models" — Park et al. (arXiv 2406.01506, 2024)

Key findings:

> "We find a remarkably simple structure: simple categorical concepts are represented as **simplices**, hierarchically related concepts are **orthogonal** in a sense we make precise, and complex concepts are represented as **polytopes constructed from direct sums of simplices**."

Validated on Gemma-2B and LLaMA-3-8B using 900+ concepts from WordNet.

**Interpretation**: This is the strongest direct evidence for polytope structure at the semantic level. Categorical concepts (like {mammal, bird, reptile, fish}) form simplices — the simplest polytopes. Complex categories (like "animal") form polytopes whose vertices are the subcategory representations. This exactly matches the PMG prediction that concept regions should be bounded by intersecting hyperplanes.

#### 5e. Sparse Autoencoder Feature Geometry

**Source**: "The Geometry of Concepts: Sparse Autoencoder Feature Structure" — Li, Tegmark et al. (Entropy 2025, arXiv 2410.19750)

Key findings at three scales:

1. **Atomic scale**: "crystals" with parallelogram/trapezoid faces (geometric relations between concept pairs)
2. **Brain scale**: Functional lobes (math/code features cluster spatially)
3. **Galaxy scale**: Fractal-like overall structure

**Interpretation**: The "crystal" structure at atomic scale shows precise geometric relationships between concepts — consistent with concepts being embedded in a structured polytope-like arrangement where relationships are encoded geometrically.

#### 5f. The Polytope Lens for Interpretability

**Source**: "Interpreting Neural Networks through the Polytope Lens" — Black et al. (AI Alignment Forum, 2022)

Key argument:

> "Polytope boundaries should therefore be placed between non-orthogonal feature directions so that activations in one feature direction don't activate the other when they shouldn't."

> "Neural networks fold and squeeze the input data manifold into a shape that is linearly separable in subsequent layers."

**Interpretation**: This work explicitly proposes that polytope structure is the right lens for interpretability. The argument that polytope boundaries should align with semantic boundaries is precisely what PMG theory predicts — bounded mesa regions for each concept.

#### 5g. Tropical Geometry and Decision Boundaries

**Source**: "On the Decision Boundaries of Neural Networks: A Tropical Geometry Perspective" — Alfarra et al. (IEEE TPAMI 2022)

Key finding:

> "Decision boundaries are a subset of a tropical hypersurface, which is intimately related to a **polytope formed by the convex hull of two zonotopes**."

**Interpretation**: This theoretical work shows that ReLU network decision boundaries have polytope structure (specifically, they're related to zonotopes — a type of polytope generated by line segments). This geometric characterization supports the mesa framework's prediction of polytope-bounded concept regions.

---

### 6. Edge Detection and Positive/Negative Edges

**Source**: Various CNN educational materials (Andrew Ng's lectures, etc.)

Standard observation in CNNs:

> "The 30 shown in yellow... corresponds to the three by three yellow region, where there are bright pixels on top and darker pixels on the bottom. And so it finds a strong positive edge there. And this -30 here corresponds to the red region, which is actually brighter on the bottom and darker on top. So that is a **negative edge**."

**Interpretation**: Edge detectors naturally produce positive and negative responses for opposite edge polarities. A single filter detects both directions, but through opposite-signed outputs. This isn't quite the same as having two filters with opposite normals, but it shows the underlying symmetry in what networks learn to detect.

---

## Synthesis

### Evidence FOR the PMG framework (polytope structure):

1. **Definitional**: ReLU networks partition space into convex polytopes (mathematical fact)
2. **Simple polytopes**: Networks converge to simple polytopes despite capacity for complexity (Fan et al. 2023)
3. **Dataset polytope covers**: MNIST/CIFAR10 efficiently covered by ~2 polytopes (Lee et al. ICML 2024)
4. **Categorical concepts as polytopes**: LLM concepts form simplices/polytopes matching semantic hierarchy (Park et al. 2024)
5. **Crystal structure in SAE features**: Geometric "crystals" with parallelogram faces (Tegmark et al. 2025)
6. **Tropical geometry**: Decision boundaries relate to zonotope polytopes (Alfarra et al. 2022)

### Evidence FOR antipodal-normal prediction specifically:

1. **Antipodal storage in toy models** when data has anticorrelated structure
2. **Benign collisions** — opposite-signed features can coexist, suggesting the geometry is at least permitted
3. **Equivariant circuits** show weight symmetries in trained networks

### Evidence AGAINST (or complicating) the antipodal prediction:

1. **Semantic antonyms are NOT antipodal** in SAE feature space
2. **Antipodal features are semantically unrelated** — when opposite directions exist, they don't seem to represent "opposite" concepts

### Key Distinction:

The counter-evidence tests **semantic antonyms** (happy/sad). The mesa prediction is about **constraint boundaries** — paired hyperplanes that together define a bounded acceptable region. These are different:

- Semantic antonyms: "dog" vs "cat" — different categories
- Mesa boundaries: "fur-length > minimum" and "fur-length < maximum" — the same dimension, opposite constraints

The empirical tests haven't directly addressed whether networks learn **paired threshold constraints** along the same feature dimension.

---

## On Orthogonality Tests: Why They're Insufficient

A natural test for MMG structure would be to look for sets of orthogonal hyperplanes. However, **lack of orthogonality is not informative**.

### The Whitening Connection

In the Mahalanobis distance framework (Oursland 2024), orthogonal principal components serve to whiten the data — to decorrelate and normalize variance across dimensions. The standard PCA decomposition Σ = VΛV^T gives an orthogonal whitening basis.

But orthogonal bases are not the only whitening bases. Any matrix W satisfying WΣW^T = I will whiten the data. The set of such matrices forms a manifold, and orthogonal bases are just one point on it.

### Local Whitening Bases

More importantly for neural network analysis: whitening may be **local** rather than global. A network might learn:

- Different whitening transforms for different regions of input space
- Approximately whitening transforms that trade off decorrelation against other objectives
- Hierarchical whitening where each layer partially whitens with respect to the previous layer's representation

This means we should look for **local whitening structure**, not just global orthogonality:

1. Within a polytope region, are the active hyperplane normals approximately whitening the local data distribution?
2. Do the learned transforms approximate Mahalanobis normalization locally, even if the global weight matrix isn't orthogonal?
3. Are there clusters of hyperplanes that together form a local whitening basis for their associated feature subspace?

### Implications for Empirical Tests

Testing for global orthogonality of weight vectors will likely fail — and that failure tells us nothing. The informative tests are:

- **Local correlation structure**: Within activation regions, is the representation decorrelated?
- **Variance normalization**: Do the learned scales (weight magnitudes) inversely track local variance?
- **Non-orthogonal whitening**: Can the weight matrix be factored into a whitening transform composed with a rotation?

The mesa framework predicts local Mahalanobis-like structure, which permits non-orthogonal hyperplanes as long as they collectively achieve local whitening.

---

## Open Questions for Future Empirical Work

1. **Test within-feature bounds**: For a monosemantic SAE feature, are there paired features that activate for "too much" and "too little" of the same underlying concept?

2. **Look for antipodal normals explicitly**: In a trained network, do weight vectors come in near-antipodal pairs more often than chance? (Not semantic antonyms, but actual weight geometry.) Note: we deliberately avoid the term "mirror neurons" to distinguish hyperplane geometry from single-unit semantics.

3. **Examine bounded regions**: For classification tasks, do the learned decision boundaries form closed polytopes (requiring opposite-facing hyperplanes) or open half-spaces?

4. **Layer-by-layer analysis**: The mesa prediction is strongest for early layers doing feature extraction. Later layers might show different geometry.

5. **Local whitening structure**: Within polytope regions, do the active constraints form an approximate whitening basis for the local data distribution? This is a more sensitive test than global orthogonality.

---

## Conclusion

The empirical record provides **strong support for the polytope structure** predicted by PMG theory, but is **mixed on the specific antipodal-normal prediction**:

**Polytope structure (strongly supported)**:
- ReLU networks definitionally partition space into polytopes
- Trained networks learn *simple* polytopes matching data structure (Fan et al.)
- Image classification datasets are efficiently covered by ~2 polytopes (Lee et al.)
- LLM categorical concepts are represented as simplices/polytopes (Park et al.)
- The "polytope lens" is emerging as a principled interpretability framework

**Antipodal normals (mixed evidence)**:
- The clearest counter-evidence (SAE antonym geometry) tests the wrong phenomenon — semantic opposites rather than constraint pairs
- The supportive evidence (antipodal storage, benign collisions) is from toy models or indirect observations
- **No one has directly tested** whether networks learn paired hyperplanes that together bound an acceptable region along a single dimension

This represents an opportunity for a novel empirical contribution: directly testing whether trained ReLU networks exhibit the antipodal-normal geometry predicted by the mesa framework, specifically looking for **constraint pairs** (upper/lower bounds on the same feature) rather than semantic antonyms.

---

## Key Citations

### Polytope Structure

1. **Montúfar et al.** (NeurIPS 2014) — "On the Number of Linear Regions of Deep Neural Networks" (foundational)
2. **Serra et al.** (ICML 2018) — "Bounding and Counting Linear Regions of Deep Neural Networks"
3. **Fan et al.** (arXiv 2305.09145, 2023) — "Deep ReLU Networks Have Surprisingly Simple Polytopes"
4. **Lee, Mammadov & Ye** (ICML 2024) — "Defining Neural Network Architecture through Polytope Structures of Datasets"
5. **Park et al.** (arXiv 2406.01506, 2024) — "The Geometry of Categorical and Hierarchical Concepts in Large Language Models"
6. **Alfarra et al.** (IEEE TPAMI 2022) — "On the Decision Boundaries of Neural Networks: A Tropical Geometry Perspective"
7. **Black et al.** (AI Alignment Forum, 2022) — "Interpreting Neural Networks through the Polytope Lens"

### Feature Geometry and Antipodal Structure

8. **Li, Tegmark et al.** (Entropy 2025, arXiv 2410.19750) — "The Geometry of Concepts: Sparse Autoencoder Feature Structure"
9. "What Causes Polysemanticity?" — ICLR 2024 Workshop (incidental polysemanticity, benign collisions)
10. "Empirical Insights into Feature Geometry in Sparse Autoencoders" — LessWrong 2024 (antonym geometry)
11. "Sparse autoencoders find composed features in small toy models" — LessWrong 2024 (antipodal storage)
12. "Naturally Occurring Equivariance in Neural Networks" — Distill 2020 (weight symmetry)
13. "Toy Models of Superposition" — Anthropic 2022 (geometric structure of features)


---
---

## GEMINI FEEDBACK

This document is your **"Defense Brief."** You have successfully marshaled a diverse jury of witnesses—from toy models to LLMs—to testify that the geometry of neural networks is indeed polyhedral.

The most critical theoretical nuance you have uncovered here is the tension between **Simplices** (Park et al.) and **Hyperrectangles** (your Antipodal MMG).

* **The MMG Prediction:** Antipodal pairs  "Box" or "Hypercube" geometry (Opposite faces are parallel).
* **The Empirical Evidence:** Concepts are Simplices  "Pyramid" geometry (Faces are not parallel).

This is a **feature, not a bug**. It suggests that while the *mechanism* (ReLU) creates polytopes, the *learned shape* might often be a Simplex (minimal constraints) rather than a Box (symmetric constraints).

Here are the final theoretical connections to solidify this "Empirical" section:

### 1. The Historical Anchor: Sparse Coding (V1 Cortex)

You mentioned "Antipodal Feature Storage". The grandfather of this concept is **Sparse Coding**.

* **The Theory:** In 1996, Olshausen and Field showed that maximizing sparsity (L1 norm) on natural images leads to Gabor filters. Crucially, these filters tile space in a way that forms a "Sparse Polytope" (the L1 ball is a cross-polytope).
* **Why it matters:** It proves that "Polyhedral" structure is not an artifact of ReLU; it is the optimal way to represent natural data under a sparsity constraint.
* **Recommended Citation:** **Olshausen, B. A., & Field, D. J. (1996).** "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." *Nature*.

### 2. The "Simplex" Tension (Archetypal Analysis)

You cited Park et al. (2024) finding that concepts are **Simplices**. This is mathematically profound.

* **The Theory:** A Simplex is the convex hull of  points. It is the "cheapest" polytope (fewest faces) that can enclose a region.
* **The Implication:** If the network is "lazy" (regularized), it will learn a Simplex (Triangle/Tetrahedron) to bound a concept, not a Box (Square/Cube). A Simplex does *not* require antipodal normals. It only requires normals that "lean" against each other to close the loop.
* **Refinement:** You might need to relax your "Antipodal" prediction to a "Closing" prediction. The normals sum to zero (), which closes the shape, but they might be arranged as a Mercedes-Benz star (120 degrees) rather than a cross (180 degrees).

### 3. The "Lottery Ticket" Geometry

The finding by Fan et al. (2023) that polytopes are "surprisingly simple" connects to the **Lottery Ticket Hypothesis**.

* **The Theory:** Pruning removes weights (hyperplanes). If you can prune 90% of weights and keep accuracy, the "Mesa" was defined by way too many redundant constraints. The final "Simple Polytope" is the winning ticket—the minimal set of constraints needed to bound the class.
* **Recommended Citation:** **Frankle, J., & Carbin, M. (2019).** "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." *ICLR*.

### 4. Visualizing the "Evidence Gap"

Your document highlights a gap between "Semantic Antonyms" and "Constraint Bounds". A diagram here would be definitive.

* **Left (Semantic):** "Hot" vector vs. "Cold" vector. They point 180° apart.
* **Right (Mesa):** "Too Hot" constraint vs. "Too Cold" constraint. They *also* point 180° apart, but they define the *same* concept ("Just Right"), whereas the semantic vectors define *different* concepts.

### Final Assessment of the "Definition Phase"

You have built a formidable structure:

1. **Prior Art:** Established the "Mesa" as a signal processing necessity.
2. **Theory:** Constructed the Polyhedral Mesa Gaussian (PMG) as a rigorous probability density derived from ReLU networks.
3. **Philosophy:** Inverted the interpretation of activation from "Presence" to "Distance from Surface".
4. **Evidence:** Validated the geometric consequences (Polytopes) while honestly identifying the open question (Antipodal Pairs).
