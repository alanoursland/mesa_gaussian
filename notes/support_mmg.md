# Empirical Evidence: Do Neural Networks Learn Antipodal Normal Pairs?

## The Prediction

If neural networks learn Multivariate Mesa Gaussians (MMGs), we would expect to observe:

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

### 5. Polytope Representations

**Source**: "The Geometry of Concepts" (arXiv 2410.19750, 2025) — Tegmark et al.

> "Another work found that representations of hierarchically related concepts are orthogonal to each other while **categorical concepts are represented as polytopes**."

**Interpretation**: The finding that categories are represented as polytopes is highly consistent with the mesa framework. A polytope is exactly what you get from the intersection of half-spaces (ReLU constraints). This supports the general mesa framework even if it doesn't directly address the antipodal-normal question.

---

### 6. Edge Detection and Positive/Negative Edges

**Source**: Various CNN educational materials (Andrew Ng's lectures, etc.)

Standard observation in CNNs:

> "The 30 shown in yellow... corresponds to the three by three yellow region, where there are bright pixels on top and darker pixels on the bottom. And so it finds a strong positive edge there. And this -30 here corresponds to the red region, which is actually brighter on the bottom and darker on top. So that is a **negative edge**."

**Interpretation**: Edge detectors naturally produce positive and negative responses for opposite edge polarities. A single filter detects both directions, but through opposite-signed outputs. This isn't quite the same as having two filters with opposite normals, but it shows the underlying symmetry in what networks learn to detect.

---

## Synthesis

### Evidence FOR antipodal-normal prediction:

1. **Antipodal storage in toy models** when data has anticorrelated structure
2. **Benign collisions** — opposite-signed features can coexist, suggesting the geometry is at least permitted
3. **Polytope representations** — categories as polytopes implies bounded regions (mesa-like)
4. **Equivariant circuits** show weight symmetries in trained networks

### Evidence AGAINST (or complicating) the prediction:

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

The empirical record is mixed but not definitively against the mesa prediction:

- The clearest counter-evidence (SAE antonym geometry) tests the wrong phenomenon — semantic opposites rather than constraint pairs.
- The supportive evidence (antipodal storage, polytopes, benign collisions) is suggestive but from toy models or indirect observations.
- **No one has directly tested** whether networks learn paired hyperplanes that together bound an acceptable region along a single dimension.

This represents an opportunity for a novel empirical contribution: directly testing whether trained ReLU networks exhibit the antipodal-normal geometry predicted by the mesa framework.

---

## Key Citations

1. "What Causes Polysemanticity?" — ICLR 2024 Workshop (incidental polysemanticity, benign collisions)
2. "Empirical Insights into Feature Geometry in Sparse Autoencoders" — LessWrong 2024 (antonym geometry)
3. "Sparse autoencoders find composed features in small toy models" — LessWrong 2024 (antipodal storage)
4. "Naturally Occurring Equivariance in Neural Networks" — Distill 2020 (weight symmetry)
5. "The Geometry of Concepts: Sparse Autoencoder Feature Structure" — arXiv 2410.19750 (polytope representations)
6. "Toy Models of Superposition" — Anthropic 2022 (geometric structure of features)

---
---

## GEMINI FEEDBACK

This document is your **"Defense Brief."** You have successfully marshaled a diverse jury of witnesses—from toy models to LLMs—to testify that the geometry of neural networks is indeed polyhedral.

The most critical theoretical nuance you have uncovered here is the tension between **Simplices** (Tegmark et al.) and **Hyperrectangles** (your Antipodal MMG).

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

You cited Tegmark et al. (2025) finding that concepts are **Polytopes**. Other work (Park et al.) specifically finds **Simplices**. This is mathematically profound.

* **The Theory:** A Simplex is the convex hull of  points. It is the "cheapest" polytope (fewest faces) that can enclose a region.
* **The Implication:** If the network is "lazy" (regularized), it will learn a Simplex (Triangle/Tetrahedron) to bound a concept, not a Box (Square/Cube). A Simplex does *not* require antipodal normals. It only requires normals that "lean" against each other to close the loop.
* **Refinement:** You might need to relax your "Antipodal" prediction to a "Closing" prediction. The normals sum to zero (), which closes the shape, but they might be arranged as a Mercedes-Benz star (120 degrees) rather than a cross (180 degrees).

### 3. The "Lottery Ticket" Geometry

The finding by Fan et al. (2023) (which you cited in a previous iteration) that polytopes are "surprisingly simple" connects to the **Lottery Ticket Hypothesis**.

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

