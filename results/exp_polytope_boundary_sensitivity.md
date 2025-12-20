## Experiment: Polytope Boundary Sensitivity

**Norm-Aggregated Model vs Matched MLP Baseline**

---

## 1. Purpose

This experiment evaluates whether **norm aggregation** in a ReLU network introduces a distinct geometric inductive bias compared to a standard Linear–ReLU–Linear MLP.

Specifically, it tests whether the norm-aggregated model learns:

* a **flat interior (plateau)** inside a polytope, and
* **sensitivity localized near true constraint boundaries**,

as opposed to the smoother, globally sensitive behavior typically associated with MLPs.

The experiment is **behavioral**, not performance-oriented.

---

## 2. Task and Geometry

The task is binary classification of membership in a randomly generated **convex polytope** in (\mathbb{R}^2), defined by linear half-space constraints:

[
n_i^\top x \le r_i
]

Ground-truth geometry is fully known, including:

* exact polytope membership
* true minimum face slack
  (distance to the nearest boundary along constraint normals)

This enables direct alignment between **model behavior** and **true geometry**.

---

## 3. Models Compared

### 3.1 Norm-Aggregation Model

```
x → Linear → ReLU → NormAggregation (learnable p) → Linear (head) → logit
```

* ReLU activations represent constraint violations.
* Norm aggregation produces a **scalar, distance-like quantity**.
* The norm parameter (p) is learned jointly with weights.

During training, (p) evolves from its initialization (p = 2.0) to approximately:

```
p ≈ 1.25
```

---

### 3.2 Matched MLP Baseline

```
x → Linear → ReLU → LinearAggregation → Linear (head) → logit
```

* Same depth and number of ReLU stages.
* Aggregation is a learned weighted sum (no norm structure).
* Same scalar bottleneck before the output head.

This baseline isolates the effect of **aggregation geometry** while holding architectural complexity constant.

---

## 4. Training

Both models are trained with identical settings:

* Optimizer: Adam
* Learning rate: 1e-3
* Epochs: 200
* Random seed matched

Final training loss:

```
Norm model ≈ 0.033
MLP model  ≈ 0.032
```

Both models fit the task equally well.

---

## 5. Boundary Sensitivity Probe

### 5.1 Probe Definition

For each trained model:

1. Sample test points uniformly.
2. Restrict to points inside the **true polytope**.
3. Partition points into:

   * **Boundary**: small true slack
   * **Interior**: large true slack
4. Apply isotropic perturbations:
   [
   x' = x + \epsilon \cdot \frac{u}{|u|}
   ]
5. Measure:

   * `model_flip`: fraction of perturbations flipping the model prediction
   * `true_flip`: fraction crossing the true boundary
   * mean |Δlogit|
   * mean |Δinternal|
   * mean input gradient norm |∇x logit|

Perturbation magnitudes tested:

```
ε ∈ {0.02, 0.05, 0.10, 0.20}
```

Interior statistics are reported only when sufficient interior probes exist.

---

## 6. Results

### 6.1 Boundary Behavior

Both models exhibit similar boundary behavior:

* Boundary flip rates increase with ε.
* Model flips track true flips.
* Boundary gradients and output changes are large.

This indicates that **both architectures learn the boundary geometry**.

---

### 6.2 Interior Behavior (Key Result)

Interior behavior differs sharply.

#### ε = 0.02

```
Interior metrics:
  |Δlogit|       norm = 0.067   mlp = 0.264
  |Δinternal|    norm = 0.029   mlp = 0.142
  |grad|         norm = 2.28    mlp = 11.21
```

#### ε = 0.05

```
Interior metrics:
  |Δlogit|       norm = 0.028   mlp = 0.530
  |Δinternal|    norm = 0.012   mlp = 0.284
  |grad|         norm = 0.31    mlp = 9.04
```

Despite identical **zero flip rates**, the MLP shows **order-of-magnitude larger interior sensitivity**.

---

### 6.3 Aggregate Interior Comparison

Mean over ε values with valid interior probes (ε = 0.02, 0.05):

```
Interior (mean):
  |Δlogit|       norm = 0.047   mlp = 0.397
  |Δinternal|    norm = 0.020   mlp = 0.213
  |grad|         norm = 1.30    mlp = 10.13
```

The norm model learns a **genuinely flat interior**, while the MLP remains smooth but sensitive.

---

## 7. Plots

The following plots support the analysis:

* `flip_rate_vs_eps.png`
  Boundary vs interior flip rates as ε varies (both models)

* `boundary_flip_rate_vs_eps.png`
  Boundary flip rate vs ε (norm vs MLP vs true)

* `interior_flip_rate_vs_eps.png`
  Interior flip rate vs ε

* `boundary_vs_interior.png`
  Summary bar chart of sensitivity metrics

* `flip_rate_vs_distance.png`
  Flip rate vs true min-face slack

(Plots are expected to be in the folder exp_polytope_boundary_sensitivity.)

---

## 8. Interpretation

The results support three conclusions:

1. **Boundary geometry is task-driven**

   * Both models learn where the boundary is.

2. **Interior geometry is architecture-driven**

   * Norm aggregation produces a flat plateau.
   * The MLP does not.

3. **Flip rates alone are insufficient**

   * Both models achieve zero interior flips.
   * Only gradient- and magnitude-based probes reveal the difference.

Norm aggregation therefore induces a **mesa-like representation**: flat interior, sharp boundary.

---

## 9. Conclusion

In a controlled polytope setting:

* Norm-aggregated ReLU networks learn distance-like representations with robust interiors.
* A matched Linear–ReLU–Linear MLP achieves the same accuracy but lacks interior flatness.
* The plateau–boundary geometry is attributable to **norm aggregation**, not generic ReLU structure.

This establishes norm aggregation as a meaningful geometric inductive bias.

---

## 10. Status

This experiment is **complete**.

No further sweeps or refinements are required to support the stated conclusions.

