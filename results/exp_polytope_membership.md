## Experiment: Polytope Membership (Smoke Test)

### Purpose

This experiment serves as a **smoke test** for the proposed norm-aggregation layer with a learnable ( p ) parameter.

The goal is *not* to demonstrate geometric behavior or inductive bias, but to answer a narrower technical question:

> **Can a neural network with a learnable L-p norm layer train stably and backpropagate through ( p ) without numerical or optimization issues?**

Before running more elaborate geometric probes, it is necessary to establish that:

* gradients flow through the norm layer,
* the learnable ( p ) parameter updates correctly,
* and training does not collapse or diverge on a simple task.

---

### Task

The task is binary classification of a **single half-space** in (\mathbb{R}^2):

[
y = \mathbb{1}[x_1 + x_2 > 0]
]

This defines a linear decision boundary with:

* one active constraint,
* no corners,
* no multi-constraint interactions.

In this setting, all L-p norms are equivalent up to scaling, so there is **no geometric pressure** for ( p ) to change.

This makes the task ideal for verifying *stability* rather than expressivity.

---

### Model

The model architecture is:

```
x → Linear → ReLU → NormLayer(p learnable) → Linear → logit
```

Key details:

* The norm layer aggregates ReLU outputs using a weighted L-p formulation.
* The parameter ( p ) is constrained to ( p ≥ 1 ) and is optimized jointly with the weights.
* Training uses Adam with a standard learning rate.

---

### Experimental Setup

* Input dimension: 2
* Hidden dimension: 32
* Batch size: 256
* Training steps: 50
* Initialization: ( p = 2.0 )
* Device: CPU or GPU (if available)

The dataset consists of randomly sampled points from a standard normal distribution.

---

### Observed Result

After training:

```
Learned p: ≈ 2.03
```

Training loss decreases smoothly with no signs of instability.

---

### Interpretation

This outcome is **expected and correct**.

* The task involves only a **single constraint**.
* For one active dimension, L1, L2, and general L-p norms are equivalent.
* Therefore, there is **no gradient signal encouraging ( p ) to move away from its initialization**.

The fact that:

* ( p ) remains near 2,
* gradients do not explode or vanish,
* and training converges normally

confirms that the norm layer is:

* differentiable in practice,
* numerically stable,
* and compatible with standard optimization.

---

### Impact and Takeaways

This experiment establishes that:

1. **Learnable L-p norm aggregation is trainable**

   * Gradients flow through ( p )
   * No special tricks are required for stability

2. **The layer does not invent geometry where none is required**

   * In a 1-constraint task, ( p ) correctly remains near its initial value

3. **The implementation is safe to use in more complex experiments**

   * Subsequent results can be attributed to geometry, not bugs or instability

This smoke test justifies proceeding to experiments where:

* multiple constraints are active simultaneously,
* corner geometry matters,
* and different aggregation norms produce measurably different behavior.

Those questions are addressed in later experiments (e.g. polytope membership with boundary sensitivity probes).

---

### Summary

This experiment confirms the **mechanical correctness** of the norm-aggregation layer with a learnable ( p ), but makes **no claims** about inductive bias or representational advantages.

Its role is foundational: it verifies that later geometric results rest on a stable and well-behaved implementation.

