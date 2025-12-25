# LpNormLayer Smoke Test

## Purpose

Verify that the `LpNormLayer` module integrates correctly into a trainable network without numerical errors.

## Setup

**Architecture:** Linear(2, 32) → ReLU → LpNormLayer(32, 1) → Linear(1, 1)

**Task:** Binary classification. Positive class if `x[0] + x[1] > 0`, else negative.

**Training:**
- Batch size: 256 (fixed random batch)
- Optimizer: Adam, lr=1e-3
- Epochs: 50
- Loss: Binary cross-entropy with logits
- p: Fixed at 1.0 (not learned)

## Results

| Run | Initial Loss | Final Loss | Accuracy | p |
|-----|--------------|------------|----------|-----|
| 1 | 1.002 | 0.093 | 99.6% | 1.0 |
| 2 | 2.255 | 0.118 | 100.0% | 1.0 |
| 3 | 1.637 | 0.158 | 99.2% | 1.0 |

## Conclusion

**PASS.** The LpNormLayer trains without numerical instability across multiple random initializations. Loss decreases consistently, accuracy exceeds 99%.

The module is ready for use in more substantive experiments.
