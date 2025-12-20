import torch
import torch.nn as nn
import torch.nn.functional as F

from norm_layer import NormLayer


class LinearNormBlock(nn.Module):
    """
    x -> linear -> relu -> norm -> (optional affine) y
    Here norm collapses hidden dimension to a scalar per example.
    """
    def __init__(self, in_dim: int, hidden_dim: int, learn_p: bool = True, p_init: float = 2.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.norm = NormLayer(hidden_dim, learn_p=learn_p, p_init=p_init)
        self.out_bias = nn.Parameter(torch.zeros(()))  # scalar bias (optional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.lin(x))     # (batch, hidden_dim), one-sided violations
        d = self.norm(v)            # (batch,)
        return d + self.out_bias    # (batch,)


class PolytopeClassifier(nn.Module):
    """
    Simple classifier:
      x -> LinearNormBlock -> linear -> sigmoid
    Produces logits.
    """
    def __init__(self, in_dim: int, hidden_dim: int, learn_p: bool = True, p_init: float = 2.0):
        super().__init__()
        self.block = LinearNormBlock(in_dim, hidden_dim, learn_p=learn_p, p_init=p_init)
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.block(x).unsqueeze(-1)  # (batch, 1)
        logits = self.head(d)            # (batch, 1)
        return logits

    def internal_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """Return the internal distance-like scalar before the head."""
        return self.block(x)


class MLPBaseline(nn.Module):
    """Matched MLP baseline: Linear -> ReLU -> LinearAggregation -> head

    Architecture mirrors PolytopeClassifier but uses learned linear aggregation
    instead of norm aggregation. This isolates the effect of the norm operation.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.agg = nn.Linear(hidden_dim, 1)  # linear aggregation to scalar
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.lin(x))  # (batch, hidden_dim)
        d = self.agg(v).squeeze(-1)  # (batch,)
        logits = self.head(d.unsqueeze(-1))
        return logits

    def internal_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """Return the internal aggregated scalar before the head."""
        v = F.relu(self.lin(x))
        return self.agg(v).squeeze(-1)
