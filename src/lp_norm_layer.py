import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LpNormLayer(nn.Module):
    """
    Weighted Lp-norm layer for neural networks.
       
    References:
        - Gulcehre et al. (2014) "Learned-Norm Pooling for Deep Feedforward 
          and Recurrent Neural Networks"
        - Sermanet et al. (2012) introduced Lp pooling for CNNs
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        learn_p: bool = True,
        p_init: float = 2.0,
        eps: float = 1e-6,
        bias=True
    ):
        super().__init__()
        
        if p_init < 1.0:
            raise ValueError(f"p_init must be >= 1.0, got {p_init}")
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.learn_p = learn_p
        self.eps = eps

        self._raw_w = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.uniform_(self._raw_w, 0.1, 1.0)

        
        if learn_p:
            raw_p_init = math.log(math.log(2 ** p_init - 1))
            self._raw_p = nn.Parameter(torch.full((out_dim,), raw_p_init))
        else:
            # Fixed p, not learned - register as buffer
            raw_p_init = p_init
            self.register_buffer("_raw_p", torch.full((out_dim,), raw_p_init))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_buffer("bias", torch.zeros(out_dim))

    @property
    def w(self) -> torch.Tensor:
        """Effective weights, shape (out_dim, in_dim). Always non-negative."""
        return torch.abs(self._raw_w)

    @property
    def p(self) -> torch.Tensor:
        if self.learn_p:
            p_val = F.softplus(torch.exp(self._raw_p)) / math.log(2)
        else:
            p_val = self._raw_p
        return p_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(-1) != self.in_dim:
            raise ValueError(f"Expected x shape (batch, {self.in_dim}), got {tuple(x.shape)}")

        # === DEBUG: Check input ===
        if torch.isnan(x).any():
            raise ValueError("NaN in input x")
        if torch.isinf(x).any():
            raise ValueError("Inf in input x")


        # Get effective parameters
        p = self.p  # (out_dim,)

        x = x.unsqueeze(1) # (batch, 1, in_dim)
        x = self._raw_w * x # (out_dim, in_dim) broadcasts -> (batch, out_dim, in_dim)
        
        # Ensure non-negative inputs (should already be if following ReLU)
        x = torch.abs(x)  # (batch, out_dim, in_dim)

        # === Numerical stability for large p ===
        # Factor out max: ||x||_p = x_max * ||x/x_max||_p
        # This keeps all values raised to p in [0, 1], preventing overflow.
        x_max = x.max(dim=-1, keepdim=True).values  # (batch, out_dim, 1)
        x_max = x_max.clamp(min=self.eps)
        x_normalized = x / x_max  # (batch, out_dim, in_dim), values in [0, 1]

        # p: (out_dim,) -> (1, out_dim, 1) for broadcasting
        p_expanded = p.view(1, self.out_dim, 1)

        # Compute (x_i / x_max)^p - safe because x_normalized in [0, 1]
        x_powered = x_normalized.pow(p_expanded)  # (batch, out_dim, in_dim)

        # === DEBUG: Check intermediate values ===
        if torch.isnan(x_powered).any():
            print(f"NaN in x_powered")
            print(f"  x_normalized range: [{x_normalized.min()}, {x_normalized.max()}]")
            print(f"  p: {p}")
            raise ValueError("NaN in x_powered")

        # Sum (weights already applied)
        summed = x_powered.sum(dim=-1)  # (batch, out_dim)

        # === DEBUG ===
        if torch.isnan(summed).any():
            print(f"NaN in summed")
            raise ValueError("NaN in summed")

        # Add eps before taking root to handle edge case of all-zero weights
        summed = summed + self.eps

        # Take p-th root
        p_recip = 1.0 / p.unsqueeze(0)  # (1, out_dim)
        result = summed.pow(p_recip)  # (batch, out_dim)

        # === DEBUG ===
        if torch.isnan(result).any():
            print(f"NaN in result")
            print(f"  summed range: [{summed.min()}, {summed.max()}]")
            print(f"  p_recip: {p_recip}")
            raise ValueError("NaN in result")

        # Restore scale
        x_max = x_max.squeeze(-1)  # (batch, out_dim)
        y = x_max * result  # (batch, out_dim)

        return y + self.bias

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"learn_p={self.learn_p}, ps={self.eps}"
        )