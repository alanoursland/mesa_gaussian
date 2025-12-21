import math
import torch
import torch.nn as nn


class LpNormLayer(nn.Module):
    """
    Weighted Lp-norm layer for neural networks.
    
    Computes a learnable weighted Lp-norm over inputs, analogous to nn.Linear
    but using norm-based aggregation instead of dot product.
    
    For each output j:
        y_j = x_max * (sum_i w_{j,i} * (x_i / x_max)^{p_j})^{1/p_j}
    
    where:
        - w_{j,i} = |raw_w_{j,i}|  (non-negative by construction)
        - p_j = clamp(exp(raw_p_j), 1, 50)  (learnable exponent in [1, 50])
        - x_max = max(|x|)  (factored out for numerical stability)
    
    Design choices:
    
        Weights via abs(): We need w >= 0 for the Lp-norm to be mathematically
        valid (non-negative terms under the sum, real-valued roots). Using abs()
        allows gradient to flow from both sides toward zero, and w=0 is a valid
        stable point meaning "ignore this input."
        
        Exponent via clamped exp(): We need p >= 1 for triangle inequality 
        (otherwise it's only a quasinorm). Using exp() compresses the useful 
        range [1, 50] into ~4 units of parameter space, making all practically
        useful p values easily reachable. The clamp at p=1 creates zero gradient
        which acts as a wall (not a hole) - raw_p cannot drift into deep negative
        territory because there's no gradient to push it there.
        
        Max-normalization: For large p, computing x^p directly causes overflow.
        Factoring out x_max ensures all terms raised to p are in [0, 1].
    
    Mathematical properties (for p >= 1):
        - Satisfies norm axioms: non-negativity, homogeneity, triangle inequality
        - 1-Lipschitz continuous
        - Convex function
        - Interpolates between sum (p=1) and max (p→∞)
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension (number of independent norms to compute)
        learn_p: If True, p is learned per output. If False, p is fixed.
        p_init: Initial value for p (default: 2.0 for L2-like behavior)
        p_max: Maximum value for p (default: 50.0)
        eps: Small constant for numerical stability
    
    Input shape: (batch, in_dim) - should be non-negative (e.g., after ReLU)
    Output shape: (batch, out_dim)
    
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
        p_max: float = 50.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        if p_init < 1.0:
            raise ValueError(f"p_init must be >= 1.0, got {p_init}")
        if p_max < p_init:
            raise ValueError(f"p_max must be >= p_init, got p_max={p_max}, p_init={p_init}")
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.learn_p = learn_p
        self.p_max = p_max
        self.eps = eps

        # Weights: (out_dim, in_dim)
        # We use abs(raw_w) to ensure non-negativity.
        # Initialize with small positive values; after abs(), these stay positive.
        self._raw_w = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.uniform_(self._raw_w, 0.1, 1.0)

        # Exponent: p = clamp(exp(raw_p), 1, p_max)
        # Initialize raw_p so that exp(raw_p) = p_init, i.e., raw_p = log(p_init)
        raw_p_init = math.log(p_init)
        
        if learn_p:
            self._raw_p = nn.Parameter(torch.full((out_dim,), raw_p_init))
        else:
            # Fixed p, not learned - register as buffer
            self.register_buffer("_raw_p", torch.full((out_dim,), raw_p_init))

    @property
    def w(self) -> torch.Tensor:
        """Effective weights, shape (out_dim, in_dim). Always non-negative."""
        return torch.abs(self._raw_w)

    @property
    def p(self) -> torch.Tensor:
        """Effective p values, shape (out_dim,). Always in [1, p_max]."""
        return torch.clamp(torch.exp(self._raw_p), min=1.0, max=self.p_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted Lp-norm.
        
        Args:
            x: Input tensor of shape (batch, in_dim). Expected non-negative.
        
        Returns:
            Output tensor of shape (batch, out_dim).
        """
        if x.dim() != 2 or x.size(-1) != self.in_dim:
            raise ValueError(f"Expected x shape (batch, {self.in_dim}), got {tuple(x.shape)}")

        # === DEBUG: Check input ===
        if torch.isnan(x).any():
            raise ValueError("NaN in input x")
        if torch.isinf(x).any():
            raise ValueError("Inf in input x")


        # Get effective parameters
        w = self.w  # (out_dim, in_dim)
        p = self.p  # (out_dim,)
        # print(self._raw_p, p)
        
        # Ensure non-negative inputs (should already be if following ReLU)
        x = torch.abs(x)  # (batch, in_dim)

        # === Numerical stability for large p ===
        # Factor out max: ||x||_p = x_max * ||x/x_max||_p
        # This keeps all values raised to p in [0, 1], preventing overflow.
        x_max = x.max(dim=-1, keepdim=True).values  # (batch, 1)
        x_max = x_max.clamp(min=self.eps)  # avoid division by zero
        x_normalized = x / x_max  # (batch, in_dim), values in [0, 1]

        # Reshape for broadcasting
        # x_normalized: (batch, in_dim) -> (batch, 1, in_dim)
        # p: (out_dim,) -> (1, out_dim, 1)
        # w: (out_dim, in_dim) -> (1, out_dim, in_dim)
        x_normalized = x_normalized.unsqueeze(1)  # (batch, 1, in_dim)
        p_expanded = p.view(1, self.out_dim, 1)   # (1, out_dim, 1)
        w_expanded = w.unsqueeze(0)               # (1, out_dim, in_dim)

        # Compute (x_i / x_max)^p - safe because x_normalized in [0, 1]
        x_powered = x_normalized.pow(p_expanded)  # (batch, out_dim, in_dim)

        # === DEBUG: Check intermediate values ===
        if torch.isnan(x_powered).any():
            print(f"NaN in x_powered")
            print(f"  x_normalized range: [{x_normalized.min()}, {x_normalized.max()}]")
            print(f"  p: {p}")
            raise ValueError("NaN in x_powered")


        # Weighted sum
        weighted_sum = (w_expanded * x_powered).sum(dim=-1)  # (batch, out_dim)

        # === DEBUG ===
        if torch.isnan(weighted_sum).any():
            print(f"NaN in weighted_sum")
            print(f"  w range: [{w.min()}, {w.max()}]")
            raise ValueError("NaN in weighted_sum")



        # Add eps before taking root to handle edge case of all-zero weights
        weighted_sum = weighted_sum + self.eps

        # Take p-th root
        p_recip = 1.0 / p.unsqueeze(0)  # (1, out_dim)
        result = weighted_sum.pow(p_recip)  # (batch, out_dim)

        # === DEBUG ===
        if torch.isnan(result).any():
            print(f"NaN in result")
            print(f"  weighted_sum range: [{weighted_sum.min()}, {weighted_sum.max()}]")
            print(f"  p_recip: {p_recip}")
            raise ValueError("NaN in result")

        # Restore scale
        y = x_max * result  # (batch, out_dim)

        return y

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"learn_p={self.learn_p}, p_max={self.p_max}, eps={self.eps}"
        )