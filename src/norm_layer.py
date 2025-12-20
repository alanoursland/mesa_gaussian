import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLayer(nn.Module):
    """
    Norm-like aggregation of one-sided violations.
    Input: v >= 0 (e.g., ReLU(Wx + b) output)
    Output: scalar (or per-output if you use groups) distance-like value.

    D(v) = ( sum_i alpha_i * (v_i + eps)^p )^(1/p)
    where alpha_i = softplus(u_i) to keep weights non-negative,
    and p = 1 + softplus(q) (optional learnable p, constrained >= 1).
    """
    def __init__(
        self,
        in_dim: int,
        learn_p: bool = True,
        p_init: float = 2.0,
        eps: float = 1e-6,
        use_exp_weights: bool = False,  # if True: alpha = exp(u), else softplus(u)
    ):
        super().__init__()
        self.in_dim = in_dim
        self.learn_p = learn_p
        self.eps = eps
        self.use_exp_weights = use_exp_weights

        # Unconstrained parameters that will be mapped to positive weights alpha
        self.u = nn.Parameter(torch.zeros(in_dim))

        # Learnable p (optional), constrained to >= 1
        if learn_p:
            # Choose q so that 1 + softplus(q) ≈ p_init
            # softplus(q) ≈ p_init - 1  => q ≈ softplus^{-1}(p_init - 1)
            target = torch.tensor(max(p_init - 1.0, 1e-6))
            q_init = torch.log(torch.expm1(target))  # inverse softplus
            self.q = nn.Parameter(q_init.clone())
        else:
            self.register_buffer("_p_fixed", torch.tensor(float(p_init)))

    def _alpha(self) -> torch.Tensor:
        if self.use_exp_weights:
            # exp can blow up; softplus is usually safer
            return torch.exp(self.u)
        return F.softplus(self.u)

    def p(self) -> torch.Tensor:
        if self.learn_p:
            return 1.0 + F.softplus(self.q)
        return self._p_fixed

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: (batch, in_dim), expected non-negative
        returns: (batch,) distance-like scalar
        """
        if v.dim() != 2 or v.size(-1) != self.in_dim:
            raise ValueError(f"Expected v shape (batch, {self.in_dim}), got {tuple(v.shape)}")

        alpha = self._alpha()  # (in_dim,)
        p = self.p()           # scalar tensor

        # Weighted power sum
        # add eps to avoid 0^p issues for tiny p/grad stability
        vp = (v + self.eps).pow(p)                # (batch, in_dim)
        s = torch.sum(vp * alpha, dim=-1)         # (batch,)
        d = (s + self.eps).pow(1.0 / p)           # (batch,)
        return d
