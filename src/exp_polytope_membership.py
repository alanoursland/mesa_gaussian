import torch
import torch.nn as nn
import torch.nn.functional as F

from lp_norm_layer import LpNormLayer
from train_utils import train_step


class PolytopeClassifier(nn.Module):
    """Norm-based binary classifier: Linear -> ReLU -> NormLayer -> Linear head."""
    def __init__(self, in_dim: int, hidden_dim: int, learn_p: bool = True, p_init: float = 2.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.norm = LpNormLayer(hidden_dim, out_dim=1, learn_p=learn_p, p_init=p_init)
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = F.relu(self.lin(x))
        d = self.norm(v)  # (batch, 1)
        logits = self.head(d)
        return logits
    
    def p_mean(self):
        return self.norm.p.mean().detach().cpu()
    
    def p_variance(self):
        return self.norm.p.var(unbiased=False).detach().cpu()


if __name__ == "__main__":
    # torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = 2
    hidden_dim = 32
    model = PolytopeClassifier(in_dim, hidden_dim, learn_p=False, p_init=1.0)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy batch
    x = torch.randn(256, in_dim, device=device)
    y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(-1)

    # Initial loss
    with torch.no_grad():
        logits = model(x)
        initial_loss = F.binary_cross_entropy_with_logits(logits, y).item()

    # Train
    for _ in range(1000):
        loss = train_step(model, opt, x, y)
    final_loss = loss.item() if hasattr(loss, 'item') else loss

    # Accuracy
    with torch.no_grad():
        preds = (model(x) > 0).float()
        accuracy = (preds == y).float().mean().item() * 100

    # Report
    print("=== Smoke Test: LpNormLayer ===")
    print(f"Initial loss: {initial_loss:.3f}")
    print(f"Final loss:   {final_loss:.3f}")
    print(f"Accuracy:     {accuracy:.1f}%")
    print(f"Learned p:    {model.norm.p.item():.3f}")
    print()
    
    passed = final_loss < initial_loss and accuracy > 90
    print("PASS" if passed else "FAIL")