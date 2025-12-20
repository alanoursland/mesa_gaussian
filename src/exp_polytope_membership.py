import torch

from blocks import PolytopeClassifier
from train_utils import train_step


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = 2
    hidden_dim = 32
    model = PolytopeClassifier(in_dim, hidden_dim, learn_p=True, p_init=2.0)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy batch
    x = torch.randn(256, in_dim, device=device)
    y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(-1)

    for _ in range(50):
        loss = train_step(model, opt, x, y)

    # Inspect learned p
    print("Learned p:", float(model.block.norm.p().detach()))
