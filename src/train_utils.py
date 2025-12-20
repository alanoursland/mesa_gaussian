import torch
import torch.nn.functional as F


def train_step(model, opt, x, y):
    """
    x: (batch, in_dim)
    y: (batch, 1) in {0,1}
    """
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    opt.step()
    return float(loss.detach())
