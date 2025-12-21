import torch
import torch.nn.functional as F


def train_step(model, opt, x, y, loss_type="ce", lambda_grad=0.1, max_grad_norm=1.0):
    """
    x: (batch, in_dim)
    y: (batch, 1) in {0,1}
    loss_type: "ce" | "grad_penalty" | "confidence"
    """
    model.train()
    opt.zero_grad(set_to_none=True)
    
    if loss_type == "ce":
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
    
    elif loss_type == "grad_penalty":
        x = x.detach().requires_grad_(True)
        logits = model(x)
        ce = F.binary_cross_entropy_with_logits(logits, y)
        
        # Gradient of logits w.r.t. input
        grad = torch.autograd.grad(
            logits.sum(), x, create_graph=True
        )[0]
        grad_penalty = (grad ** 2).sum(dim=-1).mean()
        
        loss = ce + lambda_grad * grad_penalty
    
    elif loss_type == "confidence":
        logits = model(x)
        ce = F.binary_cross_entropy_with_logits(logits, y)
        # Push toward ±2 margin
        targets = (2 * y - 1) * 2.0
        margin_loss = F.mse_loss(logits, targets)
        loss = ce + 0.1 * margin_loss
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    loss.backward()

    # Check for NaN gradients
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
            has_nan = True
    
    if has_nan:
        print(f"Skipping step due to NaN gradients (loss={loss.item():.4f})")
        opt.zero_grad(set_to_none=True)  # Clear the bad gradients
        return float('nan')
    
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    opt.step()
    return float(loss.detach())