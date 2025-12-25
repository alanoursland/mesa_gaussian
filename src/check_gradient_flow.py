import torch
from lp_norm_layer import LpNormLayer

# Setup
in_dim = 10
layer = LpNormLayer(in_dim=in_dim, out_dim=1, p_init=2.0)
optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)

# Scenario: The "Ground Truth" is a Max-Pooling operation
# We want to see if p moves from 2.0 toward a higher value (like 10+)
print(f"{'Iter':<6} | {'p value':<10} | {'p_grad':<10} | {'Loss':<10}")
print("-" * 45)

for i in range(1000):
    x = torch.rand(32, in_dim) + 0.1
    target = x.max(dim=-1, keepdim=True).values
    
    optimizer.zero_grad()
    output = layer(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    p_val = layer.p.item()
    p_grad = layer._raw_p.grad.item()
    
    optimizer.step()
    
    if i % 10 == 0:
        print(f"{i:<6} | {p_val:<10.4f} | {p_grad:<10.4f} | {loss.item():<10.6f}")
    
print(f"{i:<6} | {p_val:<10.4f} | {p_grad:<10.4f} | {loss.item():<10.6f}")
