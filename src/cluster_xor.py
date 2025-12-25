import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from lp_norm_layer import LpNormLayer

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Generate Data on GPU
def get_xor_clusters(n_points=1000):
    centers = torch.tensor([[0,0], [1,1], [0,1], [1,0]], device=device)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    
    data, targets = [], []
    for i in range(4):
        noise = torch.randn(n_points // 4, 2, device=device) * 0.15
        data.append(centers[i] + noise)
        targets.append(torch.full((n_points // 4,), labels[i], device=device))
    
    return torch.cat(data), torch.cat(targets)

# 2. Model (Moved to Device)
class XORModel(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(2, hidden_dim)
        self.relu = nn.ReLU()
        # Using 2 output norms for the 2 XOR classes
        self.lp = LpNormLayer(in_dim=hidden_dim, out_dim=2, p_init=2.0, learn_p=False).to(device)
        
    def reinit(self, sigma=0.1):
        # Standard Kaiming for the first layer
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        
        # FIX: Initialize with a RANGE [0, sigma] so weights are random
        # This allows different norms to look at different hyperplanes
        nn.init.xavier_normal_(self.lp._raw_w)
        
    def forward(self, x):
        # Activation = Violation from the Mesa
        x = self.relu(self.fc1(x))
        x = self.lp(x)
        # Return negative distance so '0' (inside Mesa) is peak probability
        return -x

def get_best_init(model, X, Y, num_trials=100):
    best_loss = float('inf')
    best_weights = None
    criterion = nn.CrossEntropyLoss()
    
    print(f"Sampling {num_trials} initializations...")
    for i in range(num_trials):
        model.reinit(sigma=0.1)
        with torch.no_grad():
            outputs = model(X)
            # Use .item() to get a standard Python float
            loss = criterion(outputs, Y).item() 
        
        if loss < best_loss:
            best_loss = loss
            # Use deep copy to ensure we aren't just saving a reference
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Trial {i}: New best loss: {best_loss:.4f}")
            
    model.load_state_dict(best_weights)
    return model   

# Initialize
X, Y = get_xor_clusters()
model = XORModel(hidden_dim=4).to(device)
model = get_best_init(model, X, Y, 1000)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 3. Training Loop
print(f"{'Epoch':<8} | {'p_min':<8} | {'p_max':<8} | {'Loss':<10} | {'Acc':<10}")
print("-" * 55)

for epoch in range(401):
    logits = model(X)
    loss = criterion(logits, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == Y).float().mean()
            p_vals = model.lp.p
            print(f"{epoch:<8} | {p_vals.min():<8.2f} | {p_vals.max():<8.2f} | {loss.item():<10.4f} | {acc:<10.2%}")

# 4. Reason about the results
def plot_fc1_hyperplanes(model):
    # Get weights and biases from the first linear layer
    weights = model.fc1.weight.detach().cpu().numpy() # (hidden_dim, 2)
    biases = model.fc1.bias.detach().cpu().numpy()   # (hidden_dim,)
    
    plt.figure(figsize=(8, 8))
    
    # Plot the data clusters for context
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    plt.scatter(X_np[:, 0], X_np[:, 1], c=Y_np, cmap='coolwarm', alpha=0.3, s=10)
    
    # Define x range for plotting lines
    x_range = np.linspace(-0.5, 1.5, 100)
    
    for i in range(weights.shape[0]):
        w = weights[i]
        b = biases[i]
        # Line equation: w0*x + w1*y + b = 0  =>  y = (-w0*x - b) / w1
        if abs(w[1]) > 1e-5:
            y_vals = (-w[0] * x_range - b) / w[1]
            plt.plot(x_range, y_vals, '--', alpha=0.5, label=f'n{i}')
        
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.title("FC1 Neuron Decision Boundaries (Hyperplanes)")
    plt.grid(True)
    plt.show()

def inspect_lp_weights(model):
    # Access the effective weights (absolute values of raw_w)
    # Shape will be (out_dim, in_dim) -> (8, 16)
    weights = model.lp.w.detach().cpu().numpy()
    p_values = model.lp.p.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(weights, cmap='viridis', aspect='auto')
    
    # Add p-values as text labels on the Y-axis
    ax.set_yticks(range(len(p_values)))
    ax.set_yticklabels([f'Norm {i}\n(p={p:.2f})' for i, p in enumerate(p_values)])
    
    ax.set_xlabel('Hidden Feature Index (from FC1)')
    ax.set_title('LpNormLayer Weight Heatmap')
    plt.colorbar(im, label='Weight Magnitude')
    plt.show()

def print_lp_weights_ordered(model):
    # Access effective weights: w = abs(raw_w)
    weights = model.lp.w.detach().cpu().numpy()  # Shape: (out_dim, in_dim)
    p_values = model.lp.p.detach().cpu().numpy() # Shape: (out_dim,)
    
    num_out, num_in = weights.shape
    
    print("\n" + "="*80)
    print(f"{'Norm Index':<12} | {'p-value':<8} | {'Feature Weights (Top 5 magnitude)':<40}")
    print("-" * 80)
    
    for i in range(num_out):
        # Sort weights for this norm to see the most 'important' hyperplanes
        row_weights = weights[i]
        sorted_indices = row_weights.argsort()[::-1] # High to low
        
        # Format the top weights for readability
        top_indices = sorted_indices[:5]
        weight_str = ", ".join([f"H{idx}: {row_weights[idx]:.3f}" for idx in top_indices])
        
        print(f"Norm {i:<7} | {p_values[i]:<8.3f} | {weight_str}")
        
    print("="*80)
    print("Note: H(idx) corresponds to the hyperplane index from your FC1 plot.\n")

def print_full_lp_sparsity(model):
    weights = model.lp.w.detach().cpu().numpy()
    p_values = model.lp.p.detach().cpu().numpy()
    
    num_out, num_in = weights.shape
    
    print("\n" + "="*100)
    print(f"{'INDEX':<6} | {'P-VAL':<6} | FULL WEIGHT DISTRIBUTION (H0 through H{num_in-1})")
    print("-" * 100)
    
    for i in range(num_out):
        # Create a visual representation for the row
        # . = negligible, - = low, + = medium, # = high
        row_str = ""
        for w in weights[i]:
            if w < 0.05: row_str += " . "
            elif w < 0.5: row_str += " - "
            elif w < 1.5: row_str += " + "
            else:         row_str += " # "
            
        print(f"Norm{i:<2} | {p_values[i]:<6.2f} | {row_str}")
        
    print("-" * 100)
    print("Scale:  ( . ) < 0.05  |  ( - ) < 0.5  |  ( + ) < 1.5  |  ( # ) > 1.5")
    print("=" * 100)
    
    # Also print raw values for numerical inspection
    print("\nRaw Weight Matrix (Values):")
    header = "      " + "".join([f"H{j:<6}" for j in range(num_in)])
    print(header)
    for i in range(num_out):
        row = f"N{i:<3}: " + "".join([f"{w:<7.2f}" for w in weights[i]])
        print(row)

print_full_lp_sparsity(model)
# Run the inspection
# print_lp_weights_ordered(model)
# inspect_lp_weights(model)
# plot_fc1_hyperplanes(model)

def plot_polytope_regions(model, X, Y, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5)):
    # 1. Extract Parameters
    W1 = model.fc1.weight.detach()
    b1 = model.fc1.bias.detach()
    W_lp = model.lp.w.detach()
    
    # Move data to CPU for plotting
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # Setup grid
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 400),
                         np.linspace(y_range[0], y_range[1], 400))
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], 
                               dtype=torch.float32, device=W1.device)

    plt.figure(figsize=(14, 6))

    for i in range(W_lp.shape[0]):  # Class 0 and Class 1 Polytopes
        ax = plt.subplot(1, 2, i+1)
        
        # Identify active constraints for this class
        active_mask = W_lp[i] > 0.5 
        active_indices = torch.where(active_mask)[0]
        
        # Calculate plateau region: intersection of half-spaces (d_j <= 0)
        dists = grid_points @ W1.T + b1
        inside_plateau = torch.all(dists[:, active_mask] <= 0, dim=1)
        inside_plateau = inside_plateau.cpu().numpy().reshape(xx.shape)
        
        # Draw the Mesa Plateau (The "Zero" Region)
        plt.contourf(xx, yy, inside_plateau, levels=[0.5, 1], colors=['#2ca02c'], alpha=0.2)
        
        # Draw Hyperplane boundaries
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        for idx in active_indices:
            w = W1[idx].cpu().numpy()
            b = b1[idx].cpu().numpy()
            if abs(w[1]) > 1e-5:
                y_line = (-w[0] * x_vals - b) / w[1]
                plt.plot(x_vals, y_line, 'k-', alpha=0.3, lw=1)
        
        # OVERLAY DATA POINTS
        # Color points by their ground-truth class to see the "fit"
        scatter = plt.scatter(X_np[:, 0], X_np[:, 1], c=Y_np, cmap='coolwarm', 
                              s=15, alpha=0.6, edgecolors='none')
        
        plt.title(f"Polytope for Class {i}\n(Active Constraints: {active_indices.tolist()})")
        plt.xlim(x_range); plt.ylim(y_range)
        plt.grid(True, linestyle=':', alpha=0.4)
        
    plt.tight_layout()
    plt.show()

# Use the best-of-100 model and plot
plot_polytope_regions(model, X, Y)