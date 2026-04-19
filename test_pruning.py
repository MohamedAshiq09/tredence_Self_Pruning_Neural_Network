"""
Quick test to verify pruning mechanism works correctly
"""
import torch
from model.network import SelfPruningNet

print("Testing Self-Pruning Network Setup\n")
print("="*60)

# Create model
model = SelfPruningNet()
print("✓ Model created successfully")

# Test forward pass
x = torch.randn(4, 3, 32, 32)
output = model(x)
print(f"✓ Forward pass works: input {x.shape} -> output {output.shape}")

# Check initial gate values
print("\nInitial Gate Statistics:")
for name, layer in [('fc1', model.fc1), ('fc2', model.fc2), ('fc3', model.fc3)]:
    gates = layer.get_gates()
    print(f"  {name}: mean={gates.mean():.4f}, std={gates.std():.4f}, min={gates.min():.4f}, max={gates.max():.4f}")

# Check sparsity calculation
sparsity_stats = model.get_overall_sparsity()
print(f"\n✓ Initial sparsity: {sparsity_stats['overall_sparsity']:.2f}%")

# Check sparsity loss
sparsity_loss = model.calculate_sparsity_loss()
print(f"✓ Sparsity loss calculation works: {sparsity_loss:.4f}")

# Simulate training steps with high lambda
print("\nSimulating 100 training steps with lambda=0.1...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

lambda_val = 0.1

for step in range(100):
    # Fake labels
    labels = torch.randint(0, 10, (4,))
    
    optimizer.zero_grad()
    
    # Forward
    outputs = model(x)
    cls_loss = criterion(outputs, labels)
    sparse_loss = model.calculate_sparsity_loss()
    
    # Total loss
    total_loss = cls_loss + lambda_val * sparse_loss
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    if step % 20 == 0:
        avg_gate = sum(layer.get_gates().mean().item() for layer in model.get_prunable_layers()) / 3
        curr_sparsity = model.get_overall_sparsity()['overall_sparsity']
        print(f"  Step {step}: Loss={total_loss.item():.4f}, Avg Gate={avg_gate:.4f}, Sparsity={curr_sparsity:.2f}%")

print(f"\n✓ Training steps completed")

# Check gates after training
print("\nGate Statistics After 100 Training Steps:")
for name, layer in [('fc1', model.fc1), ('fc2', model.fc2), ('fc3', model.fc3)]:
    gates = layer.get_gates()
    print(f"  {name}: mean={gates.mean():.4f}, std={gates.std():.4f}, min={gates.min():.4f}, max={gates.max():.4f}")

final_sparsity = model.get_overall_sparsity()
print(f"\nFinal sparsity: {final_sparsity['overall_sparsity']:.2f}%")

print("\n" + "="*60)
if final_sparsity['overall_sparsity'] > 50:
    print("✓ SUCCESS! Strong pruning is working!")
elif final_sparsity['overall_sparsity'] > 20:
    print("✓ Pruning working but moderate")
else:
    print("❌ FAILED: Pruning too weak")
print("\nRun: python train.py")
