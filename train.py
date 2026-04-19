import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from model.network import SelfPruningNet


class LambdaScheduler:
    """Dynamic lambda scheduling: linear increase over epochs"""
    
    def __init__(self, base_lambda, total_epochs):
        self.base_lambda = base_lambda
        self.total_epochs = total_epochs
    
    def get_lambda(self, epoch):
        # Simple linear increase: lambda grows with training
        progress = epoch / self.total_epochs
        return self.base_lambda * (1 + progress)


def get_dataloaders(batch_size=128):
    """Load CIFAR-10 dataset with augmentation"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


def train_epoch(model, trainloader, optimizer, criterion, device, lambda_val, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_sparse_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Classification loss
        cls_loss = criterion(outputs, labels)
        
        # Sparsity loss (L1 on gates)
        sparse_loss = model.calculate_sparsity_loss()
        
        # Total loss with warmup: no sparsity for first 2 epochs (let model learn features first)
        if epoch < 2:
            total_loss = cls_loss
        else:
            total_loss = cls_loss + lambda_val * sparse_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += total_loss.item()
        running_cls_loss += cls_loss.item()
        running_sparse_loss += sparse_loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(trainloader)
    epoch_cls_loss = running_cls_loss / len(trainloader)
    epoch_sparse_loss = running_sparse_loss / len(trainloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_cls_loss, epoch_sparse_loss, epoch_acc


def evaluate(model, testloader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = test_loss / len(testloader)
    test_acc = 100.0 * correct / total
    
    return test_loss, test_acc


def train_model(lambda_val, epochs=50, use_scheduler=False, save_dir='results'):
    """
    Train self-pruning network with given lambda value.
    
    Args:
        lambda_val: Sparsity regularization strength
        epochs: Number of training epochs
        use_scheduler: Whether to use dynamic lambda scheduling
        save_dir: Directory to save results
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = get_dataloaders()
    
    # Initialize model
    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Lambda scheduler
    if use_scheduler:
        lambda_sched = LambdaScheduler(
            base_lambda=lambda_val,
            total_epochs=epochs
        )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'sparsity': [],
        'lambda_values': []
    }
    
    print(f"\nTraining with lambda={lambda_val}, epochs={epochs}")
    print("=" * 70)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Get current lambda
        current_lambda = lambda_sched.get_lambda(epoch) if use_scheduler else lambda_val
        
        # Train
        train_loss, cls_loss, sparse_loss, train_acc = train_epoch(
            model, trainloader, optimizer, criterion, device, current_lambda, epoch
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        # Calculate sparsity
        sparsity_stats = model.get_overall_sparsity()
        sparsity = sparsity_stats['overall_sparsity']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['sparsity'].append(sparsity)
        history['lambda_values'].append(current_lambda)
        
        # Learning rate step
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == 1 or epoch == 2 or epoch == 3 or epoch == 4:
            # Get average gate value for debugging
            avg_gate = sum(layer.get_gates().mean().item() for layer in model.get_prunable_layers()) / 3
            warmup_status = "[WARMUP - No Pruning]" if epoch < 2 else ""
            print(f"Epoch [{epoch+1}/{epochs}] {warmup_status} "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | Sparsity: {sparsity:.2f}% | "
                  f"Lambda: {current_lambda:.6f} | Avg Gate: {avg_gate:.4f}")
        
        # Save best model (only after warmup to ensure pruning has started)
        if test_acc > best_acc and epoch >= 10:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'sparsity': sparsity,
                'lambda': lambda_val
            }, os.path.join(save_dir, f'best_model_lambda_{lambda_val}.pth'))
    
    print("=" * 70)
    print(f"Training completed. Best test accuracy: {best_acc:.2f}%")
    
    # Final evaluation
    final_sparsity_stats = model.get_overall_sparsity()
    print(f"\nFinal Sparsity Statistics:")
    print(f"Overall Sparsity: {final_sparsity_stats['overall_sparsity']:.2f}%")
    print(f"Total Weights: {final_sparsity_stats['total_weights']}")
    print(f"Pruned Weights: {final_sparsity_stats['pruned_weights']}")
    print("\nPer-Layer Sparsity:")
    for layer_name, stats in final_sparsity_stats['layer_stats'].items():
        print(f"  {layer_name}: {stats['sparsity']:.2f}% ({stats['pruned']}/{stats['total']})")
    
    # Save results
    results = {
        'lambda': lambda_val,
        'final_test_acc': test_acc,
        'best_test_acc': best_acc,
        'final_sparsity': final_sparsity_stats['overall_sparsity'],
        'sparsity_stats': final_sparsity_stats,
        'history': history
    }
    
    with open(os.path.join(save_dir, f'results_lambda_{lambda_val}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results, history


def plot_gate_distribution(model, save_path='results/gate_distribution.png'):
    """Plot distribution of gate values across all prunable layers"""
    
    all_gates = []
    for layer in model.get_prunable_layers():
        gates = layer.get_gates().cpu().numpy().flatten()
        all_gates.extend(gates)
    
    all_gates = np.array(all_gates)
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(all_gates, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gate Values')
    plt.grid(True, alpha=0.3)
    
    # Log scale histogram
    plt.subplot(1, 2, 2)
    plt.hist(all_gates, bins=100, edgecolor='black', alpha=0.7, log=True)
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency (log scale)')
    plt.title('Distribution of Gate Values (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gate distribution plot saved to {save_path}")
    plt.close()


def plot_training_curves(history, lambda_val, save_path='results/training_curves.png'):
    """Plot training curves"""
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['test_acc'], label='Test', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Accuracy over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['test_loss'], label='Test', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sparsity
    axes[1, 0].plot(epochs, history['sparsity'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sparsity (%)')
    axes[1, 0].set_title('Sparsity over Epochs')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Lambda values
    axes[1, 1].plot(epochs, history['lambda_values'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Lambda Value')
    axes[1, 1].set_title('Lambda Scheduling')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves (Lambda = {lambda_val})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    # Experiment with different lambda values (BALANCED for stable pruning)
    lambda_values = [0.05, 0.1, 0.2]
    
    all_results = []
    
    for lambda_val in lambda_values:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: Lambda = {lambda_val}")
        print(f"{'='*70}\n")
        
        save_dir = f'results/lambda_{lambda_val}'
        os.makedirs(save_dir, exist_ok=True)
        
        # Train model
        model, results, history = train_model(
            lambda_val=lambda_val,
            epochs=50,
            use_scheduler=True,
            save_dir=save_dir
        )
        
        # Plot results
        plot_gate_distribution(model, save_path=f'{save_dir}/gate_distribution.png')
        plot_training_curves(history, lambda_val, save_path=f'{save_dir}/training_curves.png')
        
        all_results.append(results)
        
        print(f"\nResults saved to {save_dir}/")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*70}\n")
    print(f"{'Lambda':<15} {'Test Accuracy':<20} {'Sparsity Level':<20}")
    print("-" * 70)
    for res in all_results:
        print(f"{res['lambda']:<15.6f} {res['final_test_acc']:<20.2f} {res['final_sparsity']:<20.2f}")
    
    # Save summary
    with open('results/summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to results/")
