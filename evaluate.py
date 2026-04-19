import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from model.network import SelfPruningNet
from train import get_dataloaders, evaluate


def load_and_evaluate(model_path, lambda_val):
    """Load trained model and evaluate"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SelfPruningNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nEvaluating model: {model_path}")
    print(f"Lambda: {lambda_val}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint test acc: {checkpoint['test_acc']:.2f}%")
    
    # Load test data
    _, testloader = get_dataloaders()
    
    # Evaluate
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, testloader, criterion, device)
    
    # Get sparsity
    sparsity_stats = model.get_overall_sparsity()
    
    print(f"\nCurrent Evaluation:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Overall Sparsity: {sparsity_stats['overall_sparsity']:.2f}%")
    print(f"\nLayer-wise Sparsity:")
    for layer_name, stats in sparsity_stats['layer_stats'].items():
        print(f"  {layer_name}: {stats['sparsity']:.2f}%")
    
    return {
        'lambda': lambda_val,
        'test_acc': test_acc,
        'sparsity': sparsity_stats['overall_sparsity'],
        'layer_stats': sparsity_stats['layer_stats']
    }


def compare_models():
    """Compare all trained models"""
    
    lambda_values = [0.0001, 0.001, 0.01]
    results = []
    
    for lambda_val in lambda_values:
        model_path = f'results/lambda_{lambda_val}/best_model_lambda_{lambda_val}.pth'
        if os.path.exists(model_path):
            result = load_and_evaluate(model_path, lambda_val)
            results.append(result)
        else:
            print(f"Model not found: {model_path}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"{'Lambda':<15} {'Test Accuracy (%)':<20} {'Sparsity (%)':<20}")
    print("-"*70)
    for res in results:
        print(f"{res['lambda']:<15.6f} {res['test_acc']:<20.2f} {res['sparsity']:<20.2f}")
    print("="*70)
    
    # Create comparison plot
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        lambdas = [r['lambda'] for r in results]
        accs = [r['test_acc'] for r in results]
        sparsities = [r['sparsity'] for r in results]
        
        # Accuracy vs Lambda
        ax1.plot(lambdas, accs, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Lambda (log scale)')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Test Accuracy vs Lambda')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Sparsity vs Lambda
        ax2.plot(lambdas, sparsities, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Lambda (log scale)')
        ax2.set_ylabel('Sparsity (%)')
        ax2.set_title('Sparsity vs Lambda')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plot saved to results/model_comparison.png")
        plt.close()
    
    return results


if __name__ == '__main__':
    results = compare_models()
