import torch
import torch.nn as nn
from model.prunable_layer import PrunableLinear


class SelfPruningNet(nn.Module):
    """
    Self-pruning neural network for CIFAR-10 classification.
    
    Architecture:
    - Conv layers for feature extraction
    - PrunableLinear layers that learn to prune themselves
    """
    
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = PrunableLinear(128 * 4 * 4, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x) 
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x) 
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x) 
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_prunable_layers(self):
        """Return all prunable layers for sparsity calculation"""
        return [self.fc1, self.fc2, self.fc3]
    
    def calculate_sparsity_loss(self):
        """
        Calculate L1 penalty on all gates (encourages sparsity).
        Uses mean with VERY STRONG amplification for maximum pruning signal.
        
        Returns:
            Scalar tensor representing average gate activation (amplified)
        """
        total_gates = 0
        total_activation = 0.0
        
        for layer in self.get_prunable_layers():
            gates = torch.sigmoid(5.0 * layer.gate_scores)
            total_activation += gates.sum()
            total_gates += gates.numel()
        
        # Return mean activation with 200x amplification for maximum pruning pressure
        sparsity_loss = 200.0 * (total_activation / total_gates)
        return sparsity_loss
    
    def get_overall_sparsity(self, threshold=0.3):
        """
        Calculate overall sparsity across all prunable layers.
        
        Returns:
            Dictionary with sparsity stats
        """
        total_weights = 0
        pruned_weights = 0
        layer_stats = {}
        
        for name, layer in [('fc1', self.fc1), ('fc2', self.fc2), ('fc3', self.fc3)]:
            gates = layer.get_gates()
            layer_total = gates.numel()
            layer_pruned = (gates < threshold).sum().item()
            
            total_weights += layer_total
            pruned_weights += layer_pruned
            
            layer_stats[name] = {
                'total': layer_total,
                'pruned': layer_pruned,
                'sparsity': 100.0 * layer_pruned / layer_total
            }
        
        overall_sparsity = 100.0 * pruned_weights / total_weights
        
        return {
            'overall_sparsity': overall_sparsity,
            'total_weights': total_weights,
            'pruned_weights': pruned_weights,
            'layer_stats': layer_stats
        }
    
    def hard_prune_all(self, threshold=0.3):
        """Apply hard pruning to all prunable layers"""
        for layer in self.get_prunable_layers():
            layer.hard_prune(threshold)
