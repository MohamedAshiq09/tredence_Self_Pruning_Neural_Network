import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    Custom Linear layer with learnable gates for self-pruning.
    
    Each weight has an associated gate (0 to 1) that controls its contribution.
    Gates are learned during training via gradient descent with sparsity regularization.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Gate scores - same shape as weight
        # These will be transformed to [0,1] via sigmoid
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming initialization and gates to positive values"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # Initialize gate_scores to positive values so sigmoid gives ~0.5-0.9 initially
        nn.init.uniform_(self.gate_scores, 0.5, 2.0)
    
    def forward(self, x):
        """
        Forward pass with gated weights.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Transform gate_scores to [0,1] range using sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gates to weights (element-wise multiplication)
        pruned_weights = self.weight * gates
        
        # Standard linear transformation with pruned weights
        output = F.linear(x, pruned_weights, self.bias)
        
        return output
    
    def get_gates(self):
        """Return current gate values (after sigmoid)"""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)
    
    def get_sparsity(self, threshold=1e-2):
        """
        Calculate sparsity level (percentage of pruned weights).
        
        Args:
            threshold: Gate values below this are considered pruned
            
        Returns:
            Sparsity percentage (0-100)
        """
        gates = self.get_gates()
        total_weights = gates.numel()
        pruned_weights = (gates < threshold).sum().item()
        return 100.0 * pruned_weights / total_weights
    
    def hard_prune(self, threshold=1e-2):
        """
        Permanently remove weights with gates below threshold.
        Sets corresponding weights to zero and freezes them.
        """
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            mask = gates < threshold
            self.weight[mask] = 0.0
            # Set gate_scores to very negative value (sigmoid -> 0)
            self.gate_scores[mask] = -10.0
