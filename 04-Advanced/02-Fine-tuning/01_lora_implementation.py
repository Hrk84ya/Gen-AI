"""
LoRA (Low-Rank Adaptation) Implementation

This module implements LoRA for parameter-efficient fine-tuning of large models.
LoRA decomposes weight updates into low-rank matrices, significantly reducing
the number of trainable parameters while maintaining performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer


class LoRALayer(nn.Module):
    """
    LoRA layer that can be applied to any linear layer.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank decomposition
        alpha: Scaling parameter
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer"""
        # x @ A^T @ B^T = x @ (B @ A)^T
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        return self.dropout(lora_output) * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    This replaces a standard linear layer and adds LoRA parameters
    while keeping the original weights frozen.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Store original layer (frozen)
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Add LoRA adaptation
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original output + LoRA adaptation"""
        original_output = self.original_layer(x)
        lora_output = self.lora(x)
        return original_output + lora_output


class LoRAConfig:
    """Configuration for LoRA adaptation"""
    
    def __init__(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: Optional[list] = None,
        bias: str = "none"  # "none", "all", "lora_only"
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["query", "value"]
        self.bias = bias


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: The model to adapt
        config: LoRA configuration
    
    Returns:
        Model with LoRA adaptations applied
    """
    
    def replace_linear_with_lora(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child_module, nn.Linear):
                # Check if this module should be adapted
                should_adapt = any(target in full_name for target in config.target_modules)
                
                if should_adapt:
                    # Replace with LoRA linear layer
                    lora_layer = LoRALinear(
                        original_layer=child_module,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.dropout
                    )
                    setattr(module, child_name, lora_layer)
                    print(f"Applied LoRA to: {full_name}")
            else:
                # Recursively apply to child modules
                replace_linear_with_lora(child_module, full_name)
    
    replace_linear_with_lora(model)
    return model


class AdaLoRA(nn.Module):
    """
    Adaptive LoRA that dynamically adjusts rank based on importance.
    
    This implementation includes importance scoring and rank adaptation
    during training for more efficient parameter allocation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.0,
        importance_threshold: float = 0.1
    ):
        super().__init__()
        self.max_rank = max_rank
        self.alpha = alpha
        self.importance_threshold = importance_threshold
        
        # Initialize with maximum rank
        self.lora_A = nn.Parameter(torch.randn(max_rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, max_rank))
        
        # Importance scores for each rank component
        self.importance_scores = nn.Parameter(torch.ones(max_rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Current effective rank
        self.current_rank = max_rank
    
    def compute_importance(self) -> torch.Tensor:
        """Compute importance scores for rank components"""
        # Use gradient magnitude as importance measure
        if self.lora_A.grad is not None and self.lora_B.grad is not None:
            grad_A = torch.norm(self.lora_A.grad, dim=1)
            grad_B = torch.norm(self.lora_B.grad, dim=0)
            importance = grad_A * grad_B
            
            # Update importance scores with momentum
            self.importance_scores.data = 0.9 * self.importance_scores.data + 0.1 * importance
        
        return self.importance_scores
    
    def prune_low_importance_ranks(self):
        """Remove rank components with low importance"""
        importance = self.compute_importance()
        
        # Find ranks to keep
        keep_mask = importance > self.importance_threshold
        self.current_rank = keep_mask.sum().item()
        
        if self.current_rank < self.max_rank:
            # Create pruned parameters
            keep_indices = torch.where(keep_mask)[0]
            
            with torch.no_grad():
                self.lora_A.data = self.lora_A.data[keep_indices]
                self.lora_B.data = self.lora_B.data[:, keep_indices]
                self.importance_scores.data = self.importance_scores.data[keep_indices]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with current effective rank"""
        # Use only the current effective rank
        A_active = self.lora_A[:self.current_rank]
        B_active = self.lora_B[:, :self.current_rank]
        
        lora_output = x @ A_active.T @ B_active.T
        scaling = self.alpha / self.current_rank if self.current_rank > 0 else 0
        
        return self.dropout(lora_output) * scaling


class LoRATrainer:
    """Trainer class for LoRA fine-tuning"""
    
    def __init__(
        self,
        model: nn.Module,
        config: LoRAConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.model = apply_lora_to_model(model, config)
        self.config = config
        
        # Only optimize LoRA parameters
        lora_params = []
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                lora_params.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        print(f"LoRA parameters: {sum(p.numel() for p in lora_params):,}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable ratio: {sum(p.numel() for p in lora_params) / sum(p.numel() for p in self.model.parameters()):.2%}")
    
    def save_lora_weights(self, path: str):
        """Save only LoRA parameters"""
        lora_state_dict = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                lora_state_dict[name] = param.data
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'config': self.config.__dict__
        }, path)
    
    def load_lora_weights(self, path: str):
        """Load LoRA parameters"""
        checkpoint = torch.load(path)
        lora_state_dict = checkpoint['lora_state_dict']
        
        # Load LoRA parameters
        for name, param in self.model.named_parameters():
            if name in lora_state_dict:
                param.data = lora_state_dict[name]


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into the original model for inference.
    This eliminates the computational overhead of LoRA during inference.
    """
    
    def merge_lora_layer(module):
        for name, child in module.named_children():
            if isinstance(child, LoRALinear):
                # Compute merged weight
                original_weight = child.original_layer.weight.data
                lora_weight = child.lora.lora_B @ child.lora.lora_A * child.lora.scaling
                merged_weight = original_weight + lora_weight
                
                # Create new linear layer with merged weights
                merged_layer = nn.Linear(
                    child.original_layer.in_features,
                    child.original_layer.out_features,
                    bias=child.original_layer.bias is not None
                )
                merged_layer.weight.data = merged_weight
                if child.original_layer.bias is not None:
                    merged_layer.bias.data = child.original_layer.bias.data
                
                setattr(module, name, merged_layer)
            else:
                merge_lora_layer(child)
    
    merge_lora_layer(model)
    return model


# Example usage and testing
if __name__ == "__main__":
    # Example: Apply LoRA to a transformer model
    print("Testing LoRA implementation...")
    
    # Create a simple model for testing
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=512, nhead=8):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.query = nn.Linear(d_model, d_model)
            self.key = nn.Linear(d_model, d_model)
            self.value = nn.Linear(d_model, d_model)
            self.output = nn.Linear(d_model, d_model)
        
        def forward(self, x):
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            attn_out, _ = self.attention(q, k, v)
            return self.output(attn_out)
    
    # Create model and apply LoRA
    model = SimpleTransformer()
    config = LoRAConfig(
        rank=4,
        alpha=8.0,
        target_modules=["query", "value", "output"]
    )
    
    # Apply LoRA
    lora_model = apply_lora_to_model(model, config)
    
    # Test forward pass
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = lora_model(x)
    print(f"Output shape: {output.shape}")
    
    # Test LoRA trainer
    trainer = LoRATrainer(model, config)
    
    # Test saving and loading
    trainer.save_lora_weights("test_lora.pth")
    trainer.load_lora_weights("test_lora.pth")
    
    # Test weight merging
    merged_model = merge_lora_weights(lora_model)
    merged_output = merged_model(x)
    
    print("LoRA implementation tested successfully!")
    print(f"Original output norm: {output.norm():.4f}")
    print(f"Merged output norm: {merged_output.norm():.4f}")
    
    # Clean up
    import os
    if os.path.exists("test_lora.pth"):
        os.remove("test_lora.pth")