"""
Progressive Unfreezing Implementation

This module implements progressive unfreezing strategies for fine-tuning
deep neural networks. Progressive unfreezing gradually unfreezes layers
during training to prevent catastrophic forgetting and improve adaptation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Callable, Union
import numpy as np
from collections import defaultdict


class ProgressiveUnfreezer:
    """
    Implements progressive unfreezing strategies for neural networks.
    
    Supports multiple unfreezing schedules:
    - Linear: Unfreeze layers at regular intervals
    - Exponential: Unfreeze layers with exponentially increasing intervals
    - Performance-based: Unfreeze based on validation performance
    - Custom: User-defined unfreezing schedule
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: str = "linear",
        total_epochs: int = 100,
        unfreeze_schedule: Optional[List[int]] = None,
        performance_threshold: float = 0.01,
        patience: int = 5
    ):
        self.model = model
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.unfreeze_schedule = unfreeze_schedule
        self.performance_threshold = performance_threshold
        self.patience = patience
        
        # Get all layers that can be unfrozen
        self.layers = self._get_unfrozen_layers()
        self.frozen_layers = list(range(len(self.layers)))
        
        # Performance tracking for adaptive unfreezing
        self.performance_history = []
        self.epochs_since_improvement = 0
        self.best_performance = float('-inf')
        
        # Initially freeze all layers except the last one
        self._freeze_all_layers()
        if self.layers:
            self._unfreeze_layer(-1)  # Unfreeze last layer (classifier)
    
    def _get_unfrozen_layers(self) -> List[nn.Module]:
        """Get list of layers that can be progressively unfrozen"""
        layers = []
        
        def collect_layers(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Skip certain layer types that shouldn't be unfrozen
                if isinstance(child, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    continue
                
                # If it has parameters, it's a layer we can unfreeze
                if any(p.requires_grad for p in child.parameters()):
                    layers.append((full_name, child))
                
                # Recursively collect from children
                collect_layers(child, full_name)
        
        collect_layers(self.model)
        return layers
    
    def _freeze_all_layers(self):
        """Freeze all model parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _unfreeze_layer(self, layer_idx: int):
        """Unfreeze a specific layer"""
        if layer_idx < 0:
            layer_idx = len(self.layers) + layer_idx
        
        if 0 <= layer_idx < len(self.layers):
            layer_name, layer = self.layers[layer_idx]
            for param in layer.parameters():
                param.requires_grad = True
            
            if layer_idx in self.frozen_layers:
                self.frozen_layers.remove(layer_idx)
            
            print(f"Unfroze layer {layer_idx}: {layer_name}")
    
    def _get_linear_schedule(self) -> List[int]:
        """Generate linear unfreezing schedule"""
        if len(self.layers) <= 1:
            return []
        
        # Unfreeze layers at regular intervals
        interval = max(1, self.total_epochs // (len(self.layers) - 1))
        schedule = []
        
        # Start from second-to-last layer (last is already unfrozen)
        for i in range(len(self.layers) - 2, -1, -1):
            epoch = (len(self.layers) - 2 - i + 1) * interval
            schedule.append((epoch, i))
        
        return schedule
    
    def _get_exponential_schedule(self) -> List[int]:
        """Generate exponential unfreezing schedule"""
        if len(self.layers) <= 1:
            return []
        
        schedule = []
        base_interval = max(1, self.total_epochs // (2 * len(self.layers)))
        
        for i in range(len(self.layers) - 2, -1, -1):
            layer_num = len(self.layers) - 2 - i
            epoch = int(base_interval * (2 ** layer_num))
            if epoch < self.total_epochs:
                schedule.append((epoch, i))
        
        return schedule
    
    def _should_unfreeze_adaptive(self, current_performance: float) -> bool:
        """Determine if we should unfreeze next layer based on performance"""
        if current_performance > self.best_performance + self.performance_threshold:
            self.best_performance = current_performance
            self.epochs_since_improvement = 0
            return False
        else:
            self.epochs_since_improvement += 1
            return self.epochs_since_improvement >= self.patience
    
    def step(self, epoch: int, performance: Optional[float] = None) -> bool:
        """
        Perform unfreezing step for current epoch.
        
        Args:
            epoch: Current training epoch
            performance: Current validation performance (for adaptive strategies)
        
        Returns:
            True if a layer was unfrozen, False otherwise
        """
        unfrozen = False
        
        if self.strategy == "linear":
            schedule = self._get_linear_schedule()
            for unfreeze_epoch, layer_idx in schedule:
                if epoch == unfreeze_epoch and layer_idx in self.frozen_layers:
                    self._unfreeze_layer(layer_idx)
                    unfrozen = True
                    break
        
        elif self.strategy == "exponential":
            schedule = self._get_exponential_schedule()
            for unfreeze_epoch, layer_idx in schedule:
                if epoch == unfreeze_epoch and layer_idx in self.frozen_layers:
                    self._unfreeze_layer(layer_idx)
                    unfrozen = True
                    break
        
        elif self.strategy == "adaptive" and performance is not None:
            self.performance_history.append(performance)
            
            if (self.frozen_layers and 
                self._should_unfreeze_adaptive(performance)):
                # Unfreeze the next layer (from top to bottom)
                next_layer = max(self.frozen_layers)
                self._unfreeze_layer(next_layer)
                self.epochs_since_improvement = 0
                unfrozen = True
        
        elif self.strategy == "custom" and self.unfreeze_schedule:
            for unfreeze_epoch, layer_idx in self.unfreeze_schedule:
                if epoch == unfreeze_epoch and layer_idx in self.frozen_layers:
                    self._unfreeze_layer(layer_idx)
                    unfrozen = True
                    break
        
        return unfrozen
    
    def get_unfrozen_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of currently unfrozen parameters"""
        unfrozen_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                unfrozen_params.append(param)
        return unfrozen_params
    
    def get_status(self) -> Dict:
        """Get current unfreezing status"""
        total_params = sum(p.numel() for p in self.model.parameters())
        unfrozen_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_layers': len(self.layers),
            'frozen_layers': len(self.frozen_layers),
            'unfrozen_layers': len(self.layers) - len(self.frozen_layers),
            'total_parameters': total_params,
            'unfrozen_parameters': unfrozen_params,
            'unfrozen_ratio': unfrozen_params / total_params if total_params > 0 else 0
        }


class DiscriminativeLearningRates:
    """
    Implements discriminative learning rates for different layers.
    Lower layers get lower learning rates to preserve pre-trained features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 1e-3,
        lr_decay: float = 0.5,
        layer_groups: Optional[List[List[str]]] = None
    ):
        self.model = model
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.layer_groups = layer_groups or self._auto_group_layers()
    
    def _auto_group_layers(self) -> List[List[str]]:
        """Automatically group layers for discriminative learning rates"""
        layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                layer_names.append(name)
        
        # Group layers into thirds
        n_layers = len(layer_names)
        group_size = max(1, n_layers // 3)
        
        groups = []
        for i in range(0, n_layers, group_size):
            groups.append(layer_names[i:i + group_size])
        
        return groups
    
    def get_parameter_groups(self) -> List[Dict]:
        """Get parameter groups with different learning rates"""
        param_groups = []
        
        for i, group in enumerate(self.layer_groups):
            # Calculate learning rate for this group
            lr = self.base_lr * (self.lr_decay ** (len(self.layer_groups) - 1 - i))
            
            # Collect parameters for this group
            group_params = []
            for name, param in self.model.named_parameters():
                if any(layer_name in name for layer_name in group):
                    group_params.append(param)
            
            if group_params:
                param_groups.append({
                    'params': group_params,
                    'lr': lr,
                    'group_name': f'group_{i}'
                })
        
        return param_groups


class SlantedTriangularLR:
    """
    Implements Slanted Triangular Learning Rate schedule.
    Learning rate increases linearly then decreases linearly.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: float,
        total_steps: int,
        cut_frac: float = 0.1,
        ratio: float = 32
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.cut_frac = cut_frac
        self.ratio = ratio
        
        self.cut_step = int(total_steps * cut_frac)
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.cut_step:
            # Increasing phase
            lr = self.max_lr * self.step_count / self.cut_step
        else:
            # Decreasing phase
            remaining_steps = self.total_steps - self.cut_step
            current_step = self.step_count - self.cut_step
            lr = self.max_lr * (1 - current_step / remaining_steps) / self.ratio
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class ProgressiveFinetuner:
    """
    Complete progressive fine-tuning trainer with multiple strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        unfreezing_strategy: str = "linear",
        base_lr: float = 1e-3,
        use_discriminative_lr: bool = True,
        use_slanted_triangular: bool = True,
        total_epochs: int = 100
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.total_epochs = total_epochs
        
        # Initialize progressive unfreezer
        self.unfreezer = ProgressiveUnfreezer(
            model=model,
            strategy=unfreezing_strategy,
            total_epochs=total_epochs
        )
        
        # Setup optimizer with discriminative learning rates
        if use_discriminative_lr:
            discriminative_lr = DiscriminativeLearningRates(model, base_lr)
            param_groups = discriminative_lr.get_parameter_groups()
            self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        else:
            unfrozen_params = self.unfreezer.get_unfrozen_parameters()
            self.optimizer = optim.AdamW(unfrozen_params, lr=base_lr, weight_decay=0.01)
        
        # Setup learning rate scheduler
        if use_slanted_triangular:
            total_steps = len(train_loader) * total_epochs
            self.lr_scheduler = SlantedTriangularLR(
                self.optimizer, base_lr, total_steps
            )
        else:
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs
            )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            if isinstance(self.lr_scheduler, SlantedTriangularLR):
                self.lr_scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if not isinstance(self.lr_scheduler, SlantedTriangularLR):
            self.lr_scheduler.step()
        
        return total_loss / num_batches
    
    def validate(self) -> tuple:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy
    
    def train(self) -> Dict:
        """Complete training loop with progressive unfreezing"""
        print("Starting progressive fine-tuning...")
        
        for epoch in range(self.total_epochs):
            # Check if we should unfreeze layers
            val_loss, val_acc = self.validate()
            unfrozen = self.unfreezer.step(epoch, val_acc)
            
            if unfrozen:
                # Update optimizer with new unfrozen parameters
                unfrozen_params = self.unfreezer.get_unfrozen_parameters()
                for param_group in self.optimizer.param_groups:
                    param_group['params'] = [p for p in unfrozen_params 
                                           if p in param_group['params'] or 
                                           any(torch.equal(p, existing_p) 
                                               for existing_p in param_group['params'])]
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            status = self.unfreezer.get_status()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{self.total_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Unfrozen Layers: {status['unfrozen_layers']}/{status['total_layers']}")
            print(f"  Unfrozen Params: {status['unfrozen_ratio']:.2%}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'final_status': self.unfreezer.get_status()
        }


# Example usage
if __name__ == "__main__":
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # Test progressive unfreezer
    model = SimpleModel()
    unfreezer = ProgressiveUnfreezer(model, strategy="linear", total_epochs=10)
    
    print("Initial status:")
    print(unfreezer.get_status())
    
    # Simulate training epochs
    for epoch in range(10):
        unfrozen = unfreezer.step(epoch)
        if unfrozen:
            print(f"Epoch {epoch}: Layer unfrozen")
            print(unfreezer.get_status())
    
    print("Progressive unfreezing test completed!")