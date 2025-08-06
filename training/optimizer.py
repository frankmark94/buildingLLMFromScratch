#!/usr/bin/env python3
"""
Optimizer and learning rate scheduler implementations for transformer training.
"""

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class AdamW(optim.AdamW):
    """Extended AdamW optimizer with additional features."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, amsgrad=False, decoupled_weight_decay=True):
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.decoupled_weight_decay = decoupled_weight_decay


class CosineAnnealingWithWarmupLR(_LRScheduler):
    """Cosine annealing learning rate scheduler with linear warmup."""
    
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, 
                 min_lr_ratio: float = 0.1, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [self.min_lr_ratio * base_lr + 
                   (base_lr - self.min_lr_ratio * base_lr) * 0.5 * 
                   (1 + math.cos(math.pi * progress))
                   for base_lr in self.base_lrs]


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup followed by cosine annealing decay."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps 
                   for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * 
                   (1 + math.cos(math.pi * progress))
                   for base_lr in self.base_lrs]


class InverseSqrtLR(_LRScheduler):
    """Inverse square root learning rate schedule (Transformer paper style)."""
    
    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = max(self.last_epoch + 1, 1)  # Avoid division by zero
        
        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Inverse square root decay
            return [base_lr * math.sqrt(self.warmup_steps / step) for base_lr in self.base_lrs]


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Don't apply weight decay to bias, layer norms, and embeddings
        if (param.dim() <= 1 or 
            'bias' in name or 
            'ln' in name or 
            'norm' in name or 
            'embedding' in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {
            'params': decay_params,
            'weight_decay': config.get('weight_decay', 0.1),
            'lr': config.get('learning_rate', 6e-4)
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
            'lr': config.get('learning_rate', 6e-4)
        }
    ]
    
    optimizer_type = config.get('optimizer', 'adamw').lower()
    
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=config.get('learning_rate', 6e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.95)),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 0.1)
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=config.get('learning_rate', 6e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.95)),
            eps=config.get('eps', 1e-8)
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=config.get('learning_rate', 6e-4),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.1)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer with {len(decay_params)} decay params "
               f"and {len(no_decay_params)} no-decay params")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, 
                    config: Dict[str, Any]) -> Optional[_LRScheduler]:
    """Create learning rate scheduler based on configuration."""
    
    scheduler_type = config.get('lr_scheduler', 'cosine').lower()
    
    if scheduler_type == 'none' or scheduler_type is None:
        return None
    
    warmup_steps = config.get('warmup_steps', 2000)
    max_steps = config.get('max_steps', 100000)
    
    if scheduler_type == 'cosine':
        min_lr_ratio = config.get('min_lr', 6e-5) / config.get('learning_rate', 6e-4)
        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio
        )
    elif scheduler_type == 'linear_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=max_steps,
            min_lr=config.get('min_lr', 6e-5)
        )
    elif scheduler_type == 'inverse_sqrt':
        scheduler = InverseSqrtLR(
            optimizer,
            warmup_steps=warmup_steps
        )
    elif scheduler_type == 'constant':
        from torch.optim.lr_scheduler import ConstantLR
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=warmup_steps)
    elif scheduler_type == 'exponential':
        gamma = config.get('lr_decay_gamma', 0.99)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler with warmup_steps={warmup_steps}")
    
    return scheduler


def get_parameter_groups(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """Get parameter groups for different weight decay rates."""
    
    # Parameters that should have weight decay
    decay_params = set()
    
    # Parameters that should NOT have weight decay
    no_decay_params = set()
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Apply weight decay to linear layer weights (but not biases)
        if param.dim() >= 2:  # Weight matrices
            decay_params.add(name)
        else:  # Biases and other 1D parameters
            no_decay_params.add(name)
    
    # Override for specific parameter names
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Never apply weight decay to these
        if any(substr in name for substr in ['bias', 'ln', 'norm', 'embedding']):
            if name in decay_params:
                decay_params.remove(name)
            no_decay_params.add(name)
    
    # Create parameter groups
    param_dict = {name: param for name, param in model.named_parameters()}
    
    decay_group = [param_dict[name] for name in decay_params]
    no_decay_group = [param_dict[name] for name in no_decay_params]
    
    return [
        {'params': decay_group, 'weight_decay_applied': True},
        {'params': no_decay_group, 'weight_decay_applied': False}
    ]


def scale_learning_rate_with_batch_size(base_lr: float, batch_size: int, 
                                       base_batch_size: int = 32) -> float:
    """Scale learning rate with batch size (linear scaling rule)."""
    return base_lr * (batch_size / base_batch_size)


def get_layer_wise_lr_decay_groups(model: torch.nn.Module, 
                                  base_lr: float,
                                  decay_rate: float = 0.8) -> List[Dict[str, Any]]:
    """Create layer-wise learning rate decay groups."""
    param_groups = []
    
    # Get number of layers
    if hasattr(model, 'config'):
        n_layers = model.config.n_layers
    else:
        n_layers = len([name for name, _ in model.named_modules() 
                       if 'block' in name or 'layer' in name])
    
    # Group parameters by layer
    layer_params = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Extract layer number from parameter name
        layer_num = None
        if 'blocks.' in name:
            try:
                layer_num = int(name.split('blocks.')[1].split('.')[0])
            except (IndexError, ValueError):
                pass
        
        if layer_num is not None:
            if layer_num not in layer_params:
                layer_params[layer_num] = []
            layer_params[layer_num].append(param)
        else:
            # Non-layer parameters (embeddings, final layers)
            if 'other' not in layer_params:
                layer_params['other'] = []
            layer_params['other'].append(param)
    
    # Create groups with decayed learning rates
    for layer_id, params in layer_params.items():
        if layer_id == 'other':
            lr = base_lr  # Keep base learning rate for non-layer params
        else:
            # Apply decay: later layers get lower learning rates
            lr = base_lr * (decay_rate ** (n_layers - layer_id - 1))
        
        param_groups.append({
            'params': params,
            'lr': lr,
            'layer_id': layer_id
        })
    
    logger.info(f"Created {len(param_groups)} layer-wise learning rate groups")
    return param_groups


class GradientClipping:
    """Gradient clipping utilities."""
    
    @staticmethod
    def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0):
        """Clip gradient norm."""
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    
    @staticmethod
    def clip_grad_value(parameters, clip_value: float):
        """Clip gradient values."""
        return torch.nn.utils.clip_grad_value_(parameters, clip_value)
    
    @staticmethod
    def get_grad_norm(parameters, norm_type: float = 2.0):
        """Get gradient norm."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        parameters = [p for p in parameters if p.grad is not None]
        
        if len(parameters) == 0:
            return torch.tensor(0.0)
        
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type
        )
        
        return total_norm


def print_optimizer_info(optimizer: torch.optim.Optimizer):
    """Print optimizer information."""
    print(f"\nOptimizer: {optimizer.__class__.__name__}")
    print(f"Number of parameter groups: {len(optimizer.param_groups)}")
    
    total_params = 0
    for i, group in enumerate(optimizer.param_groups):
        group_params = sum(p.numel() for p in group['params'])
        total_params += group_params
        
        print(f"Group {i}:")
        print(f"  - Parameters: {group_params:,}")
        print(f"  - Learning rate: {group['lr']:.2e}")
        print(f"  - Weight decay: {group.get('weight_decay', 0.0):.2e}")
    
    print(f"Total parameters: {total_params:,}")
    print()


def save_optimizer_checkpoint(optimizer: torch.optim.Optimizer, 
                             scheduler: Optional[_LRScheduler],
                             filepath: str):
    """Save optimizer and scheduler state."""
    checkpoint = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, filepath)


def load_optimizer_checkpoint(optimizer: torch.optim.Optimizer,
                             scheduler: Optional[_LRScheduler],
                             filepath: str):
    """Load optimizer and scheduler state."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])