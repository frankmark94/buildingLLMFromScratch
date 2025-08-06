#!/usr/bin/env python3
"""
Mixed precision training utilities for neural LLM training.
Supports both native PyTorch AMP and manual mixed precision.
"""

import torch
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MixedPrecisionTrainer:
    """Mixed precision training manager."""
    
    def __init__(self, enabled: bool = True, init_scale: float = 2**16):
        self.enabled = enabled
        self.scaler = GradScaler(init_scale=init_scale, enabled=enabled) if enabled else None
        
        if enabled:
            logger.info("Mixed precision training enabled with native PyTorch AMP")
        else:
            logger.info("Mixed precision training disabled")
    
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.enabled:
            return autocast()
        else:
            return torch.no_grad() if not torch.is_grad_enabled() else self._dummy_context()
    
    def _dummy_context(self):
        """Dummy context manager that does nothing."""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return DummyContext()
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient scaling."""
        if self.enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer):
        """Optimizer step with gradient scaling."""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update(self):
        """Update gradient scaler."""
        if self.enabled and self.scaler is not None:
            self.scaler.update()
    
    def clip_gradients(self, parameters, max_norm: float):
        """Clip gradients with proper scaling."""
        if self.enabled and self.scaler is not None:
            # Unscale gradients before clipping
            self.scaler.unscale_(parameters[0].grad.data.device)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    
    def get_scale(self) -> float:
        """Get current gradient scale."""
        if self.enabled and self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def get_scaler_state(self) -> Optional[Dict[str, Any]]:
        """Get scaler state for checkpointing."""
        if self.enabled and self.scaler is not None:
            return self.scaler.state_dict()
        return None
    
    def load_scaler_state(self, state_dict: Dict[str, Any]):
        """Load scaler state from checkpoint."""
        if self.enabled and self.scaler is not None:
            self.scaler.load_state_dict(state_dict)
    
    def is_enabled(self) -> bool:
        """Check if mixed precision is enabled."""
        return self.enabled


class ManualMixedPrecision:
    """Manual mixed precision implementation for custom control."""
    
    def __init__(self, loss_scale: float = 2**16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.loss_scale = loss_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._inf_counter = 0
        
        logger.info(f"Manual mixed precision initialized with loss_scale={loss_scale}")
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        return loss * self.loss_scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> bool:
        """Unscale gradients and check for infs/nans."""
        found_inf = False
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.loss_scale)
                    
                    # Check for inf/nan
                    if torch.isinf(param.grad.data).any() or torch.isnan(param.grad.data).any():
                        found_inf = True
                        break
            if found_inf:
                break
        
        return found_inf
    
    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Optimizer step with gradient scaling handling."""
        found_inf = self.unscale_gradients(optimizer)
        
        if found_inf:
            # Skip optimizer step and reduce scale
            self.loss_scale *= self.backoff_factor
            self._growth_tracker = 0
            self._inf_counter += 1
            logger.warning(f"Gradient overflow detected, scaling down to {self.loss_scale}")
            return False
        else:
            # Normal optimizer step
            optimizer.step()
            
            # Potentially grow the scale
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.loss_scale *= self.growth_factor
                self._growth_tracker = 0
                logger.debug(f"Scaling up loss scale to {self.loss_scale}")
            
            return True
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        return self.loss_scale


class AutocastContext:
    """Custom autocast context for specific operations."""
    
    def __init__(self, enabled: bool = True, dtype: torch.dtype = torch.float16):
        self.enabled = enabled
        self.dtype = dtype
        self.prev_autocast_state = None
    
    def __enter__(self):
        if self.enabled and torch.cuda.is_available():
            self.prev_autocast_state = torch.is_autocast_enabled()
            torch.set_autocast_enabled(True)
            torch.set_autocast_gpu_dtype(self.dtype)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and torch.cuda.is_available():
            torch.set_autocast_enabled(self.prev_autocast_state)


class GradientAccumulator:
    """Gradient accumulation with mixed precision support."""
    
    def __init__(self, accumulation_steps: int, mixed_precision: bool = True):
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.step_count = 0
        
        if mixed_precision:
            self.mp_trainer = MixedPrecisionTrainer(enabled=True)
        else:
            self.mp_trainer = None
    
    def accumulate_gradients(self, loss: torch.Tensor, model: torch.nn.Module):
        """Accumulate gradients for the current step."""
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        
        # Backward pass with mixed precision
        if self.mp_trainer:
            self.mp_trainer.backward(loss)
        else:
            loss.backward()
        
        self.step_count += 1
    
    def should_step(self) -> bool:
        """Check if we should take an optimizer step."""
        return self.step_count % self.accumulation_steps == 0
    
    def step(self, optimizer: torch.optim.Optimizer):
        """Take optimizer step after gradient accumulation."""
        if self.should_step():
            if self.mp_trainer:
                self.mp_trainer.step(optimizer)
                self.mp_trainer.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
    
    def reset(self):
        """Reset the accumulation counter."""
        self.step_count = 0


class MemoryEfficientAttention:
    """Memory efficient attention computation."""
    
    @staticmethod
    def compute_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         mask: Optional[torch.Tensor] = None, 
                         chunk_size: int = 1024) -> torch.Tensor:
        """Compute attention in chunks to save memory."""
        B, H, T, D = query.shape
        
        if T <= chunk_size:
            # Use regular attention for small sequences
            scores = torch.matmul(query, key.transpose(-2, -1)) / (D ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, value)
            return output
        
        # Chunked attention computation
        output = torch.zeros_like(value)
        
        for i in range(0, T, chunk_size):
            end_i = min(i + chunk_size, T)
            q_chunk = query[:, :, i:end_i, :]
            
            # Compute attention scores for this query chunk
            scores = torch.matmul(q_chunk, key.transpose(-2, -1)) / (D ** 0.5)
            
            if mask is not None:
                scores = scores.masked_fill(mask[:, :, i:end_i, :] == 0, float('-inf'))
            
            attn_weights = torch.softmax(scores, dim=-1)
            output[:, :, i:end_i, :] = torch.matmul(attn_weights, value)
        
        return output


def check_mixed_precision_support() -> Dict[str, bool]:
    """Check mixed precision support on current device."""
    support_info = {
        'cuda_available': torch.cuda.is_available(),
        'amp_available': hasattr(torch.cuda.amp, 'autocast'),
        'tensor_cores': False,
        'bf16_support': False,
        'fp16_support': False
    }
    
    if torch.cuda.is_available():
        # Check for Tensor Core support (compute capability >= 7.0)
        capability = torch.cuda.get_device_capability()
        support_info['tensor_cores'] = capability[0] >= 7
        
        # Check dtype support
        try:
            test_tensor = torch.randn(2, 2, device='cuda', dtype=torch.float16)
            support_info['fp16_support'] = True
        except:
            pass
        
        try:
            test_tensor = torch.randn(2, 2, device='cuda', dtype=torch.bfloat16)
            support_info['bf16_support'] = True
        except:
            pass
    
    return support_info


def print_mixed_precision_info():
    """Print mixed precision support information."""
    support = check_mixed_precision_support()
    
    print("Mixed Precision Support:")
    print(f"  CUDA Available: {support['cuda_available']}")
    print(f"  AMP Available: {support['amp_available']}")
    print(f"  Tensor Cores: {support['tensor_cores']}")
    print(f"  FP16 Support: {support['fp16_support']}")
    print(f"  BF16 Support: {support['bf16_support']}")
    
    if support['cuda_available']:
        device_name = torch.cuda.get_device_name()
        print(f"  Device: {device_name}")
        
        if support['tensor_cores']:
            print("  Recommendation: Use FP16 mixed precision for optimal performance")
        else:
            print("  Recommendation: Mixed precision may provide limited benefits")


class DynamicLossScaling:
    """Dynamic loss scaling for mixed precision training."""
    
    def __init__(self, init_scale: float = 2**16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000,
                 min_scale: float = 1.0, max_scale: float = 2**24):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.growth_tracker = 0
        self.overflow_count = 0
    
    def update_scale(self, found_inf: bool):
        """Update loss scale based on gradient overflow status."""
        if found_inf:
            # Overflow detected - reduce scale
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self.growth_tracker = 0
            self.overflow_count += 1
        else:
            # No overflow - potentially increase scale
            self.growth_tracker += 1
            if self.growth_tracker >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self.growth_tracker = 0
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        return self.scale
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        return {
            'current_scale': self.scale,
            'overflow_count': self.overflow_count,
            'growth_tracker': self.growth_tracker
        }