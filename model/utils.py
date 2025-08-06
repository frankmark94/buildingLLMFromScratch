#!/usr/bin/env python3
"""
Utility functions for model operations and configurations.
"""

import os
import yaml
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging

from .transformer import ModelConfig, GPTModel

logger = logging.getLogger(__name__)


def load_model_config(config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """Load model configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('model', {})


def create_model_from_config_file(config_path: str = "config/model_config.yaml") -> GPTModel:
    """Create model from configuration file."""
    config_dict = load_model_config(config_path)
    config = ModelConfig.from_dict(config_dict)
    return GPTModel(config)


def get_model_variants() -> Dict[str, Dict[str, Any]]:
    """Get predefined model size variants."""
    return {
        "nano": {
            "n_layers": 6,
            "n_heads": 6,
            "n_embd": 384,
            "vocab_size": 50304,
            "block_size": 1024,
            "dropout": 0.1,
            "bias": True
        },
        "micro": {
            "n_layers": 12,
            "n_heads": 12,
            "n_embd": 768,
            "vocab_size": 50304,
            "block_size": 1024,
            "dropout": 0.1,
            "bias": True
        },
        "small": {
            "n_layers": 12,
            "n_heads": 12,
            "n_embd": 768,
            "vocab_size": 50304,
            "block_size": 1024,
            "dropout": 0.1,
            "bias": True
        },
        "medium": {
            "n_layers": 24,
            "n_heads": 16,
            "n_embd": 1024,
            "vocab_size": 50304,
            "block_size": 1024,
            "dropout": 0.1,
            "bias": True
        },
        "large": {
            "n_layers": 24,
            "n_heads": 16,
            "n_embd": 1536,
            "vocab_size": 50304,
            "block_size": 1024,
            "dropout": 0.1,
            "bias": True
        },
        "xl": {
            "n_layers": 48,
            "n_heads": 25,
            "n_embd": 1600,
            "vocab_size": 50304,
            "block_size": 1024,
            "dropout": 0.1,
            "bias": True
        }
    }


def create_model_by_size(size: str = "small") -> GPTModel:
    """Create model by predefined size variant."""
    variants = get_model_variants()
    if size not in variants:
        raise ValueError(f"Unknown model size: {size}. Available: {list(variants.keys())}")
    
    config = ModelConfig.from_dict(variants[size])
    return GPTModel(config)


def estimate_model_params(
    n_layers: int,
    n_heads: int,
    n_embd: int,
    vocab_size: int,
    block_size: int
) -> int:
    """Estimate number of parameters for given architecture."""
    # Token embeddings
    token_emb_params = vocab_size * n_embd
    
    # Position embeddings
    pos_emb_params = block_size * n_embd
    
    # Transformer blocks
    # Each block has:
    # - Multi-head attention: 4 * n_embd^2 (q, k, v, out projections)
    # - MLP: 2 * n_embd * (4 * n_embd) = 8 * n_embd^2 (up and down projections)
    # - Layer norms: 2 * n_embd (small, negligible)
    block_params = n_layers * (4 * n_embd * n_embd + 8 * n_embd * n_embd + 4 * n_embd)
    
    # Final layer norm
    ln_f_params = n_embd
    
    # Language modeling head (tied with token embeddings, so don't double count)
    lm_head_params = 0  # vocab_size * n_embd (tied)
    
    total_params = token_emb_params + pos_emb_params + block_params + ln_f_params + lm_head_params
    
    return total_params


def calculate_model_memory(
    batch_size: int,
    seq_length: int,
    n_layers: int,
    n_heads: int,
    n_embd: int,
    vocab_size: int,
    dtype_size: int = 2  # 2 for fp16, 4 for fp32
) -> Dict[str, float]:
    """Calculate approximate model memory usage in MB."""
    
    # Parameters
    param_count = estimate_model_params(n_layers, n_heads, n_embd, vocab_size, seq_length)
    param_memory = param_count * dtype_size / (1024**2)
    
    # Activations (rough estimate)
    # Forward pass activations
    activation_memory = batch_size * seq_length * n_embd * n_layers * dtype_size / (1024**2)
    
    # Attention scores
    attention_memory = batch_size * n_heads * seq_length * seq_length * n_layers * dtype_size / (1024**2)
    
    # Gradients (same size as parameters)
    gradient_memory = param_memory
    
    # Optimizer states (AdamW has 2 states per parameter)
    optimizer_memory = param_memory * 2
    
    return {
        "parameters_mb": param_memory,
        "activations_mb": activation_memory,
        "attention_mb": attention_memory,
        "gradients_mb": gradient_memory,
        "optimizer_mb": optimizer_memory,
        "total_mb": param_memory + activation_memory + attention_memory + gradient_memory + optimizer_memory
    }


def find_optimal_batch_size(
    model: GPTModel,
    seq_length: int,
    available_memory_gb: float = 16,  # Typical GPU memory
    dtype_size: int = 2  # fp16
) -> int:
    """Find optimal batch size that fits in available GPU memory."""
    
    memory_info = calculate_model_memory(
        batch_size=1,
        seq_length=seq_length,
        n_layers=model.config.n_layers,
        n_heads=model.config.n_heads,
        n_embd=model.config.n_embd,
        vocab_size=model.config.vocab_size,
        dtype_size=dtype_size
    )
    
    # Memory per sample (activations scale with batch size)
    memory_per_sample_mb = memory_info["activations_mb"] + memory_info["attention_mb"]
    
    # Fixed memory (parameters, gradients, optimizer)
    fixed_memory_mb = (memory_info["parameters_mb"] + 
                      memory_info["gradients_mb"] + 
                      memory_info["optimizer_mb"])
    
    available_memory_mb = available_memory_gb * 1024
    
    # Leave some headroom (80% of available memory)
    usable_memory_mb = available_memory_mb * 0.8
    
    # Calculate maximum batch size
    memory_for_batch = usable_memory_mb - fixed_memory_mb
    max_batch_size = int(memory_for_batch / memory_per_sample_mb)
    
    return max(1, max_batch_size)


def count_model_flops(model: GPTModel, seq_length: int) -> int:
    """Count approximate FLOPs for forward pass."""
    cfg = model.config
    
    # Embedding lookups (not counted as FLOPs)
    
    # Transformer blocks
    flops = 0
    for _ in range(cfg.n_layers):
        # Multi-head attention
        # Q, K, V projections: 3 * seq_length * n_embd * n_embd
        flops += 3 * seq_length * cfg.n_embd * cfg.n_embd
        
        # Attention computation: seq_length * seq_length * n_embd
        flops += seq_length * seq_length * cfg.n_embd
        
        # Attention output projection: seq_length * n_embd * n_embd
        flops += seq_length * cfg.n_embd * cfg.n_embd
        
        # MLP
        # Up projection: seq_length * n_embd * (4 * n_embd)
        flops += seq_length * cfg.n_embd * (4 * cfg.n_embd)
        # Down projection: seq_length * (4 * n_embd) * n_embd
        flops += seq_length * (4 * cfg.n_embd) * cfg.n_embd
    
    # Language modeling head: seq_length * n_embd * vocab_size
    flops += seq_length * cfg.n_embd * cfg.vocab_size
    
    return flops


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_model_parallel(model: GPTModel, device_ids: Optional[List[int]] = None) -> nn.Module:
    """Setup model for data parallel training."""
    device = get_device()
    model = model.to(device)
    
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        logger.info(f"Using DataParallel with devices: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
    
    return model


def save_model_info(model: GPTModel, filepath: str) -> None:
    """Save detailed model information to file."""
    info = {
        "config": model.config.to_dict(),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "non_embedding_parameters": model.get_num_params(),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    }
    
    # Add memory estimates for different batch sizes
    info["memory_estimates"] = {}
    for batch_size in [1, 4, 8, 16, 32]:
        memory_info = calculate_model_memory(
            batch_size=batch_size,
            seq_length=model.config.block_size,
            n_layers=model.config.n_layers,
            n_heads=model.config.n_heads,
            n_embd=model.config.n_embd,
            vocab_size=model.config.vocab_size
        )
        info["memory_estimates"][f"batch_size_{batch_size}"] = memory_info
    
    with open(filepath, 'w') as f:
        yaml.dump(info, f, default_flow_style=False)
    
    logger.info(f"Model info saved to {filepath}")


def load_pretrained_weights(model: GPTModel, pretrained_path: str) -> GPTModel:
    """Load pretrained weights into model (with size matching)."""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
    
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    
    if 'model_state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_state_dict']
    
    model_dict = model.state_dict()
    
    # Filter out unnecessary keys and size mismatches
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            logger.warning(f"Skipping {k}: shape mismatch or not found in current model")
    
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    logger.info(f"Loaded {len(filtered_dict)} layers from pretrained weights")
    return model


def print_model_comparison(models: Dict[str, GPTModel]) -> None:
    """Print comparison table of multiple models."""
    print("\nModel Comparison:")
    print("-" * 80)
    print(f"{'Model':<15} {'Layers':<8} {'Heads':<8} {'Embed':<8} {'Params':<12} {'Size (MB)':<12}")
    print("-" * 80)
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        print(f"{name:<15} {model.config.n_layers:<8} {model.config.n_heads:<8} "
              f"{model.config.n_embd:<8} {total_params:<12,} {model_size_mb:<12.2f}")
    
    print("-" * 80)