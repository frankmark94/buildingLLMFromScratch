#!/usr/bin/env python3
"""
Compare different model configurations and show parameter counts and memory usage.
"""

import yaml
import math
from pathlib import Path
from typing import Dict, Any


def estimate_parameters(config: Dict[str, Any]) -> int:
    """Estimate parameter count for a model configuration."""
    model_config = config.get('model', {})
    
    n_layers = model_config.get('n_layers', 6)
    n_embd = model_config.get('n_embd', 512)
    vocab_size = model_config.get('vocab_size', 32000)
    block_size = model_config.get('block_size', 256)
    
    # Token embeddings: vocab_size * n_embd
    token_emb_params = vocab_size * n_embd
    
    # Position embeddings: block_size * n_embd
    pos_emb_params = block_size * n_embd
    
    # Transformer blocks (each layer):
    # - Multi-head attention: 4 * n_embd^2 (q, k, v, out projections)
    # - MLP: 8 * n_embd^2 (up 4x expansion, then down)
    # - Layer norms: ~2 * n_embd (negligible)
    block_params = n_layers * (4 * n_embd * n_embd + 8 * n_embd * n_embd + 2 * n_embd)
    
    # Final layer norm
    ln_f_params = n_embd
    
    # Language modeling head (usually tied with token embeddings)
    # So we don't double count
    
    total_params = token_emb_params + pos_emb_params + block_params + ln_f_params
    
    return total_params


def estimate_memory(config: Dict[str, Any]) -> Dict[str, float]:
    """Estimate memory usage for training."""
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    param_count = estimate_parameters(config)
    batch_size = training_config.get('batch_size', 1)
    block_size = model_config.get('block_size', 256)
    n_embd = model_config.get('n_embd', 512)
    n_layers = model_config.get('n_layers', 6)
    use_amp = training_config.get('use_amp', False)
    
    # Data type size (FP16 vs FP32)
    dtype_size = 2 if use_amp else 4
    
    # Parameter memory
    param_memory_mb = param_count * dtype_size / (1024**2)
    
    # Gradient memory (same as parameters)
    grad_memory_mb = param_memory_mb
    
    # Optimizer states (AdamW: 2 states per parameter)
    optimizer_memory_mb = param_memory_mb * 2
    
    # Activation memory (rough estimate)
    activation_memory_mb = batch_size * block_size * n_embd * n_layers * dtype_size / (1024**2)
    
    # Total
    total_memory_mb = param_memory_mb + grad_memory_mb + optimizer_memory_mb + activation_memory_mb
    
    return {
        'parameters_mb': param_memory_mb,
        'gradients_mb': grad_memory_mb,
        'optimizer_mb': optimizer_memory_mb,
        'activations_mb': activation_memory_mb,
        'total_mb': total_memory_mb,
        'total_gb': total_memory_mb / 1024,
        'inference_mb': param_memory_mb + activation_memory_mb * 0.3  # Rough inference estimate
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def compare_configurations():
    """Compare all available configurations."""
    
    configs_to_compare = [
        ("Tiny (Auto-detected)", "config/model_config_optimized.yaml", "config/training_config_optimized.yaml"),
        ("Surface Pro (32GB)", "config/surface_pro_config.yaml", None),
        ("Consumer Laptop", "config/hardware_configs.yaml", "consumer_laptop"),
        ("High-end", "config/hardware_configs.yaml", "high_end")
    ]
    
    print("\n" + "="*80)
    print("MODEL CONFIGURATION COMPARISON")
    print("="*80)
    
    print(f"{'Config Name':<20} {'Params':<10} {'Layers':<8} {'Embed':<8} {'Context':<9} {'Memory (GB)':<12} {'Train Time':<15}")
    print("-" * 80)
    
    for name, config_path, subset in configs_to_compare:
        if subset and subset != "consumer_laptop" and subset != "high_end":
            continue
            
        config = {}
        
        if config_path == "config/hardware_configs.yaml" and subset:
            # Load from hardware configs
            try:
                with open(config_path, 'r') as f:
                    hardware_configs = yaml.safe_load(f)
                    config = hardware_configs.get(subset, {})
            except FileNotFoundError:
                continue
        elif subset is None:
            # Load single config file
            config = load_config(config_path)
        else:
            # Load and merge training config
            model_config = load_config(config_path)
            if subset:
                training_config = load_config(subset)
                config = {**model_config, **training_config}
            else:
                config = model_config
        
        if not config:
            continue
        
        # Calculate metrics
        param_count = estimate_parameters(config)
        memory = estimate_memory(config)
        
        model_cfg = config.get('model', {})
        n_layers = model_cfg.get('n_layers', 0)
        n_embd = model_cfg.get('n_embd', 0)
        block_size = model_cfg.get('block_size', 0)
        
        # Rough training time estimates (very approximate)
        if param_count < 10_000_000:  # < 10M params
            train_time = "1-2 hours"
        elif param_count < 50_000_000:  # < 50M params
            train_time = "4-8 hours" 
        elif param_count < 200_000_000:  # < 200M params
            train_time = "12-24 hours"
        else:
            train_time = "1-3 days"
        
        print(f"{name:<20} {param_count/1e6:.1f}M{'':<5} {n_layers:<8} {n_embd:<8} {block_size:<9} {memory['total_gb']:.1f}{'':<10} {train_time:<15}")
    
    print("-" * 80)
    
    # Show detailed breakdown for Surface Pro config
    surface_config = load_config("config/surface_pro_config.yaml")
    if surface_config:
        print(f"\n📋 SURFACE PRO CONFIGURATION DETAILS:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        params = estimate_parameters(surface_config)
        memory = estimate_memory(surface_config)
        model_cfg = surface_config.get('model', {})
        training_cfg = surface_config.get('training', {})
        
        print(f"🎯 Model Architecture:")
        print(f"   • Parameters: {params/1e6:.1f}M ({params:,})")
        print(f"   • Layers: {model_cfg.get('n_layers')}")
        print(f"   • Attention Heads: {model_cfg.get('n_heads')}")
        print(f"   • Hidden Size: {model_cfg.get('n_embd')}")
        print(f"   • Context Length: {model_cfg.get('block_size')} tokens")
        print(f"   • Vocabulary: {model_cfg.get('vocab_size'):,} tokens")
        
        print(f"\n💾 Memory Usage:")
        print(f"   • Training: {memory['total_gb']:.1f} GB")
        print(f"   • Inference: {memory['inference_mb']/1024:.1f} GB")
        print(f"   • Parameters: {memory['parameters_mb']:.0f} MB")
        
        print(f"\n⚙️ Training Settings:")
        print(f"   • Batch Size: {training_cfg.get('batch_size')}")
        print(f"   • Effective Batch Size: {training_cfg.get('gradient_accumulation_steps')}")
        print(f"   • Learning Rate: {training_cfg.get('learning_rate')}")
        print(f"   • Max Steps: {training_cfg.get('max_steps'):,}")
        print(f"   • Mixed Precision: {'✅' if training_cfg.get('use_amp') else '❌'}")
        print(f"   • Gradient Checkpointing: {'✅' if model_cfg.get('gradient_checkpointing') else '❌'}")
        
        print(f"\n⏱️ Expected Timeline:")
        print(f"   • Training Time: 6-12 hours (CPU)")
        print(f"   • With GPU: 1-2 hours")
        print(f"   • Dataset Processing: 30-60 minutes")
        
        print(f"\n🎯 What This Model Can Do:")
        print(f"   ✅ Complete sentences and paragraphs")
        print(f"   ✅ Basic reasoning and facts")
        print(f"   ✅ Text generation in training domain") 
        print(f"   ✅ Fine-tune on custom datasets")
        print(f"   ✅ Perfect for learning ML concepts")
        
    print("="*80)


if __name__ == "__main__":
    compare_configurations()