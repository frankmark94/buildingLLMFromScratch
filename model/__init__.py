#!/usr/bin/env python3
"""
Neural LLM model package.
Contains transformer architecture, attention mechanisms, and utilities.
"""

from .transformer import GPTModel, ModelConfig, create_model_from_config
from .layers import (
    LayerNorm, TransformerBlock, TokenEmbedding, PositionalEmbedding,
    SinusoidalPositionalEmbedding, init_weights, count_parameters, get_model_size_mb
)
from .attention import MultiHeadAttention, GroupedQueryAttention, SlidingWindowAttention, get_attention_class
from .utils import (
    load_model_config, create_model_from_config_file, get_model_variants,
    create_model_by_size, estimate_model_params, calculate_model_memory,
    find_optimal_batch_size, count_model_flops, get_device, setup_model_parallel,
    save_model_info, load_pretrained_weights, print_model_comparison
)

__all__ = [
    # Core model classes
    'GPTModel',
    'ModelConfig',
    'create_model_from_config',
    
    # Layer components
    'LayerNorm',
    'TransformerBlock', 
    'TokenEmbedding',
    'PositionalEmbedding',
    'SinusoidalPositionalEmbedding',
    
    # Attention mechanisms
    'MultiHeadAttention',
    'GroupedQueryAttention', 
    'SlidingWindowAttention',
    'get_attention_class',
    
    # Utilities
    'load_model_config',
    'create_model_from_config_file',
    'get_model_variants',
    'create_model_by_size',
    'estimate_model_params',
    'calculate_model_memory',
    'find_optimal_batch_size',
    'count_model_flops',
    'get_device',
    'setup_model_parallel',
    'save_model_info',
    'load_pretrained_weights',
    'print_model_comparison',
    
    # Helper functions
    'init_weights',
    'count_parameters',
    'get_model_size_mb',
]