#!/usr/bin/env python3
"""
Core neural network layers for transformer architecture.
Includes embeddings, layer normalization, feed-forward networks, and utilities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNorm(nn.Module):
    """Layer normalization with optional bias (similar to GPT-2)."""
    
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention mechanism."""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_heads
        
        # Key, query, value projections for all heads (batched)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size, config.block_size))
                  .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        
        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Multi-layer perceptron (feed-forward network)."""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (like GPT-2)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TokenEmbedding(nn.Module):
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.n_embd = n_embd
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding."""
    
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.embedding = nn.Embedding(block_size, n_embd)
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t = x.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)  # shape (1, t)
        return self.embedding(pos)


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embeddings (like the original Transformer paper)."""
    
    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Create sinusoidal position encoding
        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, block_size, n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t = x.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        return self.pe[:, :t]


def init_weights(module: nn.Module, initializer_range: float = 0.02):
    """Initialize weights following GPT-2 style."""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
    elif isinstance(module, LayerNorm):
        torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb