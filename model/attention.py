#!/usr/bin/env python3
"""
Advanced attention mechanisms for transformer models.
Includes FlashAttention integration and memory-efficient attention variants.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTENTION = True
except ImportError:
    HAS_FLASH_ATTENTION = False


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional FlashAttention support."""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout
        self.use_flash_attention = getattr(config, 'use_flash_attention', False) and HAS_FLASH_ATTENTION
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask for regular attention
        if not self.use_flash_attention:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                      .view(1, 1, config.block_size, config.block_size)
            )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        
        if self.use_flash_attention:
            # Use FlashAttention for memory efficiency
            attn_output = self._flash_attention(q, k, v)
        else:
            # Standard scaled dot-product attention
            attn_output = self._scaled_dot_product_attention(q, k, v, attention_mask)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.out_proj(attn_output))
        
        return output
    
    def _scaled_dot_product_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        _, _, T, _ = scores.size()
        causal_mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output
    
    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """FlashAttention implementation."""
        # FlashAttention expects (batch, seqlen, nheads, headdim)
        q = q.transpose(1, 2)  # (B, T, n_heads, head_dim)
        k = k.transpose(1, 2)  # (B, T, n_heads, head_dim)
        v = v.transpose(1, 2)  # (B, T, n_heads, head_dim)
        
        # Apply FlashAttention
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            softmax_scale=1.0 / math.sqrt(self.head_dim)
        )
        
        # Convert back to (B, n_heads, T, head_dim)
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) for more efficient inference."""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout
        
        # Number of key-value heads (fewer than query heads for efficiency)
        self.n_kv_heads = getattr(config, 'n_kv_heads', config.n_heads)
        assert config.n_heads % self.n_kv_heads == 0
        self.n_rep = config.n_heads // self.n_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                  .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat k and v to match number of query heads
        k = k.repeat_interleave(self.n_rep, dim=1)  # (B, n_heads, T, head_dim)
        v = v.repeat_interleave(self.n_rep, dim=1)  # (B, n_heads, T, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.out_proj(attn_output))
        
        return output


class SlidingWindowAttention(nn.Module):
    """Sliding window attention for handling longer sequences."""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout
        self.window_size = getattr(config, 'sliding_window_size', config.block_size // 2)
        
        # Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Create sliding window mask
        mask = self._create_sliding_window_mask(T, x.device)
        
        # Scaled dot-product attention with sliding window
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.out_proj(attn_output))
        
        return output
    
    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sliding window attention mask."""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            # Causal attention with sliding window
            start = max(0, i - self.window_size + 1)
            end = i + 1
            mask[i, start:end] = 1
        
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def get_attention_class(config):
    """Get the appropriate attention class based on config."""
    attention_type = getattr(config, 'attention_type', 'standard')
    
    if attention_type == 'standard':
        return MultiHeadAttention
    elif attention_type == 'grouped_query':
        return GroupedQueryAttention
    elif attention_type == 'sliding_window':
        return SlidingWindowAttention
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")