#!/usr/bin/env python3
"""
GPT-style transformer model implementation.
Complete autoregressive language model with configurable architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from .layers import (
    LayerNorm, TransformerBlock, TokenEmbedding, PositionalEmbedding,
    init_weights, count_parameters, get_model_size_mb
)
from .attention import get_attention_class


@dataclass
class ModelConfig:
    """Configuration class for the transformer model."""
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    vocab_size: int = 50304
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True
    
    # Advanced options
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    attention_type: str = 'standard'  # 'standard', 'grouped_query', 'sliding_window'
    n_kv_heads: Optional[int] = None  # For grouped query attention
    sliding_window_size: Optional[int] = None  # For sliding window attention
    
    # Initialization
    initializer_range: float = 0.02
    
    # Positional encoding type
    pos_encoding_type: str = 'learned'  # 'learned' or 'sinusoidal'
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()


class GPTModel(nn.Module):
    """GPT-style autoregressive transformer model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = TokenEmbedding(config.vocab_size, config.n_embd)
        
        if config.pos_encoding_type == 'learned':
            self.position_embedding = PositionalEmbedding(config.block_size, config.n_embd)
        else:
            from .layers import SinusoidalPositionalEmbedding
            self.position_embedding = SinusoidalPositionalEmbedding(config.block_size, config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight sharing between token embeddings and lm_head (optional)
        if hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.embedding.weight
        
        # Initialize weights
        self.apply(lambda module: init_weights(module, config.initializer_range))
        
        # Apply special scaled initialization to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.initializer_range / math.sqrt(2 * config.n_layers))
        
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token indices of shape (batch_size, sequence_length)
            labels: Labels for language modeling loss (batch_size, sequence_length)
            attention_mask: Mask to avoid performing attention on padding tokens
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        
        # Token embeddings
        tok_emb = self.token_embedding(input_ids)  # (b, t, n_embd)
        
        # Position embeddings
        pos_emb = self.position_embedding(input_ids)  # (b, t, n_embd)
        
        # Combine embeddings
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (b, t, vocab_size)
        
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Initial token sequence (batch_size, sequence_length)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to use sampling vs greedy decoding
            
        Returns:
            Generated token sequence (batch_size, sequence_length + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop input_ids if it exceeds block_size
                input_ids_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
                
                # Forward pass
                outputs = self.forward(input_ids_cond)
                logits = outputs['logits']
                
                # Get logits for the last position
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample from the distribution
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to the sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), we subtract the position/token embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.embedding.weight.numel()
            n_params -= self.token_embedding.embedding.weight.numel()
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        # First estimate the number of flops we do per iteration
        # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.n_embd//cfg.n_heads, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    @torch.no_grad()
    def generate_from_prompt(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """Generate text from a string prompt."""
        # Tokenize prompt
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(prompt)
        else:
            tokens = tokenizer.tokenize_text(prompt)
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)
        
        # Generate
        generated_ids = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Decode generated text
        generated_tokens = generated_ids[0].tolist()
        
        if hasattr(tokenizer, 'decode'):
            return tokenizer.decode(generated_tokens)
        else:
            return tokenizer.decode_tokens(generated_tokens)
    
    def save_checkpoint(self, filepath: str, optimizer_state: Optional[Dict] = None, 
                       step: int = 0, loss: float = 0.0) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'step': step,
            'loss': loss,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cpu') -> Tuple['GPTModel', Dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        
        config = ModelConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint
    
    def print_model_info(self) -> None:
        """Print model information."""
        total_params, trainable_params = count_parameters(self)
        model_size_mb = get_model_size_mb(self)
        
        print(f"Model Configuration:")
        print(f"  - Layers: {self.config.n_layers}")
        print(f"  - Heads: {self.config.n_heads}")
        print(f"  - Embedding dim: {self.config.n_embd}")
        print(f"  - Vocab size: {self.config.vocab_size}")
        print(f"  - Block size: {self.config.block_size}")
        print(f"  - Dropout: {self.config.dropout}")
        print(f"Model Statistics:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size: {model_size_mb:.2f} MB")
        print(f"  - Non-embedding parameters: {self.get_num_params():,}")


def create_model_from_config(config_dict: Dict[str, Any]) -> GPTModel:
    """Create model from configuration dictionary."""
    config = ModelConfig.from_dict(config_dict)
    return GPTModel(config)