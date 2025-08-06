#!/usr/bin/env python3
"""
Advanced text generation and inference pipeline with multiple sampling strategies.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
import re
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextGenerator:
    """Advanced text generator with multiple sampling strategies and optimizations."""
    
    def __init__(self, model: torch.nn.Module, tokenizer, device: torch.device, 
                 config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or self._default_config()
        
        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(device)
        
        # Cache for KV-cache optimization
        self.use_kv_cache = self.config.get('use_kv_cache', False)
        self.past_key_values = None
        
        logger.info(f"TextGenerator initialized on {device}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default generation configuration."""
        return {
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'max_new_tokens': 256,
            'min_length': 10,
            'do_sample': True,
            'early_stopping': True,
            'pad_token_id': 0,
            'eos_token_id': 1,
            'use_kv_cache': False
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt with configurable parameters."""
        # Merge config with kwargs
        generation_config = {**self.config, **kwargs}
        
        # Tokenize prompt
        input_tokens = self._tokenize_prompt(prompt)
        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generate tokens
        with torch.no_grad():
            generated_ids = self._generate_tokens(input_ids, generation_config)
        
        # Decode generated text
        generated_text = self._decode_tokens(generated_ids[0].cpu().tolist())
        
        # Extract only the new tokens (remove prompt)
        if generation_config.get('return_prompt', False):
            return generated_text
        else:
            # Remove the prompt from the beginning
            prompt_text = self._decode_tokens(input_tokens)
            if generated_text.startswith(prompt_text):
                return generated_text[len(prompt_text):].lstrip()
            return generated_text
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts simultaneously."""
        generation_config = {**self.config, **kwargs}
        batch_size = generation_config.get('batch_size', len(prompts))
        
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch
            batch_tokens = []
            max_length = 0
            
            for prompt in batch_prompts:
                tokens = self._tokenize_prompt(prompt)
                batch_tokens.append(tokens)
                max_length = max(max_length, len(tokens))
            
            # Pad sequences to same length
            padded_batch = []
            attention_mask = []
            
            for tokens in batch_tokens:
                padded_tokens = tokens + [generation_config['pad_token_id']] * (max_length - len(tokens))
                padded_batch.append(padded_tokens)
                
                mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
                attention_mask.append(mask)
            
            input_ids = torch.tensor(padded_batch, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool).to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self._generate_tokens(input_ids, generation_config, attention_mask)
            
            # Decode results
            for j, prompt in enumerate(batch_prompts):
                generated_text = self._decode_tokens(generated_ids[j].cpu().tolist())
                
                if not generation_config.get('return_prompt', False):
                    # Remove prompt
                    prompt_text = self._decode_tokens(batch_tokens[j])
                    if generated_text.startswith(prompt_text):
                        generated_text = generated_text[len(prompt_text):].lstrip()
                
                results.append(generated_text)
        
        return results
    
    def _generate_tokens(self, input_ids: torch.Tensor, config: Dict[str, Any],
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Core token generation with advanced sampling."""
        batch_size, seq_len = input_ids.shape
        max_new_tokens = config['max_new_tokens']
        max_length = seq_len + max_new_tokens
        
        # Initialize generation state
        generated = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Generation loop
        for step in range(max_new_tokens):
            # Prepare input for this step
            if past_key_values is not None and config.get('use_kv_cache', False):
                # Only use the last token for subsequent steps with KV cache
                current_input = generated[:, -1:]
            else:
                current_input = generated
            
            # Forward pass
            outputs = self.model(current_input)
            logits = outputs['logits']
            
            # Get logits for the last position
            next_token_logits = logits[:, -1, :]
            
            # Apply generation constraints and sampling
            next_tokens = self._sample_next_tokens(next_token_logits, generated, config)
            
            # Update generated sequence
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for early stopping
            if config.get('early_stopping', True):
                eos_mask = (next_tokens == config['eos_token_id'])
                finished = finished | eos_mask
                
                if finished.all():
                    break
        
        return generated
    
    def _sample_next_tokens(self, logits: torch.Tensor, generated: torch.Tensor,
                           config: Dict[str, Any]) -> torch.Tensor:
        """Advanced sampling with multiple strategies."""
        batch_size, vocab_size = logits.shape
        
        # Apply temperature
        temperature = config.get('temperature', 1.0)
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply repetition penalty
        repetition_penalty = config.get('repetition_penalty', 1.0)
        if repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
        
        # Apply top-k filtering
        top_k = config.get('top_k', None)
        if top_k is not None and top_k > 0:
            logits = self._top_k_filtering(logits, top_k)
        
        # Apply top-p (nucleus) filtering
        top_p = config.get('top_p', None)
        if top_p is not None and top_p < 1.0:
            logits = self._top_p_filtering(logits, top_p)
        
        # Sample or take argmax
        if config.get('do_sample', True):
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(logits, dim=-1)
        
        return next_tokens
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, generated: torch.Tensor,
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        batch_size, seq_len = generated.shape
        
        for batch_idx in range(batch_size):
            for token_id in set(generated[batch_idx].tolist()):
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= penalty
                else:
                    logits[batch_idx, token_id] /= penalty
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k = min(top_k, logits.size(-1))
        
        # Get the top-k values and their indices
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        
        # Create a mask for values below the top-k threshold
        min_top_k_values = top_k_values[:, -1:].expand_as(logits)
        logits = torch.where(logits < min_top_k_values, 
                           torch.full_like(logits, float('-inf')), 
                           logits)
        
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        # Scatter sorted tensors back to original indexing
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    def _tokenize_prompt(self, prompt: str) -> List[int]:
        """Tokenize a text prompt."""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(prompt)
        elif hasattr(self.tokenizer, 'tokenize_text'):
            return self.tokenizer.tokenize_text(prompt)
        else:
            raise ValueError("Tokenizer must have 'encode' or 'tokenize_text' method")
    
    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to text."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(tokens)
        elif hasattr(self.tokenizer, 'decode_tokens'):
            return self.tokenizer.decode_tokens(tokens)
        else:
            raise ValueError("Tokenizer must have 'decode' or 'decode_tokens' method")
    
    def generate_with_constraints(self, prompt: str, 
                                 banned_words: Optional[List[str]] = None,
                                 required_words: Optional[List[str]] = None,
                                 **kwargs) -> str:
        """Generate text with lexical constraints."""
        generation_config = {**self.config, **kwargs}
        
        # Convert words to token IDs
        banned_token_ids = set()
        required_token_ids = set()
        
        if banned_words:
            for word in banned_words:
                word_tokens = self._tokenize_prompt(word)
                banned_token_ids.update(word_tokens)
        
        if required_words:
            for word in required_words:
                word_tokens = self._tokenize_prompt(word)
                required_token_ids.update(word_tokens)
        
        # Custom sampling function with constraints
        def constrained_sampling(logits, generated, config):
            # Apply banned words constraint
            if banned_token_ids:
                for token_id in banned_token_ids:
                    logits[:, token_id] = float('-inf')
            
            # Boost required words (if not yet used)
            if required_token_ids:
                generated_tokens = set(generated.view(-1).tolist())
                remaining_required = required_token_ids - generated_tokens
                
                if remaining_required:
                    boost_factor = 2.0
                    for token_id in remaining_required:
                        logits[:, token_id] *= boost_factor
            
            return self._sample_next_tokens(logits, generated, config)
        
        # Temporarily replace sampling function
        original_sample_fn = self._sample_next_tokens
        self._sample_next_tokens = constrained_sampling
        
        try:
            result = self.generate(prompt, **kwargs)
        finally:
            # Restore original sampling function
            self._sample_next_tokens = original_sample_fn
        
        return result
    
    def generate_with_stopping_criteria(self, prompt: str, 
                                       stop_strings: Optional[List[str]] = None,
                                       max_sentences: Optional[int] = None,
                                       **kwargs) -> str:
        """Generate text with custom stopping criteria."""
        generation_config = {**self.config, **kwargs}
        stop_strings = stop_strings or []
        
        # Generate with a larger max_tokens first
        original_max_tokens = generation_config.get('max_new_tokens', 256)
        generation_config['max_new_tokens'] = min(original_max_tokens * 2, 1000)
        
        generated_text = self.generate(prompt, **generation_config)
        
        # Apply stopping criteria
        final_text = generated_text
        
        # Stop at specific strings
        for stop_string in stop_strings:
            if stop_string in final_text:
                stop_index = final_text.find(stop_string)
                final_text = final_text[:stop_index]
        
        # Stop at sentence boundary
        if max_sentences:
            sentences = re.split(r'[.!?]+', final_text)
            if len(sentences) > max_sentences:
                final_text = '.'.join(sentences[:max_sentences]) + '.'
        
        return final_text
    
    def interactive_generate(self, initial_prompt: str = "", **kwargs) -> None:
        """Interactive text generation session."""
        print("🤖 Interactive Text Generation")
        print("=" * 50)
        print("Commands:")
        print("  /quit - Exit")
        print("  /config - Show current settings")
        print("  /set <param> <value> - Change parameter")
        print("  /reset - Reset to default settings")
        print("=" * 50)
        
        current_config = {**self.config, **kwargs}
        conversation_history = initial_prompt
        
        while True:
            try:
                user_input = input("\n💬 Enter prompt (or command): ").strip()
                
                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == '/config':
                    print("\n⚙️ Current Configuration:")
                    for key, value in current_config.items():
                        print(f"  {key}: {value}")
                    continue
                
                elif user_input.startswith('/set '):
                    parts = user_input.split(' ', 2)
                    if len(parts) >= 3:
                        param, value = parts[1], parts[2]
                        try:
                            # Try to convert to appropriate type
                            if value.lower() in ['true', 'false']:
                                current_config[param] = value.lower() == 'true'
                            elif '.' in value:
                                current_config[param] = float(value)
                            else:
                                current_config[param] = int(value)
                            print(f"✅ Set {param} = {current_config[param]}")
                        except ValueError:
                            current_config[param] = value
                            print(f"✅ Set {param} = {value}")
                    continue
                
                elif user_input.lower() == '/reset':
                    current_config = self._default_config()
                    print("🔄 Reset to default configuration")
                    continue
                
                elif user_input.startswith('/'):
                    print("❌ Unknown command")
                    continue
                
                # Generate response
                if conversation_history and not user_input:
                    prompt = conversation_history
                else:
                    prompt = conversation_history + " " + user_input if conversation_history else user_input
                
                print("\n🤖 Generating...")
                start_time = time.time()
                
                generated_text = self.generate(prompt, **current_config)
                
                generation_time = time.time() - start_time
                
                print(f"\n📝 Generated text:")
                print("-" * 30)
                print(generated_text)
                print("-" * 30)
                print(f"⏱️ Generation time: {generation_time:.2f}s")
                
                # Update conversation history
                conversation_history = prompt + generated_text
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue


class StreamingGenerator:
    """Streaming text generator for real-time output."""
    
    def __init__(self, generator: TextGenerator):
        self.generator = generator
    
    def generate_stream(self, prompt: str, callback: Callable[[str], None], **kwargs):
        """Generate text with streaming callback for each token."""
        generation_config = {**self.generator.config, **kwargs}
        
        # Tokenize prompt
        input_tokens = self.generator._tokenize_prompt(prompt)
        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.generator.device)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for step in range(generation_config.get('max_new_tokens', 256)):
                # Forward pass
                outputs = self.generator.model(generated)
                logits = outputs['logits']
                
                # Sample next token
                next_token_logits = logits[:, -1, :]
                next_token = self.generator._sample_next_tokens(
                    next_token_logits, generated, generation_config
                )
                
                # Update sequence
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                
                # Decode and call callback with new token
                new_token_text = self.generator._decode_tokens([next_token.item()])
                callback(new_token_text)
                
                # Check for stopping
                if next_token.item() == generation_config.get('eos_token_id', 1):
                    break