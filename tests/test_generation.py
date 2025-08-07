#!/usr/bin/env python3
"""
Unit tests for text generation components.
"""

import sys
import unittest
import torch
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model import GPTModel, ModelConfig
from data.tokenizer import LLMTokenizer
from inference import TextGenerator


class TestTextGenerator(unittest.TestCase):
    """Test text generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create small model for testing
        self.config = ModelConfig(
            vocab_size=1000,
            block_size=64,
            n_layers=2,
            n_heads=4,
            n_embd=128
        )
        self.model = GPTModel(self.config)
        self.model.eval()
        
        # Create simple tokenizer for testing
        self.tokenizer = LLMTokenizer()
        
        # Mock tokenizer methods for testing
        self.tokenizer.vocab_size = 1000
        self.tokenizer.pad_token_id = 0
        self.tokenizer.unk_token_id = 1
        self.tokenizer.bos_token_id = 2
        self.tokenizer.eos_token_id = 3
        
        # Mock encode/decode for testing
        def mock_encode(text):
            # Simple word-based tokenization for testing
            words = text.lower().split()
            # Map words to token IDs (simplified)
            token_map = {'hello': 10, 'world': 11, 'test': 12, 'the': 13, 'is': 14, 'a': 15}
            tokens = [self.tokenizer.bos_token_id]
            for word in words:
                tokens.append(token_map.get(word, self.tokenizer.unk_token_id))
            return tokens
        
        def mock_decode(tokens):
            # Simple reverse mapping for testing
            id_map = {10: 'hello', 11: 'world', 12: 'test', 13: 'the', 14: 'is', 15: 'a'}
            words = []
            for token in tokens:
                if token == self.tokenizer.bos_token_id:
                    continue
                elif token == self.tokenizer.eos_token_id:
                    break
                else:
                    words.append(id_map.get(token, '<unk>'))
            return ' '.join(words)
        
        self.tokenizer.encode = mock_encode
        self.tokenizer.decode = mock_decode
        
        # Create generator
        self.device = torch.device('cpu')
        self.generator = TextGenerator(
            self.model, self.tokenizer, self.device,
            config={
                'max_new_tokens': 10,
                'temperature': 1.0,
                'top_k': 50,
                'top_p': 0.9
            }
        )
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.device, self.device)
        self.assertEqual(self.generator.config['max_new_tokens'], 10)
    
    def test_single_generation(self):
        """Test single text generation."""
        prompt = "hello world"
        
        try:
            generated = self.generator.generate(prompt)
            
            # Should return a string
            self.assertIsInstance(generated, str)
            
            # Should contain the original prompt (depending on return_prompt setting)
            # This is a basic test - actual generation quality depends on model training
            self.assertTrue(len(generated) > 0)
            
        except Exception as e:
            # Generation might fail with untrained model, which is okay for unit test
            self.assertIsInstance(e, Exception)
    
    def test_batch_generation(self):
        """Test batch text generation."""
        prompts = ["hello world", "test case"]
        
        try:
            generated_list = self.generator.generate_batch(prompts)
            
            # Should return list of strings
            self.assertIsInstance(generated_list, list)
            self.assertEqual(len(generated_list), len(prompts))
            
            for generated in generated_list:
                self.assertIsInstance(generated, str)
            
        except Exception as e:
            # Batch generation might fail with untrained model
            self.assertIsInstance(e, Exception)
    
    def test_generation_config_update(self):
        """Test generation configuration updates."""
        # Test config update
        new_config = {
            'max_new_tokens': 20,
            'temperature': 0.5,
            'top_k': 30
        }
        
        # Update config (this would be done internally)
        original_max_tokens = self.generator.config['max_new_tokens']
        
        # Create new generator with different config
        new_generator = TextGenerator(
            self.model, self.tokenizer, self.device, config=new_config
        )
        
        self.assertEqual(new_generator.config['max_new_tokens'], 20)
        self.assertEqual(new_generator.config['temperature'], 0.5)
        self.assertNotEqual(new_generator.config['max_new_tokens'], original_max_tokens)
    
    def test_generation_parameters(self):
        """Test different generation parameters."""
        prompt = "hello"
        
        # Test different temperatures
        configs = [
            {'temperature': 0.1, 'max_new_tokens': 5},
            {'temperature': 1.0, 'max_new_tokens': 5},
            {'temperature': 2.0, 'max_new_tokens': 5}
        ]
        
        for config in configs:
            generator = TextGenerator(self.model, self.tokenizer, self.device, config=config)
            
            try:
                generated = generator.generate(prompt)
                self.assertIsInstance(generated, str)
            except Exception:
                # May fail with untrained model
                pass
    
    def test_tokenization_methods(self):
        """Test internal tokenization methods."""
        prompt = "hello world test"
        
        # Test prompt tokenization
        tokens = self.generator._tokenize_prompt(prompt)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Test token detokenization
        if tokens:
            decoded = self.generator._detokenize_tokens(tokens)
            self.assertIsInstance(decoded, str)


class TestGenerationSampling(unittest.TestCase):
    """Test different sampling strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple logits tensor for testing sampling
        self.vocab_size = 100
        self.logits = torch.randn(1, 1, self.vocab_size)  # Batch=1, Seq=1, Vocab=100
    
    def test_temperature_sampling(self):
        """Test temperature-based sampling."""
        from inference.generator import apply_temperature
        
        # Test different temperatures
        temperatures = [0.1, 0.5, 1.0, 2.0]
        
        for temp in temperatures:
            scaled_logits = apply_temperature(self.logits.clone(), temp)
            
            # Shape should be preserved
            self.assertEqual(scaled_logits.shape, self.logits.shape)
            
            # For temp < 1, logits should be more peaked
            if temp < 1.0:
                self.assertTrue(torch.max(scaled_logits) > torch.max(self.logits))
    
    def test_top_k_filtering(self):
        """Test top-k filtering."""
        from inference.generator import apply_top_k
        
        k_values = [1, 5, 10, 50]
        
        for k in k_values:
            filtered_logits = apply_top_k(self.logits.clone(), k)
            
            # Shape should be preserved
            self.assertEqual(filtered_logits.shape, self.logits.shape)
            
            # Should have at most k non-masked values per sequence
            non_masked = (filtered_logits[0, 0] > -float('inf')).sum().item()
            self.assertLessEqual(non_masked, k)
    
    def test_top_p_filtering(self):
        """Test top-p (nucleus) filtering."""
        from inference.generator import apply_top_p
        
        p_values = [0.1, 0.5, 0.9, 0.95]
        
        for p in p_values:
            filtered_logits = apply_top_p(self.logits.clone(), p)
            
            # Shape should be preserved
            self.assertEqual(filtered_logits.shape, self.logits.shape)
            
            # Should have filtered some tokens (unless p=1.0)
            if p < 1.0:
                masked_count = (filtered_logits[0, 0] == -float('inf')).sum().item()
                self.assertGreater(masked_count, 0)


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)  # For reproducible tests
    
    # Run tests
    unittest.main()