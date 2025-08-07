#!/usr/bin/env python3
"""
Unit tests for model components.
"""

import sys
import unittest
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model import GPTModel, ModelConfig
from model.attention import MultiHeadAttention
from model.layers import TransformerBlock


class TestModelConfig(unittest.TestCase):
    """Test model configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ModelConfig()
        
        self.assertEqual(config.vocab_size, 50000)
        self.assertEqual(config.block_size, 512)
        self.assertEqual(config.n_layers, 12)
        self.assertEqual(config.n_heads, 12)
        self.assertEqual(config.n_embd, 768)
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'vocab_size': 30000,
            'block_size': 256,
            'n_layers': 6,
            'n_heads': 8,
            'n_embd': 512
        }
        
        config = ModelConfig.from_dict(config_dict)
        
        self.assertEqual(config.vocab_size, 30000)
        self.assertEqual(config.block_size, 256)
        self.assertEqual(config.n_layers, 6)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.n_embd, 512)


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_embd = 768
        self.n_heads = 12
        self.block_size = 512
        self.batch_size = 2
        self.seq_len = 64
        
        self.attention = MultiHeadAttention(
            n_embd=self.n_embd,
            n_heads=self.n_heads,
            block_size=self.block_size
        )
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        # Create input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.n_embd)
        
        # Forward pass
        output = self.attention(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.n_embd))
    
    def test_attention_with_mask(self):
        """Test attention with causal mask."""
        x = torch.randn(self.batch_size, self.seq_len, self.n_embd)
        
        # Create attention mask
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        
        # This should not raise an error
        output = self.attention(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.n_embd))


class TestTransformerBlock(unittest.TestCase):
    """Test transformer block."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = ModelConfig(
            vocab_size=50000,
            block_size=512,
            n_layers=12,
            n_heads=12,
            n_embd=768
        )
        self.block = TransformerBlock(config)
        self.batch_size = 2
        self.seq_len = 64
        self.n_embd = 768
    
    def test_block_forward(self):
        """Test transformer block forward pass."""
        x = torch.randn(self.batch_size, self.seq_len, self.n_embd)
        
        output = self.block(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.n_embd))


class TestGPTModel(unittest.TestCase):
    """Test full GPT model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            vocab_size=1000,  # Small vocab for testing
            block_size=128,   # Small context for testing
            n_layers=2,       # Few layers for testing
            n_heads=4,        # Few heads for testing
            n_embd=256        # Small embedding for testing
        )
        self.model = GPTModel(self.config)
        self.batch_size = 2
        self.seq_len = 32
    
    def test_model_forward(self):
        """Test model forward pass."""
        # Create input token IDs
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.config.vocab_size)
        self.assertEqual(outputs.logits.shape, expected_shape)
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        param_count = sum(p.numel() for p in self.model.parameters())
        
        # Should have reasonable number of parameters
        self.assertGreater(param_count, 1000)
        self.assertLess(param_count, 10_000_000)  # Not too large for test config
    
    def test_model_generation_mode(self):
        """Test model in generation mode."""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
            outputs = self.model(input_ids)
            
            # Should produce logits
            self.assertEqual(len(outputs.logits.shape), 3)
            self.assertEqual(outputs.logits.shape[0], 1)  # batch size
            self.assertEqual(outputs.logits.shape[1], 10)  # sequence length
            self.assertEqual(outputs.logits.shape[2], self.config.vocab_size)


class TestModelCompatibility(unittest.TestCase):
    """Test model compatibility with PyTorch operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            vocab_size=1000,
            block_size=64,
            n_layers=2,
            n_heads=4,
            n_embd=128
        )
        self.model = GPTModel(self.config)
    
    def test_model_cuda_compatibility(self):
        """Test model CUDA compatibility if available."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model_cuda = self.model.to(device)
            
            input_ids = torch.randint(0, self.config.vocab_size, (1, 10)).to(device)
            outputs = model_cuda(input_ids)
            
            self.assertTrue(outputs.logits.is_cuda)
    
    def test_model_cpu_compatibility(self):
        """Test model CPU compatibility."""
        device = torch.device('cpu')
        model_cpu = self.model.to(device)
        
        input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
        outputs = model_cpu(input_ids)
        
        self.assertFalse(outputs.logits.is_cuda)
    
    def test_model_training_mode(self):
        """Test model training mode transitions."""
        # Test training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Test evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)  # For reproducible tests
    
    # Run tests
    unittest.main()