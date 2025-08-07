#!/usr/bin/env python3
"""
Unit tests for tokenizer components.
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.tokenizer import LLMTokenizer


class TestLLMTokenizer(unittest.TestCase):
    """Test LLM tokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = LLMTokenizer()
        self.sample_texts = [
            "Hello, world! This is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing is fascinating.",
            "Machine learning models require training data."
        ]
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        self.assertIsNotNone(self.tokenizer)
        self.assertEqual(self.tokenizer.tokenizer_type, 'bpe')
    
    def test_tokenizer_training(self):
        """Test tokenizer training on sample data."""
        # Create temporary file with sample data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for text in self.sample_texts:
                f.write(text + '\n')
            temp_file = f.name
        
        try:
            # Train tokenizer
            self.tokenizer.train_tokenizer([temp_file], vocab_size=1000)
            
            # Check that tokenizer was trained
            self.assertGreater(self.tokenizer.vocab_size, 0)
            self.assertLessEqual(self.tokenizer.vocab_size, 1000)
            
        finally:
            # Clean up
            Path(temp_file).unlink()
    
    def test_encode_decode_consistency(self):
        """Test that encode->decode is consistent."""
        # Train on sample data first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for text in self.sample_texts:
                f.write(text + '\n')
            temp_file = f.name
        
        try:
            self.tokenizer.train_tokenizer([temp_file], vocab_size=1000)
            
            # Test encode/decode consistency
            test_text = "Hello world, this is a test."
            
            # Encode
            tokens = self.tokenizer.encode(test_text)
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)
            
            # Decode
            decoded_text = self.tokenizer.decode(tokens)
            self.assertIsInstance(decoded_text, str)
            
            # Check that basic content is preserved (allowing for some tokenization differences)
            self.assertIn('Hello', decoded_text)
            self.assertIn('world', decoded_text)
            self.assertIn('test', decoded_text)
            
        finally:
            Path(temp_file).unlink()
    
    def test_special_tokens(self):
        """Test special token handling."""
        # Train tokenizer first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for text in self.sample_texts:
                f.write(text + '\n')
            temp_file = f.name
        
        try:
            self.tokenizer.train_tokenizer([temp_file], vocab_size=1000)
            
            # Check special tokens
            self.assertIsNotNone(self.tokenizer.pad_token_id)
            self.assertIsNotNone(self.tokenizer.unk_token_id)
            self.assertIsNotNone(self.tokenizer.bos_token_id)
            self.assertIsNotNone(self.tokenizer.eos_token_id)
            
            # Special token IDs should be different
            special_tokens = [
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id, 
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id
            ]
            self.assertEqual(len(special_tokens), len(set(special_tokens)))
            
        finally:
            Path(temp_file).unlink()
    
    def test_tokenizer_save_load(self):
        """Test tokenizer saving and loading."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train tokenizer
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for text in self.sample_texts:
                    f.write(text + '\n')
                temp_file = f.name
            
            try:
                self.tokenizer.train_tokenizer([temp_file], vocab_size=1000)
                
                # Save tokenizer
                save_path = Path(temp_dir) / "tokenizer"
                self.tokenizer.save_tokenizer(str(save_path))
                
                # Load tokenizer
                new_tokenizer = LLMTokenizer()
                new_tokenizer.load_tokenizer(str(save_path))
                
                # Test that loaded tokenizer works the same
                test_text = "This is a test sentence."
                
                original_tokens = self.tokenizer.encode(test_text)
                loaded_tokens = new_tokenizer.encode(test_text)
                
                self.assertEqual(original_tokens, loaded_tokens)
                
            finally:
                Path(temp_file).unlink()
    
    def test_batch_processing(self):
        """Test batch encoding and decoding."""
        # Train tokenizer first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for text in self.sample_texts:
                f.write(text + '\n')
            temp_file = f.name
        
        try:
            self.tokenizer.train_tokenizer([temp_file], vocab_size=1000)
            
            # Test batch encoding
            test_texts = ["Hello world", "This is a test", "Batch processing"]
            
            batch_tokens = []
            for text in test_texts:
                tokens = self.tokenizer.encode(text)
                batch_tokens.append(tokens)
            
            # Should have encoded all texts
            self.assertEqual(len(batch_tokens), len(test_texts))
            
            # Test batch decoding
            decoded_texts = []
            for tokens in batch_tokens:
                decoded = self.tokenizer.decode(tokens)
                decoded_texts.append(decoded)
            
            self.assertEqual(len(decoded_texts), len(test_texts))
            
        finally:
            Path(temp_file).unlink()


if __name__ == '__main__':
    # Set up test environment
    unittest.main()