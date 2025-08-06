#!/usr/bin/env python3
"""
Tokenizer training and management for neural LLM.
Supports BPE, SentencePiece, and custom tokenization strategies.
"""

import os
import json
import yaml
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
import logging
from collections import Counter

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer  
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

import sentencepiece as spm
from transformers import GPT2TokenizerFast
from datasets import load_from_disk, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMTokenizer:
    """Unified tokenizer class for different tokenization strategies."""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize tokenizer with configuration."""
        self.config = self._load_config(config_path)
        self.tokenizer_config = self.config['data']['tokenizer']
        self.cache_dir = Path(self.config['data']['cache_dir'])
        self.data_dir = Path(self.config['data']['data_dir'])
        
        self.tokenizer = None
        self.vocab_size = self.tokenizer_config['vocab_size']
        self.special_tokens = self.tokenizer_config['special_tokens']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_training_data_iterator(self) -> Iterator[str]:
        """Get iterator over training text data."""
        datasets_config = self.config['data']['datasets']
        
        for dataset_config in datasets_config:
            if not dataset_config.get('enabled', False):
                continue
                
            name = dataset_config['name']
            info_file = self.data_dir / f"{name.replace('_', '')}_*_info.yaml"
            info_files = list(self.data_dir.glob(info_file.name))
            
            if not info_files:
                logger.warning(f"No info file found for {name}, skipping")
                continue
            
            # Load cached dataset
            try:
                if name == "the_pile":
                    from datasets import load_dataset
                    dataset = load_dataset("EleutherAI/pile", split="train", 
                                         cache_dir=str(self.cache_dir), streaming=True)
                    
                    logger.info(f"Streaming text from {name}")
                    for i, example in enumerate(dataset):
                        if i % 10000 == 0:
                            logger.info(f"Processed {i} examples from {name}")
                        yield example['text']
                        
                        # Limit for training tokenizer (can be memory intensive)
                        if i >= 100000:  # 100k examples should be sufficient for tokenizer training
                            break
                            
                elif name == "wikipedia":
                    from datasets import load_dataset
                    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", 
                                         split="train", cache_dir=str(self.cache_dir))
                    
                    logger.info(f"Loading text from {name}")
                    for example in tqdm(dataset.select(range(min(50000, len(dataset)))), 
                                       desc=f"Processing {name}"):
                        yield example['text']
                        
            except Exception as e:
                logger.error(f"Error loading dataset {name}: {e}")
                continue
    
    def train_bpe_tokenizer(self) -> Tokenizer:
        """Train a Byte-Pair Encoding tokenizer."""
        logger.info("Training BPE tokenizer...")
        
        # Initialize BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        
        # Set pre-tokenizer
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        
        # Set normalizer
        tokenizer.normalizer = Sequence([NFD(), StripAccents()])
        
        # Set decoder
        tokenizer.decoder = ByteLevelDecoder()
        
        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.tokenizer_config.get('min_frequency', 2),
            special_tokens=self.special_tokens,
            show_progress=self.tokenizer_config.get('show_progress', True)
        )
        
        # Train on data
        tokenizer.train_from_iterator(self._get_training_data_iterator(), trainer)
        
        # Set post processor
        tokenizer.post_processor = TemplateProcessing(
            single="<|endoftext|> $A",
            special_tokens=[("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))]
        )
        
        return tokenizer
    
    def train_sentencepiece_tokenizer(self) -> str:
        """Train a SentencePiece tokenizer."""
        logger.info("Training SentencePiece tokenizer...")
        
        # Prepare training data
        training_file = self.data_dir / "sentencepiece_training.txt"
        
        with open(training_file, 'w', encoding='utf-8') as f:
            for text in tqdm(self._get_training_data_iterator(), desc="Preparing SentencePiece data"):
                f.write(text + "\n")
        
        # SentencePiece model file
        model_file = self.data_dir / "sentencepiece.model"
        
        # Train SentencePiece
        spm.SentencePieceTrainer.train(
            input=str(training_file),
            model_prefix=str(model_file.with_suffix("")),
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type='bpe',
            user_defined_symbols=self.special_tokens,
            pad_id=0,
            unk_id=1,
            bos_id=2, 
            eos_id=3,
            normalization_rule_name='nmt_nfkc_cf'
        )
        
        # Clean up training file
        training_file.unlink()
        
        return str(model_file)
    
    def train_tokenizer(self) -> None:
        """Train tokenizer based on configuration."""
        tokenizer_type = self.tokenizer_config['type'].lower()
        
        if tokenizer_type == 'bpe':
            tokenizer = self.train_bpe_tokenizer()
            tokenizer_file = self.data_dir / "tokenizer.json"
            tokenizer.save(str(tokenizer_file))
            
            # Also save as transformers tokenizer for compatibility
            fast_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
            fast_tokenizer.save_pretrained(str(self.data_dir / "tokenizer"))
            
            logger.info(f"BPE tokenizer saved to {tokenizer_file}")
            
        elif tokenizer_type == 'sentencepiece':
            model_file = self.train_sentencepiece_tokenizer()
            logger.info(f"SentencePiece tokenizer saved to {model_file}")
            
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        
        # Save tokenizer metadata
        self._save_tokenizer_metadata(tokenizer_type)
    
    def _save_tokenizer_metadata(self, tokenizer_type: str) -> None:
        """Save tokenizer metadata."""
        metadata = {
            'type': tokenizer_type,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'config': self.tokenizer_config
        }
        
        metadata_file = self.data_dir / "tokenizer_metadata.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info(f"Tokenizer metadata saved to {metadata_file}")
    
    def load_tokenizer(self) -> Union[Tokenizer, smp.SentencePieceProcessor]:
        """Load trained tokenizer."""
        metadata_file = self.data_dir / "tokenizer_metadata.yaml"
        
        if not metadata_file.exists():
            raise FileNotFoundError("No trained tokenizer found. Run train_tokenizer() first.")
        
        with open(metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)
        
        tokenizer_type = metadata['type']
        
        if tokenizer_type == 'bpe':
            tokenizer_file = self.data_dir / "tokenizer.json"
            if tokenizer_file.exists():
                return Tokenizer.from_file(str(tokenizer_file))
            else:
                # Load transformers tokenizer
                return GPT2TokenizerFast.from_pretrained(str(self.data_dir / "tokenizer"))
                
        elif tokenizer_type == 'sentencepiece':
            model_file = self.data_dir / "sentencepiece.model"
            sp = spm.SentencePieceProcessor()
            sp.load(str(model_file))
            return sp
            
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using the trained tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        
        if hasattr(self.tokenizer, 'encode'):
            # Transformers tokenizer or HuggingFace tokenizer
            return self.tokenizer.encode(text)
        elif hasattr(self.tokenizer, 'encode_as_ids'):
            # SentencePiece tokenizer
            return self.tokenizer.encode_as_ids(text)
        else:
            raise ValueError("Unknown tokenizer type")
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to text."""
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        
        if hasattr(self.tokenizer, 'decode'):
            # Transformers or HuggingFace tokenizer
            return self.tokenizer.decode(tokens)
        elif hasattr(self.tokenizer, 'decode_ids'):
            # SentencePiece tokenizer  
            return self.tokenizer.decode_ids(tokens)
        else:
            raise ValueError("Unknown tokenizer type")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs."""
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        
        special_ids = {}
        
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            # Transformers tokenizer
            for token in self.special_tokens:
                special_ids[token] = self.tokenizer.convert_tokens_to_ids(token)
        elif hasattr(self.tokenizer, 'piece_to_id'):
            # SentencePiece tokenizer
            for token in self.special_tokens:
                special_ids[token] = self.tokenizer.piece_to_id(token)
        
        return special_ids


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train tokenizer for LLM")
    parser.add_argument("--config", default="config/data_config.yaml",
                       help="Path to data configuration file")
    parser.add_argument("--test-text", type=str,
                       help="Test text to tokenize and decode")
    
    args = parser.parse_args()
    
    tokenizer_manager = LLMTokenizer(args.config)
    
    if args.test_text:
        # Test tokenization
        try:
            tokens = tokenizer_manager.tokenize_text(args.test_text)
            decoded = tokenizer_manager.decode_tokens(tokens)
            
            print(f"Original: {args.test_text}")
            print(f"Tokens: {tokens}")
            print(f"Decoded: {decoded}")
            print(f"Token count: {len(tokens)}")
            
        except Exception as e:
            logger.error(f"Error testing tokenizer: {e}")
            logger.info("Please train tokenizer first")
    else:
        # Train tokenizer
        tokenizer_manager.train_tokenizer()


if __name__ == "__main__":
    main()