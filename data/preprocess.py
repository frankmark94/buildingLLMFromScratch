#!/usr/bin/env python3
"""
Data preprocessing pipeline for neural LLM training.
Handles cleaning, deduplication, filtering, and tokenization of text data.
"""

import os
import re
import json
import yaml
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Iterator, Set, Tuple
import logging
from collections import defaultdict
import multiprocessing as mp
from functools import partial

from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Try to download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Text cleaning utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_length = config.get('min_length', 50)
        self.max_length = config.get('max_length', 100000)
        self.filter_languages = config.get('filter_languages', ['en'])
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.excessive_whitespace = re.compile(r'\s+')
        self.non_printable = re.compile(r'[^\x20-\x7E\n\r\t]')
        
    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs and emails
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        
        # Remove non-printable characters (keep basic ASCII + newlines/tabs)
        text = self.non_printable.sub('', text)
        
        # Normalize whitespace
        text = self.excessive_whitespace.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def filter_text(self, text: str) -> bool:
        """Filter text based on length and other criteria."""
        if not text or len(text) < self.min_length:
            return False
        
        if len(text) > self.max_length:
            return False
        
        # Check for minimum word count
        word_count = len(text.split())
        if word_count < 10:  # At least 10 words
            return False
        
        # Check for reasonable character-to-space ratio (detect gibberish)
        space_ratio = text.count(' ') / len(text)
        if space_ratio < 0.1 or space_ratio > 0.3:  # Reasonable bounds
            return False
        
        # Check for excessive repetition
        if self._has_excessive_repetition(text):
            return False
        
        return True
    
    def _has_excessive_repetition(self, text: str, max_repeat_ratio: float = 0.3) -> bool:
        """Check if text has excessive character or word repetition."""
        # Check character repetition
        char_counts = defaultdict(int)
        for char in text.lower():
            if char.isalnum():
                char_counts[char] += 1
        
        if char_counts:
            most_common_char_count = max(char_counts.values())
            char_repeat_ratio = most_common_char_count / len(text)
            if char_repeat_ratio > max_repeat_ratio:
                return True
        
        # Check word repetition  
        words = text.lower().split()
        if len(words) > 10:
            word_counts = defaultdict(int)
            for word in words:
                if len(word) > 3:  # Only check longer words
                    word_counts[word] += 1
            
            if word_counts:
                most_common_word_count = max(word_counts.values())
                word_repeat_ratio = most_common_word_count / len(words)
                if word_repeat_ratio > max_repeat_ratio:
                    return True
        
        return False


class TextDeduplicator:
    """Text deduplication utilities."""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.seen_prefixes: Set[str] = set()
    
    def get_text_hash(self, text: str) -> str:
        """Get hash of text for exact deduplication."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_text_prefix_hash(self, text: str, prefix_length: int = 100) -> str:
        """Get hash of text prefix for near-duplicate detection."""
        prefix = text[:prefix_length].lower().strip()
        return hashlib.md5(prefix.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate (exact or near-duplicate)."""
        # Exact duplicate check
        text_hash = self.get_text_hash(text)
        if text_hash in self.seen_hashes:
            return True
        
        # Near-duplicate check (same prefix)
        prefix_hash = self.get_text_prefix_hash(text)
        if prefix_hash in self.seen_prefixes:
            return True
        
        # Add to seen sets
        self.seen_hashes.add(text_hash)
        self.seen_prefixes.add(prefix_hash)
        
        return False


class DataPreprocessor:
    """Main data preprocessing pipeline."""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        self.config = self._load_config(config_path)
        self.preprocessing_config = self.config['data']['preprocessing']
        self.data_dir = Path(self.config['data']['data_dir'])
        self.cache_dir = Path(self.config['data']['cache_dir'])
        
        self.cleaner = TextCleaner(self.preprocessing_config)
        self.deduplicator = TextDeduplicator()
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def process_text(self, text: str) -> Tuple[str, bool]:
        """Process a single text: clean, filter, and deduplicate."""
        # Clean text
        if self.preprocessing_config.get('clean_text', True):
            text = self.cleaner.clean_text(text)
        
        # Filter text
        if not self.cleaner.filter_text(text):
            return "", False
        
        # Check for duplicates
        if self.preprocessing_config.get('dedupe', True):
            if self.deduplicator.is_duplicate(text):
                return "", False
        
        return text, True
    
    def process_dataset_batch(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Process a batch of examples from a dataset."""
        processed_texts = []
        valid_indices = []
        
        texts = examples.get('text', examples.get('content', []))
        
        for i, text in enumerate(texts):
            processed_text, is_valid = self.process_text(text)
            if is_valid:
                processed_texts.append(processed_text)
                valid_indices.append(i)
        
        # Return only valid examples
        result = {'text': processed_texts}
        
        # Copy over other fields for valid examples
        for key, values in examples.items():
            if key not in ['text', 'content']:
                result[key] = [values[i] for i in valid_indices]
        
        return result
    
    def preprocess_dataset(self, dataset_name: str) -> Dataset:
        """Preprocess a specific dataset."""
        logger.info(f"Preprocessing dataset: {dataset_name}")
        
        # Load dataset based on name
        try:
            if dataset_name == "the_pile":
                dataset = load_dataset("EleutherAI/pile", split="train", 
                                     cache_dir=str(self.cache_dir), streaming=True)
                # Convert streaming to regular dataset for processing
                # Note: This might be memory intensive for large datasets
                examples = []
                for i, example in enumerate(tqdm(dataset, desc="Loading The Pile")):
                    examples.append(example)
                    if i >= 100000:  # Limit for demo purposes
                        break
                dataset = Dataset.from_list(examples)
                
            elif dataset_name == "wikipedia":
                dataset = load_dataset("wikimedia/wikipedia", "20231101.en", 
                                     split="train", cache_dir=str(self.cache_dir))
                
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Process dataset
            logger.info(f"Processing {len(dataset)} examples...")
            processed_dataset = dataset.map(
                self.process_dataset_batch,
                batched=True,
                batch_size=1000,
                num_proc=min(4, mp.cpu_count()),
                desc=f"Processing {dataset_name}"
            )
            
            # Filter out empty texts
            processed_dataset = processed_dataset.filter(
                lambda example: len(example['text']) > 0,
                desc="Filtering empty texts"
            )
            
            logger.info(f"Processed dataset size: {len(processed_dataset)} examples")
            
            # Save processed dataset
            output_path = self.data_dir / f"{dataset_name}_processed"
            processed_dataset.save_to_disk(str(output_path))
            logger.info(f"Saved processed dataset to {output_path}")
            
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Error preprocessing {dataset_name}: {e}")
            raise
    
    def create_training_data(self, block_size: int = 1024) -> None:
        """Create final training data files in binary format."""
        logger.info("Creating training data files...")
        
        # Find all processed datasets
        processed_datasets = []
        for dataset_config in self.config['data']['datasets']:
            if not dataset_config.get('enabled', False):
                continue
            
            dataset_name = dataset_config['name']
            dataset_path = self.data_dir / f"{dataset_name}_processed"
            
            if dataset_path.exists():
                logger.info(f"Loading processed dataset: {dataset_name}")
                dataset = load_from_disk(str(dataset_path))
                processed_datasets.append(dataset)
            else:
                logger.warning(f"Processed dataset not found: {dataset_path}")
        
        if not processed_datasets:
            logger.error("No processed datasets found!")
            return
        
        # Combine datasets
        if len(processed_datasets) == 1:
            combined_dataset = processed_datasets[0]
        else:
            from datasets import concatenate_datasets
            combined_dataset = concatenate_datasets(processed_datasets)
        
        logger.info(f"Combined dataset size: {len(combined_dataset)} examples")
        
        # Split into train/val
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        split_dataset = combined_dataset.train_test_split(
            test_size=val_split,
            train_size=train_split,
            seed=42
        )
        
        # Tokenize and create binary files
        from .tokenizer import LLMTokenizer
        tokenizer_manager = LLMTokenizer()
        
        try:
            tokenizer = tokenizer_manager.load_tokenizer()
        except FileNotFoundError:
            logger.error("Tokenizer not found! Please train tokenizer first.")
            return
        
        # Process train and validation sets
        for split_name, split_data in [("train", split_dataset['train']), ("val", split_dataset['test'])]:
            logger.info(f"Creating {split_name} data...")
            
            all_tokens = []
            
            for example in tqdm(split_data, desc=f"Tokenizing {split_name}"):
                tokens = tokenizer_manager.tokenize_text(example['text'])
                all_tokens.extend(tokens)
                
                # Add end-of-text token between documents
                eos_ids = tokenizer_manager.get_special_token_ids()
                if '<|endoftext|>' in eos_ids:
                    all_tokens.append(eos_ids['<|endoftext|>'])
            
            # Convert to numpy array and save
            all_tokens = np.array(all_tokens, dtype=np.uint16)  # Assuming vocab_size < 65536
            
            output_file = self.data_dir / f"{split_name}.bin"
            all_tokens.tofile(str(output_file))
            
            logger.info(f"Saved {split_name} data: {len(all_tokens):,} tokens to {output_file}")
        
        # Save metadata
        metadata = {
            'vocab_size': tokenizer_manager.get_vocab_size(),
            'block_size': block_size,
            'train_tokens': len(np.fromfile(str(self.data_dir / "train.bin"), dtype=np.uint16)),
            'val_tokens': len(np.fromfile(str(self.data_dir / "val.bin"), dtype=np.uint16)),
            'special_token_ids': tokenizer_manager.get_special_token_ids()
        }
        
        with open(self.data_dir / "data_metadata.yaml", 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info("Data preprocessing completed!")
    
    def preprocess_all_datasets(self) -> None:
        """Preprocess all enabled datasets."""
        datasets_config = self.config['data']['datasets']
        
        for dataset_config in datasets_config:
            if not dataset_config.get('enabled', False):
                continue
            
            dataset_name = dataset_config['name']
            try:
                self.preprocess_dataset(dataset_name)
            except Exception as e:
                logger.error(f"Failed to preprocess {dataset_name}: {e}")
                continue
        
        # Create final training data
        self.create_training_data(
            block_size=self.config['data']['block_size']
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess data for LLM training")
    parser.add_argument("--config", default="config/data_config.yaml",
                       help="Path to data configuration file")
    parser.add_argument("--dataset", type=str,
                       help="Preprocess specific dataset")
    parser.add_argument("--create-training-data", action="store_true",
                       help="Create final training data files")
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.config)
    
    if args.dataset:
        preprocessor.preprocess_dataset(args.dataset)
    elif args.create_training_data:
        preprocessor.create_training_data()
    else:
        preprocessor.preprocess_all_datasets()


if __name__ == "__main__":
    main()