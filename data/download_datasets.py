#!/usr/bin/env python3
"""
Dataset downloading script for neural LLM training.
Supports The Pile, Common Crawl, Wikipedia, and other public datasets.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List
import logging
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetDownloader:
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize the dataset downloader with configuration."""
        self.config = self._load_config(config_path)
        self.cache_dir = Path(self.config['data']['cache_dir'])
        self.data_dir = Path(self.config['data']['data_dir'])
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def download_the_pile(self, subset: str = "all", split: str = "train") -> None:
        """Download The Pile dataset."""
        logger.info(f"Downloading The Pile dataset (subset: {subset}, split: {split})")
        
        try:
            if subset == "all":
                dataset_name = "EleutherAI/pile"
            else:
                dataset_name = f"EleutherAI/pile-{subset}"
            
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # Save dataset info
            info_file = self.data_dir / f"pile_{subset}_{split}_info.yaml"
            with open(info_file, 'w') as f:
                yaml.dump({
                    'dataset_name': dataset_name,
                    'subset': subset,
                    'split': split,
                    'num_examples': len(dataset),
                    'features': list(dataset.features.keys()),
                    'cache_files': dataset.cache_files
                }, f)
            
            logger.info(f"The Pile downloaded successfully. Info saved to {info_file}")
            
        except Exception as e:
            logger.error(f"Error downloading The Pile: {e}")
            raise
    
    def download_wikipedia(self, language: str = "en", date: str = "20231101") -> None:
        """Download Wikipedia dataset."""
        logger.info(f"Downloading Wikipedia dataset ({language}, {date})")
        
        try:
            dataset = load_dataset(
                "wikimedia/wikipedia",
                f"{date}.{language}",
                split="train",
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # Save dataset info
            info_file = self.data_dir / f"wikipedia_{language}_{date}_info.yaml"
            with open(info_file, 'w') as f:
                yaml.dump({
                    'dataset_name': "wikimedia/wikipedia",
                    'language': language,
                    'date': date,
                    'num_examples': len(dataset),
                    'features': list(dataset.features.keys()),
                    'cache_files': dataset.cache_files
                }, f)
            
            logger.info(f"Wikipedia downloaded successfully. Info saved to {info_file}")
            
        except Exception as e:
            logger.error(f"Error downloading Wikipedia: {e}")
            raise
    
    def download_common_crawl(self, subset: str = "2023-06") -> None:
        """Download Common Crawl dataset."""
        logger.info(f"Downloading Common Crawl dataset (subset: {subset})")
        
        try:
            # Note: This is a placeholder - actual Common Crawl access might need different approach
            dataset = load_dataset(
                "mc4",
                "en",
                split="train",
                cache_dir=str(self.cache_dir),
                streaming=True  # CC is very large, use streaming
            )
            
            logger.info("Common Crawl dataset prepared for streaming")
            
        except Exception as e:
            logger.error(f"Error downloading Common Crawl: {e}")
            logger.info("Note: Common Crawl might require special access or preprocessing")
            raise
    
    def download_custom_dataset(self, name: str, config: Dict[str, Any]) -> None:
        """Download custom dataset based on configuration."""
        logger.info(f"Loading custom dataset: {name}")
        
        try:
            path = config['path']
            text_column = config.get('text_column', 'text')
            
            if path.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=path, split='train')
            elif path.endswith('.csv'):
                dataset = load_dataset('csv', data_files=path, split='train')
            elif path.endswith('.txt'):
                # Read as plain text file
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # Create dataset from text
                dataset = Dataset.from_dict({text_column: [text]})
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            # Save dataset info
            info_file = self.data_dir / f"custom_{name}_info.yaml"
            with open(info_file, 'w') as f:
                yaml.dump({
                    'dataset_name': f"custom_{name}",
                    'path': path,
                    'text_column': text_column,
                    'num_examples': len(dataset),
                    'features': list(dataset.features.keys())
                }, f)
            
            logger.info(f"Custom dataset {name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading custom dataset {name}: {e}")
            raise
    
    def download_all_enabled_datasets(self) -> None:
        """Download all enabled datasets from configuration."""
        datasets_config = self.config['data']['datasets']
        custom_datasets_config = self.config.get('custom_datasets', {})
        
        for dataset_config in datasets_config:
            if not dataset_config.get('enabled', False):
                continue
            
            name = dataset_config['name']
            logger.info(f"Processing dataset: {name}")
            
            try:
                if name == "the_pile":
                    self.download_the_pile(
                        subset=dataset_config.get('subset', 'all'),
                        split=dataset_config.get('split', 'train')
                    )
                elif name == "wikipedia":
                    subset = dataset_config.get('subset', '20231101.en')
                    date, lang = subset.split('.')
                    self.download_wikipedia(language=lang, date=date)
                elif name == "common_crawl":
                    self.download_common_crawl(
                        subset=dataset_config.get('subset', '2023-06')
                    )
                else:
                    logger.warning(f"Unknown dataset: {name}")
                    
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
                continue
        
        # Download custom datasets
        for name, config in custom_datasets_config.items():
            if config.get('enabled', False):
                try:
                    self.download_custom_dataset(name, config)
                except Exception as e:
                    logger.error(f"Failed to download custom dataset {name}: {e}")
                    continue
        
        logger.info("Dataset downloading completed!")
    
    def list_downloaded_datasets(self) -> List[str]:
        """List all downloaded datasets."""
        info_files = list(self.data_dir.glob("*_info.yaml"))
        datasets = []
        
        for info_file in info_files:
            with open(info_file, 'r') as f:
                info = yaml.safe_load(f)
                datasets.append({
                    'name': info.get('dataset_name', 'unknown'),
                    'num_examples': info.get('num_examples', 0),
                    'info_file': str(info_file)
                })
        
        return datasets


def main():
    parser = argparse.ArgumentParser(description="Download datasets for LLM training")
    parser.add_argument("--config", default="config/data_config.yaml", 
                       help="Path to data configuration file")
    parser.add_argument("--list", action="store_true", 
                       help="List downloaded datasets")
    parser.add_argument("--dataset", type=str, 
                       help="Download specific dataset (the_pile, wikipedia, common_crawl)")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.config)
    
    if args.list:
        datasets = downloader.list_downloaded_datasets()
        print(f"\nDownloaded datasets ({len(datasets)}):")
        for dataset in datasets:
            print(f"  - {dataset['name']}: {dataset['num_examples']:,} examples")
        return
    
    if args.dataset:
        if args.dataset == "the_pile":
            downloader.download_the_pile()
        elif args.dataset == "wikipedia":
            downloader.download_wikipedia()
        elif args.dataset == "common_crawl":
            downloader.download_common_crawl()
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            sys.exit(1)
    else:
        downloader.download_all_enabled_datasets()


if __name__ == "__main__":
    main()