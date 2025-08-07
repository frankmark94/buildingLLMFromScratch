#!/usr/bin/env python3
"""
Fine-tune a pre-trained neural LLM on custom data.
"""

import sys
import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.fine_tuner import FineTuner, FineTuningConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_fine_tuning_config(args, config_dict: Dict[str, Any]) -> FineTuningConfig:
    """Create fine-tuning configuration from args and config file."""
    
    # Override config with command line args
    if args.model:
        config_dict['model_path'] = args.model
    if args.dataset:
        config_dict['dataset_path'] = args.dataset
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir
    if args.learning_rate:
        config_dict['learning_rate'] = args.learning_rate
    if args.batch_size:
        config_dict['batch_size'] = args.batch_size
    if args.epochs:
        config_dict['num_epochs'] = args.epochs
    
    return FineTuningConfig(**config_dict)


def main():
    """Main entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune neural LLM on custom data")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to pre-trained model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to fine-tuning dataset (JSON/JSONL)")
    
    # Optional arguments
    parser.add_argument("--config", help="Path to fine-tuning config YAML file")
    parser.add_argument("--output-dir", default="fine_tuned_models", help="Output directory for fine-tuned model")
    parser.add_argument("--tokenizer", help="Path to tokenizer (default: data/tokenizer)")
    
    # Training parameters
    parser.add_argument("--task-type", choices=["text_generation", "classification"], 
                       default="text_generation", help="Type of fine-tuning task")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    
    # Data parameters
    parser.add_argument("--text-column", default="text", help="Name of text column in dataset")
    parser.add_argument("--label-column", help="Name of label column (for classification)")
    parser.add_argument("--train-split", type=float, default=0.9, help="Training split ratio")
    
    # Advanced options
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, 
                       help="Gradient accumulation steps")
    parser.add_argument("--freeze-embeddings", action="store_true", 
                       help="Freeze embedding layers")
    parser.add_argument("--freeze-layers", nargs="+", type=int, 
                       help="Layer indices to freeze (e.g., 0 1 2)")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    
    args = parser.parse_args()
    
    # Load base config
    config_dict = {}
    if args.config:
        config_dict = load_config(args.config)
    
    # Set defaults
    config_defaults = {
        'model_path': args.model,
        'dataset_path': args.dataset,
        'output_dir': args.output_dir,
        'tokenizer_path': args.tokenizer or "data/tokenizer",
        'task_type': args.task_type,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'max_seq_length': args.max_seq_length,
        'text_column': args.text_column,
        'label_column': args.label_column,
        'train_split': args.train_split,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'freeze_embeddings': args.freeze_embeddings,
        'freeze_layers': args.freeze_layers,
        'use_amp': not args.no_amp,
    }
    
    # Merge configs
    for key, value in config_defaults.items():
        if key not in config_dict or value is not None:
            config_dict[key] = value
    
    # Create fine-tuning config
    ft_config = FineTuningConfig(**{k: v for k, v in config_dict.items() if v is not None})
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("🚀 Neural LLM Fine-Tuning")
    print("=" * 50)
    print(f"📁 Model: {ft_config.model_path}")
    print(f"📊 Dataset: {ft_config.dataset_path}")
    print(f"🎯 Task: {ft_config.task_type}")
    print(f"🖥️  Device: {device}")
    print(f"⚙️  Learning Rate: {ft_config.learning_rate}")
    print(f"📦 Batch Size: {ft_config.batch_size}")
    print(f"🔄 Epochs: {ft_config.num_epochs}")
    print("=" * 50)
    
    try:
        # Create and run fine-tuner
        fine_tuner = FineTuner(ft_config, device)
        fine_tuner.fine_tune()
        
        print("\n🎉 Fine-tuning completed successfully!")
        print(f"📁 Fine-tuned model saved to: {ft_config.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Fine-tuning interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())