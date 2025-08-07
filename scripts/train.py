#!/usr/bin/env python3
"""
Main training script for neural LLM.
Supports single-GPU and distributed training with comprehensive configuration.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model import GPTModel, ModelConfig
from data.tokenizer import LLMTokenizer
from data.data_loader import DataLoader
from training.trainer import Trainer, TrainingConfig
from training.optimizer import create_optimizer, create_scheduler
from evaluation.perplexity import PerplexityEvaluator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_distributed() -> tuple[int, int, int]:
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def create_model(config: ModelConfig, device: torch.device) -> GPTModel:
    """Create and initialize model."""
    model = GPTModel(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    model.apply(init_weights)
    model = model.to(device)
    
    return model


def setup_data_loaders(data_config: Dict[str, Any], tokenizer: LLMTokenizer, 
                      world_size: int, rank: int) -> tuple:
    """Setup training and validation data loaders."""
    
    # Create data loader
    data_loader = DataLoader(
        tokenizer=tokenizer,
        data_dir=data_config.get('data_dir', 'data'),
        **data_config.get('loader_params', {})
    )
    
    # Get datasets
    train_dataset = data_loader.get_dataset('train')
    val_dataset = data_loader.get_dataset('validation')
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 4),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=data_config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 4),
        sampler=val_sampler,
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_sampler


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train neural LLM")
    
    # Configuration
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--training-config", help="Path to training config YAML")
    parser.add_argument("--data-config", help="Path to data config YAML")
    
    # Override options
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--evaluate-only", action="store_true", help="Only run evaluation")
    
    # Distributed training
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    # Hardware profile
    parser.add_argument("--profile", help="Hardware profile from hardware_configs.yaml")
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    
    # Load configurations
    print("📋 Loading configurations...")
    
    # Model config
    model_config_dict = load_config(args.config)
    if args.profile:
        # Load hardware configs and apply profile
        hardware_configs = load_config("config/hardware_configs.yaml")
        if args.profile in hardware_configs:
            profile_config = hardware_configs[args.profile]
            # Merge model config with profile
            model_config_dict.update(profile_config.get('model', {}))
    
    model_config = ModelConfig.from_dict(model_config_dict)
    
    # Training config
    training_config_dict = {}
    if args.training_config:
        training_config_dict = load_config(args.training_config)
    elif 'training' in model_config_dict:
        training_config_dict = model_config_dict['training']
    
    training_config = TrainingConfig.from_dict(training_config_dict)
    
    # Data config
    data_config = {}
    if args.data_config:
        data_config = load_config(args.data_config)
    elif 'data' in model_config_dict:
        data_config = model_config_dict['data']
    
    # Override output directory
    if args.output_dir:
        training_config.output_dir = args.output_dir
    
    # Setup device
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    if is_main_process:
        print("🚀 Neural LLM Training")
        print("=" * 50)
        print(f"🖥️  Device: {device}")
        print(f"🌍 World Size: {world_size}")
        print(f"🎯 Model: {model_config.n_layers} layers, {model_config.n_embd} dims")
        print(f"📦 Batch Size: {training_config.batch_size}")
        print(f"📊 Learning Rate: {training_config.learning_rate}")
        print(f"💾 Output: {training_config.output_dir}")
        print("=" * 50)
    
    # Load tokenizer
    if is_main_process:
        print("🔤 Loading tokenizer...")
    
    tokenizer = LLMTokenizer()
    tokenizer_path = data_config.get('tokenizer_path', 'data/tokenizer')
    if Path(tokenizer_path).exists():
        tokenizer.load_tokenizer(tokenizer_path)
    
    # Update model config with tokenizer info
    model_config.vocab_size = tokenizer.vocab_size
    
    # Create model
    if is_main_process:
        print("🏗️  Creating model...")
    
    model = create_model(model_config, device)
    
    if is_main_process:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {param_count:,}")
    
    # Setup data loaders
    if is_main_process:
        print("📚 Setting up data loaders...")
    
    train_loader, val_loader, train_sampler = setup_data_loaders(
        data_config, tokenizer, world_size, rank
    )
    
    if is_main_process:
        print(f"📈 Training batches: {len(train_loader)}")
        print(f"📉 Validation batches: {len(val_loader)}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, len(train_loader), training_config)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if is_main_process:
            print(f"🔄 Resuming from {args.resume}")
        
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
    
    # Setup distributed model
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        device=device,
        rank=rank,
        world_size=world_size
    )
    
    # Evaluation only mode
    if args.evaluate_only:
        if is_main_process:
            print("📊 Running evaluation only...")
        
        evaluator = PerplexityEvaluator(model, tokenizer, device)
        val_loss, perplexity = evaluator.evaluate_dataset(val_loader)
        
        if is_main_process:
            print(f"📈 Validation Loss: {val_loss:.4f}")
            print(f"🎯 Perplexity: {perplexity:.2f}")
        
        return 0
    
    # Training loop
    if is_main_process:
        print("🚄 Starting training...")
    
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            start_epoch=start_epoch,
            train_sampler=train_sampler
        )
        
        if is_main_process:
            print("🎉 Training completed successfully!")
    
    except KeyboardInterrupt:
        if is_main_process:
            print("\n⏹️ Training interrupted by user")
        return 1
    
    except Exception as e:
        if is_main_process:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        # Cleanup distributed training
        if world_size > 1:
            dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())