#!/usr/bin/env python3
"""
Main training loop with support for distributed training, mixed precision, and advanced optimization.
"""

import os
import time
import math
import yaml
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.distributed import init_process_group, destroy_process_group
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path

from ..model import GPTModel, ModelConfig
from .optimizer import create_optimizer, create_scheduler
from .mixed_precision import MixedPrecisionTrainer
from .checkpointing import CheckpointManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class with distributed training and mixed precision support."""
    
    def __init__(
        self,
        model: GPTModel,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        
        # Training configuration
        self.max_steps = config['max_steps']
        self.eval_interval = config['eval_interval']
        self.save_interval = config['save_interval']
        self.log_interval = config['log_interval']
        self.gradient_accumulation_steps = config['gradient_accumulation_steps']
        self.grad_clip = config['grad_clip']
        
        # Setup model for distributed training
        self.model = self.model.to(device)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        
        # Mixed precision setup
        self.use_amp = config.get('use_amp', False)
        self.mixed_precision_trainer = MixedPrecisionTrainer(enabled=self.use_amp)
        
        # Optimizer and scheduler
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        
        # Checkpointing
        checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir,
            max_checkpoints=config.get('save_total_limit', 3)
        )
        
        # Logging
        self.use_wandb = config.get('use_wandb', False) and self.is_main_process
        if self.use_wandb:
            self._setup_wandb(config)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.tokens_seen = 0
        
        # Performance tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        logger.info(f"Trainer initialized - Rank: {rank}/{world_size}, Device: {device}")
    
    def _setup_wandb(self, config: Dict[str, Any]) -> None:
        """Initialize Weights & Biases logging."""
        run_name = config.get('wandb_run_name') or f"neural-llm-{int(time.time())}"
        
        wandb.init(
            project=config.get('wandb_project', 'neural-llm'),
            name=run_name,
            config={
                'model_config': self.model.config.to_dict(),
                'training_config': config,
                'world_size': self.world_size,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
            }
        )
        
        # Watch model gradients
        if config.get('log_grad_norm', False):
            wandb.watch(self.model, log='gradients', log_freq=self.log_interval)
    
    def train(self) -> None:
        """Main training loop."""
        logger.info(f"Starting training for {self.max_steps} steps")
        
        self.model.train()
        start_time = time.time()
        
        # Training loop
        data_iter = iter(self.train_loader)
        
        for step in range(self.step, self.max_steps):
            self.step = step
            
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
                self.epoch += 1
            
            # Forward and backward pass
            loss = self._training_step(batch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            if step % self.log_interval == 0:
                self._log_training_step(step, loss, start_time)
            
            # Evaluation
            if step % self.eval_interval == 0:
                val_loss = self._evaluate()
                self._log_evaluation(step, val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.is_main_process:
                        self._save_best_model()
            
            # Checkpointing
            if step % self.save_interval == 0 and self.is_main_process:
                self._save_checkpoint(step, loss)
        
        # Final evaluation and save
        final_val_loss = self._evaluate()
        if self.is_main_process:
            self._save_checkpoint(self.max_steps, final_val_loss, is_final=True)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        if self.use_wandb:
            wandb.finish()
    
    def _training_step(self, batch) -> float:
        """Single training step with gradient accumulation."""
        total_loss = 0.0
        
        for micro_step in range(self.gradient_accumulation_steps):
            # Get micro-batch
            if isinstance(batch, dict):
                micro_batch = {k: v[micro_step::self.gradient_accumulation_steps] 
                             for k, v in batch.items()}
            else:
                micro_batch = batch[micro_step::self.gradient_accumulation_steps]
            
            # Forward pass with mixed precision
            with self.mixed_precision_trainer.autocast():
                if isinstance(micro_batch, dict):
                    outputs = self.model(**micro_batch)
                else:
                    input_ids = micro_batch.to(self.device)
                    labels = input_ids.clone()
                    outputs = self.model(input_ids, labels=labels)
                
                loss = outputs['loss']
                loss = loss / self.gradient_accumulation_steps  # Scale loss
            
            # Backward pass
            self.mixed_precision_trainer.backward(loss)
            total_loss += loss.item()
        
        # Optimizer step
        if self.grad_clip > 0:
            self.mixed_precision_trainer.clip_gradients(self.model.parameters(), self.grad_clip)
        
        self.mixed_precision_trainer.step(self.optimizer)
        self.mixed_precision_trainer.update()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update token count
        if hasattr(batch, 'size'):
            self.tokens_seen += batch.size(0) * batch.size(1) * self.world_size
        elif isinstance(batch, dict) and 'input_ids' in batch:
            self.tokens_seen += batch['input_ids'].size(0) * batch['input_ids'].size(1) * self.world_size
        
        return total_loss
    
    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate model on validation set."""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        eval_steps = self.config.get('eval_steps', 200)
        
        for i, batch in enumerate(self.val_loader):
            if i >= eval_steps:
                break
            
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
            else:
                input_ids = batch.to(self.device)
                labels = input_ids.clone()
                outputs = self.model(input_ids, labels=labels)
            
            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Gather losses from all processes
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        
        self.model.train()
        return avg_loss
    
    def _log_training_step(self, step: int, loss: float, start_time: float) -> None:
        """Log training metrics."""
        elapsed_time = time.time() - start_time
        tokens_per_sec = self.tokens_seen / elapsed_time if elapsed_time > 0 else 0
        
        lr = self.optimizer.param_groups[0]['lr']
        
        if self.is_main_process:
            # Calculate MFU (Model FLOPS Utilization)
            if hasattr(self.model, 'estimate_mfu') and tokens_per_sec > 0:
                dt = elapsed_time / (step + 1)
                mfu = self.model.estimate_mfu(1, dt)  # Rough estimate
            else:
                mfu = 0.0
            
            logger.info(
                f"Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e} | "
                f"Tokens/sec: {tokens_per_sec:.0f} | MFU: {mfu:.2%}"
            )
            
            # WandB logging
            if self.use_wandb:
                wandb.log({
                    'train/loss': loss,
                    'train/learning_rate': lr,
                    'train/tokens_per_second': tokens_per_sec,
                    'train/mfu': mfu,
                    'train/step': step,
                    'train/tokens_seen': self.tokens_seen,
                }, step=step)
        
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
    
    def _log_evaluation(self, step: int, val_loss: float) -> None:
        """Log evaluation metrics."""
        perplexity = math.exp(min(val_loss, 10))  # Cap for numerical stability
        
        if self.is_main_process:
            logger.info(f"Step {step:6d} | Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
            
            if self.use_wandb:
                wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': perplexity,
                }, step=step)
        
        self.val_losses.append(val_loss)
    
    def _save_checkpoint(self, step: int, loss: float, is_final: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_step_{step}.pt" if not is_final else "final_checkpoint.pt"
        
        # Get model state dict (handle DDP)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'step': step,
            'epoch': self.epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.mixed_precision_trainer.get_scaler_state(),
            'config': model_to_save.config.to_dict(),
            'loss': loss,
            'best_val_loss': self.best_val_loss,
            'tokens_seen': self.tokens_seen,
            'train_losses': self.train_losses[-1000:],  # Keep last 1000 losses
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates[-1000:],
        }
        
        filepath = self.checkpoint_manager.save_checkpoint(checkpoint, checkpoint_name)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def _save_best_model(self) -> None:
        """Save the best model based on validation loss."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'step': self.step,
            'model_state_dict': model_to_save.state_dict(),
            'config': model_to_save.config.to_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        filepath = self.checkpoint_manager.checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, filepath)
        logger.info(f"Best model saved: {filepath}")
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load mixed precision scaler state
        if checkpoint.get('scaler_state_dict'):
            self.mixed_precision_trainer.load_scaler_state(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.step = checkpoint['step']
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.tokens_seen = checkpoint.get('tokens_seen', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        logger.info(f"Resumed from step {self.step}, best val loss: {self.best_val_loss:.4f}")


def setup_distributed_training():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    return rank, world_size, device


def cleanup_distributed_training():
    """Clean up distributed training."""
    if torch.distributed.is_initialized():
        destroy_process_group()