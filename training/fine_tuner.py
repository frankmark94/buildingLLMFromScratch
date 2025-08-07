#!/usr/bin/env python3
"""
Fine-tuning system for neural LLM models.
Supports task-specific adaptation with custom datasets.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

from .trainer import Trainer
from ..model import GPTModel, ModelConfig
from ..data.tokenizer import LLMTokenizer
from ..data.data_loader import DataLoader


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    # Model and data
    model_path: str
    dataset_path: str
    tokenizer_path: str = "data/tokenizer"
    output_dir: str = "fine_tuned_models"
    
    # Fine-tuning parameters
    learning_rate: float = 1e-5  # Lower than pre-training
    warmup_ratio: float = 0.1
    num_epochs: int = 3
    max_seq_length: int = 512
    
    # Data parameters
    train_split: float = 0.9
    text_column: str = "text"
    label_column: Optional[str] = None  # For classification tasks
    
    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Regularization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    
    # Task type
    task_type: str = "text_generation"  # text_generation, classification, qa
    num_labels: Optional[int] = None  # For classification
    
    # Memory optimization
    use_amp: bool = True
    gradient_checkpointing: bool = True
    freeze_embeddings: bool = False
    freeze_layers: Optional[List[int]] = None  # Layer indices to freeze


class FineTuner:
    """Fine-tuning system for neural LLMs."""
    
    def __init__(self, config: FineTuningConfig, device: torch.device):
        """Initialize fine-tuner."""
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_base_model(self) -> None:
        """Load pre-trained base model."""
        self.logger.info(f"Loading base model from {self.config.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        
        # Load model config
        if 'config' in checkpoint:
            model_config = ModelConfig.from_dict(checkpoint['config'])
        else:
            # Default config
            model_config = ModelConfig()
        
        # Adjust for fine-tuning task
        if self.config.task_type == "classification" and self.config.num_labels:
            model_config.vocab_size = self.config.num_labels
        
        # Create model
        self.model = GPTModel(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Apply fine-tuning modifications
        self._modify_model_for_finetuning()
        
        self.model = self.model.to(self.device)
        self.logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _modify_model_for_finetuning(self) -> None:
        """Modify model architecture for fine-tuning task."""
        if self.config.task_type == "classification":
            # Replace language modeling head with classification head
            if hasattr(self.model, 'lm_head'):
                in_features = self.model.lm_head.in_features
                self.model.lm_head = nn.Linear(in_features, self.config.num_labels)
                nn.init.normal_(self.model.lm_head.weight, std=0.02)
        
        # Apply freezing
        if self.config.freeze_embeddings:
            if hasattr(self.model, 'wte'):
                self.model.wte.weight.requires_grad = False
            if hasattr(self.model, 'wpe'):
                self.model.wpe.weight.requires_grad = False
        
        if self.config.freeze_layers:
            for layer_idx in self.config.freeze_layers:
                if hasattr(self.model, 'h') and layer_idx < len(self.model.h):
                    for param in self.model.h[layer_idx].parameters():
                        param.requires_grad = False
        
        # Apply gradient checkpointing
        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def load_tokenizer(self) -> None:
        """Load tokenizer."""
        self.logger.info(f"Loading tokenizer from {self.config.tokenizer_path}")
        self.tokenizer = LLMTokenizer()
        # Load tokenizer from path if it exists
        if Path(self.config.tokenizer_path).exists():
            self.tokenizer.load_tokenizer(self.config.tokenizer_path)
    
    def prepare_dataset(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Prepare training and validation datasets."""
        self.logger.info(f"Loading dataset from {self.config.dataset_path}")
        
        # Load dataset
        if self.config.dataset_path.endswith('.jsonl'):
            data = []
            with open(self.config.dataset_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        elif self.config.dataset_path.endswith('.json'):
            with open(self.config.dataset_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported dataset format: {self.config.dataset_path}")
        
        # Split data
        split_idx = int(len(data) * self.config.train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        self.logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} validation")
        
        # Create datasets
        train_dataset = self._create_dataset(train_data)
        val_dataset = self._create_dataset(val_data)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _create_dataset(self, data: List[Dict]) -> torch.utils.data.Dataset:
        """Create dataset for fine-tuning task."""
        if self.config.task_type == "text_generation":
            return TextGenerationDataset(
                data, self.tokenizer, self.config.text_column, self.config.max_seq_length
            )
        elif self.config.task_type == "classification":
            return ClassificationDataset(
                data, self.tokenizer, self.config.text_column, 
                self.config.label_column, self.config.max_seq_length
            )
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
    
    def setup_optimizer(self, train_loader) -> None:
        """Setup optimizer and scheduler."""
        # Calculate total steps
        total_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Setup optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        self.logger.info(f"Optimizer setup: {total_steps} total steps, {warmup_steps} warmup steps")
    
    def fine_tune(self) -> None:
        """Run fine-tuning process."""
        self.logger.info("Starting fine-tuning process")
        
        # Load components
        self.load_base_model()
        self.load_tokenizer()
        
        # Prepare data
        train_loader, val_loader = self.prepare_dataset()
        
        # Setup training
        self.setup_optimizer(train_loader)
        
        # Setup AMP if enabled
        scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None
        
        # Training loop
        best_val_loss = float('inf')
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._compute_loss(batch)
                        loss = outputs['loss'] / self.config.gradient_accumulation_steps
                else:
                    outputs = self._compute_loss(batch)
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                
                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_amp:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        self.logger.info(
                            f"Step {global_step}, Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f}, "
                            f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                        )
                    
                    # Evaluation
                    if global_step % self.config.eval_steps == 0:
                        val_loss = self._evaluate(val_loader)
                        self.logger.info(f"Validation loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self._save_model(f"best_model_step_{global_step}")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self._save_model(f"checkpoint_step_{global_step}")
                
                num_batches += 1
            
            # End of epoch
            avg_loss = total_loss / num_batches
            val_loss = self._evaluate(val_loader)
            
            self.logger.info(
                f"Epoch {epoch + 1} complete - Train Loss: {avg_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
        
        # Save final model
        self._save_model("final_model")
        self.logger.info("Fine-tuning completed!")
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for the current batch."""
        if self.config.task_type == "text_generation":
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Compute language modeling loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return {'loss': loss, 'logits': outputs.logits}
        
        elif self.config.task_type == "classification":
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')
            labels = batch['labels']
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Classification loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.logits, labels)
            
            return {'loss': loss, 'logits': outputs.logits}
    
    def _evaluate(self, val_loader) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self._compute_loss(batch)
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _save_model(self, name: str) -> None:
        """Save model checkpoint."""
        output_path = Path(self.config.output_dir) / f"{name}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'model_config': self.model.config.__dict__,
        }
        
        torch.save(checkpoint, output_path)
        self.logger.info(f"Model saved to {output_path}")


class TextGenerationDataset(torch.utils.data.Dataset):
    """Dataset for text generation fine-tuning."""
    
    def __init__(self, data: List[Dict], tokenizer: LLMTokenizer, 
                 text_column: str, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item[self.text_column]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        # Create attention mask
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in tokens]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset for classification fine-tuning."""
    
    def __init__(self, data: List[Dict], tokenizer: LLMTokenizer, 
                 text_column: str, label_column: str, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item[self.text_column]
        label = item[self.label_column]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        # Create attention mask
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in tokens]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }