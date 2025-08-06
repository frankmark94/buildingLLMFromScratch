#!/usr/bin/env python3
"""
Perplexity calculation and language modeling evaluation metrics.
"""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """Evaluator for calculating perplexity and related metrics."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                 use_amp: bool = False):
        self.model = model
        self.device = device
        self.use_amp = use_amp
        
    @torch.no_grad()
    def calculate_perplexity(self, dataloader: DataLoader, 
                           max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate perplexity on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            max_batches: Maximum number of batches to evaluate (None for all)
            
        Returns:
            Dictionary with perplexity metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        total_batches = 0
        
        # Track token-level accuracy
        correct_predictions = 0
        total_predictions = 0
        
        for i, batch in enumerate(tqdm(dataloader, desc="Calculating perplexity")):
            if max_batches is not None and i >= max_batches:
                break
            
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_ids = batch['input_ids']
                labels = batch.get('labels', input_ids)
            else:
                input_ids = batch.to(self.device)
                labels = input_ids.clone()
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, labels=labels)
            else:
                outputs = self.model(input_ids, labels=labels)
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Accumulate loss
            batch_size, seq_len = input_ids.shape
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
            total_batches += 1
            
            # Calculate token accuracy
            predictions = torch.argmax(logits[:, :-1], dim=-1)
            targets = labels[:, 1:]
            
            # Mask out padding tokens if present
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'pad_token_id'):
                pad_token_id = self.model.config.pad_token_id
                mask = (targets != pad_token_id)
                correct_predictions += ((predictions == targets) * mask).sum().item()
                total_predictions += mask.sum().item()
            else:
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += predictions.numel()
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        # Calculate bits per character (approximate)
        bits_per_token = avg_loss / math.log(2)
        
        # Token accuracy
        token_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        metrics = {
            'perplexity': perplexity,
            'loss': avg_loss,
            'bits_per_token': bits_per_token,
            'token_accuracy': token_accuracy,
            'total_tokens': total_tokens,
            'total_batches': total_batches
        }
        
        logger.info(f"Perplexity: {perplexity:.2f}, Loss: {avg_loss:.4f}, "
                   f"Token Accuracy: {token_accuracy:.4f}")
        
        return metrics
    
    @torch.no_grad()
    def calculate_perplexity_by_length(self, dataloader: DataLoader,
                                     length_buckets: List[int] = [128, 256, 512, 1024]) -> Dict[str, Dict[str, float]]:
        """Calculate perplexity for different sequence lengths."""
        self.model.eval()
        
        # Initialize buckets
        buckets = {f"length_{bucket}": {'total_loss': 0.0, 'total_tokens': 0, 'count': 0} 
                  for bucket in length_buckets}
        buckets['other'] = {'total_loss': 0.0, 'total_tokens': 0, 'count': 0}
        
        for batch in tqdm(dataloader, desc="Calculating length-based perplexity"):
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_ids = batch['input_ids']
                labels = batch.get('labels', input_ids)
            else:
                input_ids = batch.to(self.device)
                labels = input_ids.clone()
            
            seq_len = input_ids.size(1)
            
            # Find appropriate bucket
            bucket_key = 'other'
            for bucket_len in length_buckets:
                if seq_len <= bucket_len:
                    bucket_key = f"length_{bucket_len}"
                    break
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, labels=labels)
            else:
                outputs = self.model(input_ids, labels=labels)
            
            loss = outputs['loss']
            batch_size = input_ids.size(0)
            
            # Update bucket
            buckets[bucket_key]['total_loss'] += loss.item() * batch_size * seq_len
            buckets[bucket_key]['total_tokens'] += batch_size * seq_len
            buckets[bucket_key]['count'] += 1
        
        # Calculate perplexity for each bucket
        results = {}
        for bucket_key, bucket_data in buckets.items():
            if bucket_data['total_tokens'] > 0:
                avg_loss = bucket_data['total_loss'] / bucket_data['total_tokens']
                perplexity = math.exp(avg_loss)
                
                results[bucket_key] = {
                    'perplexity': perplexity,
                    'loss': avg_loss,
                    'total_tokens': bucket_data['total_tokens'],
                    'count': bucket_data['count']
                }
        
        return results


class LanguageModelingMetrics:
    """Additional language modeling evaluation metrics."""
    
    @staticmethod
    def calculate_entropy(logits: torch.Tensor) -> float:
        """Calculate average entropy of predictions."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy.item()
    
    @staticmethod
    def calculate_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, 
                               k: int = 5) -> float:
        """Calculate top-k accuracy."""
        _, top_k_pred = torch.topk(logits, k, dim=-1)
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_pred)
        correct = (top_k_pred == targets_expanded).any(dim=-1)
        return correct.float().mean().item()
    
    @staticmethod 
    def calculate_calibration_error(logits: torch.Tensor, targets: torch.Tensor,
                                  num_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        probs = F.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1)[0]
        predictions = probs.argmax(dim=-1)
        accuracies = (predictions == targets).float()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean().item()
                avg_confidence_in_bin = confidences[in_bin].mean().item()
                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


def evaluate_model_comprehensive(model: torch.nn.Module, 
                                eval_dataloader: DataLoader,
                                device: torch.device,
                                use_amp: bool = False,
                                max_batches: Optional[int] = None) -> Dict[str, Any]:
    """Comprehensive model evaluation with multiple metrics."""
    
    evaluator = PerplexityEvaluator(model, device, use_amp)
    
    # Basic perplexity metrics
    perplexity_metrics = evaluator.calculate_perplexity(eval_dataloader, max_batches)
    
    # Additional detailed metrics
    model.eval()
    total_entropy = 0.0
    total_top5_acc = 0.0
    total_calibration_error = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader, desc="Detailed evaluation")):
            if max_batches is not None and i >= max_batches:
                break
            
            # Prepare batch
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch['input_ids']
                labels = batch.get('labels', input_ids)
            else:
                input_ids = batch.to(device)
                labels = input_ids.clone()
            
            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, labels=labels)
            else:
                outputs = model(input_ids, labels=labels)
            
            logits = outputs['logits'][:, :-1].contiguous()  # Exclude last token
            targets = labels[:, 1:].contiguous()  # Exclude first token
            
            # Flatten for metric calculations
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            # Calculate additional metrics
            entropy = LanguageModelingMetrics.calculate_entropy(logits_flat)
            top5_acc = LanguageModelingMetrics.calculate_top_k_accuracy(logits_flat, targets_flat, k=5)
            calibration_error = LanguageModelingMetrics.calculate_calibration_error(logits_flat, targets_flat)
            
            total_entropy += entropy
            total_top5_acc += top5_acc
            total_calibration_error += calibration_error
            num_batches += 1
    
    # Combine all metrics
    comprehensive_metrics = {
        **perplexity_metrics,
        'entropy': total_entropy / num_batches,
        'top5_accuracy': total_top5_acc / num_batches,
        'calibration_error': total_calibration_error / num_batches
    }
    
    return comprehensive_metrics


def compare_model_performance(models: Dict[str, torch.nn.Module],
                            eval_dataloader: DataLoader,
                            device: torch.device,
                            use_amp: bool = False) -> Dict[str, Dict[str, Any]]:
    """Compare performance of multiple models on the same dataset."""
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        
        # Move model to device
        model = model.to(device)
        
        # Evaluate
        metrics = evaluate_model_comprehensive(model, eval_dataloader, device, use_amp)
        results[model_name] = metrics
        
        logger.info(f"{model_name} - Perplexity: {metrics['perplexity']:.2f}, "
                   f"Loss: {metrics['loss']:.4f}")
    
    return results


def perplexity_over_training(model: torch.nn.Module,
                           eval_dataloader: DataLoader,
                           checkpoint_paths: List[str],
                           device: torch.device) -> Dict[str, List[float]]:
    """Calculate perplexity progression over training checkpoints."""
    
    perplexities = []
    losses = []
    steps = []
    
    evaluator = PerplexityEvaluator(model, device)
    
    for checkpoint_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        metrics = evaluator.calculate_perplexity(eval_dataloader)
        
        perplexities.append(metrics['perplexity'])
        losses.append(metrics['loss'])
        steps.append(checkpoint.get('step', 0))
    
    return {
        'steps': steps,
        'perplexities': perplexities,
        'losses': losses
    }


def calculate_bits_per_byte(model: torch.nn.Module,
                          text_data: List[str],
                          tokenizer,
                          device: torch.device) -> float:
    """Calculate bits per byte for text compression evaluation."""
    
    model.eval()
    total_bits = 0.0
    total_bytes = 0
    
    with torch.no_grad():
        for text in tqdm(text_data, desc="Calculating bits per byte"):
            # Tokenize
            if hasattr(tokenizer, 'encode'):
                tokens = tokenizer.encode(text)
            else:
                tokens = tokenizer.tokenize_text(text)
            
            # Convert to tensor
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            
            if input_ids.size(1) <= 1:
                continue
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs['logits'][0, :-1]  # Exclude last position
            targets = input_ids[0, 1:]  # Exclude first position
            
            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Convert to bits (log base 2)
            bits = -token_log_probs.sum().item() / math.log(2)
            
            total_bits += bits
            total_bytes += len(text.encode('utf-8'))
    
    return total_bits / total_bytes if total_bytes > 0 else float('inf')