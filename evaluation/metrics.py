#!/usr/bin/env python3
"""
Evaluation metrics and utilities for language model assessment.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class LanguageModelMetrics:
    """Collection of language model evaluation metrics."""
    
    @staticmethod
    def perplexity(logits: torch.Tensor, targets: torch.Tensor, 
                   pad_token_id: Optional[int] = None) -> float:
        """Calculate perplexity from logits and targets."""
        # Flatten logits and targets
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Create mask for non-padding tokens
        if pad_token_id is not None:
            mask = (targets_flat != pad_token_id)
            logits_flat = logits_flat[mask]
            targets_flat = targets_flat[mask]
        
        if targets_flat.numel() == 0:
            return float('inf')
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Convert to perplexity
        return math.exp(loss.item())
    
    @staticmethod
    def bits_per_character(logits: torch.Tensor, targets: torch.Tensor, 
                          text_lengths: List[int],
                          pad_token_id: Optional[int] = None) -> float:
        """Calculate bits per character."""
        total_nll = 0.0
        total_chars = sum(text_lengths)
        
        # Calculate negative log likelihood
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        if pad_token_id is not None:
            mask = (targets_flat != pad_token_id)
            logits_flat = logits_flat[mask]
            targets_flat = targets_flat[mask]
        
        if targets_flat.numel() == 0:
            return float('inf')
        
        log_probs = F.log_softmax(logits_flat, dim=-1)
        nll = F.nll_loss(log_probs, targets_flat, reduction='sum')
        
        total_nll += nll.item()
        
        # Convert to bits per character
        return total_nll / (total_chars * math.log(2))
    
    @staticmethod
    def token_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                      pad_token_id: Optional[int] = None) -> float:
        """Calculate token-level accuracy."""
        predictions = torch.argmax(logits, dim=-1)
        
        if pad_token_id is not None:
            mask = (targets != pad_token_id)
            correct = ((predictions == targets) * mask).sum().item()
            total = mask.sum().item()
        else:
            correct = (predictions == targets).sum().item()
            total = targets.numel()
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5,
                      pad_token_id: Optional[int] = None) -> float:
        """Calculate top-k accuracy."""
        _, top_k_pred = torch.topk(logits, k, dim=-1)
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_pred)
        correct_mask = (top_k_pred == targets_expanded).any(dim=-1)
        
        if pad_token_id is not None:
            pad_mask = (targets != pad_token_id)
            correct = (correct_mask * pad_mask).sum().item()
            total = pad_mask.sum().item()
        else:
            correct = correct_mask.sum().item()
            total = targets.numel()
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def entropy(logits: torch.Tensor) -> float:
        """Calculate average entropy of predictions."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean().item()


class GenerationMetrics:
    """Metrics for evaluating text generation quality."""
    
    @staticmethod
    def repetition_penalty(text: str, n: int = 4) -> float:
        """Calculate n-gram repetition penalty."""
        words = text.split()
        if len(words) < n:
            return 0.0
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
        
        return repeated_ngrams / total_ngrams
    
    @staticmethod
    def diversity_score(texts: List[str], n: int = 3) -> float:
        """Calculate diversity score across generated texts."""
        all_ngrams = set()
        total_ngrams = 0
        
        for text in texts:
            words = text.split()
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                all_ngrams.add(ngram)
                total_ngrams += 1
        
        if total_ngrams == 0:
            return 0.0
        
        return len(all_ngrams) / total_ngrams
    
    @staticmethod
    def self_bleu(texts: List[str], n: int = 4) -> float:
        """Calculate Self-BLEU score to measure diversity."""
        if len(texts) <= 1:
            return 0.0
        
        total_bleu = 0.0
        count = 0
        
        for i, text in enumerate(texts):
            other_texts = texts[:i] + texts[i + 1:]
            bleu_scores = []
            
            for other_text in other_texts:
                bleu = GenerationMetrics._simple_bleu(text, other_text, n)
                bleu_scores.append(bleu)
            
            if bleu_scores:
                total_bleu += max(bleu_scores)
                count += 1
        
        return total_bleu / count if count > 0 else 0.0
    
    @staticmethod
    def _simple_bleu(candidate: str, reference: str, n: int = 4) -> float:
        """Simple BLEU score implementation."""
        candidate_words = candidate.split()
        reference_words = reference.split()
        
        if len(candidate_words) == 0 or len(reference_words) == 0:
            return 0.0
        
        # Calculate n-gram precision for each n
        precisions = []
        
        for i in range(1, n + 1):
            candidate_ngrams = []
            reference_ngrams = []
            
            # Extract n-grams
            for j in range(len(candidate_words) - i + 1):
                candidate_ngrams.append(' '.join(candidate_words[j:j + i]))
            
            for j in range(len(reference_words) - i + 1):
                reference_ngrams.append(' '.join(reference_words[j:j + i]))
            
            if not candidate_ngrams:
                precisions.append(0.0)
                continue
            
            # Count matches
            candidate_counter = Counter(candidate_ngrams)
            reference_counter = Counter(reference_ngrams)
            
            matches = 0
            for ngram, count in candidate_counter.items():
                matches += min(count, reference_counter.get(ngram, 0))
            
            precision = matches / len(candidate_ngrams)
            precisions.append(precision)
        
        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0
        
        bleu = np.exp(np.mean(np.log(precisions)))
        
        # Brevity penalty
        brevity_penalty = min(1.0, len(candidate_words) / len(reference_words))
        
        return bleu * brevity_penalty


class CalibrationMetrics:
    """Metrics for evaluating model calibration."""
    
    @staticmethod
    def expected_calibration_error(logits: torch.Tensor, targets: torch.Tensor,
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
    
    @staticmethod
    def reliability_diagram_data(logits: torch.Tensor, targets: torch.Tensor,
                               num_bins: int = 10) -> Tuple[List[float], List[float], List[int]]:
        """Generate data for reliability diagram."""
        probs = F.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1)[0]
        predictions = probs.argmax(dim=-1)
        accuracies = (predictions == targets).float()
        
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = accuracies[in_bin].mean().item()
                bin_confidence = confidences[in_bin].mean().item()
                bin_count = in_bin.sum().item()
            else:
                bin_accuracy = 0.0
                bin_confidence = (bin_lower + bin_upper).item() / 2
                bin_count = 0
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
        
        return bin_accuracies, bin_confidences, bin_counts


class LossAnalysis:
    """Analysis of training and validation losses."""
    
    @staticmethod
    def loss_components(logits: torch.Tensor, targets: torch.Tensor,
                       vocab_size: int) -> Dict[str, float]:
        """Analyze loss components."""
        # Basic cross-entropy loss
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                targets.view(-1))
        
        # Uniform baseline (random predictions)
        uniform_loss = math.log(vocab_size)
        
        # Unigram baseline (based on target distribution)
        target_counts = torch.bincount(targets.view(-1), minlength=vocab_size).float()
        target_probs = target_counts / target_counts.sum()
        unigram_loss = -(target_probs * torch.log(target_probs + 1e-10)).sum()
        
        return {
            'cross_entropy_loss': ce_loss.item(),
            'uniform_baseline': uniform_loss,
            'unigram_baseline': unigram_loss.item(),
            'improvement_over_uniform': uniform_loss - ce_loss.item(),
            'improvement_over_unigram': unigram_loss.item() - ce_loss.item()
        }
    
    @staticmethod
    def loss_by_position(logits: torch.Tensor, targets: torch.Tensor,
                        max_positions: int = 100) -> List[float]:
        """Calculate loss by sequence position."""
        batch_size, seq_len, vocab_size = logits.shape
        
        position_losses = []
        
        for pos in range(min(seq_len - 1, max_positions)):
            pos_logits = logits[:, pos, :]  # Shape: (batch_size, vocab_size)
            pos_targets = targets[:, pos + 1]  # Shape: (batch_size,)
            
            loss = F.cross_entropy(pos_logits, pos_targets)
            position_losses.append(loss.item())
        
        return position_losses
    
    @staticmethod
    def loss_by_frequency(logits: torch.Tensor, targets: torch.Tensor,
                         token_frequencies: Dict[int, int]) -> Dict[str, float]:
        """Calculate loss by token frequency buckets."""
        # Create frequency buckets
        all_freqs = list(token_frequencies.values())
        freq_percentiles = np.percentile(all_freqs, [25, 50, 75])
        
        buckets = {
            'rare': [],      # Bottom 25%
            'uncommon': [],  # 25-50%
            'common': [],    # 50-75%
            'frequent': []   # Top 25%
        }
        
        # Flatten logits and targets
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Categorize each target token
        for i, target_token in enumerate(targets_flat):
            token_id = target_token.item()
            freq = token_frequencies.get(token_id, 0)
            
            if freq <= freq_percentiles[0]:
                bucket = 'rare'
            elif freq <= freq_percentiles[1]:
                bucket = 'uncommon'
            elif freq <= freq_percentiles[2]:
                bucket = 'common'
            else:
                bucket = 'frequent'
            
            buckets[bucket].append(i)
        
        # Calculate loss for each bucket
        bucket_losses = {}
        for bucket_name, indices in buckets.items():
            if indices:
                bucket_logits = logits_flat[indices]
                bucket_targets = targets_flat[indices]
                bucket_loss = F.cross_entropy(bucket_logits, bucket_targets)
                bucket_losses[bucket_name] = bucket_loss.item()
            else:
                bucket_losses[bucket_name] = float('nan')
        
        return bucket_losses


def compute_comprehensive_metrics(model_outputs: Dict[str, torch.Tensor],
                                targets: torch.Tensor,
                                pad_token_id: Optional[int] = None,
                                text_lengths: Optional[List[int]] = None,
                                token_frequencies: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    
    logits = model_outputs['logits'][:, :-1].contiguous()  # Exclude last token
    targets = targets[:, 1:].contiguous()  # Exclude first token
    
    metrics = {}
    
    # Basic metrics
    metrics['perplexity'] = LanguageModelMetrics.perplexity(logits, targets, pad_token_id)
    metrics['token_accuracy'] = LanguageModelMetrics.token_accuracy(logits, targets, pad_token_id)
    metrics['top5_accuracy'] = LanguageModelMetrics.top_k_accuracy(logits, targets, k=5, pad_token_id=pad_token_id)
    metrics['entropy'] = LanguageModelMetrics.entropy(logits)
    
    # Calibration metrics
    metrics['expected_calibration_error'] = CalibrationMetrics.expected_calibration_error(
        logits.view(-1, logits.size(-1)), targets.view(-1)
    )
    
    # Loss analysis
    if token_frequencies:
        metrics['loss_by_frequency'] = LossAnalysis.loss_by_frequency(
            logits, targets, token_frequencies
        )
    
    metrics['loss_components'] = LossAnalysis.loss_components(
        logits, targets, logits.size(-1)
    )
    
    # Position analysis
    metrics['loss_by_position'] = LossAnalysis.loss_by_position(logits, targets)
    
    # Bits per character (if text lengths provided)
    if text_lengths:
        metrics['bits_per_character'] = LanguageModelMetrics.bits_per_character(
            logits, targets, text_lengths, pad_token_id
        )
    
    return metrics


def compare_model_metrics(metrics_list: List[Dict[str, Any]], 
                         model_names: List[str]) -> Dict[str, Any]:
    """Compare metrics across multiple models."""
    
    comparison = {
        'model_names': model_names,
        'metrics_comparison': defaultdict(dict)
    }
    
    # Extract common metrics
    common_metrics = set(metrics_list[0].keys())
    for metrics in metrics_list[1:]:
        common_metrics &= set(metrics.keys())
    
    for metric_name in common_metrics:
        for i, (model_name, metrics) in enumerate(zip(model_names, metrics_list)):
            if isinstance(metrics[metric_name], (int, float)):
                comparison['metrics_comparison'][metric_name][model_name] = metrics[metric_name]
    
    # Find best model for each metric
    comparison['best_models'] = {}
    for metric_name, model_scores in comparison['metrics_comparison'].items():
        if model_scores:
            # Determine if lower is better (for loss-like metrics)
            lower_is_better = metric_name in ['perplexity', 'loss', 'expected_calibration_error', 'bits_per_character']
            
            if lower_is_better:
                best_model = min(model_scores.items(), key=lambda x: x[1])
            else:
                best_model = max(model_scores.items(), key=lambda x: x[1])
            
            comparison['best_models'][metric_name] = best_model
    
    return comparison


def print_metrics_summary(metrics: Dict[str, Any], model_name: str = "Model"):
    """Print a formatted summary of metrics."""
    print(f"\n{model_name} Evaluation Metrics")
    print("=" * 50)
    
    # Basic metrics
    if 'perplexity' in metrics:
        print(f"Perplexity: {metrics['perplexity']:.2f}")
    if 'token_accuracy' in metrics:
        print(f"Token Accuracy: {metrics['token_accuracy']:.4f}")
    if 'top5_accuracy' in metrics:
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    if 'entropy' in metrics:
        print(f"Entropy: {metrics['entropy']:.4f}")
    if 'expected_calibration_error' in metrics:
        print(f"Calibration Error: {metrics['expected_calibration_error']:.4f}")
    
    # Loss components
    if 'loss_components' in metrics:
        print("\nLoss Analysis:")
        components = metrics['loss_components']
        print(f"  Cross-entropy Loss: {components['cross_entropy_loss']:.4f}")
        print(f"  Improvement over Random: {components['improvement_over_uniform']:.4f}")
        print(f"  Improvement over Unigram: {components['improvement_over_unigram']:.4f}")
    
    # Frequency-based analysis
    if 'loss_by_frequency' in metrics:
        print("\nLoss by Token Frequency:")
        freq_losses = metrics['loss_by_frequency']
        for bucket, loss in freq_losses.items():
            if not math.isnan(loss):
                print(f"  {bucket.capitalize()}: {loss:.4f}")
    
    print("=" * 50)