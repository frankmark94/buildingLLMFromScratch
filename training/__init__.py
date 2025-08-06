#!/usr/bin/env python3
"""
Training package for neural LLM.
Contains trainer, optimizer, mixed precision, and checkpointing utilities.
"""

from .trainer import Trainer, setup_distributed_training, cleanup_distributed_training
from .optimizer import (
    AdamW, CosineAnnealingWithWarmupLR, LinearWarmupCosineAnnealingLR, InverseSqrtLR,
    create_optimizer, create_scheduler, GradientClipping, 
    print_optimizer_info, save_optimizer_checkpoint, load_optimizer_checkpoint
)
from .mixed_precision import (
    MixedPrecisionTrainer, ManualMixedPrecision, AutocastContext,
    GradientAccumulator, MemoryEfficientAttention, DynamicLossScaling,
    check_mixed_precision_support, print_mixed_precision_info
)
from .checkpointing import (
    CheckpointManager, ModelCheckpoint, find_latest_checkpoint,
    compare_checkpoints, migrate_checkpoint_format
)

__all__ = [
    # Main trainer
    'Trainer',
    'setup_distributed_training', 
    'cleanup_distributed_training',
    
    # Optimizers and schedulers
    'AdamW',
    'CosineAnnealingWithWarmupLR',
    'LinearWarmupCosineAnnealingLR', 
    'InverseSqrtLR',
    'create_optimizer',
    'create_scheduler',
    'GradientClipping',
    'print_optimizer_info',
    'save_optimizer_checkpoint',
    'load_optimizer_checkpoint',
    
    # Mixed precision
    'MixedPrecisionTrainer',
    'ManualMixedPrecision',
    'AutocastContext',
    'GradientAccumulator',
    'MemoryEfficientAttention', 
    'DynamicLossScaling',
    'check_mixed_precision_support',
    'print_mixed_precision_info',
    
    # Checkpointing
    'CheckpointManager',
    'ModelCheckpoint',
    'find_latest_checkpoint',
    'compare_checkpoints',
    'migrate_checkpoint_format',
]