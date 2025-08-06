#!/usr/bin/env python3
"""
Checkpointing utilities for model training and inference.
Handles saving, loading, and managing model checkpoints.
"""

import os
import glob
import torch
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup and best model tracking."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3, 
                 save_best: bool = True, metric_name: str = 'val_loss',
                 metric_mode: str = 'min'):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best checkpoint separately
            metric_name: Metric name for best checkpoint selection
            metric_mode: 'min' or 'max' for best metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        self.best_metric = float('inf') if metric_mode == 'min' else float('-inf')
        self.checkpoint_history = []
        
        logger.info(f"CheckpointManager initialized at {checkpoint_dir}")
    
    def save_checkpoint(self, state: Dict[str, Any], filename: str, 
                       metric_value: Optional[float] = None) -> str:
        """
        Save a checkpoint and manage cleanup.
        
        Args:
            state: State dictionary to save
            filename: Filename for the checkpoint
            metric_value: Value of the tracking metric
            
        Returns:
            Path to saved checkpoint
        """
        # Add timestamp to state
        state['timestamp'] = datetime.now().isoformat()
        state['checkpoint_filename'] = filename
        
        # Save checkpoint
        filepath = self.checkpoint_dir / filename
        torch.save(state, filepath)
        
        # Update checkpoint history
        checkpoint_info = {
            'filepath': str(filepath),
            'filename': filename,
            'step': state.get('step', 0),
            'metric_value': metric_value,
            'timestamp': state['timestamp']
        }
        self.checkpoint_history.append(checkpoint_info)
        
        # Save best checkpoint if applicable
        if self.save_best and metric_value is not None:
            is_best = self._is_best_metric(metric_value)
            if is_best:
                self.best_metric = metric_value
                best_path = self.checkpoint_dir / 'best_checkpoint.pt'
                shutil.copy2(filepath, best_path)
                logger.info(f"New best checkpoint saved: {best_path} (metric: {metric_value})")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # Save checkpoint history
        self._save_checkpoint_history()
        
        logger.info(f"Checkpoint saved: {filepath}")
        return str(filepath)
    
    def load_checkpoint(self, checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
        """Load checkpoint from file."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        return checkpoint
    
    def load_best_checkpoint(self, device: str = 'cpu') -> Optional[Dict[str, Any]]:
        """Load the best checkpoint if it exists."""
        best_path = self.checkpoint_dir / 'best_checkpoint.pt'
        if best_path.exists():
            return self.load_checkpoint(str(best_path), device)
        return None
    
    def load_latest_checkpoint(self, device: str = 'cpu') -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        if not self.checkpoint_history:
            self._load_checkpoint_history()
        
        if self.checkpoint_history:
            latest = max(self.checkpoint_history, key=lambda x: x['step'])
            return self.load_checkpoint(latest['filepath'], device)
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        if not self.checkpoint_history:
            self._load_checkpoint_history()
        
        return sorted(self.checkpoint_history, key=lambda x: x['step'])
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            info = {
                'filepath': checkpoint_path,
                'step': checkpoint.get('step', 0),
                'epoch': checkpoint.get('epoch', 0),
                'loss': checkpoint.get('loss', 0.0),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'config': checkpoint.get('config', {}),
                'tokens_seen': checkpoint.get('tokens_seen', 0),
            }
            
            # Add model parameters count if available
            if 'model_state_dict' in checkpoint:
                total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
                info['total_parameters'] = total_params
            
            return info
            
        except Exception as e:
            logger.error(f"Error loading checkpoint info: {e}")
            return None
    
    def delete_checkpoint(self, checkpoint_path: str):
        """Delete a specific checkpoint."""
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            
            # Remove from history
            self.checkpoint_history = [
                cp for cp in self.checkpoint_history 
                if cp['filepath'] != checkpoint_path
            ]
            self._save_checkpoint_history()
            
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
    
    def cleanup_all_checkpoints(self):
        """Delete all checkpoints in the directory."""
        checkpoint_files = glob.glob(str(self.checkpoint_dir / "*.pt"))
        for filepath in checkpoint_files:
            os.remove(filepath)
        
        self.checkpoint_history = []
        self._save_checkpoint_history()
        
        logger.info(f"Cleaned up all checkpoints in {self.checkpoint_dir}")
    
    def _is_best_metric(self, metric_value: float) -> bool:
        """Check if the current metric is the best so far."""
        if self.metric_mode == 'min':
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        # Sort by step (keep most recent)
        sorted_checkpoints = sorted(
            self.checkpoint_history, 
            key=lambda x: x['step'], 
            reverse=True
        )
        
        # Remove old checkpoints
        if len(sorted_checkpoints) > self.max_checkpoints:
            to_remove = sorted_checkpoints[self.max_checkpoints:]
            
            for checkpoint_info in to_remove:
                filepath = checkpoint_info['filepath']
                if os.path.exists(filepath):
                    # Don't delete best checkpoint
                    best_path = str(self.checkpoint_dir / 'best_checkpoint.pt')
                    if os.path.samefile(filepath, best_path):
                        continue
                    
                    os.remove(filepath)
                    logger.debug(f"Removed old checkpoint: {filepath}")
            
            # Update history
            self.checkpoint_history = sorted_checkpoints[:self.max_checkpoints]
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to JSON file."""
        history_file = self.checkpoint_dir / 'checkpoint_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from JSON file."""
        history_file = self.checkpoint_dir / 'checkpoint_history.json'
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.checkpoint_history = json.load(f)
                
                # Filter out checkpoints that no longer exist
                existing_checkpoints = []
                for cp in self.checkpoint_history:
                    if os.path.exists(cp['filepath']):
                        existing_checkpoints.append(cp)
                
                self.checkpoint_history = existing_checkpoints
                
            except Exception as e:
                logger.error(f"Error loading checkpoint history: {e}")
                self.checkpoint_history = []
        
        # Also scan directory for any checkpoints not in history
        self._scan_directory_for_checkpoints()
    
    def _scan_directory_for_checkpoints(self):
        """Scan directory for checkpoint files and add to history if missing."""
        checkpoint_files = glob.glob(str(self.checkpoint_dir / "checkpoint_step_*.pt"))
        
        existing_files = {cp['filepath'] for cp in self.checkpoint_history}
        
        for filepath in checkpoint_files:
            if filepath not in existing_files:
                # Extract step from filename
                filename = os.path.basename(filepath)
                try:
                    step = int(filename.split('checkpoint_step_')[1].split('.pt')[0])
                except:
                    step = 0
                
                checkpoint_info = {
                    'filepath': filepath,
                    'filename': filename,
                    'step': step,
                    'metric_value': None,
                    'timestamp': 'unknown'
                }
                self.checkpoint_history.append(checkpoint_info)


class ModelCheckpoint:
    """Simple checkpoint saving utility."""
    
    @staticmethod
    def save_model(model: torch.nn.Module, filepath: str, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None,
                   step: int = 0, loss: float = 0.0,
                   metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint with optional training state."""
        
        # Get model state (handle DDP)
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'step': step,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add model config if available
        if hasattr(model, 'config'):
            checkpoint['config'] = model.config.to_dict()
        elif hasattr(model, 'module') and hasattr(model.module, 'config'):
            checkpoint['config'] = model.module.config.to_dict()
        
        # Add optimizer state
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add metadata
        if metadata is not None:
            checkpoint.update(metadata)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint saved: {filepath}")
    
    @staticmethod
    def load_model(model: torch.nn.Module, filepath: str, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None,
                   device: str = 'cpu',
                   strict: bool = True) -> Dict[str, Any]:
        """Load model checkpoint and optionally training state."""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        logger.info(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        model_state = checkpoint['model_state_dict']
        
        # Handle loading into DDP model
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state, strict=strict)
        else:
            model.load_state_dict(model_state, strict=strict)
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded successfully from step {checkpoint.get('step', 0)}")
        
        return checkpoint


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "checkpoint_step_*.pt") -> Optional[str]:
    """Find the latest checkpoint file in a directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        return None
    
    # Extract step numbers and find the maximum
    def extract_step(filepath):
        filename = os.path.basename(filepath)
        try:
            return int(filename.split('checkpoint_step_')[1].split('.pt')[0])
        except:
            return 0
    
    latest_file = max(checkpoint_files, key=extract_step)
    return latest_file


def compare_checkpoints(checkpoint_paths: List[str]) -> Dict[str, Any]:
    """Compare multiple checkpoints and return summary information."""
    comparison = {
        'checkpoints': [],
        'best_checkpoint': None,
        'latest_checkpoint': None,
        'total_checkpoints': len(checkpoint_paths)
    }
    
    best_loss = float('inf')
    latest_step = -1
    
    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            info = {
                'path': path,
                'step': checkpoint.get('step', 0),
                'loss': checkpoint.get('loss', 0.0),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
                'tokens_seen': checkpoint.get('tokens_seen', 0),
            }
            
            comparison['checkpoints'].append(info)
            
            # Track best and latest
            if info['loss'] < best_loss:
                best_loss = info['loss']
                comparison['best_checkpoint'] = info
            
            if info['step'] > latest_step:
                latest_step = info['step']
                comparison['latest_checkpoint'] = info
                
        except Exception as e:
            logger.error(f"Error loading checkpoint {path}: {e}")
    
    return comparison


def migrate_checkpoint_format(old_checkpoint_path: str, new_checkpoint_path: str,
                             config_updates: Optional[Dict[str, Any]] = None):
    """Migrate checkpoint from old format to new format."""
    logger.info(f"Migrating checkpoint from {old_checkpoint_path} to {new_checkpoint_path}")
    
    # Load old checkpoint
    old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    
    # Create new checkpoint with updated format
    new_checkpoint = {
        'model_state_dict': old_checkpoint.get('model_state_dict', {}),
        'optimizer_state_dict': old_checkpoint.get('optimizer_state_dict', {}),
        'scheduler_state_dict': old_checkpoint.get('scheduler_state_dict', {}),
        'step': old_checkpoint.get('step', 0),
        'epoch': old_checkpoint.get('epoch', 0),
        'loss': old_checkpoint.get('loss', 0.0),
        'best_val_loss': old_checkpoint.get('best_val_loss', float('inf')),
        'config': old_checkpoint.get('config', {}),
        'timestamp': datetime.now().isoformat(),
        'migrated_from': old_checkpoint_path,
    }
    
    # Apply config updates if provided
    if config_updates:
        if 'config' not in new_checkpoint:
            new_checkpoint['config'] = {}
        new_checkpoint['config'].update(config_updates)
    
    # Save new checkpoint
    torch.save(new_checkpoint, new_checkpoint_path)
    logger.info(f"Checkpoint migration completed: {new_checkpoint_path}")