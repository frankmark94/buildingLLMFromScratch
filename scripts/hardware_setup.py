#!/usr/bin/env python3
"""
Hardware detection and automatic configuration setup.
Detects available hardware and suggests optimal configurations.
"""

import os
import psutil
import torch
import platform
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardwareProfiler:
    """Detect and profile available hardware for optimal configuration."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.gpu_info = self._get_gpu_info()
        self.memory_info = self._get_memory_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            'platform': platform.system(),
            'cpu_count': os.cpu_count(),
            'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'architecture': platform.machine(),
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': 0,
            'gpu_names': [],
            'total_gpu_memory': 0,
            'gpu_compute_capability': None
        }
        
        if torch.cuda.is_available():
            gpu_info['cuda_device_count'] = torch.cuda.device_count()
            
            for i in range(gpu_info['cuda_device_count']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                
                gpu_info['gpu_names'].append(gpu_name)
                gpu_info['total_gpu_memory'] += gpu_memory
                
                # Get compute capability
                if i == 0:  # Use first GPU's capability
                    capability = torch.cuda.get_device_capability(i)
                    gpu_info['gpu_compute_capability'] = f"{capability[0]}.{capability[1]}"
        
        # Convert to GB
        gpu_info['total_gpu_memory_gb'] = gpu_info['total_gpu_memory'] / (1024**3)
        
        return gpu_info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        
        return {
            'total_ram_gb': memory.total / (1024**3),
            'available_ram_gb': memory.available / (1024**3),
            'ram_usage_percent': memory.percent
        }
    
    def classify_hardware(self) -> str:
        """Classify hardware into predefined categories."""
        gpu_memory = self.gpu_info['total_gpu_memory_gb']
        system_ram = self.memory_info['total_ram_gb']
        has_cuda = self.gpu_info['cuda_available']
        
        # Hardware classification logic
        if not has_cuda or gpu_memory < 4:
            if system_ram >= 32:
                return "consumer_laptop"  # Good RAM, no/weak GPU
            else:
                return "dev_tiny"  # Limited resources
        elif gpu_memory >= 12:
            return "high_end"  # High-end GPU
        elif gpu_memory >= 6:
            return "mid_range_gpu"  # Mid-range GPU
        else:
            return "consumer_laptop"  # Basic GPU
    
    def get_recommended_config(self) -> Tuple[str, Dict[str, Any]]:
        """Get recommended configuration based on hardware."""
        hardware_class = self.classify_hardware()
        
        # Load hardware configurations
        config_path = Path("config/hardware_configs.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                hardware_configs = yaml.safe_load(f)
            
            config = hardware_configs.get(hardware_class, {})
        else:
            # Fallback configuration
            config = self._get_fallback_config(hardware_class)
        
        return hardware_class, config
    
    def _get_fallback_config(self, hardware_class: str) -> Dict[str, Any]:
        """Fallback configuration if file is not available."""
        if hardware_class == "dev_tiny":
            return {
                "model": {
                    "n_layers": 4,
                    "n_heads": 4,
                    "n_embd": 256,
                    "vocab_size": 16000,
                    "block_size": 128,
                    "gradient_checkpointing": True
                },
                "training": {
                    "batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "max_steps": 1000,
                    "use_amp": True
                }
            }
        elif hardware_class == "consumer_laptop":
            return {
                "model": {
                    "n_layers": 6,
                    "n_heads": 8,
                    "n_embd": 512,
                    "vocab_size": 32000,
                    "block_size": 256,
                    "gradient_checkpointing": True
                },
                "training": {
                    "batch_size": 1,
                    "gradient_accumulation_steps": 32,
                    "max_steps": 10000,
                    "use_amp": True
                }
            }
        else:
            # Default to conservative settings
            return self._get_fallback_config("consumer_laptop")
    
    def estimate_memory_usage(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate memory usage for given configuration."""
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        # Model parameters
        n_layers = model_config.get('n_layers', 6)
        n_embd = model_config.get('n_embd', 512)
        vocab_size = model_config.get('vocab_size', 32000)
        block_size = model_config.get('block_size', 256)
        
        # Estimate parameter count (simplified)
        # Token embeddings: vocab_size * n_embd
        # Position embeddings: block_size * n_embd
        # Transformer blocks: ~12 * n_embd^2 per layer (attention + MLP)
        # LM head: vocab_size * n_embd (tied with token embeddings)
        
        param_count = (
            vocab_size * n_embd +          # Token embeddings
            block_size * n_embd +          # Position embeddings  
            n_layers * 12 * n_embd * n_embd +  # Transformer blocks
            n_embd                         # Layer norm
        )
        
        # Memory estimates (in GB)
        dtype_size = 2 if training_config.get('use_amp', True) else 4  # FP16 vs FP32
        
        # Parameter memory
        param_memory_gb = param_count * dtype_size / (1024**3)
        
        # Gradient memory (same as parameters)
        grad_memory_gb = param_memory_gb
        
        # Optimizer state (AdamW has 2 states per parameter)
        optimizer_memory_gb = param_memory_gb * 2
        
        # Activation memory (rough estimate)
        batch_size = training_config.get('batch_size', 1)
        activation_memory_gb = (
            batch_size * block_size * n_embd * n_layers * dtype_size / (1024**3)
        )
        
        # Total training memory
        total_training_gb = (
            param_memory_gb + grad_memory_gb + 
            optimizer_memory_gb + activation_memory_gb
        )
        
        return {
            'parameters_gb': param_memory_gb,
            'gradients_gb': grad_memory_gb,
            'optimizer_gb': optimizer_memory_gb,
            'activations_gb': activation_memory_gb,
            'total_training_gb': total_training_gb,
            'inference_gb': param_memory_gb + activation_memory_gb * 0.5,  # Rough inference estimate
            'parameter_count_millions': param_count / 1e6
        }
    
    def print_hardware_report(self):
        """Print detailed hardware report."""
        print("\n" + "="*60)
        print("HARDWARE ANALYSIS REPORT")
        print("="*60)
        
        # System info
        print(f"\nSystem Information:")
        print(f"  Platform: {self.system_info['platform']}")
        print(f"  CPU Cores: {self.system_info['cpu_count']}")
        print(f"  Architecture: {self.system_info['architecture']}")
        
        # Memory info
        print(f"\nMemory Information:")
        print(f"  Total RAM: {self.memory_info['total_ram_gb']:.1f} GB")
        print(f"  Available RAM: {self.memory_info['available_ram_gb']:.1f} GB")
        print(f"  RAM Usage: {self.memory_info['ram_usage_percent']:.1f}%")
        
        # GPU info
        print(f"\nGPU Information:")
        print(f"  CUDA Available: {self.gpu_info['cuda_available']}")
        if self.gpu_info['cuda_available']:
            print(f"  GPU Count: {self.gpu_info['cuda_device_count']}")
            print(f"  Total GPU Memory: {self.gpu_info['total_gpu_memory_gb']:.1f} GB")
            print(f"  GPU Names: {', '.join(self.gpu_info['gpu_names'])}")
            print(f"  Compute Capability: {self.gpu_info['gpu_compute_capability']}")
        else:
            print("  No CUDA GPUs detected")
        
        # Recommendations
        hardware_class, config = self.get_recommended_config()
        memory_est = self.estimate_memory_usage(config)
        
        print(f"\nRecommended Configuration: {hardware_class}")
        print(f"  Model Size: ~{memory_est['parameter_count_millions']:.1f}M parameters")
        print(f"  Estimated Training Memory: {memory_est['total_training_gb']:.1f} GB")
        print(f"  Estimated Inference Memory: {memory_est['inference_gb']:.1f} GB")
        
        # Warnings and tips
        print(f"\nRecommendations:")
        
        if not self.gpu_info['cuda_available']:
            print("  ⚠️  No CUDA GPU detected - training will be VERY slow on CPU")
            print("  💡 Consider using Google Colab or cloud GPU instances for training")
        
        if memory_est['total_training_gb'] > self.memory_info['available_ram_gb']:
            print("  ⚠️  Estimated memory usage exceeds available RAM")
            print("  💡 Consider using an even smaller model configuration")
        
        if hardware_class == "consumer_laptop":
            print("  💡 Your hardware is suitable for small-scale experimentation")
            print("  💡 Training will take longer but is definitely feasible")
            print("  💡 Consider training overnight or using cloud resources for larger models")
        
        print("="*60)


def create_hardware_specific_configs():
    """Create configuration files optimized for detected hardware."""
    profiler = HardwareProfiler()
    hardware_class, recommended_config = profiler.get_recommended_config()
    
    # Create optimized model config
    model_config_path = Path("config/model_config_optimized.yaml")
    if 'model' in recommended_config:
        with open(model_config_path, 'w') as f:
            yaml.dump({'model': recommended_config['model']}, f, default_flow_style=False)
        logger.info(f"Created optimized model config: {model_config_path}")
    
    # Create optimized training config
    training_config_path = Path("config/training_config_optimized.yaml")
    if 'training' in recommended_config:
        with open(training_config_path, 'w') as f:
            yaml.dump({'training': recommended_config['training']}, f, default_flow_style=False)
        logger.info(f"Created optimized training config: {training_config_path}")
    
    return hardware_class, recommended_config


def main():
    """Run hardware analysis and create optimized configurations."""
    print("Analyzing your hardware...")
    
    profiler = HardwareProfiler()
    profiler.print_hardware_report()
    
    print("\nCreating optimized configurations...")
    hardware_class, config = create_hardware_specific_configs()
    
    print(f"\n✅ Hardware-optimized configurations created!")
    print(f"📁 Check config/model_config_optimized.yaml")
    print(f"📁 Check config/training_config_optimized.yaml")
    
    print(f"\n🚀 To use optimized settings, run training with:")
    print(f"   python scripts/train.py --model-config config/model_config_optimized.yaml --training-config config/training_config_optimized.yaml")


if __name__ == "__main__":
    main()