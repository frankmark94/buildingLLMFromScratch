#!/usr/bin/env python3
"""
Comprehensive evaluation script for neural LLM models.
Supports perplexity, downstream tasks, and generation quality evaluation.
"""

import os
import sys
import argparse
import yaml
import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model import GPTModel, ModelConfig
from data.tokenizer import LLMTokenizer
from data.data_loader import DataLoader
from evaluation.perplexity import PerplexityEvaluator
from evaluation.benchmarks import BenchmarkEvaluator
from evaluation.metrics import MetricsCalculator
from inference import TextGenerator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_path: str, tokenizer_path: str, 
                            device: torch.device) -> tuple:
    """Load model and tokenizer."""
    print(f"🔄 Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    if 'config' in checkpoint:
        config = ModelConfig.from_dict(checkpoint['config'])
    else:
        # Fallback config
        config = ModelConfig()
    
    model = GPTModel(config)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {param_count:,} parameters")
    
    # Load tokenizer
    print(f"🔄 Loading tokenizer from {tokenizer_path}")
    tokenizer = LLMTokenizer()
    
    if Path(tokenizer_path).exists():
        tokenizer.load_tokenizer(tokenizer_path)
    
    print("✅ Tokenizer loaded")
    
    return model, tokenizer, config


def evaluate_perplexity(model: GPTModel, tokenizer: LLMTokenizer, 
                       data_config: Dict[str, Any], device: torch.device,
                       max_samples: Optional[int] = None) -> Dict[str, float]:
    """Evaluate model perplexity on datasets."""
    print("\n📊 Evaluating Perplexity...")
    print("-" * 30)
    
    results = {}
    
    # Create data loader
    data_loader = DataLoader(
        tokenizer=tokenizer,
        data_dir=data_config.get('data_dir', 'data'),
        **data_config.get('loader_params', {})
    )
    
    evaluator = PerplexityEvaluator(model, tokenizer, device)
    
    # Evaluate on different splits
    for split in ['validation', 'test']:
        try:
            dataset = data_loader.get_dataset(split)
            if len(dataset) == 0:
                continue
            
            # Create data loader
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=data_config.get('eval_batch_size', 8),
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            print(f"📈 Evaluating on {split} set ({len(dataset)} samples)...")
            
            loss, perplexity = evaluator.evaluate_dataset(loader, max_samples)
            
            results[f'{split}_loss'] = loss
            results[f'{split}_perplexity'] = perplexity
            
            print(f"  Loss: {loss:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            
        except Exception as e:
            print(f"⚠️ Could not evaluate {split}: {e}")
    
    return results


def evaluate_generation_quality(model: GPTModel, tokenizer: LLMTokenizer,
                               device: torch.device, 
                               prompts: List[str]) -> Dict[str, Any]:
    """Evaluate generation quality with sample prompts."""
    print("\n🎯 Evaluating Generation Quality...")
    print("-" * 35)
    
    generator = TextGenerator(
        model, tokenizer, device,
        config={
            'max_new_tokens': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }
    )
    
    metrics_calc = MetricsCalculator()
    results = {
        'samples': [],
        'avg_length': 0,
        'repetition_score': 0,
        'diversity_score': 0
    }
    
    total_length = 0
    all_generations = []
    
    for i, prompt in enumerate(prompts[:10]):  # Limit to 10 prompts
        print(f"\n[{i+1}] Prompt: {prompt[:50]}...")
        
        try:
            generated = generator.generate(prompt)
            generation = generated.replace(prompt, '').strip()
            
            # Store sample
            results['samples'].append({
                'prompt': prompt,
                'generation': generation,
                'length': len(generation.split())
            })
            
            total_length += len(generation.split())
            all_generations.append(generation)
            
            print(f"Generated: {generation[:100]}...")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    if all_generations:
        # Calculate metrics
        results['avg_length'] = total_length / len(all_generations)
        results['repetition_score'] = metrics_calc.calculate_repetition_score(all_generations)
        results['diversity_score'] = metrics_calc.calculate_diversity_score(all_generations)
        
        print(f"\n📊 Generation Metrics:")
        print(f"  Average Length: {results['avg_length']:.1f} words")
        print(f"  Repetition Score: {results['repetition_score']:.3f}")
        print(f"  Diversity Score: {results['diversity_score']:.3f}")
    
    return results


def evaluate_downstream_tasks(model: GPTModel, tokenizer: LLMTokenizer,
                             device: torch.device,
                             task_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate on downstream tasks."""
    print("\n🎯 Evaluating Downstream Tasks...")
    print("-" * 32)
    
    evaluator = BenchmarkEvaluator(model, tokenizer, device)
    results = {}
    
    for task_name, task_config in task_configs.items():
        try:
            print(f"\n📋 Running {task_name}...")
            
            task_results = evaluator.evaluate_task(task_name, task_config)
            results[task_name] = task_results
            
            # Print key metrics
            if 'accuracy' in task_results:
                print(f"  Accuracy: {task_results['accuracy']:.3f}")
            if 'f1_score' in task_results:
                print(f"  F1 Score: {task_results['f1_score']:.3f}")
            if 'bleu_score' in task_results:
                print(f"  BLEU Score: {task_results['bleu_score']:.3f}")
            
        except Exception as e:
            print(f"❌ Task {task_name} failed: {e}")
            results[task_name] = {'error': str(e)}
    
    return results


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_path}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate neural LLM model")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    
    # Optional arguments
    parser.add_argument("--tokenizer", default="data/tokenizer", help="Path to tokenizer")
    parser.add_argument("--config", help="Path to evaluation config YAML")
    parser.add_argument("--data-config", help="Path to data config YAML")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file for results")
    
    # Evaluation modes
    parser.add_argument("--skip-perplexity", action="store_true", help="Skip perplexity evaluation")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation quality evaluation")
    parser.add_argument("--skip-downstream", action="store_true", help="Skip downstream task evaluation")
    
    # Options
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--max-samples", type=int, help="Maximum samples for evaluation")
    parser.add_argument("--prompts-file", help="File containing evaluation prompts")
    
    args = parser.parse_args()
    
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
    
    print("📊 Neural LLM Evaluation")
    print("=" * 40)
    print(f"📁 Model: {args.model}")
    print(f"🖥️  Device: {device}")
    print(f"💾 Output: {args.output}")
    print("=" * 40)
    
    # Load model and tokenizer
    try:
        model, tokenizer, model_config = load_model_and_tokenizer(
            args.model, args.tokenizer, device
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Load configurations
    eval_config = {}
    if args.config:
        eval_config = load_config(args.config)
    
    data_config = {}
    if args.data_config:
        data_config = load_config(args.data_config)
    elif 'data' in eval_config:
        data_config = eval_config['data']
    
    # Initialize results
    results = {
        'model_path': args.model,
        'model_config': model_config.__dict__,
        'evaluation_timestamp': datetime.now().isoformat(),
        'device': str(device)
    }
    
    try:
        # Perplexity evaluation
        if not args.skip_perplexity and data_config:
            perplexity_results = evaluate_perplexity(
                model, tokenizer, data_config, device, args.max_samples
            )
            results['perplexity'] = perplexity_results
        
        # Generation quality evaluation
        if not args.skip_generation:
            # Load prompts
            if args.prompts_file:
                with open(args.prompts_file, 'r') as f:
                    prompts = [line.strip() for line in f if line.strip()]
            else:
                # Default prompts
                prompts = [
                    "The future of artificial intelligence is",
                    "Climate change affects the planet by",
                    "The most important invention in history was",
                    "In the year 2050, technology will",
                    "The key to happiness is",
                    "Science helps us understand",
                    "The greatest challenge facing humanity is",
                    "Education should focus on",
                    "The role of government should be",
                    "Art and creativity are important because"
                ]
            
            generation_results = evaluate_generation_quality(
                model, tokenizer, device, prompts
            )
            results['generation_quality'] = generation_results
        
        # Downstream task evaluation
        if not args.skip_downstream and 'downstream_tasks' in eval_config:
            downstream_results = evaluate_downstream_tasks(
                model, tokenizer, device, eval_config['downstream_tasks']
            )
            results['downstream_tasks'] = downstream_results
        
        # Save results
        save_results(results, args.output)
        
        print("\n🎉 Evaluation completed successfully!")
        
        # Print summary
        print("\n📈 Summary:")
        if 'perplexity' in results:
            for key, value in results['perplexity'].items():
                if 'perplexity' in key:
                    print(f"  {key}: {value:.2f}")
        
        if 'generation_quality' in results:
            gen_results = results['generation_quality']
            print(f"  Avg Generation Length: {gen_results.get('avg_length', 0):.1f} words")
            print(f"  Diversity Score: {gen_results.get('diversity_score', 0):.3f}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())