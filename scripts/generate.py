#!/usr/bin/env python3
"""
Main text generation script with command-line interface.
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model import GPTModel, ModelConfig
from data.tokenizer import LLMTokenizer
from inference import TextGenerator


def load_model_and_tokenizer(model_path: str, tokenizer_path: str, device: torch.device):
    """Load model and tokenizer from paths."""
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load tokenizer
    print(f"🔄 Loading tokenizer from {tokenizer_path}")
    tokenizer = LLMTokenizer()
    print("✅ Tokenizer loaded")
    
    return model, tokenizer


def generate_single(generator: TextGenerator, prompt: str, config: Dict[str, Any]) -> None:
    """Generate text for a single prompt."""
    print(f"\n💭 Prompt: {prompt}")
    print("🤖 Generating...")
    
    try:
        generated_text = generator.generate(prompt, **config)
        
        print("\n📝 Generated Text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
        # Count tokens
        prompt_tokens = len(generator._tokenize_prompt(prompt))
        output_tokens = len(generator._tokenize_prompt(generated_text))
        
        print(f"📊 Prompt: {prompt_tokens} tokens | Output: {output_tokens} tokens")
        
    except KeyboardInterrupt:
        print("\n⏹️ Generation interrupted by user")
    except Exception as e:
        print(f"❌ Generation failed: {e}")


def generate_batch(generator: TextGenerator, prompts: List[str], 
                  config: Dict[str, Any]) -> None:
    """Generate text for multiple prompts."""
    print(f"\n🔄 Batch generating for {len(prompts)} prompts...")
    
    try:
        generated_texts = generator.generate_batch(prompts, **config)
        
        print("\n📝 Batch Generation Results:")
        print("=" * 60)
        
        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts), 1):
            print(f"\n[{i}] Prompt: {prompt}")
            print(f"Generated: {generated}")
            print("-" * 40)
        
        total_output_tokens = sum(len(generator._tokenize_prompt(text)) 
                                for text in generated_texts)
        print(f"📊 Total output tokens: {total_output_tokens:,}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Generation interrupted by user")
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")


def generate_from_file(generator: TextGenerator, file_path: str, 
                      config: Dict[str, Any]) -> None:
    """Generate text from prompts in a file."""
    try:
        with open(file_path, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"📂 Loaded {len(prompts)} prompts from {file_path}")
        generate_batch(generator, prompts, config)
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def interactive_mode(generator: TextGenerator, config: Dict[str, Any]) -> None:
    """Run interactive generation mode."""
    print("\n🎮 Interactive Generation Mode")
    print("=" * 40)
    print("Commands:")
    print("  /quit - Exit")
    print("  /config - Show settings")
    print("  /set <param> <value> - Change parameter")
    print("=" * 40)
    
    while True:
        try:
            prompt = input("\n💬 Enter prompt: ").strip()
            
            if prompt.lower() in ['/quit', '/exit', '/q']:
                print("👋 Goodbye!")
                break
            
            elif prompt.lower() == '/config':
                print("\n⚙️ Current Configuration:")
                for key, value in config.items():
                    print(f"  {key}: {value}")
                continue
            
            elif prompt.startswith('/set '):
                parts = prompt.split(' ', 2)
                if len(parts) >= 3:
                    param, value = parts[1], parts[2]
                    try:
                        # Convert to appropriate type
                        if value.lower() in ['true', 'false']:
                            config[param] = value.lower() == 'true'
                        elif '.' in value:
                            config[param] = float(value)
                        else:
                            config[param] = int(value)
                        print(f"✅ Set {param} = {config[param]}")
                    except ValueError:
                        config[param] = value
                        print(f"✅ Set {param} = {value}")
                continue
            
            elif prompt.startswith('/'):
                print("❌ Unknown command")
                continue
            
            if not prompt:
                print("💭 Please enter a prompt")
                continue
            
            # Generate
            generate_single(generator, prompt, config)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate text with trained neural LLM")
    
    # Model and tokenizer
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer", help="Path to tokenizer")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda, mps)")
    
    # Generation modes
    parser.add_argument("--prompt", help="Single prompt to generate from")
    parser.add_argument("--prompts", nargs="+", help="Multiple prompts for batch generation")
    parser.add_argument("--file", help="File containing prompts (one per line)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--no-sample", action="store_true", help="Use greedy decoding")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    # Output options
    parser.add_argument("--output", help="Output file to save results")
    parser.add_argument("--return-prompt", action="store_true", help="Include prompt in output")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"🎲 Random seed set to {args.seed}")
    
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
    
    print(f"🖥️  Using device: {device}")
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, args.tokenizer, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Create generator
    generation_config = {
        'max_new_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
        'do_sample': not args.no_sample,
        'return_prompt': args.return_prompt
    }
    
    generator = TextGenerator(model, tokenizer, device, generation_config)
    
    print(f"🎯 Generation settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    
    # Determine generation mode
    if args.interactive:
        interactive_mode(generator, generation_config)
    
    elif args.prompt:
        generate_single(generator, args.prompt, generation_config)
    
    elif args.prompts:
        generate_batch(generator, args.prompts, generation_config)
    
    elif args.file:
        generate_from_file(generator, args.file, generation_config)
    
    else:
        # Default to single prompt input
        prompt = input("💬 Enter prompt: ").strip()
        if prompt:
            generate_single(generator, prompt, generation_config)
        else:
            print("❌ No prompt provided")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())