#!/usr/bin/env python3
"""
Interactive text generation interface with advanced features.
"""

import os
import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model import GPTModel, ModelConfig
from data.tokenizer import LLMTokenizer
from inference.generator import TextGenerator, StreamingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveSession:
    """Advanced interactive text generation session."""
    
    def __init__(self, model_path: str, tokenizer_path: str = None, 
                 config_path: str = None, device: str = 'auto'):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or "data/tokenizer"
        self.config_path = config_path
        self.device = self._get_device(device)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Initialize generator
        generation_config = self._load_generation_config()
        self.generator = TextGenerator(self.model, self.tokenizer, self.device, generation_config)
        self.streaming_generator = StreamingGenerator(self.generator)
        
        # Session state
        self.conversation_history = []
        self.session_stats = {
            'generations': 0,
            'total_tokens': 0,
            'start_time': datetime.now()
        }
        
        logger.info(f"Interactive session initialized on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _load_model_and_tokenizer(self) -> tuple:
        """Load model and tokenizer from checkpoints."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model
        if 'config' in checkpoint:
            config = ModelConfig.from_dict(checkpoint['config'])
        else:
            # Fallback to default config
            config = ModelConfig()
        
        model = GPTModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(self.device)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = LLMTokenizer()
        
        return model, tokenizer
    
    def _load_generation_config(self) -> Dict[str, Any]:
        """Load generation configuration."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('generation', {})
        
        # Default generation config
        return {
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'max_new_tokens': 200,
            'repetition_penalty': 1.1,
            'do_sample': True
        }
    
    def print_welcome(self):
        """Print welcome message."""
        print("\n" + "="*70)
        print("🤖 Neural LLM Interactive Generation")
        print("="*70)
        print(f"📱 Model: {Path(self.model_path).name}")
        print(f"🖥️  Device: {self.device}")
        print(f"🧠 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("\n📋 Available Commands:")
        print("  /help          - Show this help")
        print("  /config        - Show current generation settings")
        print("  /set <k> <v>   - Change generation parameter")
        print("  /presets       - Show available presets")
        print("  /preset <name> - Load a preset configuration")
        print("  /history       - Show conversation history")
        print("  /save <file>   - Save conversation to file")
        print("  /load <file>   - Load conversation from file")
        print("  /clear         - Clear conversation history")
        print("  /stats         - Show session statistics")
        print("  /stream        - Toggle streaming mode")
        print("  /quit          - Exit the session")
        print("\n💡 Tips:")
        print("  - Press Enter with empty input to continue last response")
        print("  - Use Ctrl+C to interrupt generation")
        print("  - Longer prompts generally produce better results")
        print("="*70)
    
    def run(self):
        """Main interactive loop."""
        self.print_welcome()
        
        streaming_mode = False
        
        while True:
            try:
                # Get user input
                prompt = input("\n💬 You: ").strip()
                
                # Handle commands
                if prompt.startswith('/'):
                    if self._handle_command(prompt, streaming_mode):
                        streaming_mode = not streaming_mode
                    continue
                
                # Handle empty input (continue conversation)
                if not prompt:
                    if self.conversation_history:
                        prompt = self.conversation_history[-1]['content']
                    else:
                        print("💭 Please enter a prompt to get started!")
                        continue
                
                # Generate response
                self._generate_response(prompt, streaming_mode)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session ended by user.")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Session error: {e}", exc_info=True)
    
    def _handle_command(self, command: str, streaming_mode: bool) -> bool:
        """Handle user commands. Returns True if streaming mode should be toggled."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/help':
            self.print_welcome()
        
        elif cmd == '/config':
            self._show_config()
        
        elif cmd == '/set' and len(parts) >= 3:
            self._set_parameter(parts[1], ' '.join(parts[2:]))
        
        elif cmd == '/presets':
            self._show_presets()
        
        elif cmd == '/preset' and len(parts) >= 2:
            self._load_preset(parts[1])
        
        elif cmd == '/history':
            self._show_history()
        
        elif cmd == '/save' and len(parts) >= 2:
            self._save_conversation(parts[1])
        
        elif cmd == '/load' and len(parts) >= 2:
            self._load_conversation(parts[1])
        
        elif cmd == '/clear':
            self.conversation_history = []
            print("🗑️ Conversation history cleared.")
        
        elif cmd == '/stats':
            self._show_stats()
        
        elif cmd == '/stream':
            print(f"🔄 Streaming mode {'disabled' if streaming_mode else 'enabled'}")
            return True
        
        elif cmd in ['/quit', '/exit', '/q']:
            print("👋 Goodbye!")
            sys.exit(0)
        
        else:
            print("❌ Unknown command. Type /help for available commands.")
        
        return False
    
    def _generate_response(self, prompt: str, streaming: bool):
        """Generate and display response."""
        print("\n🤖 Assistant: ", end='', flush=True)
        
        start_time = datetime.now()
        
        if streaming:
            # Streaming generation
            response_tokens = []
            
            def stream_callback(token: str):
                print(token, end='', flush=True)
                response_tokens.append(token)
            
            try:
                self.streaming_generator.generate_stream(prompt, stream_callback)
                response = ''.join(response_tokens)
            except KeyboardInterrupt:
                print("\n⏹️ Generation interrupted.")
                response = ''.join(response_tokens)
        else:
            # Standard generation
            try:
                response = self.generator.generate(prompt)
                print(response)
            except KeyboardInterrupt:
                print("\n⏹️ Generation interrupted.")
                return
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Update session stats
        self.session_stats['generations'] += 1
        response_tokens = len(self.generator._tokenize_prompt(response))
        self.session_stats['total_tokens'] += response_tokens
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': start_time.isoformat(),
            'prompt': prompt,
            'response': response,
            'generation_time': generation_time,
            'tokens': response_tokens
        })
        
        print(f"\n⏱️ {generation_time:.1f}s | 🎯 {response_tokens} tokens")
    
    def _show_config(self):
        """Show current generation configuration."""
        print("\n⚙️ Current Generation Configuration:")
        print("-" * 40)
        for key, value in self.generator.config.items():
            print(f"  {key:<20}: {value}")
        print("-" * 40)
    
    def _set_parameter(self, param: str, value: str):
        """Set a generation parameter."""
        try:
            # Convert value to appropriate type
            if value.lower() in ['true', 'false']:
                typed_value = value.lower() == 'true'
            elif '.' in value:
                typed_value = float(value)
            else:
                try:
                    typed_value = int(value)
                except ValueError:
                    typed_value = value
            
            self.generator.config[param] = typed_value
            print(f"✅ Set {param} = {typed_value}")
            
        except Exception as e:
            print(f"❌ Error setting parameter: {e}")
    
    def _show_presets(self):
        """Show available generation presets."""
        presets = {
            'creative': {
                'temperature': 1.0,
                'top_k': 40,
                'top_p': 0.9,
                'repetition_penalty': 1.1
            },
            'balanced': {
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.9,
                'repetition_penalty': 1.1
            },
            'precise': {
                'temperature': 0.3,
                'top_k': 10,
                'top_p': 0.7,
                'repetition_penalty': 1.2
            },
            'deterministic': {
                'temperature': 0.1,
                'do_sample': False,
                'repetition_penalty': 1.0
            }
        }
        
        print("\n🎨 Available Presets:")
        print("-" * 40)
        for name, config in presets.items():
            print(f"  {name}:")
            for key, value in config.items():
                print(f"    {key}: {value}")
            print()
    
    def _load_preset(self, preset_name: str):
        """Load a generation preset."""
        presets = {
            'creative': {
                'temperature': 1.0,
                'top_k': 40,
                'top_p': 0.9,
                'repetition_penalty': 1.1
            },
            'balanced': {
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.9,
                'repetition_penalty': 1.1
            },
            'precise': {
                'temperature': 0.3,
                'top_k': 10,
                'top_p': 0.7,
                'repetition_penalty': 1.2
            },
            'deterministic': {
                'temperature': 0.1,
                'do_sample': False,
                'repetition_penalty': 1.0
            }
        }
        
        if preset_name in presets:
            self.generator.config.update(presets[preset_name])
            print(f"✅ Loaded preset '{preset_name}'")
        else:
            print(f"❌ Unknown preset '{preset_name}'. Available: {list(presets.keys())}")
    
    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print(f"\n📚 Conversation History ({len(self.conversation_history)} entries):")
        print("=" * 60)
        
        for i, entry in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            print(f"\n[{i}] {entry['timestamp']}")
            print(f"💬 Prompt: {entry['prompt'][:100]}{'...' if len(entry['prompt']) > 100 else ''}")
            print(f"🤖 Response: {entry['response'][:200]}{'...' if len(entry['response']) > 200 else ''}")
            print(f"📊 {entry['tokens']} tokens in {entry['generation_time']:.1f}s")
    
    def _save_conversation(self, filename: str):
        """Save conversation to file."""
        try:
            filepath = Path(filename)
            if not filepath.suffix:
                filepath = filepath.with_suffix('.json')
            
            save_data = {
                'session_info': {
                    'model_path': self.model_path,
                    'device': str(self.device),
                    'start_time': self.session_stats['start_time'].isoformat(),
                    'save_time': datetime.now().isoformat()
                },
                'conversation_history': self.conversation_history,
                'session_stats': {
                    **self.session_stats,
                    'start_time': self.session_stats['start_time'].isoformat()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"💾 Conversation saved to {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
    
    def _load_conversation(self, filename: str):
        """Load conversation from file."""
        try:
            filepath = Path(filename)
            if not filepath.suffix:
                filepath = filepath.with_suffix('.json')
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.conversation_history = data.get('conversation_history', [])
            print(f"📂 Loaded conversation with {len(self.conversation_history)} entries from {filepath}")
            
        except Exception as e:
            print(f"❌ Error loading conversation: {e}")
    
    def _show_stats(self):
        """Show session statistics."""
        runtime = datetime.now() - self.session_stats['start_time']
        
        print(f"\n📊 Session Statistics:")
        print("-" * 30)
        print(f"  Runtime: {runtime}")
        print(f"  Generations: {self.session_stats['generations']}")
        print(f"  Total tokens: {self.session_stats['total_tokens']:,}")
        
        if self.session_stats['generations'] > 0:
            avg_tokens = self.session_stats['total_tokens'] / self.session_stats['generations']
            print(f"  Avg tokens/gen: {avg_tokens:.1f}")
        
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 30)


def main():
    """Main entry point for interactive session."""
    parser = argparse.ArgumentParser(description="Interactive LLM text generation")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", help="Path to tokenizer (default: data/tokenizer)")
    parser.add_argument("--config", help="Path to generation config")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    try:
        session = InteractiveSession(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            config_path=args.config,
            device=args.device
        )
        session.run()
        
    except Exception as e:
        logger.error(f"Failed to start interactive session: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()