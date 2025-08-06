#!/usr/bin/env python3
"""
Launch interactive text generation session.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.interactive import InteractiveSession


def main():
    """Main entry point for interactive generation."""
    parser = argparse.ArgumentParser(description="Interactive LLM text generation")
    
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", help="Path to tokenizer (default: data/tokenizer)")
    parser.add_argument("--config", help="Path to generation config YAML file")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting interactive LLM session...")
        
        session = InteractiveSession(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            config_path=args.config,
            device=args.device
        )
        
        session.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Session ended by user.")
    except Exception as e:
        print(f"❌ Failed to start interactive session: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())