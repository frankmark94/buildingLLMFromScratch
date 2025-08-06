#!/usr/bin/env python3
"""
Launch FastAPI server for LLM inference.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.api import LLMServer


def main():
    """Main entry point for API server."""
    parser = argparse.ArgumentParser(description="Neural LLM API Server")
    
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", help="Path to tokenizer (default: data/tokenizer)")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting LLM API server...")
        print(f"📡 Server will be available at http://{args.host}:{args.port}")
        print("📚 API docs will be available at http://{args.host}:{args.port}/docs")
        
        server = LLMServer(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            device=args.device,
            host=args.host,
            port=args.port
        )
        
        server.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user.")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())