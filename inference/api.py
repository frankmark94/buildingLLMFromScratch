#!/usr/bin/env python3
"""
FastAPI server for LLM text generation with REST endpoints.
"""

import os
import sys
import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.responses import StreamingResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("FastAPI dependencies not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

import torch
from model import GPTModel, ModelConfig
from data.tokenizer import LLMTokenizer
from inference.generator import TextGenerator, StreamingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(50, ge=1, le=200, description="Top-k filtering")
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Use sampling vs greedy decoding")
    stream: bool = Field(False, description="Stream response tokens")
    stop_strings: Optional[List[str]] = Field(None, description="Strings that stop generation")
    return_prompt: bool = Field(False, description="Include prompt in response")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str = Field(..., description="Generated text")
    prompt: Optional[str] = Field(None, description="Original prompt (if requested)")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class BatchGenerationRequest(BaseModel):
    """Request model for batch text generation."""
    prompts: List[str] = Field(..., description="List of input prompts")
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_k: Optional[int] = Field(50, ge=1, le=200)
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    do_sample: bool = Field(True)
    return_prompt: bool = Field(False)


class BatchGenerationResponse(BaseModel):
    """Response model for batch generation."""
    results: List[GenerationResponse] = Field(..., description="List of generation results")
    total_time: float = Field(..., description="Total processing time")


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    parameters: int
    architecture: Dict[str, Any]
    device: str
    loaded_at: str


class LLMServer:
    """FastAPI server for LLM inference."""
    
    def __init__(self, model_path: str, tokenizer_path: str = None, 
                 device: str = 'auto', host: str = '0.0.0.0', port: int = 8000):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or "data/tokenizer"
        self.device = self._get_device(device)
        self.host = host
        self.port = port
        
        # Load model and tokenizer
        self.model, self.tokenizer, self.model_info = self._load_model_and_tokenizer()
        
        # Initialize generator
        self.generator = TextGenerator(self.model, self.tokenizer, self.device)
        self.streaming_generator = StreamingGenerator(self.generator)
        
        # FastAPI app
        self.app = FastAPI(
            title="Neural LLM API",
            description="REST API for neural language model text generation",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"LLM Server initialized on {self.device}")
    
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
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model
        if 'config' in checkpoint:
            config = ModelConfig.from_dict(checkpoint['config'])
        else:
            config = ModelConfig()
        
        model = GPTModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(self.device)
        
        # Load tokenizer
        tokenizer = LLMTokenizer()
        
        # Model info
        model_info = {
            'model_name': Path(self.model_path).name,
            'parameters': sum(p.numel() for p in model.parameters()),
            'architecture': {
                'n_layers': config.n_layers,
                'n_heads': config.n_heads,
                'n_embd': config.n_embd,
                'vocab_size': config.vocab_size,
                'block_size': config.block_size
            },
            'device': str(self.device),
            'loaded_at': datetime.now().isoformat()
        }
        
        return model, tokenizer, model_info
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve a simple web interface."""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Neural LLM API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 Neural LLM API</h1>
                    <p>REST API for neural language model text generation</p>
                    
                    <h2>Available Endpoints:</h2>
                    
                    <div class="endpoint">
                        <h3>GET /info</h3>
                        <p>Get model information and status</p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>POST /generate</h3>
                        <p>Generate text from a prompt</p>
                        <p><strong>Body:</strong> <code>{"prompt": "Your text here", "max_new_tokens": 256}</code></p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>POST /generate/batch</h3>
                        <p>Generate text for multiple prompts</p>
                        <p><strong>Body:</strong> <code>{"prompts": ["Prompt 1", "Prompt 2"]}</code></p>
                    </div>
                    
                    <div class="endpoint">
                        <h3>POST /generate/stream</h3>
                        <p>Stream generated tokens in real-time</p>
                        <p><strong>Body:</strong> <code>{"prompt": "Your text here", "stream": true}</code></p>
                    </div>
                    
                    <p><a href="/docs">📚 View Interactive API Documentation</a></p>
                </div>
            </body>
            </html>
            """
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "model_loaded": True
            }
        
        @self.app.get("/info", response_model=ModelInfo)
        async def model_info():
            """Get model information."""
            return ModelInfo(**self.model_info)
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest):
            """Generate text from a prompt."""
            if request.stream:
                raise HTTPException(status_code=400, detail="Use /generate/stream for streaming")
            
            start_time = datetime.now()
            
            try:
                # Generate text
                generated_text = self.generator.generate(
                    request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    return_prompt=request.return_prompt
                )
                
                # Handle stop strings
                if request.stop_strings:
                    for stop_string in request.stop_strings:
                        if stop_string in generated_text:
                            generated_text = generated_text[:generated_text.find(stop_string)]
                
                generation_time = (datetime.now() - start_time).total_seconds()
                
                # Count tokens
                tokens_generated = len(self.generator._tokenize_prompt(generated_text))
                
                return GenerationResponse(
                    generated_text=generated_text,
                    prompt=request.prompt if request.return_prompt else None,
                    tokens_generated=tokens_generated,
                    generation_time=generation_time,
                    model_info=self.model_info
                )
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        
        @self.app.post("/generate/batch", response_model=BatchGenerationResponse)
        async def batch_generate(request: BatchGenerationRequest):
            """Generate text for multiple prompts."""
            start_time = datetime.now()
            
            try:
                # Generate for all prompts
                generated_texts = self.generator.generate_batch(
                    request.prompts,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    return_prompt=request.return_prompt
                )
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                # Create responses
                results = []
                for i, (prompt, generated_text) in enumerate(zip(request.prompts, generated_texts)):
                    tokens_generated = len(self.generator._tokenize_prompt(generated_text))
                    
                    results.append(GenerationResponse(
                        generated_text=generated_text,
                        prompt=prompt if request.return_prompt else None,
                        tokens_generated=tokens_generated,
                        generation_time=total_time / len(request.prompts),  # Average time
                        model_info=self.model_info
                    ))
                
                return BatchGenerationResponse(
                    results=results,
                    total_time=total_time
                )
                
            except Exception as e:
                logger.error(f"Batch generation error: {e}")
                raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")
        
        @self.app.post("/generate/stream")
        async def stream_generate(request: GenerationRequest):
            """Stream generated tokens."""
            if not request.stream:
                request.stream = True
            
            async def generate_stream():
                try:
                    # Stream generation
                    tokens = []
                    
                    def stream_callback(token: str):
                        tokens.append(token)
                        return f"data: {{'token': '{token.replace(chr(10), '\\n').replace(chr(13), '\\r')}'}}\n\n"
                    
                    # Generate with streaming callback
                    response_parts = []
                    
                    def collect_callback(token: str):
                        response_parts.append(stream_callback(token))
                    
                    self.streaming_generator.generate_stream(
                        request.prompt,
                        collect_callback,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        repetition_penalty=request.repetition_penalty,
                        do_sample=request.do_sample
                    )
                    
                    # Yield all collected parts
                    for part in response_parts:
                        yield part
                    
                    # Final message
                    yield f"data: {{'done': true, 'tokens_generated': {len(tokens)}}}\n\n"
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: {{'error': '{str(e)}'}}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )
    
    def run(self):
        """Run the FastAPI server."""
        logger.info(f"Starting server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


def main():
    """Main entry point for the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural LLM API Server")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", help="Path to tokenizer (default: data/tokenizer)")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    try:
        server = LLMServer(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            device=args.device,
            host=args.host,
            port=args.port
        )
        server.run()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()