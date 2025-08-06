# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Dependencies
```bash
pip install -r requirements.txt
```

### Data Pipeline
```bash
# Download datasets (The Pile, Wikipedia, etc.)
python data/download_datasets.py

# Train tokenizer on downloaded data
python data/tokenizer.py

# Preprocess and clean data, create train/val binary files
python data/preprocess.py

# List downloaded datasets
python data/download_datasets.py --list
```

### Training
```bash
# Single GPU training
python scripts/train.py

# Multi-GPU distributed training
python scripts/distributed_train.py --gpus 4

# Resume from checkpoint
python scripts/train.py --resume checkpoints/model_step_5000.pt
```

### Evaluation and Inference
```bash
# Calculate perplexity on validation set
python scripts/eval.py

# Generate text interactively
python scripts/generate.py --prompt "The future of AI is"

# Fine-tune on custom dataset
python scripts/finetune.py --dataset path/to/custom_data.jsonl
```

### Development Tools
```bash
# Code formatting
black .
isort .

# Type checking
mypy .

# Linting
flake8 .

# Run tests
pytest tests/

# Run specific test
pytest tests/test_model.py::TestTransformer::test_forward_pass
```

## Architecture Overview

This is a complete neural language model training pipeline implementing GPT-style autoregressive transformers from scratch. The architecture follows a modular design with clear separation of concerns:

### Configuration System
All hyperparameters are managed through YAML files in `config/`:
- `model_config.yaml`: Architecture parameters (layers, heads, embedding size)
- `training_config.yaml`: Training hyperparameters (learning rate, batch size, mixed precision)
- `data_config.yaml`: Dataset sources and preprocessing settings
- `generation_config.yaml`: Text generation parameters (temperature, top-k, top-p)

### Data Pipeline Flow
1. **Download** (`data/download_datasets.py`): Fetches public datasets (The Pile, Wikipedia, Common Crawl)
2. **Tokenization** (`data/tokenizer.py`): Trains BPE or SentencePiece tokenizers on the data
3. **Preprocessing** (`data/preprocess.py`): Cleans, deduplicates, filters, and converts to binary format
4. **Loading** (`data/data_loader.py`): Efficient batch loading for training

### Model Architecture
The transformer implementation in `model/` follows standard GPT architecture:
- Decoder-only transformer blocks with multi-head self-attention
- Configurable model sizes (125M to 1B+ parameters)
- Optional FlashAttention and gradient checkpointing for memory efficiency
- Custom initialization and layer normalization

### Training System
Training infrastructure in `training/` supports:
- Mixed precision training with native AMP
- Distributed training across multiple GPUs using PyTorch DDP
- Advanced optimizers (AdamW) with cosine learning rate scheduling
- Gradient accumulation for large effective batch sizes
- Automatic checkpointing and resume capability
- Weights & Biases integration for experiment tracking

### Key Implementation Details

- **Binary Data Format**: Training data is stored as numpy arrays in `.bin` files for fast loading
- **Streaming Support**: Large datasets can be processed in streaming mode to manage memory
- **Modular Tokenizers**: Supports both BPE and SentencePiece with consistent interfaces
- **Configuration Inheritance**: YAML configs support model size variants and easy experimentation
- **Memory Management**: Gradient checkpointing and mixed precision reduce memory usage for larger models

### Development Workflow

1. Configure datasets and model parameters in YAML files
2. Run data pipeline: download → tokenize → preprocess
3. Train model with automatic checkpointing and logging
4. Evaluate on validation set and downstream tasks
5. Generate text and fine-tune for specific applications

The codebase prioritizes reproducibility, scalability, and ease of experimentation while maintaining clean separation between data processing, model definition, training, and inference components.