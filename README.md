# Neural LLM Training From Scratch

A complete implementation for training GPT-style autoregressive transformer models from scratch, targeting 500M-1B parameter models for manageable experimentation.
<img width="1555" height="946" alt="image" src="https://github.com/user-attachments/assets/8e79b108-088e-4460-aa53-4e79d137b619" />


## Features

- **Scalable Training**: Single-GPU to multi-GPU distributed training with PyTorch DDP
- **Mixed Precision**: Native AMP support for faster training and reduced memory usage
- **Configurable Architecture**: YAML-based configuration for all hyperparameters
- **Data Pipeline**: Support for The Pile, Common Corpus, and other public datasets
- **Evaluation**: Perplexity calculation and downstream task benchmarking
- **Inference**: Text generation with multiple sampling strategies
- **Fine-tuning**: Task-specific adaptation capabilities

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and Prepare Data**
   ```bash
   python data/download_datasets.py
   python data/preprocess.py
   ```

3. **Train Model**
   ```bash
   # Single GPU
   python scripts/train.py
   
   # Multi-GPU
   python scripts/distributed_train.py --gpus 4
   ```

4. **Generate Text**
   ```bash
   python scripts/generate.py --prompt "The future of AI is"
   ```

## Project Structure

- `config/`: YAML configuration files
- `data/`: Data downloading, preprocessing, and loading
- `model/`: Transformer architecture implementation  
- `training/`: Training loop, optimization, and checkpointing
- `evaluation/`: Perplexity calculation and benchmarking
- `inference/`: Text generation and API
- `scripts/`: Main training, evaluation, and generation scripts
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for analysis

## Model Configuration

Default configuration targets a 500M parameter model:
- 12 transformer layers
- 768 hidden dimensions  
- 12 attention heads
- 1024 context length
- 50K vocabulary (BPE tokenization)

## Dataset Support

- The Pile (English text corpus)
- Common Corpus (web crawl data)
- Wikipedia dumps
- Custom datasets via configuration

## Training Features

- **Optimization**: AdamW with cosine learning rate scheduling
- **Mixed Precision**: Automatic mixed precision for efficiency
- **Gradient Accumulation**: Support for large effective batch sizes
- **Checkpointing**: Regular model saving with resume capability
- **Monitoring**: Integration with Weights & Biases

## License

This project is designed to work with openly licensed and public domain data sources to avoid copyright issues.
