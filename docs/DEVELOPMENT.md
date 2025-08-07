# Development Guide

This guide covers development practices, testing, and contributing to the Neural LLM project.

## 🧪 Testing

### Running Tests

The project includes comprehensive unit tests for core components:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test modules
python -m unittest tests.test_model
python -m unittest tests.test_tokenizer
python -m unittest tests.test_generation

# Run with different verbosity levels
python tests/run_tests.py --quiet          # Minimal output
python tests/run_tests.py --verbose        # Detailed output

# Stop on first failure
python tests/run_tests.py --failfast

# List all available tests
python tests/run_tests.py --list
```

### Test Coverage

The test suite covers:

- **Model Architecture** (`test_model.py`):
  - Model configuration and initialization
  - Multi-head attention mechanisms
  - Transformer blocks and full GPT model
  - CUDA/CPU compatibility
  - Parameter counting and training modes

- **Tokenization** (`test_tokenizer.py`):
  - Tokenizer training and initialization
  - Encode/decode consistency
  - Special token handling
  - Save/load functionality
  - Batch processing

- **Text Generation** (`test_generation.py`):
  - Text generator initialization
  - Single and batch generation
  - Configuration updates
  - Sampling strategies (temperature, top-k, top-p)

### Adding New Tests

When adding new functionality, please include corresponding tests:

1. Create test files following the `test_*.py` naming convention
2. Use `unittest.TestCase` as the base class
3. Include setUp and tearDown methods for test fixtures
4. Test both success and failure cases
5. Mock external dependencies when appropriate

Example test structure:
```python
import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from your_module import YourClass

class TestYourClass(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.instance = YourClass()
    
    def test_functionality(self):
        """Test specific functionality."""
        result = self.instance.method()
        self.assertEqual(result, expected_value)

if __name__ == '__main__':
    unittest.main()
```

## 🏗️ Code Structure

### Project Organization

```
buildingLLMFromScratch/
├── config/          # Configuration files (YAML)
├── data/           # Data processing pipeline
├── model/          # Neural network architecture
├── training/       # Training infrastructure  
├── evaluation/     # Evaluation and benchmarking
├── inference/      # Text generation and serving
├── scripts/        # Main execution scripts
├── tests/          # Unit tests
└── docs/           # Documentation
```

### Coding Standards

- **Python Style**: Follow PEP 8 guidelines
- **Documentation**: Include docstrings for all classes and functions
- **Type Hints**: Use type hints for function parameters and return values
- **Error Handling**: Include appropriate try/catch blocks and error messages
- **Logging**: Use the logging module for debug/info messages

### Configuration Management

All configuration is managed through YAML files:

- **Model Config**: Architecture parameters (layers, dimensions, vocab size)
- **Training Config**: Hyperparameters (learning rate, batch size, epochs)
- **Data Config**: Data processing parameters (tokenization, preprocessing)
- **Hardware Config**: Hardware-specific optimizations

## 🔧 Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/frankmark94/buildingLLMFromScratch.git
cd buildingLLMFromScratch

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-cov black flake8 mypy

# Run tests to verify setup
python tests/run_tests.py
```

### Making Changes

1. **Create Feature Branch**: 
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Implement your feature following coding standards

3. **Add Tests**: Include unit tests for new functionality

4. **Run Tests**: Ensure all tests pass
   ```bash
   python tests/run_tests.py
   ```

5. **Update Documentation**: Update relevant documentation files

6. **Commit Changes**:
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

### Hardware-Specific Development

When developing hardware-specific features:

1. Test on multiple hardware configurations if possible
2. Use the hardware detection utilities in `scripts/hardware_setup.py`
3. Add appropriate fallbacks for unsupported hardware
4. Update hardware configuration files as needed

## 📊 Performance Optimization

### Profiling

Use PyTorch's built-in profiling tools:

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Your training/inference code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory Optimization

- Use gradient checkpointing for large models
- Enable mixed precision training (AMP)
- Monitor memory usage during development
- Use memory-mapped files for large datasets

### Speed Optimization

- Use compiled models (`torch.compile`) when available
- Optimize data loading with appropriate `num_workers`
- Use efficient attention implementations (FlashAttention)
- Profile bottlenecks regularly

## 🐛 Debugging

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Slow Training**:
   - Check data loading bottlenecks
   - Verify GPU utilization
   - Enable model compilation

3. **Poor Generation Quality**:
   - Check tokenizer training
   - Verify model architecture
   - Adjust generation parameters

### Debug Utilities

The project includes several debugging utilities:

```bash
# Hardware diagnostics
python scripts/hardware_setup.py

# Model architecture inspection
python -c "from model import GPTModel, ModelConfig; model = GPTModel(ModelConfig()); print(model)"

# Tokenizer debugging
python -c "from data.tokenizer import LLMTokenizer; tokenizer = LLMTokenizer(); print(tokenizer.encode('test'))"
```

## 📝 Documentation

### Updating Documentation

- Update `README.md` for user-facing changes
- Update `DEVELOPMENT.md` for developer-facing changes
- Include docstrings for all new functions and classes
- Add examples for new features

### Documentation Standards

- Use clear, concise language
- Include code examples where appropriate
- Keep documentation in sync with code changes
- Use markdown formatting consistently

## 🚀 Deployment

### Model Deployment

The project supports several deployment options:

1. **Local Generation**: Command-line tools and interactive interfaces
2. **API Server**: FastAPI-based REST API
3. **Batch Processing**: Scripts for batch inference
4. **Fine-tuned Models**: Domain-specific model deployment

### Production Considerations

- Monitor model performance and quality
- Implement proper error handling and logging
- Use appropriate hardware for production workloads
- Consider model quantization for deployment efficiency

## 🤝 Contributing

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request with clear description
5. Respond to review feedback

### Code Review Process

All contributions go through code review:

- Code follows project standards
- Tests are included and passing
- Documentation is updated
- Changes are backwards compatible when possible

### Issue Reporting

When reporting issues:

1. Include system information (OS, Python version, hardware)
2. Provide minimal reproducible example
3. Include relevant error messages and logs
4. Check existing issues first

---

Happy coding! 🎉