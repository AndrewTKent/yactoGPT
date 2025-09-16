# README.md

# yacto-GPT 🚀

A Clean and Tiny OpenAI GPT - A minimal, clean, and scalable implementation of GPT.

## Features

- **Clean Architecture**: Modular design with clear separation of concerns
- **Configuration-Driven**: YAML-based configuration for easy experimentation
- **Production Ready**: Docker support for both training and inference
- **Experiment Tracking**: Integrated Weights & Biases (wandb) support
- **Scalable Design**: Easy to extend with new datasets, tokenizers, and models
- **GPU/CPU Support**: Automatic device detection with graceful fallback
- **REST API**: Built-in inference server with streaming support

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/andrewtkent/yacto-gpt.git
cd yacto-gpt

# Install dependencies
pip install -r requirements/base.txt -r requirements/train.txt

# Download Shakespeare dataset
python scripts/prepare_data.py --dataset shakespeare
```

### Training

```bash
# Train with default configuration
python scripts/train.py --config config/train_shakespeare.yaml

# Resume from checkpoint
python scripts/train.py --config config/train_shakespeare.yaml --resume
```

### Generation

```bash
# Generate text
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "To be or not to be"

# Stream generation
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "Hello" --stream

# Multiple samples
python scripts/generate.py --checkpoint checkpoints/best_model.pt --num-samples 3
```

### Inference Server

```bash
# Start the inference server
python scripts/serve.py --checkpoint checkpoints/best_model.pt

# Or with Docker
docker-compose up yacto-gpt-inference
```

#### API Usage

```python
import requests

# Generate text
response = requests.post('http://localhost:8080/generate', json={
    'prompt': 'To be or not to be',
    'max_new_tokens': 100,
    'temperature': 0.8
})
print(response.json()['generated'])

# Stream generation
response = requests.post('http://localhost:8080/generate/stream', json={
    'prompt': 'Hello',
    'max_new_tokens': 100
}, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## Docker Deployment

### Build Images

```bash
# Build inference image
docker build -f docker/Dockerfile.inference -t yacto-gpt:inference .

# Build training image
docker build -f docker/Dockerfile.training -t yacto-gpt:training .
```

### Run with Docker Compose

```bash
# Start inference server
docker-compose up yacto-gpt-inference

# Run training (with GPU)
docker-compose --profile training up yacto-gpt-training
```

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Project Structure

```
yacto-gpt/
├── config/              # Configuration files
├── data/                # Dataset storage
├── models/              # Model architecture
├── training/            # Training logic and callbacks
├── inference/           # Inference and serving
├── tokenizers/          # Tokenization
├── utils/               # Utilities
├── scripts/             # Main scripts
├── docker/              # Docker files
└── checkpoints/         # Model checkpoints
```

## Configuration

### Model Configuration

Edit `config/train_shakespeare.yaml`:

```yaml
model:
  n_embd: 384       # Embedding dimension
  n_head: 6         # Number of attention heads
  n_layer: 6        # Number of transformer layers
  block_size: 256   # Context window size
  dropout: 0.2      # Dropout rate
```

### Training Configuration

```yaml
training:
  batch_size: 64
  learning_rate: 3e-4
  max_iters: 10000
  eval_interval: 500
  compile_model: true  # PyTorch 2.0+ compilation
```

## Experiment Tracking

### Setup Weights & Biases

1. Create an account at [wandb.ai](https://wandb.ai)
2. Get your API key
3. Set environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

4. Training will automatically log to wandb

### Disable wandb

Set in config:

```yaml
logging:
  wandb_enabled: false
```

## Model Sizes

| Configuration | Parameters | Training Time (V100) | 
|--------------|------------|---------------------|
| Tiny (default) | ~10M | ~30 min |
| Small | ~50M | ~2 hours |
| Medium | ~150M | ~8 hours |
| Large | ~350M | ~24 hours |

## Advanced Features

### Custom Datasets

1. Create a new data loader in `data/loaders/`:

```python
from .base_loader import BaseDataLoader

class MyDataLoader(BaseDataLoader):
    def prepare_data(self):
        # Load and process your data
        pass
```

2. Update configuration to use your dataset

### Custom Tokenizers

1. Implement in `tokenizers/`:

```python
from .base_tokenizer import BaseTokenizer

class MyTokenizer(BaseTokenizer):
    def encode(self, text):
        # Your encoding logic
        pass
```

### Distributed Training

For multi-GPU training, use PyTorch's distributed launcher:

```bash
torchrun --nproc_per_node=4 scripts/train.py --config config/train_shakespeare.yaml
```

## Performance Optimizations

- **Flash Attention**: Automatically used when available (PyTorch 2.0+)
- **Mixed Precision**: FP16/BF16 training for faster computation
- **Model Compilation**: PyTorch 2.0 compilation for ~30% speedup
- **Gradient Accumulation**: For larger effective batch sizes
- **CPU Offloading**: For training larger models on limited GPU memory

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in config
- Reduce `block_size` for smaller context
- Use gradient accumulation
- Enable CPU offloading

### Slow Training

- Enable model compilation: `compile_model: true`
- Use mixed precision: `dtype: float16`
- Check GPU utilization: `nvidia-smi`

### Poor Generation Quality

- Train for more iterations
- Increase model size
- Tune sampling parameters (temperature, top_k, top_p)

## Citation

If you use yacto-GPT in your research, please cite:

```bibtex
@software{yacto-gpt,
  title = {yacto-GPT: Yet Another Clean and Tiny OpenAI GPT},
  year = {2024},
  url = {https://github.com/yourusername/yacto-gpt}
}
```

## Acknowledgments

- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- Built with PyTorch and Hugging Face tools
- Shakespeare dataset from [char-rnn](https://github.com/karpathy/char-rnn)

## License

MIT License - see LICENSE file for details

---

# setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yacto-gpt",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Yet Another Clean and Tiny OpenAI GPT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yacto-gpt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "train": [
            "wandb>=0.15.0",
            "matplotlib>=3.7.0",
            "tensorboard>=2.13.0",
        ],
        "inference": [
            "flask>=3.0.0",
            "gunicorn>=21.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "yacto-train=scripts.train:main",
            "yacto-generate=scripts.generate:main",
            "yacto-serve=scripts.serve:main",
            "yacto-prepare=scripts.prepare_data:main",
        ],
    },
)

---

# .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pt
*.pth
*.ckpt
checkpoints/
logs/
runs/

# Data
data/raw/
data/processed/
*.txt
*.csv
*.json

# Weights & Biases
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Environment
.env
.env.local

# Docker
.dockerignore

# Notebooks
.ipynb_checkpoints/
*.ipynb

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

---

# pyproject.toml

[build-system]
requires = ["setuptools>=45", "wheel", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
