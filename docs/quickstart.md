# yacto-GPT Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Set up the project structure

```bash
# Create project directory
mkdir yacto-gpt
cd yacto-gpt

# Create all necessary directories
mkdir -p config data/raw data/processed models training inference tokenizers utils scripts docker checkpoints logs
mkdir -p data/loaders requirements tests
```

### 2. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install numpy pyyaml tqdm wandb flask
```

### 3. Download Shakespeare dataset

```bash
python scripts/prepare_data.py --dataset shakespeare
```

### 4. Train your first model

```bash
# Small model for quick testing (5-10 minutes on GPU)
python scripts/train.py --config config/train_shakespeare.yaml
```

### 5. Generate text

```bash
# Generate from your trained model
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "To be or not to be"
```

## 📊 Monitor Training with W&B

1. Sign up at [wandb.ai](https://wandb.ai)
2. Get your API key from settings
3. Set environment variable:
```bash
export WANDB_API_KEY=your_key_here
```
4. Training will automatically log metrics, loss curves, and sample generations

## 🐳 Docker Deployment (Production)

```bash
# Build and run inference server
docker build -f docker/Dockerfile.inference -t yacto-gpt:inference .
docker run -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints yacto-gpt:inference

# Test the API
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_new_tokens": 50}'
```

## ⚙️ Configuration Examples

### Tiny Model (Testing - 5 min training)
```yaml
model:
  n_embd: 128
  n_head: 4
  n_layer: 4
  block_size: 128
training:
  max_iters: 1000
```

### Small Model (Default - 30 min training)
```yaml
model:
  n_embd: 384
  n_head: 6
  n_layer: 6
  block_size: 256
training:
  max_iters: 10000
```

### Medium Model (Better quality - 2-4 hours)
```yaml
model:
  n_embd: 768
  n_head: 12
  n_layer: 12
  block_size: 512
training:
  max_iters: 20000
```

## 📝 Common Commands

```bash
# Training
python scripts/train.py --config config/train_shakespeare.yaml
python scripts/train.py --config config/train_shakespeare.yaml --resume  # Resume from checkpoint

# Generation
python scripts/generate.py --prompt "Hello world" --max-tokens 100
python scripts/generate.py --prompt "Hello" --stream  # Stream output
python scripts/generate.py --prompt "Test" --num-samples 5  # Multiple samples

# Serving
python scripts/serve.py --checkpoint checkpoints/best_model.pt  # Start API server
```

## 🎯 Tips for Best Results

1. **Start Small**: Use tiny config for testing your setup
2. **GPU Recommended**: Training is 10-100x faster on GPU
3. **Monitor Loss**: Validation loss should decrease steadily
4. **Experiment with Sampling**: Try different temperature (0.5-1.0) and top-k (10-100) values
5. **Save Best Model**: The checkpoint callback automatically saves the best model

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `batch_size` or `block_size` in config |
| Slow training | Enable `compile_model: true` (PyTorch 2.0+) |
| Poor generation | Train longer or increase model size |
| Can't import modules | Add project root to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)` |

## 📚 Next Steps

1. **Custom Dataset**: Implement your own data loader in `data/loaders/`
2. **Scale Up**: Try larger models and longer training
3. **Deploy**: Use Docker for production deployment
4. **Optimize**: Enable Flash Attention, mixed precision training
5. **Experiment**: Try different architectures, tokenizers, datasets

Happy training! 🎉