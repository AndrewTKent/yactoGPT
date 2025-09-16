# scripts/train.py
#!/usr/bin/env python3
"""Main training script for yacto-GPT."""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GPT, GPTConfig
from data.loaders import ShakespeareDataLoader
from training import Trainer, WandbCallback, CheckpointCallback
from utils import setup_logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Handle inheritance
    if 'inherit' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['inherit'])
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        # Merge configs (override base with current)
        for key in config:
            if key != 'inherit':
                if key in base_config and isinstance(base_config[key], dict):
                    base_config[key].update(config[key])
                else:
                    base_config[key] = config[key]
        config = base_config
        
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train yacto-GPT model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_shakespeare.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Create data loader
    logger.info("Loading dataset...")
    data_loader = ShakespeareDataLoader(
        data_path=config['data']['data_path'],
        train_split=config['data']['train_split'],
        block_size=config['model']['block_size'],
        device=config['training']['device']
    )
    
    # Update vocab size in config
    config['model']['vocab_size'] = data_loader.get_tokenizer().vocab_size()
    
    # Create model
    logger.info("Creating model...")
    model_config = GPTConfig(**config['model'])
    model = GPT(model_config)
    
    logger.info(f"Model has {model.get_num_params():,} parameters")
    
    # Save tokenizer
    tokenizer_path = os.path.join('checkpoints', 'tokenizer.txt')
    os.makedirs('checkpoints', exist_ok=True)
    data_loader.get_tokenizer().save(tokenizer_path)
    logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    # Create callbacks
    callbacks = [
        CheckpointCallback(checkpoint_dir='checkpoints'),
        WandbCallback(config, enabled=config['logging'].get('wandb_enabled', True))
    ]
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=data_loader,
        val_loader=data_loader,  # Same loader for train/val split
        config=config,
        callbacks=callbacks
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()