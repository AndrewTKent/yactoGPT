# utils/checkpoint.py
import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_val_loss: float,
    config: Dict[str, Any],
    tokenizer_state: Optional[Dict] = None
):
    """Save model checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        iteration: Current iteration
        best_val_loss: Best validation loss
        config: Training configuration
        tokenizer_state: Tokenizer state dict
    """
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'best_val_loss': best_val_loss,
        'config': config,
        'model_config': {
            'vocab_size': model.config.vocab_size,
            'block_size': model.config.block_size,
            'n_embd': model.config.n_embd,
            'n_head': model.config.n_head,
            'n_layer': model.config.n_layer,
            'dropout': model.config.dropout,
            'bias': model.config.bias,
        }
    }
    
    if tokenizer_state:
        checkpoint['tokenizer_state'] = tokenizer_state
        
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True
) -> int:
    """Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load into
        optimizer: Optimizer to load state into
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Iteration number from checkpoint
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint.get('iteration', 0)