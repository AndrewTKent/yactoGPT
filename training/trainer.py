# training/trainer.py
import os
import time
import math
from typing import Dict, Any, List, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from models.utils import configure_optimizers
from utils.device import get_device, get_dtype
from utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """Main training loop."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        callbacks: List = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            config: Training configuration
            callbacks: List of callback objects
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.callbacks = callbacks or []
        
        # Setup device and dtype
        self.device = get_device(config['training'].get('device', 'auto'))
        self.dtype = get_dtype(config['training'].get('dtype', 'float16'))
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=self.dtype)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Compile model if requested (PyTorch 2.0+)
        if config['training'].get('compile_model', False) and hasattr(torch, 'compile'):
            print("Compiling model...")
            self.model = torch.compile(self.model)
            
        # Setup optimizer
        self.optimizer = configure_optimizers(
            self.model,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            device_type=self.device
        )
        
        # Setup gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == torch.float16))
        
        # Training state
        self.iter_num = 0
        self.best_val_loss = float('inf')
        
        # Initialize callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
            
    def train(self):
        """Run training loop."""
        config = self.config['training']
        
        # Training loop
        train_start_time = time.time()
        
        for iter_num in range(config['max_iters']):
            self.iter_num = iter_num
            
            # Determine learning rate
            lr = self._get_lr(iter_num, config)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # Evaluate periodically
            if iter_num % config['eval_interval'] == 0:
                losses = self._estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                # Save best model
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    for callback in self.callbacks:
                        callback.on_best_model(self, losses)
                        
                # Log metrics
                for callback in self.callbacks:
                    callback.on_eval_end(self, losses)
                    
            # Sample from model periodically
            if iter_num % config.get('sample_interval', 500) == 0:
                for callback in self.callbacks:
                    callback.on_sample(self, iter_num)
                    
            # Training step
            self.model.train()
            
            # Get batch
            X, Y = self.train_loader.get_batch(
                'train',
                batch_size=config['batch_size'],
                device=self.device
            )
            
            # Forward pass
            with self.ctx:
                logits, loss = self.model(X, Y)
                
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.get('grad_clip', 0.0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['grad_clip'])
                
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Log training metrics
            if iter_num % config.get('log_interval', 100) == 0:
                elapsed = time.time() - train_start_time
                print(f"iter {iter_num}: loss {loss.item():.4f}, time {elapsed:.2f}s, lr {lr:.2e}")
                for callback in self.callbacks:
                    callback.on_train_batch_end(self, loss.item(), lr)
                    
            # Checkpoint periodically
            if iter_num % config.get('checkpoint_interval', 1000) == 0:
                for callback in self.callbacks:
                    callback.on_checkpoint(self, iter_num)
                    
        # Training complete
        for callback in self.callbacks:
            callback.on_train_end(self)
            
        print(f"Training complete! Best validation loss: {self.best_val_loss:.4f}")
        
    def _get_lr(self, iter_num: int, config: Dict[str, Any]) -> float:
        """Calculate learning rate with warmup and cosine decay."""
        # Warmup
        if iter_num < config['warmup_iters']:
            return config['learning_rate'] * iter_num / config['warmup_iters']
        
        # Cosine decay
        if iter_num > config['warmup_iters']:
            decay_ratio = (iter_num - config['warmup_iters']) / (config['max_iters'] - config['warmup_iters'])
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return config['learning_rate'] * coeff
            
        return config['learning_rate']
    
    @torch.no_grad()
    def _estimate_loss(self) -> Dict[str, float]:
        """Estimate loss on train and validation sets."""
        out = {}
        self.model.eval()
        
        for split in ['train', 'val']:
            losses = torch.zeros(self.config['training']['eval_iters'])
            
            for k in range(self.config['training']['eval_iters']):
                X, Y = self.train_loader.get_batch(
                    split,
                    batch_size=self.config['training']['batch_size'],
                    device=self.device
                )
                
                with self.ctx:
                    logits, loss = self.model(X, Y)
                    
                losses[k] = loss.item()
                
            out[split] = losses.mean().item()
            
        self.model.train()
        return out