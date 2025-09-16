# training/callbacks.py
import os
from typing import Dict, Any

import wandb
import torch

from utils.checkpoint import save_checkpoint


class WandbCallback:
    """Weights & Biases logging callback."""
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """Initialize W&B callback.
        
        Args:
            config: Full configuration dictionary
            enabled: Whether to enable W&B logging
        """
        self.config = config
        self.enabled = enabled and config['logging'].get('wandb_enabled', True)
        
    def on_train_begin(self, trainer):
        """Initialize W&B run."""
        if not self.enabled:
            return
            
        wandb.init(
            project=self.config['logging'].get('wandb_project', 'yacto-gpt'),
            name=self.config['logging'].get('wandb_run_name'),
            config=self.config
        )
        
        # Log model architecture
        wandb.watch(trainer.model)
        
    def on_train_batch_end(self, trainer, loss: float, lr: float):
        """Log batch metrics."""
        if not self.enabled:
            return
            
        wandb.log({
            'train/loss': loss,
            'train/lr': lr,
            'train/iter': trainer.iter_num
        })
        
    def on_eval_end(self, trainer, losses: Dict[str, float]):
        """Log evaluation metrics."""
        if not self.enabled:
            return
            
        wandb.log({
            'eval/train_loss': losses['train'],
            'eval/val_loss': losses['val'],
            'eval/iter': trainer.iter_num
        })
        
    def on_sample(self, trainer, iter_num: int):
        """Log generated samples."""
        if not self.enabled:
            return
            
        # Generate sample
        trainer.model.eval()
        
        context = "\n"  # Start with newline
        tokenizer = trainer.train_loader.get_tokenizer()
        
        # Encode context
        context_tokens = torch.tensor(
            tokenizer.encode(context),
            dtype=torch.long,
            device=trainer.device
        ).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            generated = trainer.model.generate(
                context_tokens,
                max_new_tokens=500,
                temperature=0.8,
                top_k=40
            )
            
        # Decode and log
        generated_text = tokenizer.decode(generated[0].tolist())
        
        wandb.log({
            'samples/text': wandb.Html(f"<pre>{generated_text}</pre>"),
            'samples/iter': iter_num
        })
        
        trainer.model.train()
        
    def on_best_model(self, trainer, losses: Dict[str, float]):
        """Log best model metrics."""
        if not self.enabled:
            return
            
        wandb.log({
            'best/val_loss': losses['val'],
            'best/iter': trainer.iter_num
        })
        
    def on_checkpoint(self, trainer, iter_num: int):
        """Log checkpoint."""
        if not self.enabled:
            return
            
        wandb.log({'checkpoint/iter': iter_num})
        
    def on_train_end(self, trainer):
        """Finish W&B run."""
        if not self.enabled:
            return
            
        wandb.finish()


class CheckpointCallback:
    """Model checkpointing callback."""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        """Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def on_train_begin(self, trainer):
        """Check for existing checkpoints."""
        latest_checkpoint = self._get_latest_checkpoint()
        if latest_checkpoint:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            trainer.iter_num = load_checkpoint(
                latest_checkpoint,
                trainer.model,
                trainer.optimizer
            )
            
    def on_checkpoint(self, trainer, iter_num: int):
        """Save checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_{iter_num}.pt'
        )
        
        save_checkpoint(
            checkpoint_path,
            trainer.model,
            trainer.optimizer,
            iter_num,
            trainer.best_val_loss,
            trainer.config
        )
        
        print(f"Saved checkpoint to {checkpoint_path}")
        
    def on_best_model(self, trainer, losses: Dict[str, float]):
        """Save best model."""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        
        save_checkpoint(
            best_path,
            trainer.model,
            trainer.optimizer,
            trainer.iter_num,
            losses['val'],
            trainer.config
        )
        
        print(f"Saved best model to {best_path}")
        
    def on_train_end(self, trainer):
        """Save final checkpoint."""
        final_path = os.path.join(self.checkpoint_dir, 'final_model.pt')
        
        save_checkpoint(
            final_path,
            trainer.model,
            trainer.optimizer,
            trainer.iter_num,
            trainer.best_val_loss,
            trainer.config
        )
        
        print(f"Saved final model to {final_path}")
        
    def _get_latest_checkpoint(self) -> str:
        """Find the latest checkpoint file."""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ]
        
        if not checkpoints:
            return None
            
        # Sort by iteration number
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(self.checkpoint_dir, checkpoints[-1])