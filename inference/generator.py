# inference/generator.py
from typing import Optional, List
import torch

from models import GPT, GPTConfig
from tokenizers import CharTokenizer
from utils.device import get_device, get_dtype


class Generator:
    """Text generation interface."""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        dtype: str = 'float16'
    ):
        """Initialize generator.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use
            dtype: Data type for inference
        """
        self.device = get_device(device)
        self.dtype = get_dtype(dtype)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load tokenizer
        self.tokenizer = CharTokenizer()
        if 'tokenizer_state' in checkpoint:
            self.tokenizer.chars = checkpoint['tokenizer_state']['chars']
            self.tokenizer._build_vocab(self.tokenizer.chars)
        else:
            # Try to load from saved file
            import os
            tokenizer_path = os.path.join(
                os.path.dirname(model_path),
                'tokenizer.txt'
            )
            if os.path.exists(tokenizer_path):
                self.tokenizer.load(tokenizer_path)
                
        # Create model
        config = GPTConfig(**checkpoint['model_config'])
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Convert to specified dtype
        if self.dtype == torch.float16:
            self.model = self.model.half()
        elif self.dtype == torch.bfloat16:
            self.model = self.model.bfloat16()
            
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.95,
        num_samples: int = 1
    ) -> List[str]:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            num_samples: Number of samples to generate
            
        Returns:
            List of generated text samples
        """
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=self.device)
        x = x.unsqueeze(0).repeat(num_samples, 1)
        
        # Generate
        y = self.model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Decode results
        results = []
        for i in range(num_samples):
            tokens = y[i].tolist()
            text = self.tokenizer.decode(tokens)
            results.append(text)
            
        return results
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.95
    ):
        """Generate text token by token (streaming).
        
        Yields:
            Generated tokens one at a time
        """
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate token by token
        for _ in range(max_new_tokens):
            # Get next token
            with torch.no_grad():
                # Crop to block size if needed
                if x.size(1) > self.model.config.block_size:
                    x_cond = x[:, -self.model.config.block_size:]
                else:
                    x_cond = x
                    
                logits, _ = self.model(x_cond)
                logits = logits[:, -1, :] / temperature
                
                # Apply sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float('Inf')
                    
                # Sample
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append and yield
                x = torch.cat((x, idx_next), dim=1)
                
                # Decode and yield the new token
                token = idx_next[0].item()
                char = self.tokenizer.decode([token])
                yield char