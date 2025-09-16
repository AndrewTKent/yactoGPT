# data/loaders/shakespeare.py
import os
from typing import Tuple, Optional

import torch
import numpy as np

from .base_loader import BaseDataLoader
from tokenizers import CharTokenizer


class ShakespeareDataLoader(BaseDataLoader):
    """Shakespeare dataset loader."""
    
    def __init__(
        self,
        data_path: str,
        train_split: float = 0.9,
        block_size: int = 256,
        device: str = 'cpu',
        tokenizer: Optional[CharTokenizer] = None
    ):
        """Initialize Shakespeare data loader.
        
        Args:
            data_path: Path to Shakespeare text file
            train_split: Fraction of data for training
            block_size: Context window size
            device: Device to load data to
            tokenizer: Tokenizer instance (will create if None)
        """
        self.data_path = data_path
        self.train_split = train_split
        self.block_size = block_size
        self.device = device
        
        # Load and prepare data
        self.tokenizer = tokenizer or CharTokenizer()
        self.train_data, self.val_data = self.prepare_data()
        
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and prepare Shakespeare dataset."""
        # Load text
        if not os.path.exists(self.data_path):
            # Download Shakespeare if not present
            self._download_shakespeare()
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Fit tokenizer if needed
        if self.tokenizer.vocab_size() == 0:
            self.tokenizer.fit(text)
            
        # Encode entire text
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        # Split into train and validation
        n = int(self.train_split * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        return train_data, val_data
    
    def _download_shakespeare(self):
        """Download Shakespeare dataset."""
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        urllib.request.urlretrieve(url, self.data_path)
        print(f"Downloaded Shakespeare dataset to {self.data_path}")
        
    def get_batch(
        self, 
        split: str, 
        batch_size: int, 
        block_size: int = None,
        device: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data.
        
        Args:
            split: 'train' or 'val'
            batch_size: Batch size
            block_size: Context window size
            device: Device to load to
            
        Returns:
            Tuple of input and target tensors
        """
        block_size = block_size or self.block_size
        device = device or self.device
        
        data = self.train_data if split == 'train' else self.val_data
        
        # Generate random positions
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
        # Get input and target sequences
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        
        # Move to device
        if device != 'cpu':
            x, y = x.to(device), y.to(device)
            
        return x, y
    
    def get_tokenizer(self) -> CharTokenizer:
        """Get tokenizer."""
        return self.tokenizer
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.train_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        # Get a sequence of length block_size
        if idx + self.block_size + 1 >= len(self.train_data):
            idx = np.random.randint(0, len(self.train_data) - self.block_size - 1)
            
        x = self.train_data[idx:idx + self.block_size]
        y = self.train_data[idx + 1:idx + self.block_size + 1]
        
        return x, y