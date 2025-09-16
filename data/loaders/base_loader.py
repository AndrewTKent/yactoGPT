# data/loaders/base_loader.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import Dataset


class BaseDataLoader(Dataset, ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare and return train and validation data."""
        pass
    
    @abstractmethod
    def get_batch(self, split: str, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data."""
        pass
    
    @abstractmethod
    def get_tokenizer(self):
        """Get the tokenizer used by this loader."""
        pass