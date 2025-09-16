# tokenizers/base_tokenizer.py
from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token indices."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token indices to text."""
        pass
    
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
    
    def __len__(self) -> int:
        return self.vocab_size()
