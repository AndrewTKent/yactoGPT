# tokenizers/char_tokenizer.py
from typing import List
from .base_tokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer."""
    
    def __init__(self, chars: str = None):
        """Initialize tokenizer.
        
        Args:
            chars: String containing all characters to include in vocabulary.
                   If None, will be set from data.
        """
        self.chars = chars
        if chars:
            self._build_vocab(chars)
    
    def _build_vocab(self, chars: str):
        """Build vocabulary from characters."""
        self.chars = sorted(list(set(chars)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
    def fit(self, text: str):
        """Fit tokenizer to text data."""
        self._build_vocab(text)
        return self
        
    def encode(self, text: str) -> List[int]:
        """Encode text to indices."""
        if not hasattr(self, 'char_to_idx'):
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode indices to text."""
        if not hasattr(self, 'idx_to_char'):
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        return ''.join([self.idx_to_char.get(idx, '') for idx in tokens])
    
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.chars) if self.chars else 0
    
    def save(self, path: str):
        """Save tokenizer vocabulary."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(''.join(self.chars))
            
    def load(self, path: str):
        """Load tokenizer vocabulary."""
        with open(path, 'r', encoding='utf-8') as f:
            chars = f.read()
            self._build_vocab(chars)
        return self
