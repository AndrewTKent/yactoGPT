# utils/device.py
import torch


def get_device(device: str = 'auto') -> str:
    """Get the appropriate device.
    
    Args:
        device: Device specification ('auto', 'cuda', 'cpu')
        
    Returns:
        Device string
    """
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device


def get_dtype(dtype: str) -> torch.dtype:
    """Convert string dtype to torch dtype.
    
    Args:
        dtype: String representation of dtype
        
    Returns:
        torch.dtype object
    """
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    return dtype_map.get(dtype, torch.float32)
