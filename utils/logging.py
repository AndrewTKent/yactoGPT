# utils/logging.py
import logging
import sys
from typing import Optional


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None
):
    """Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional file to log to
    """
    # Create logger
    logger = logging.getLogger('yacto-gpt')
    logger.setLevel(getattr(logging, level))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger