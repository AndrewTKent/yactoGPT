# data/loaders/__init__.py
from .base_loader import BaseDataLoader
from .shakespeare import ShakespeareDataLoader

__all__ = ['BaseDataLoader', 'ShakespeareDataLoader']