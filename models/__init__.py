"""
Models package initialization.

This module imports all model classes to ensure they are registered
with the model registry when the package is imported.
"""

from .registry import model_registry, register_model
from .encoders import ProbCLIPModel
from .gpo import GPO
# Import model module to ensure registration happens
from . import model

__all__ = [
    'model_registry',
    'register_model', 
    'ProbCLIPModel',
    'GPO'
]