"""
ACL25 CoPE: Composed Image Retrieval Models

This package provides implementations of various composed image retrieval models
including Probabilistic CLIP (ProbCLIP) with CoPE loss and GPO aggregation.
"""

__version__ = "0.1.0"
__author__ = "Haomiao Tang - thm23@mails.tsinghua.edu.cn"

# Import main modules
from . import models
from . import util
from . import engine

# Import main classes and functions for easy access
from .models import (
    model_registry,
    register_model,
    ProbCLIPModel,
    GPO
)

# Import training and evaluation functions
from .engine import (
    train_one_epoch,
    evaluate_probabilistic,
    update_ema,
    compute_probabilistic_distances
)

# Import training function
from .train import main as train_main

__all__ = [
    "__version__",
    "__author__",
    "models",
    "util", 
    "engine",
    "model_registry",
    "register_model",
    "ProbCLIPModel",
    "GPO",
    "train_one_epoch",
    "evaluate_probabilistic", 
    "update_ema",
    "compute_probabilistic_distances",
    "train_main"
]
