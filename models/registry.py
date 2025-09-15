"""
Model Registry System

This module provides a decorator-based system to register models with string names,
allowing for dynamic model instantiation based on configuration.
"""

import inspect
from typing import Dict, Any, Callable


class ModelRegistry:
    """Registry for storing and retrieving model classes by name."""
    
    def __init__(self):
        self._models: Dict[str, Callable] = {}
    
    def register(self, name: str):
        """
        Decorator to register a model class with a given name.
        
        Args:
            name (str): The name to register the model under
            
        Returns:
            The decorator function
        """
        def decorator(model_class):
            if name in self._models:
                raise ValueError(f"Model '{name}' is already registered")
            
            self._models[name] = model_class
            return model_class
        
        return decorator
    
    def get_model(self, name: str, *args, **kwargs):
        """
        Get and instantiate a model by name.
        
        Args:
            name (str): The name of the model to retrieve
            *args: Positional arguments to pass to the model constructor
            **kwargs: Keyword arguments to pass to the model constructor
            
        Returns:
            An instance of the requested model
            
        Raises:
            KeyError: If the model name is not registered
        """
        if name not in self._models:
            available_models = list(self._models.keys())
            raise KeyError(f"Model '{name}' not found. Available models: {available_models}")
        
        model_class = self._models[name]
        return model_class(*args, **kwargs)
    
    def list_models(self):
        """
        List all registered model names.
        
        Returns:
            List of registered model names
        """
        return list(self._models.keys())
    
    def get_model_info(self, name: str):
        """
        Get information about a registered model.
        
        Args:
            name (str): The name of the model
            
        Returns:
            Dict containing model information
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        
        model_class = self._models[name]
        return {
            'name': name,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'docstring': model_class.__doc__,
            'signature': str(inspect.signature(model_class.__init__))
        }


# Global model registry instance
model_registry = ModelRegistry()

# Convenience function for the decorator
def register_model(name: str):
    """
    Convenience function to register a model.
    
    Args:
        name (str): The name to register the model under
        
    Returns:
        The decorator function
    """
    return model_registry.register(name) 