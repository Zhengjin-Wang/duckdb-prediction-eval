"""
Model registry and factory for managing different baseline models.
"""
from typing import Dict, Type, Union, Any
from abc import ABC, abstractmethod

from .flat_vector import FlatVectorModel, create_flat_vector_model
from .mscn import MSCNModel, create_mscn_model
from .qppnet import QPPNetModel, create_qppnet_model


class ModelConfig:
    """Configuration for model instantiation."""

    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def train(self, workload_results: list, **kwargs) -> dict:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, workload_results: list) -> list:
        """Make predictions."""
        pass

    @abstractmethod
    def evaluate(self, workload_results: list) -> dict:
        """Evaluate model performance."""
        pass

    @abstractmethod
    def save(self, model_path: str):
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_path: str, **kwargs):
        """Load model from disk."""
        pass


class ModelRegistry:
    """Registry for managing different model types."""

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}

    def register(self, model_type: str, factory_func: callable, config_class: Type = ModelConfig):
        """Register a new model type."""
        self.models[model_type] = {
            'factory': factory_func,
            'config_class': config_class
        }

    def create_model(self, model_type: str, **kwargs) -> BaseModel:
        """Create a model instance."""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        factory_func = self.models[model_type]['factory']
        return factory_func(**kwargs)

    def get_model_config(self, model_type: str, **kwargs) -> ModelConfig:
        """Get model configuration."""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        config_class = self.models[model_type]['config_class']
        return config_class(model_type, **kwargs)

    def list_models(self) -> list:
        """List available model types."""
        return list(self.models.keys())


# Global model registry
model_registry = ModelRegistry()

# Register models
model_registry.register('flat_vector', create_flat_vector_model)
model_registry.register('mscn', create_mscn_model)
model_registry.register('qppnet', create_qppnet_model)


def create_model(model_type: str, **kwargs) -> BaseModel:
    """Create a model instance using the registry."""
    return model_registry.create_model(model_type, **kwargs)


def get_available_models() -> list:
    """Get list of available model types."""
    return model_registry.list_models()


def train_model(model_type: str, workload_results: list, **kwargs) -> dict:
    """Train a model of the specified type."""
    model = create_model(model_type, **kwargs)
    return model.train(workload_results, **kwargs)


def evaluate_model(model_path: str, model_type: str, workload_results: list, **kwargs) -> dict:
    """Evaluate a saved model."""
    model = load_model(model_path, model_type, **kwargs)
    return model.evaluate(workload_results)


def load_model(model_path: str, model_type: str, **kwargs) -> BaseModel:
    """Load a saved model."""
    if model_type == 'flat_vector':
        return FlatVectorModel.load(model_path)
    elif model_type == 'mscn':
        return MSCNModel.load(model_path, **kwargs)
    elif model_type == 'qppnet':
        return QPPNetModel.load(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")