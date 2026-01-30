"""Model adapter registry."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from sharingan.models.base import ModelAdapter, DefaultAdapter

if TYPE_CHECKING:
    from transformers import PreTrainedModel

# Registry of model adapters
_ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register_adapter(pattern: str) -> Callable:
    """Decorator to register a model adapter.

    Args:
        pattern: Model name pattern to match (case-insensitive substring)

    Example:
        @register_adapter("qwen")
        class QwenAdapter(ModelAdapter):
            ...
    """

    def decorator(cls: type[ModelAdapter]) -> type[ModelAdapter]:
        _ADAPTERS[pattern.lower()] = cls
        return cls

    return decorator


def get_adapter(model: "PreTrainedModel", model_name: str) -> ModelAdapter:
    """Get appropriate adapter for a model.

    Args:
        model: The loaded model
        model_name: Model name or path

    Returns:
        ModelAdapter instance for the model
    """
    model_name_lower = model_name.lower()

    # Check registered adapters
    for pattern, adapter_cls in _ADAPTERS.items():
        if pattern in model_name_lower:
            return adapter_cls(model, model_name)

    # Check model architecture type
    arch = getattr(model.config, "architectures", [""])[0].lower()
    for pattern, adapter_cls in _ADAPTERS.items():
        if pattern in arch:
            return adapter_cls(model, model_name)

    # Fall back to default
    return DefaultAdapter(model, model_name)


# Import adapters to register them
from sharingan.models import qwen  # noqa: F401, E402
