"""Model adapters for architecture-specific handling."""

from sharingan.models.base import ModelAdapter
from sharingan.models.registry import get_adapter, register_adapter

__all__ = ["ModelAdapter", "get_adapter", "register_adapter"]
