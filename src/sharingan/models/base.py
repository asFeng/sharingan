"""Base model adapter class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


class ModelAdapter(ABC):
    """Abstract base class for model-specific adapters.

    Adapters handle architecture-specific details like:
    - GQA (Grouped Query Attention) expansion
    - Layer/head naming conventions
    - Special token handling
    """

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.config = model.config

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of attention layers."""
        pass

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Number of attention heads (query heads)."""
        pass

    @property
    def num_kv_heads(self) -> int:
        """Number of key-value heads (for GQA models).

        Returns num_heads if not GQA.
        """
        return self.num_heads

    @property
    def gqa_ratio(self) -> int:
        """Ratio of query heads to KV heads."""
        return self.num_heads // self.num_kv_heads

    @property
    def uses_gqa(self) -> bool:
        """Whether model uses Grouped Query Attention."""
        return self.num_kv_heads < self.num_heads

    def expand_gqa_attention(self, attention: "torch.Tensor") -> "torch.Tensor":
        """Expand GQA attention from KV heads to full Q heads.

        Args:
            attention: Attention tensor [batch, kv_heads, seq, seq]

        Returns:
            Expanded attention [batch, q_heads, seq, seq]
        """
        if not self.uses_gqa:
            return attention

        # Repeat each KV head for its group of Q heads
        # [batch, kv_heads, seq, seq] -> [batch, q_heads, seq, seq]
        return attention.repeat_interleave(self.gqa_ratio, dim=1)

    def process_attention(self, attention: "torch.Tensor") -> np.ndarray:
        """Process raw attention tensor to numpy array.

        Args:
            attention: Raw attention from model [batch, heads, seq, seq]

        Returns:
            Processed attention as numpy [heads, seq, seq]
        """
        # Remove batch dimension (assume batch=1)
        attn = attention[0]

        # Expand GQA if needed
        if self.uses_gqa:
            attn = self.expand_gqa_attention(attn.unsqueeze(0))[0]

        return attn.cpu().float().numpy()

    def get_layer_name(self, layer_idx: int) -> str:
        """Get human-readable name for a layer."""
        return f"Layer {layer_idx}"

    def get_head_name(self, head_idx: int) -> str:
        """Get human-readable name for a head."""
        return f"Head {head_idx}"


class DefaultAdapter(ModelAdapter):
    """Default adapter for models without specific handling."""

    @property
    def num_layers(self) -> int:
        return getattr(self.config, "num_hidden_layers", 12)

    @property
    def num_heads(self) -> int:
        return getattr(self.config, "num_attention_heads", 12)

    @property
    def num_kv_heads(self) -> int:
        # Check for GQA configuration
        kv_heads = getattr(self.config, "num_key_value_heads", None)
        if kv_heads is not None:
            return kv_heads
        return self.num_heads
