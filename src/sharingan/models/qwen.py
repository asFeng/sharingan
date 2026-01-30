"""Qwen model adapter with GQA handling."""

from sharingan.models.base import ModelAdapter
from sharingan.models.registry import register_adapter


@register_adapter("qwen")
class QwenAdapter(ModelAdapter):
    """Adapter for Qwen models.

    Qwen3 models use Grouped Query Attention (GQA):
    - Qwen3-0.6B: 16 Q heads, 8 KV heads (2:1 ratio)
    - Qwen3-1.8B: 16 Q heads, 8 KV heads (2:1 ratio)
    - Qwen3-4B: 32 Q heads, 8 KV heads (4:1 ratio)
    """

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return getattr(self.config, "num_key_value_heads", self.num_heads)

    def get_layer_name(self, layer_idx: int) -> str:
        return f"Qwen Layer {layer_idx}"

    def get_head_name(self, head_idx: int) -> str:
        if self.uses_gqa:
            kv_group = head_idx // self.gqa_ratio
            return f"Head {head_idx} (KV group {kv_group})"
        return f"Head {head_idx}"
