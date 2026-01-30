"""Configuration for Sharingan."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SharinganConfig:
    """Configuration for Sharingan analyzer.

    Attributes:
        device: Device to run model on ("auto", "cuda", "cpu", "mps")
        dtype: Data type for model ("auto", "float16", "bfloat16", "float32")
        cache_dir: Directory for model cache (None uses HF default)
        max_memory: Maximum GPU memory to use (e.g., "8GB")
        offload_to_cpu: Whether to offload attention to CPU for memory savings
        global_max_size: Maximum size for global view downsampling
        local_window_size: Window size for local view
        default_colormap: Default colormap for heatmaps
        theme: Color theme ("dark", "light")
    """

    device: str = "auto"
    dtype: str = "auto"
    cache_dir: Path | None = None
    max_memory: str | None = None
    offload_to_cpu: bool = False
    global_max_size: int = 256
    local_window_size: int = 512
    default_colormap: str = "Reds"
    theme: str = "dark"

    # Theme colors (Sharingan-inspired)
    colors: dict = field(
        default_factory=lambda: {
            "primary": "#B91C1C",  # Deep red
            "secondary": "#1F2937",  # Dark gray
            "accent": "#EF4444",  # Bright red
            "background": "#111827",  # Near black
            "text": "#F9FAFB",  # Off white
            "grid": "#374151",  # Medium gray
        }
    )

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != "auto":
            return self.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def resolve_dtype(self):
        """Resolve 'auto' dtype to actual dtype."""
        import torch

        if self.dtype != "auto":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            return dtype_map.get(self.dtype, torch.float32)

        device = self.resolve_device()
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
