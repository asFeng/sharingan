"""Multi-level attention downsampling for long sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage


@dataclass
class MultiScaleAttention:
    """Container for multi-scale attention views.

    Attributes:
        global_view: Heavily downsampled overview (max 256x256)
        segment_view: Attention between logical segments
        local_views: Dict of local full-resolution windows
        original_shape: Original attention shape
        tokens: Token labels
    """

    global_view: np.ndarray
    segment_view: np.ndarray | None
    local_views: dict[str, np.ndarray]
    original_shape: tuple[int, ...]
    tokens: list[str]
    segment_boundaries: list[int] | None = None


def downsample_attention(
    attention: np.ndarray,
    target_size: int = 256,
    method: Literal["mean", "max", "sample"] = "mean",
) -> np.ndarray:
    """Downsample attention matrix to target size.

    Args:
        attention: Attention matrix [seq, seq] or [heads, seq, seq]
        target_size: Target size for each dimension
        method: Downsampling method
            - "mean": Average pooling (preserves overall patterns)
            - "max": Max pooling (preserves hotspots)
            - "sample": Uniform sampling (fastest)

    Returns:
        Downsampled attention matrix
    """
    if attention.ndim == 2:
        return _downsample_2d(attention, target_size, method)
    elif attention.ndim == 3:
        # [heads, seq, seq]
        return np.stack(
            [_downsample_2d(attention[h], target_size, method) for h in range(attention.shape[0])]
        )
    else:
        raise ValueError(f"Expected 2D or 3D array, got {attention.ndim}D")


def _downsample_2d(
    attention: np.ndarray,
    target_size: int,
    method: str,
) -> np.ndarray:
    """Downsample 2D attention matrix."""
    h, w = attention.shape

    if h <= target_size and w <= target_size:
        return attention

    if method == "sample":
        # Uniform sampling
        row_indices = np.linspace(0, h - 1, target_size, dtype=int)
        col_indices = np.linspace(0, w - 1, target_size, dtype=int)
        return attention[np.ix_(row_indices, col_indices)]

    elif method == "mean":
        # Block average using scipy
        factors = (max(1, h // target_size), max(1, w // target_size))
        return ndimage.uniform_filter(attention, size=factors)[:: factors[0], :: factors[1]][
            :target_size, :target_size
        ]

    elif method == "max":
        # Block max pooling
        factors = (max(1, h // target_size), max(1, w // target_size))
        return ndimage.maximum_filter(attention, size=factors)[:: factors[0], :: factors[1]][
            :target_size, :target_size
        ]

    else:
        raise ValueError(f"Unknown method: {method}")


def create_segment_view(
    attention: np.ndarray,
    tokens: list[str],
    segment_size: int = 128,
) -> tuple[np.ndarray, list[int]]:
    """Create segment-level attention view.

    Groups tokens into segments and computes inter-segment attention.

    Args:
        attention: Attention matrix [seq, seq]
        tokens: Token strings (used for potential smart segmentation)
        segment_size: Approximate tokens per segment

    Returns:
        Tuple of (segment_attention, segment_boundaries)
    """
    seq_len = attention.shape[0]
    num_segments = max(1, seq_len // segment_size)

    # Simple uniform segmentation for now
    boundaries = np.linspace(0, seq_len, num_segments + 1, dtype=int).tolist()

    segment_attention = np.zeros((num_segments, num_segments))

    for i in range(num_segments):
        for j in range(num_segments):
            # Average attention from segment i to segment j
            seg_attn = attention[boundaries[i] : boundaries[i + 1], boundaries[j] : boundaries[j + 1]]
            segment_attention[i, j] = seg_attn.mean()

    return segment_attention, boundaries


def create_local_view(
    attention: np.ndarray,
    center: int,
    window_size: int = 512,
) -> tuple[np.ndarray, int, int]:
    """Create local full-resolution view around a position.

    Args:
        attention: Full attention matrix [seq, seq]
        center: Center position for the window
        window_size: Size of the local window

    Returns:
        Tuple of (local_attention, start_idx, end_idx)
    """
    seq_len = attention.shape[0]
    half_window = window_size // 2

    start = max(0, center - half_window)
    end = min(seq_len, center + half_window)

    # Adjust if we hit boundaries
    if start == 0:
        end = min(seq_len, window_size)
    if end == seq_len:
        start = max(0, seq_len - window_size)

    return attention[start:end, start:end], start, end


def create_multiscale_attention(
    attention: np.ndarray,
    tokens: list[str],
    global_size: int = 256,
    segment_size: int = 128,
    local_positions: list[int] | None = None,
    local_window_size: int = 512,
) -> MultiScaleAttention:
    """Create multi-scale attention views.

    Args:
        attention: Full attention [layers, heads, seq, seq] or [seq, seq]
        tokens: Token strings
        global_size: Max size for global view
        segment_size: Tokens per segment
        local_positions: Positions for local views (auto-detect hotspots if None)
        local_window_size: Size of local view windows

    Returns:
        MultiScaleAttention with all views
    """
    # Handle different input shapes
    if attention.ndim == 4:
        # [layers, heads, seq, seq] -> aggregate
        attn_2d = attention.mean(axis=(0, 1))
    elif attention.ndim == 3:
        # [heads, seq, seq] -> aggregate
        attn_2d = attention.mean(axis=0)
    else:
        attn_2d = attention

    seq_len = attn_2d.shape[0]

    # Global view
    global_view = downsample_attention(attn_2d, global_size, method="mean")

    # Segment view
    segment_view = None
    segment_boundaries = None
    if seq_len > segment_size * 2:
        segment_view, segment_boundaries = create_segment_view(attn_2d, tokens, segment_size)

    # Local views
    local_views = {}
    if local_positions is None and seq_len > local_window_size:
        # Auto-detect interesting positions (hotspots)
        # Look at diagonal and find high-attention regions
        attention_sum = attn_2d.sum(axis=1)
        top_positions = np.argsort(attention_sum)[-3:]  # Top 3 positions
        local_positions = top_positions.tolist()

    if local_positions:
        for pos in local_positions:
            local_attn, start, end = create_local_view(attn_2d, pos, local_window_size)
            local_views[f"pos_{pos}"] = {
                "attention": local_attn,
                "start": start,
                "end": end,
                "tokens": tokens[start:end] if tokens else [],
            }

    return MultiScaleAttention(
        global_view=global_view,
        segment_view=segment_view,
        local_views=local_views,
        original_shape=attention.shape,
        tokens=tokens,
        segment_boundaries=segment_boundaries,
    )
