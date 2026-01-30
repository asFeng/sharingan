"""Attention extraction and analysis."""

from sharingan.attention.extractor import extract_attention
from sharingan.attention.downsampler import downsample_attention, MultiScaleAttention
from sharingan.attention.metrics import (
    compute_entropy,
    find_attention_sinks,
    compute_token_importance,
)

__all__ = [
    "extract_attention",
    "downsample_attention",
    "MultiScaleAttention",
    "compute_entropy",
    "find_attention_sinks",
    "compute_token_importance",
]
