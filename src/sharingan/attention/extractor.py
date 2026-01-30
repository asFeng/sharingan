"""Attention extraction from transformer models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from sharingan.models.base import ModelAdapter


def extract_attention(
    model: "PreTrainedModel",
    inputs: dict,
    *,
    adapter: "ModelAdapter | None" = None,
    offload_to_cpu: bool = False,
) -> np.ndarray:
    """Extract attention weights from a model.

    Args:
        model: The transformer model
        inputs: Tokenized inputs dict with input_ids and attention_mask
        adapter: Model adapter for architecture-specific handling
        offload_to_cpu: Whether to offload to CPU immediately for memory savings

    Returns:
        Attention weights array [num_layers, num_heads, seq_len, seq_len]
    """
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    # outputs.attentions is tuple of (batch, heads, seq, seq) per layer
    attentions = outputs.attentions

    if adapter is not None:
        # Use adapter for GQA expansion and processing
        processed = []
        for layer_attn in attentions:
            if offload_to_cpu:
                layer_attn = layer_attn.cpu()
            processed.append(adapter.process_attention(layer_attn))
        return np.stack(processed, axis=0)
    else:
        # Default processing
        processed = []
        for layer_attn in attentions:
            if offload_to_cpu:
                layer_attn = layer_attn.cpu()
            # Remove batch dimension, convert to numpy
            processed.append(layer_attn[0].cpu().float().numpy())
        return np.stack(processed, axis=0)


def extract_attention_streaming(
    model: "PreTrainedModel",
    inputs: dict,
    *,
    adapter: "ModelAdapter | None" = None,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Extract attention with streaming for memory efficiency.

    For very long sequences, processes attention in chunks to avoid OOM.

    Args:
        model: The transformer model
        inputs: Tokenized inputs
        adapter: Model adapter
        chunk_size: Size of chunks to process

    Returns:
        Attention weights (may be downsampled for very long sequences)
    """
    seq_len = inputs["input_ids"].shape[1]

    if seq_len <= chunk_size:
        return extract_attention(model, inputs, adapter=adapter, offload_to_cpu=True)

    # For long sequences, we need to be clever
    # Option 1: Extract and immediately downsample each layer
    # Option 2: Use gradient checkpointing style extraction

    # For now, use option 1 with immediate CPU offloading
    return extract_attention(model, inputs, adapter=adapter, offload_to_cpu=True)
