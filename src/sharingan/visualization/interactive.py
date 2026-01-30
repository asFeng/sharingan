"""Plotly-based interactive attention visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from sharingan.core.result import AttentionResult

# Sharingan color scheme for Plotly
SHARINGAN_TEMPLATE = {
    "paper_bgcolor": "#111827",
    "plot_bgcolor": "#111827",
    "font": {"color": "#F9FAFB"},
}

# Custom colorscale (dark to red)
SHARINGAN_COLORSCALE = [
    [0.0, "#111827"],
    [0.2, "#1F2937"],
    [0.4, "#7F1D1D"],
    [0.6, "#B91C1C"],
    [0.8, "#EF4444"],
    [1.0, "#FCA5A5"],
]


def plot_interactive(
    result: "AttentionResult",
    layer: int | None = None,
    head: int | None = None,
    level: str = "auto",
    width: int = 800,
    height: int = 800,
    show_tokens: bool = True,
) -> go.Figure:
    """Create interactive Plotly attention heatmap.

    Args:
        result: AttentionResult object
        layer: Specific layer to plot
        head: Specific head to plot
        level: Visualization level
        width: Figure width in pixels
        height: Figure height in pixels
        show_tokens: Whether to show token labels

    Returns:
        Plotly Figure with interactive heatmap
    """
    attention = result.get_attention(layer=layer, head=head, aggregate="mean")
    seq_len = attention.shape[0]

    # Auto level selection
    if level == "auto":
        level = "global" if seq_len > 256 else "local"

    if level == "global" and seq_len > 256:
        from sharingan.attention.downsampler import downsample_attention

        attention = downsample_attention(attention, target_size=256)
        tokens = None
    else:
        tokens = result.tokens if show_tokens else None

    # Create hover text
    if tokens:
        hover_text = [
            [f"Query: {tokens[i]}<br>Key: {tokens[j]}<br>Attention: {attention[i, j]:.4f}"
             for j in range(len(tokens))]
            for i in range(len(tokens))
        ]
    else:
        hover_text = [
            [f"Query pos: {i}<br>Key pos: {j}<br>Attention: {attention[i, j]:.4f}"
             for j in range(attention.shape[1])]
            for i in range(attention.shape[0])
        ]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=attention,
            colorscale=SHARINGAN_COLORSCALE,
            hoverinfo="text",
            text=hover_text,
            x=tokens if tokens and len(tokens) <= 100 else None,
            y=tokens if tokens and len(tokens) <= 100 else None,
        )
    )

    # Title
    if layer is not None and head is not None:
        title = f"Attention: Layer {layer}, Head {head}"
    elif layer is not None:
        title = f"Attention: Layer {layer}"
    else:
        title = "Attention (Aggregated)"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        width=width,
        height=height,
        **SHARINGAN_TEMPLATE,
    )

    # Reverse y-axis to match matrix convention
    fig.update_yaxes(autorange="reversed")

    return fig


def plot_layer_head_grid(
    result: "AttentionResult",
    max_layers: int = 8,
    max_heads: int = 8,
    width: int = 1200,
    height: int = 1000,
) -> go.Figure:
    """Create interactive grid of layer/head attention patterns.

    Args:
        result: AttentionResult object
        max_layers: Maximum layers to show
        max_heads: Maximum heads to show
        width: Figure width
        height: Figure height

    Returns:
        Plotly Figure with subplots grid
    """
    n_layers = min(result.num_layers, max_layers)
    n_heads = min(result.num_heads, max_heads)

    # Sample if needed
    layer_indices = (
        list(range(n_layers))
        if result.num_layers <= max_layers
        else np.linspace(0, result.num_layers - 1, max_layers, dtype=int).tolist()
    )
    head_indices = (
        list(range(n_heads))
        if result.num_heads <= max_heads
        else np.linspace(0, result.num_heads - 1, max_heads, dtype=int).tolist()
    )

    fig = make_subplots(
        rows=n_layers,
        cols=n_heads,
        subplot_titles=[f"L{l}H{h}" for l in layer_indices for h in head_indices],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for row, layer_idx in enumerate(layer_indices, 1):
        for col, head_idx in enumerate(head_indices, 1):
            attn = result.attention[layer_idx, head_idx]

            # Downsample if needed
            if attn.shape[0] > 64:
                from sharingan.attention.downsampler import downsample_attention

                attn = downsample_attention(attn, target_size=64)

            fig.add_trace(
                go.Heatmap(
                    z=attn,
                    colorscale=SHARINGAN_COLORSCALE,
                    showscale=False,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title="Attention by Layer and Head",
        width=width,
        height=height,
        **SHARINGAN_TEMPLATE,
    )

    # Update all axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    return fig


def plot_metrics_dashboard(
    result: "AttentionResult",
    width: int = 1200,
    height: int = 600,
) -> go.Figure:
    """Create interactive metrics dashboard.

    Args:
        result: AttentionResult object
        width: Figure width
        height: Figure height

    Returns:
        Plotly Figure with metrics plots
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Attention Entropy",
            "Token Importance",
            "Attention Distance",
            "Layer Similarity",
        ],
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "heatmap"}],
        ],
    )

    # Entropy plot
    entropy = result.attention_entropy()
    fig.add_trace(
        go.Scatter(
            y=entropy,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#EF4444"),
            fillcolor="rgba(239, 68, 68, 0.3)",
            name="Entropy",
        ),
        row=1,
        col=1,
    )

    # Token importance
    importance = result.token_importance()
    fig.add_trace(
        go.Bar(
            y=importance[:50] if len(importance) > 50 else importance,
            x=result.tokens[:50] if len(result.tokens) > 50 else result.tokens,
            marker_color="#B91C1C",
            name="Importance",
        ),
        row=1,
        col=2,
    )

    # Attention distance (per layer)
    from sharingan.attention.metrics import compute_attention_distance

    distances = []
    for layer in range(result.num_layers):
        attn = result.get_attention(layer=layer, aggregate="mean")
        dist = compute_attention_distance(attn)
        distances.append(dist["mean_distance"])

    fig.add_trace(
        go.Scatter(
            y=distances,
            mode="lines+markers",
            line=dict(color="#EF4444"),
            marker=dict(color="#B91C1C", size=8),
            name="Mean Distance",
        ),
        row=2,
        col=1,
    )

    # Layer similarity
    from sharingan.attention.metrics import layer_head_similarity

    similarity = layer_head_similarity(result.attention)
    # Downsample if too large
    if similarity.shape[0] > 64:
        step = similarity.shape[0] // 64
        similarity = similarity[::step, ::step]

    fig.add_trace(
        go.Heatmap(
            z=similarity,
            colorscale=SHARINGAN_COLORSCALE,
            name="Similarity",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Attention Metrics Dashboard",
        width=width,
        height=height,
        showlegend=False,
        **SHARINGAN_TEMPLATE,
    )

    return fig
