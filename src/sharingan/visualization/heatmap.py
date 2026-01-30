"""Matplotlib-based attention heatmap visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

if TYPE_CHECKING:
    from sharingan.core.result import AttentionResult

# Sharingan-inspired color scheme
SHARINGAN_COLORS = {
    "primary": "#B91C1C",
    "secondary": "#1F2937",
    "accent": "#EF4444",
    "background": "#111827",
    "text": "#F9FAFB",
}


def get_sharingan_cmap():
    """Create Sharingan-themed colormap (dark to red)."""
    colors = ["#111827", "#1F2937", "#7F1D1D", "#B91C1C", "#EF4444", "#FCA5A5"]
    return mcolors.LinearSegmentedColormap.from_list("sharingan", colors)


def plot_heatmap(
    result: "AttentionResult",
    layer: int | None = None,
    head: int | None = None,
    level: str = "auto",
    figsize: tuple[int, int] | None = None,
    cmap: str | None = None,
    show_tokens: bool = True,
    max_tokens_shown: int = 50,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot attention heatmap using matplotlib.

    Args:
        result: AttentionResult object
        layer: Specific layer to plot (None for aggregate)
        head: Specific head to plot (None for aggregate)
        level: Visualization level ("global", "local", "auto")
        figsize: Figure size (width, height)
        cmap: Colormap name (default: Sharingan theme)
        show_tokens: Whether to show token labels
        max_tokens_shown: Maximum tokens to show on axes
        title: Custom title
        ax: Existing axes to plot on

    Returns:
        Matplotlib Figure
    """
    # Get attention data
    attention = result.get_attention(layer=layer, head=head, aggregate="mean")

    # Apply downsampling if needed
    seq_len = attention.shape[0]
    if level == "auto":
        level = "global" if seq_len > 256 else "local"

    if level == "global" and seq_len > 256:
        from sharingan.attention.downsampler import downsample_attention

        attention = downsample_attention(attention, target_size=256)
        tokens_to_show = None  # Can't show tokens after heavy downsampling
    else:
        tokens_to_show = result.tokens if show_tokens else None

    # Create figure if needed
    if ax is None:
        if figsize is None:
            size = min(12, max(6, seq_len // 20))
            figsize = (size, size)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Set style
    fig.patch.set_facecolor(SHARINGAN_COLORS["background"])
    ax.set_facecolor(SHARINGAN_COLORS["background"])

    # Plot heatmap
    colormap = get_sharingan_cmap() if cmap is None else plt.get_cmap(cmap)
    im = ax.imshow(attention, cmap=colormap, aspect="equal")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=SHARINGAN_COLORS["text"])
    cbar.outline.set_edgecolor(SHARINGAN_COLORS["secondary"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=SHARINGAN_COLORS["text"])

    # Token labels
    if tokens_to_show and len(tokens_to_show) <= max_tokens_shown:
        # Truncate tokens for display
        display_tokens = [t[:10] if len(t) > 10 else t for t in tokens_to_show]

        ax.set_xticks(range(len(display_tokens)))
        ax.set_yticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(display_tokens, fontsize=8)

        # Style tick labels
        ax.tick_params(colors=SHARINGAN_COLORS["text"])
    else:
        ax.tick_params(colors=SHARINGAN_COLORS["text"])

    # Title
    if title is None:
        if layer is not None and head is not None:
            title = f"Attention: Layer {layer}, Head {head}"
        elif layer is not None:
            title = f"Attention: Layer {layer} (all heads)"
        elif head is not None:
            title = f"Attention: Head {head} (all layers)"
        else:
            title = "Attention: Aggregated"

    ax.set_title(title, color=SHARINGAN_COLORS["text"], fontsize=12, pad=10)
    ax.set_xlabel("Key Position", color=SHARINGAN_COLORS["text"])
    ax.set_ylabel("Query Position", color=SHARINGAN_COLORS["text"])

    # Grid
    ax.grid(False)

    plt.tight_layout()
    return fig


def plot_layer_summary(
    result: "AttentionResult",
    figsize: tuple[int, int] = (14, 10),
    max_layers: int = 12,
) -> plt.Figure:
    """Plot attention summary across layers.

    Args:
        result: AttentionResult object
        figsize: Figure size
        max_layers: Maximum layers to show (samples if more)

    Returns:
        Matplotlib Figure with layer grid
    """
    n_layers = result.num_layers
    layer_indices = (
        list(range(n_layers))
        if n_layers <= max_layers
        else np.linspace(0, n_layers - 1, max_layers, dtype=int).tolist()
    )

    n_cols = 4
    n_rows = (len(layer_indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.patch.set_facecolor(SHARINGAN_COLORS["background"])

    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    colormap = get_sharingan_cmap()

    for idx, (ax, layer_idx) in enumerate(zip(axes, layer_indices)):
        attn = result.get_attention(layer=layer_idx, aggregate="mean")

        # Downsample if needed
        if attn.shape[0] > 64:
            from sharingan.attention.downsampler import downsample_attention

            attn = downsample_attention(attn, target_size=64)

        ax.set_facecolor(SHARINGAN_COLORS["background"])
        ax.imshow(attn, cmap=colormap, aspect="equal")
        ax.set_title(f"Layer {layer_idx}", color=SHARINGAN_COLORS["text"], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ax in axes[len(layer_indices) :]:
        ax.axis("off")

    fig.suptitle(
        f"Attention by Layer ({result.model_name})",
        color=SHARINGAN_COLORS["text"],
        fontsize=14,
    )
    plt.tight_layout()
    return fig


def plot_entropy(
    result: "AttentionResult",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot attention entropy across positions.

    Args:
        result: AttentionResult object
        figsize: Figure size

    Returns:
        Matplotlib Figure with entropy plot
    """
    entropy = result.attention_entropy()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SHARINGAN_COLORS["background"])
    ax.set_facecolor(SHARINGAN_COLORS["background"])

    positions = range(len(entropy))
    ax.fill_between(positions, entropy, alpha=0.3, color=SHARINGAN_COLORS["accent"])
    ax.plot(positions, entropy, color=SHARINGAN_COLORS["primary"], linewidth=2)

    # Mark prompt/generated boundary
    if result.generated_length > 0:
        ax.axvline(
            result.prompt_length - 0.5,
            color=SHARINGAN_COLORS["text"],
            linestyle="--",
            alpha=0.5,
            label="Generation start",
        )
        ax.legend(facecolor=SHARINGAN_COLORS["secondary"], labelcolor=SHARINGAN_COLORS["text"])

    ax.set_xlabel("Position", color=SHARINGAN_COLORS["text"])
    ax.set_ylabel("Entropy", color=SHARINGAN_COLORS["text"])
    ax.set_title("Attention Entropy by Position", color=SHARINGAN_COLORS["text"])
    ax.tick_params(colors=SHARINGAN_COLORS["text"])

    for spine in ax.spines.values():
        spine.set_color(SHARINGAN_COLORS["secondary"])

    plt.tight_layout()
    return fig
