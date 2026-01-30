"""Matplotlib-based attention heatmap visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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

# Scaling methods for attention visualization
ScaleMethod = Literal["none", "log", "sqrt", "row", "percentile", "rank"]


def scale_attention(
    attention: np.ndarray,
    method: ScaleMethod = "none",
    percentile: float = 98,
) -> np.ndarray:
    """Scale attention values for better visualization.

    Args:
        attention: Attention matrix [seq, seq]
        method: Scaling method
            - "none": No scaling (raw values)
            - "log": Log scale (log(1 + x * 100))
            - "sqrt": Square root scaling
            - "row": Row-wise normalization (max per row = 1)
            - "percentile": Clip to percentile and normalize
            - "rank": Rank-based scaling (uniform distribution)
        percentile: Percentile for clipping (used with "percentile" method)

    Returns:
        Scaled attention values in [0, 1] range
    """
    if method == "none":
        return attention

    attn = attention.copy()

    if method == "log":
        # Log scale: emphasizes differences in small values
        attn = np.log1p(attn * 100)  # log(1 + x*100)
        attn = attn / attn.max() if attn.max() > 0 else attn

    elif method == "sqrt":
        # Square root: moderate compression
        attn = np.sqrt(attn)
        attn = attn / attn.max() if attn.max() > 0 else attn

    elif method == "row":
        # Row-wise normalization: each row's max becomes 1
        row_max = attn.max(axis=1, keepdims=True)
        row_max = np.where(row_max == 0, 1, row_max)
        attn = attn / row_max

    elif method == "percentile":
        # Clip to percentile and normalize
        threshold = np.percentile(attn, percentile)
        attn = np.clip(attn, 0, threshold)
        attn = attn / threshold if threshold > 0 else attn

    elif method == "rank":
        # Rank-based: convert to percentile ranks
        flat = attn.flatten()
        ranks = np.argsort(np.argsort(flat))
        attn = (ranks / len(ranks)).reshape(attn.shape)

    return attn


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
    show_generation_boundary: bool = True,
    token_format: str = "content",  # "content", "index", "both"
    scale: ScaleMethod = "sqrt",  # Scaling method for better contrast
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
        show_generation_boundary: Whether to mark prompt/generation boundary
        token_format: How to format token labels ("content", "index", "both")
        scale: Scaling method for attention values
            - "none": Raw values (may look faint with many tokens)
            - "sqrt": Square root (default, good balance)
            - "log": Logarithmic (emphasizes small differences)
            - "row": Row-wise normalization (max per row = 1)
            - "percentile": Clip to 98th percentile
            - "rank": Rank-based (uniform color distribution)

    Returns:
        Matplotlib Figure
    """
    # Get attention data
    attention = result.get_attention(layer=layer, head=head, aggregate="mean")

    # Apply downsampling if needed
    seq_len = attention.shape[0]
    if level == "auto":
        level = "global" if seq_len > 256 else "local"

    downsampled = False
    if level == "global" and seq_len > 256:
        from sharingan.attention.downsampler import downsample_attention

        attention = downsample_attention(attention, target_size=256)
        tokens_to_show = None  # Can't show tokens after heavy downsampling
        downsampled = True
    else:
        tokens_to_show = result.tokens if show_tokens else None

    # Apply scaling for better contrast
    attention = scale_attention(attention, method=scale)

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
        # Format token labels
        display_tokens = _format_token_labels(
            tokens_to_show, token_format, result.prompt_length, result.generated_length
        )

        ax.set_xticks(range(len(display_tokens)))
        ax.set_yticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(display_tokens, fontsize=8)

        # Color generated tokens differently
        if result.generated_length > 0:
            for i, label in enumerate(ax.get_xticklabels()):
                if i >= result.prompt_length:
                    label.set_color(SHARINGAN_COLORS["accent"])
            for i, label in enumerate(ax.get_yticklabels()):
                if i >= result.prompt_length:
                    label.set_color(SHARINGAN_COLORS["accent"])

        ax.tick_params(colors=SHARINGAN_COLORS["text"])
    else:
        ax.tick_params(colors=SHARINGAN_COLORS["text"])

    # Mark generation boundary
    if show_generation_boundary and result.generated_length > 0 and not downsampled:
        boundary = result.prompt_length - 0.5
        ax.axhline(boundary, color=SHARINGAN_COLORS["accent"], linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(boundary, color=SHARINGAN_COLORS["accent"], linestyle="--", linewidth=1.5, alpha=0.8)
        # Add labels for quadrants
        ax.text(
            result.prompt_length / 2, -1.5, "Prompt",
            ha="center", va="bottom", color=SHARINGAN_COLORS["text"], fontsize=8
        )
        if result.generated_length > 0:
            ax.text(
                result.prompt_length + result.generated_length / 2, -1.5, "Generated",
                ha="center", va="bottom", color=SHARINGAN_COLORS["accent"], fontsize=8
            )

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
        if result.generated_length > 0:
            title += f" (prompt: {result.prompt_length}, gen: {result.generated_length})"

    ax.set_title(title, color=SHARINGAN_COLORS["text"], fontsize=12, pad=10)
    ax.set_xlabel("Key Position (attending to)", color=SHARINGAN_COLORS["text"])
    ax.set_ylabel("Query Position (attending from)", color=SHARINGAN_COLORS["text"])

    # Grid
    ax.grid(False)

    plt.tight_layout()
    return fig


def _format_token_labels(
    tokens: list[str],
    format_type: str,
    prompt_length: int,
    generated_length: int,
) -> list[str]:
    """Format token labels for display.

    Args:
        tokens: List of token strings
        format_type: "content", "index", or "both"
        prompt_length: Number of prompt tokens
        generated_length: Number of generated tokens

    Returns:
        Formatted token labels
    """
    labels = []
    for i, token in enumerate(tokens):
        # Clean token for display
        clean_token = token.replace("\n", "↵").replace("\t", "→")
        if len(clean_token) > 8:
            clean_token = clean_token[:7] + "…"

        # Add generation marker
        marker = "" if i < prompt_length else "*"

        if format_type == "content":
            labels.append(f"{clean_token}{marker}")
        elif format_type == "index":
            labels.append(f"{i}{marker}")
        else:  # both
            labels.append(f"{i}:{clean_token}{marker}")

    return labels


def plot_generation_attention(
    result: "AttentionResult",
    layer: int | None = None,
    head: int | None = None,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot what each generated token attends to.

    Shows a focused view of how generated tokens attend to the prompt
    and to each other.

    Args:
        result: AttentionResult object (must have generated tokens)
        layer: Specific layer to plot
        head: Specific head to plot
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if result.generated_length == 0:
        raise ValueError("No generated tokens to visualize")

    attention = result.get_attention(layer=layer, head=head, aggregate="mean")
    prompt_len = result.prompt_length
    gen_len = result.generated_length

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(SHARINGAN_COLORS["background"])

    colormap = get_sharingan_cmap()

    # Left: Generated tokens attending to prompt
    gen_to_prompt = attention[prompt_len:, :prompt_len]
    ax1 = axes[0]
    ax1.set_facecolor(SHARINGAN_COLORS["background"])
    im1 = ax1.imshow(gen_to_prompt, cmap=colormap, aspect="auto")
    ax1.set_title("Generated → Prompt", color=SHARINGAN_COLORS["text"], fontsize=11)
    ax1.set_xlabel("Prompt tokens", color=SHARINGAN_COLORS["text"])
    ax1.set_ylabel("Generated tokens", color=SHARINGAN_COLORS["text"])

    # Label axes with actual tokens
    if gen_len <= 30:
        gen_tokens = [t[:8] for t in result.tokens[prompt_len:]]
        ax1.set_yticks(range(gen_len))
        ax1.set_yticklabels(gen_tokens, fontsize=7, color=SHARINGAN_COLORS["accent"])
    if prompt_len <= 30:
        prompt_tokens = [t[:8] for t in result.tokens[:prompt_len]]
        ax1.set_xticks(range(prompt_len))
        ax1.set_xticklabels(prompt_tokens, rotation=45, ha="right", fontsize=7, color=SHARINGAN_COLORS["text"])

    ax1.tick_params(colors=SHARINGAN_COLORS["text"])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Right: Generated tokens attending to each other
    gen_to_gen = attention[prompt_len:, prompt_len:]
    ax2 = axes[1]
    ax2.set_facecolor(SHARINGAN_COLORS["background"])
    im2 = ax2.imshow(gen_to_gen, cmap=colormap, aspect="equal")
    ax2.set_title("Generated → Generated", color=SHARINGAN_COLORS["text"], fontsize=11)
    ax2.set_xlabel("Generated tokens (key)", color=SHARINGAN_COLORS["text"])
    ax2.set_ylabel("Generated tokens (query)", color=SHARINGAN_COLORS["text"])

    if gen_len <= 30:
        gen_tokens = [t[:8] for t in result.tokens[prompt_len:]]
        ax2.set_xticks(range(gen_len))
        ax2.set_yticks(range(gen_len))
        ax2.set_xticklabels(gen_tokens, rotation=45, ha="right", fontsize=7, color=SHARINGAN_COLORS["accent"])
        ax2.set_yticklabels(gen_tokens, fontsize=7, color=SHARINGAN_COLORS["accent"])

    ax2.tick_params(colors=SHARINGAN_COLORS["text"])
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Add title info
    title = "Generation Attention Analysis"
    if layer is not None:
        title += f" (Layer {layer}"
        if head is not None:
            title += f", Head {head}"
        title += ")"

    fig.suptitle(title, color=SHARINGAN_COLORS["text"], fontsize=13)
    plt.tight_layout()
    return fig


def plot_token_attention_summary(
    result: "AttentionResult",
    top_k: int = 10,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot summary of which tokens receive/give most attention.

    Args:
        result: AttentionResult object
        top_k: Number of top tokens to show
        figsize: Figure size

    Returns:
        Matplotlib Figure with token attention summary
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor(SHARINGAN_COLORS["background"])

    attention = result.get_attention(aggregate="mean")
    tokens = result.tokens

    # Top tokens receiving attention (sinks)
    ax1 = axes[0, 0]
    attn_received = attention.sum(axis=0)
    top_receivers = np.argsort(attn_received)[-top_k:][::-1]
    ax1.set_facecolor(SHARINGAN_COLORS["background"])
    bars1 = ax1.barh(
        range(top_k),
        attn_received[top_receivers],
        color=SHARINGAN_COLORS["primary"]
    )
    ax1.set_yticks(range(top_k))
    labels1 = [f"{i}:{tokens[i][:10]}" for i in top_receivers]
    ax1.set_yticklabels(labels1, fontsize=8, color=SHARINGAN_COLORS["text"])
    ax1.set_title("Top Attention Receivers", color=SHARINGAN_COLORS["text"])
    ax1.set_xlabel("Total Attention Received", color=SHARINGAN_COLORS["text"])
    ax1.tick_params(colors=SHARINGAN_COLORS["text"])
    ax1.invert_yaxis()

    # Top tokens giving attention
    ax2 = axes[0, 1]
    attn_given = attention.sum(axis=1)
    top_givers = np.argsort(attn_given)[-top_k:][::-1]
    ax2.set_facecolor(SHARINGAN_COLORS["background"])
    ax2.barh(
        range(top_k),
        attn_given[top_givers],
        color=SHARINGAN_COLORS["accent"]
    )
    ax2.set_yticks(range(top_k))
    labels2 = [f"{i}:{tokens[i][:10]}" for i in top_givers]
    ax2.set_yticklabels(labels2, fontsize=8, color=SHARINGAN_COLORS["text"])
    ax2.set_title("Top Attention Givers", color=SHARINGAN_COLORS["text"])
    ax2.set_xlabel("Total Attention Given", color=SHARINGAN_COLORS["text"])
    ax2.tick_params(colors=SHARINGAN_COLORS["text"])
    ax2.invert_yaxis()

    # Token importance
    ax3 = axes[1, 0]
    importance = result.token_importance()
    ax3.set_facecolor(SHARINGAN_COLORS["background"])
    ax3.plot(importance, color=SHARINGAN_COLORS["primary"], linewidth=1.5)
    ax3.fill_between(range(len(importance)), importance, alpha=0.3, color=SHARINGAN_COLORS["accent"])
    ax3.set_title("Token Importance Score", color=SHARINGAN_COLORS["text"])
    ax3.set_xlabel("Token Position", color=SHARINGAN_COLORS["text"])
    ax3.set_ylabel("Importance", color=SHARINGAN_COLORS["text"])
    ax3.tick_params(colors=SHARINGAN_COLORS["text"])
    if result.generated_length > 0:
        ax3.axvline(result.prompt_length - 0.5, color=SHARINGAN_COLORS["accent"], linestyle="--", alpha=0.7)

    # Entropy
    ax4 = axes[1, 1]
    entropy = result.attention_entropy()
    ax4.set_facecolor(SHARINGAN_COLORS["background"])
    ax4.plot(entropy, color=SHARINGAN_COLORS["primary"], linewidth=1.5)
    ax4.fill_between(range(len(entropy)), entropy, alpha=0.3, color=SHARINGAN_COLORS["accent"])
    ax4.set_title("Attention Entropy", color=SHARINGAN_COLORS["text"])
    ax4.set_xlabel("Token Position", color=SHARINGAN_COLORS["text"])
    ax4.set_ylabel("Entropy", color=SHARINGAN_COLORS["text"])
    ax4.tick_params(colors=SHARINGAN_COLORS["text"])
    if result.generated_length > 0:
        ax4.axvline(result.prompt_length - 0.5, color=SHARINGAN_COLORS["accent"], linestyle="--", alpha=0.7)

    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_color(SHARINGAN_COLORS["secondary"])

    fig.suptitle(
        f"Token Attention Summary - {result.model_name}",
        color=SHARINGAN_COLORS["text"],
        fontsize=13
    )
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
