"""AttentionResult container for attention data and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go


@dataclass
class AttentionResult:
    """Container for attention extraction results.

    Attributes:
        attention: Raw attention weights [num_layers, num_heads, seq_len, seq_len]
        tokens: List of token strings
        prompt_length: Number of tokens in the original prompt
        generated_length: Number of generated tokens (0 if no generation)
        model_name: Name of the model used
        config: Configuration used for extraction
    """

    attention: np.ndarray  # [layers, heads, seq, seq]
    tokens: list[str]
    prompt_length: int
    generated_length: int = 0
    model_name: str = ""
    config: dict = field(default_factory=dict)

    # Cached computations
    _entropy: np.ndarray | None = field(default=None, repr=False)
    _sinks: list[dict] | None = field(default=None, repr=False)
    _importance: np.ndarray | None = field(default=None, repr=False)

    @property
    def num_layers(self) -> int:
        """Number of attention layers."""
        return self.attention.shape[0]

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self.attention.shape[1]

    @property
    def seq_len(self) -> int:
        """Sequence length."""
        return self.attention.shape[2]

    @property
    def total_length(self) -> int:
        """Total sequence length (prompt + generated)."""
        return self.prompt_length + self.generated_length

    def get_attention(
        self,
        layer: int | None = None,
        head: int | None = None,
        aggregate: str = "mean",
    ) -> np.ndarray:
        """Get attention weights with optional layer/head selection.

        Args:
            layer: Specific layer index (None for all)
            head: Specific head index (None for all)
            aggregate: How to aggregate ("mean", "max", "none")

        Returns:
            Attention array with shape depending on selections
        """
        attn = self.attention

        if layer is not None:
            attn = attn[layer : layer + 1]
        if head is not None:
            attn = attn[:, head : head + 1]

        if aggregate == "mean":
            return attn.mean(axis=(0, 1))
        elif aggregate == "max":
            return attn.max(axis=(0, 1))
        return attn

    def attention_entropy(
        self,
        layer: int | None = None,
        head: int | None = None,
    ) -> np.ndarray:
        """Compute attention entropy per position.

        Higher entropy = more distributed attention
        Lower entropy = more focused attention

        Args:
            layer: Specific layer (None for average across layers)
            head: Specific head (None for average across heads)

        Returns:
            Array of entropy values [seq_len] or [layers, heads, seq_len]
        """
        if self._entropy is None:
            # Compute entropy: -sum(p * log(p))
            eps = 1e-10
            attn = self.attention + eps
            entropy = -np.sum(attn * np.log(attn), axis=-1)
            self._entropy = entropy

        entropy = self._entropy

        if layer is not None:
            entropy = entropy[layer : layer + 1]
        if head is not None:
            entropy = entropy[:, head : head + 1]

        if layer is None and head is None:
            return entropy.mean(axis=(0, 1))
        return entropy.squeeze()

    def attention_sinks(
        self,
        threshold: float = 0.1,
        min_positions: int = 3,
    ) -> list[dict]:
        """Detect attention sink positions.

        Attention sinks are positions that receive disproportionate attention
        from many other positions (commonly first tokens, special tokens).

        Args:
            threshold: Minimum average attention to be considered a sink
            min_positions: Minimum number of positions attending to qualify

        Returns:
            List of dicts with sink information
        """
        if self._sinks is not None:
            return self._sinks

        # Average attention received by each position
        attn_received = self.attention.mean(axis=(0, 1)).sum(axis=0)
        attn_received /= self.seq_len  # Normalize

        sinks = []
        for pos in range(self.seq_len):
            if attn_received[pos] > threshold:
                # Count how many positions attend to this one
                attending = (self.attention.mean(axis=(0, 1))[:, pos] > 0.05).sum()
                if attending >= min_positions:
                    sinks.append(
                        {
                            "position": pos,
                            "token": self.tokens[pos] if pos < len(self.tokens) else "<unk>",
                            "attention_received": float(attn_received[pos]),
                            "attending_positions": int(attending),
                        }
                    )

        self._sinks = sorted(sinks, key=lambda x: -x["attention_received"])
        return self._sinks

    def token_importance(self, method: str = "attention_sum") -> np.ndarray:
        """Compute token importance scores.

        Args:
            method: Scoring method
                - "attention_sum": Sum of attention received
                - "attention_max": Max attention received
                - "entropy_weighted": Weighted by entropy

        Returns:
            Array of importance scores [seq_len]
        """
        if self._importance is not None and method == "attention_sum":
            return self._importance

        attn = self.attention.mean(axis=(0, 1))

        if method == "attention_sum":
            importance = attn.sum(axis=0)
        elif method == "attention_max":
            importance = attn.max(axis=0)
        elif method == "entropy_weighted":
            entropy = self.attention_entropy()
            # Low entropy positions are more "decisive"
            weights = 1.0 / (entropy + 0.1)
            importance = (attn * weights[:, None]).sum(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize to [0, 1]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)

        if method == "attention_sum":
            self._importance = importance
        return importance

    def hotspots(
        self,
        layer: int | None = None,
        head: int | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Find attention hotspots (high-attention token pairs).

        Args:
            layer: Specific layer (None for aggregate)
            head: Specific head (None for aggregate)
            top_k: Number of top hotspots to return

        Returns:
            List of dicts with (from_pos, to_pos, attention, from_token, to_token)
        """
        attn = self.get_attention(layer, head, aggregate="mean")

        # Find top-k positions
        flat_indices = np.argsort(attn.flatten())[-top_k:][::-1]
        positions = np.unravel_index(flat_indices, attn.shape)

        hotspots = []
        for from_pos, to_pos in zip(positions[0], positions[1]):
            hotspots.append(
                {
                    "from_position": int(from_pos),
                    "to_position": int(to_pos),
                    "attention": float(attn[from_pos, to_pos]),
                    "from_token": self.tokens[from_pos] if from_pos < len(self.tokens) else "<unk>",
                    "to_token": self.tokens[to_pos] if to_pos < len(self.tokens) else "<unk>",
                }
            )
        return hotspots

    def plot(
        self,
        layer: int | None = None,
        head: int | None = None,
        level: str = "auto",
        interactive: bool = False,
        scale: str = "sqrt",
        **kwargs,
    ) -> "plt.Figure | go.Figure":
        """Plot attention heatmap.

        Args:
            layer: Specific layer to plot (None for aggregate)
            head: Specific head to plot (None for aggregate)
            level: Visualization level ("global", "local", "auto")
            interactive: Whether to use Plotly for interactive visualization
            scale: Scaling method for better contrast with many tokens
                - "none": Raw values (may look faint)
                - "sqrt": Square root (default, good balance)
                - "log": Logarithmic (emphasizes small differences)
                - "row": Row-wise normalization (max per row = 1)
                - "percentile": Clip to 98th percentile
                - "rank": Rank-based (uniform color distribution)
            **kwargs: Additional arguments passed to plotting function

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        if interactive:
            from sharingan.visualization.interactive import plot_interactive

            return plot_interactive(self, layer=layer, head=head, level=level, scale=scale, **kwargs)
        else:
            from sharingan.visualization.heatmap import plot_heatmap

            return plot_heatmap(self, layer=layer, head=head, level=level, scale=scale, **kwargs)

    def to_html(self, path: str, include_metrics: bool = True) -> None:
        """Export visualization to standalone HTML file.

        Args:
            path: Output file path
            include_metrics: Whether to include metrics panel
        """
        from sharingan.visualization.html_export import export_html

        export_html(self, path, include_metrics=include_metrics)

    def summary(self) -> dict:
        """Get summary statistics of attention patterns.

        Returns:
            Dict with summary statistics
        """
        entropy = self.attention_entropy()
        sinks = self.attention_sinks()

        return {
            "model": self.model_name,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "seq_len": self.seq_len,
            "prompt_length": self.prompt_length,
            "generated_length": self.generated_length,
            "mean_entropy": float(entropy.mean()),
            "min_entropy": float(entropy.min()),
            "max_entropy": float(entropy.max()),
            "num_sinks": len(sinks),
            "top_sink": sinks[0] if sinks else None,
        }

    def __repr__(self) -> str:
        return (
            f"AttentionResult(layers={self.num_layers}, heads={self.num_heads}, "
            f"seq_len={self.seq_len}, tokens={len(self.tokens)})"
        )
