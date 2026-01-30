"""Sharingan - Attention visualization for transformers."""

from sharingan.core.analyzer import Sharingan
from sharingan.core.result import AttentionResult
from sharingan.core.config import SharinganConfig

__version__ = "0.1.0"
__all__ = ["Sharingan", "AttentionResult", "SharinganConfig", "visualize"]


def visualize(
    model_name: str,
    prompt: str,
    *,
    generate: bool = False,
    max_new_tokens: int = 50,
    layer: int | None = None,
    head: int | None = None,
    level: str = "auto",
    show: bool = True,
    save_path: str | None = None,
) -> AttentionResult:
    """One-liner attention visualization.

    Args:
        model_name: HuggingFace model name or path
        prompt: Input text to analyze
        generate: Whether to generate new tokens
        max_new_tokens: Maximum tokens to generate if generate=True
        layer: Specific layer to visualize (None for all)
        head: Specific head to visualize (None for all)
        level: Visualization level ("global", "local", "auto")
        show: Whether to display the plot
        save_path: Optional path to save the visualization

    Returns:
        AttentionResult object with attention data and analysis methods
    """
    analyzer = Sharingan(model_name)
    result = analyzer.analyze(prompt, generate=generate, max_new_tokens=max_new_tokens)

    if show or save_path:
        fig = result.plot(layer=layer, head=head, level=level)
        if save_path:
            if save_path.endswith(".html"):
                result.to_html(save_path)
            else:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            import matplotlib.pyplot as plt

            plt.show()

    return result
