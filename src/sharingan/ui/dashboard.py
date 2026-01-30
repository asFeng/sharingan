"""Gradio-based interactive dashboard for attention visualization."""

from __future__ import annotations

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from sharingan.core.analyzer import Sharingan
from sharingan.core.result import AttentionResult
from sharingan.visualization.interactive import (
    SHARINGAN_COLORSCALE,
    SHARINGAN_TEMPLATE,
)

# Sharingan theme CSS
THEME_CSS = """
.gradio-container {
    background-color: #111827 !important;
}
.dark {
    --background-fill-primary: #111827 !important;
    --background-fill-secondary: #1F2937 !important;
    --border-color-primary: #374151 !important;
    --color-accent: #B91C1C !important;
    --button-primary-background-fill: #B91C1C !important;
    --button-primary-background-fill-hover: #EF4444 !important;
}
.sharingan-header {
    background: linear-gradient(135deg, #B91C1C 0%, #7F1D1D 100%);
    padding: 1rem 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.sharingan-header h1 {
    color: white;
    margin: 0;
    font-size: 1.5rem;
}
.metric-box {
    background: #1F2937;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 3px solid #B91C1C;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #EF4444;
}
.metric-label {
    font-size: 0.75rem;
    color: #9CA3AF;
    text-transform: uppercase;
}
"""

# Global state
_current_analyzer: Sharingan | None = None
_current_result: AttentionResult | None = None


def load_model(model_name: str, progress=gr.Progress()) -> str:
    """Load a model for analysis."""
    global _current_analyzer

    if not model_name.strip():
        return "Please enter a model name"

    progress(0, desc="Loading model...")

    try:
        _current_analyzer = Sharingan(model_name.strip())
        progress(0.5, desc="Loading tokenizer and model...")
        _current_analyzer.load()
        progress(1.0, desc="Done!")
        return f"✓ Loaded {model_name}"
    except Exception as e:
        return f"✗ Error loading model: {str(e)}"


def analyze_prompt(
    prompt: str,
    generate: bool,
    max_tokens: int,
    progress=gr.Progress(),
) -> tuple[str, go.Figure, go.Figure, str]:
    """Analyze a prompt and return visualizations."""
    global _current_result

    if _current_analyzer is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(**SHARINGAN_TEMPLATE)
        return "Please load a model first", empty_fig, empty_fig, ""

    if not prompt.strip():
        empty_fig = go.Figure()
        empty_fig.update_layout(**SHARINGAN_TEMPLATE)
        return "Please enter a prompt", empty_fig, empty_fig, ""

    progress(0, desc="Analyzing...")

    try:
        progress(0.3, desc="Running model...")
        _current_result = _current_analyzer.analyze(
            prompt.strip(),
            generate=generate,
            max_new_tokens=int(max_tokens),
        )

        progress(0.7, desc="Creating visualizations...")

        # Create attention heatmap
        attn_fig = create_attention_figure(_current_result)

        # Create entropy plot
        entropy_fig = create_entropy_figure(_current_result)

        # Summary text
        summary = _current_result.summary()
        summary_text = (
            f"**Layers:** {summary['num_layers']} | "
            f"**Heads:** {summary['num_heads']} | "
            f"**Sequence:** {summary['seq_len']} tokens\n\n"
            f"**Mean Entropy:** {summary['mean_entropy']:.3f} | "
            f"**Attention Sinks:** {summary['num_sinks']}"
        )

        # Tokens display
        tokens_display = " ".join(
            f"`{t}`" for t in _current_result.tokens[:100]
        )
        if len(_current_result.tokens) > 100:
            tokens_display += f" ... (+{len(_current_result.tokens) - 100} more)"

        progress(1.0, desc="Done!")
        return summary_text, attn_fig, entropy_fig, tokens_display

    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(**SHARINGAN_TEMPLATE)
        return f"Error: {str(e)}", empty_fig, empty_fig, ""


def create_attention_figure(
    result: AttentionResult,
    layer: int | None = None,
    head: int | None = None,
) -> go.Figure:
    """Create Plotly attention heatmap."""
    attention = result.get_attention(layer=layer, head=head, aggregate="mean")

    # Downsample if needed
    if attention.shape[0] > 256:
        from sharingan.attention.downsampler import downsample_attention

        attention = downsample_attention(attention, target_size=256)

    fig = go.Figure(
        data=go.Heatmap(
            z=attention,
            colorscale=SHARINGAN_COLORSCALE,
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Attention: %{z:.4f}<extra></extra>",
        )
    )

    title = "Attention (Aggregated)"
    if layer is not None:
        title = f"Layer {layer}" + (f", Head {head}" if head is not None else "")

    fig.update_layout(
        title=title,
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        height=500,
        **SHARINGAN_TEMPLATE,
    )
    fig.update_yaxes(autorange="reversed")

    return fig


def create_entropy_figure(result: AttentionResult) -> go.Figure:
    """Create entropy plot."""
    entropy = result.attention_entropy()

    fig = go.Figure(
        data=go.Scatter(
            y=entropy,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#EF4444"),
            fillcolor="rgba(239, 68, 68, 0.3)",
        )
    )

    fig.update_layout(
        title="Attention Entropy by Position",
        xaxis_title="Position",
        yaxis_title="Entropy",
        height=300,
        **SHARINGAN_TEMPLATE,
    )

    return fig


def update_layer_view(layer: int, head: str) -> go.Figure:
    """Update attention view for specific layer/head."""
    if _current_result is None:
        fig = go.Figure()
        fig.update_layout(**SHARINGAN_TEMPLATE)
        return fig

    head_idx = None if head == "All (mean)" else int(head.split()[-1])
    return create_attention_figure(_current_result, layer=int(layer), head=head_idx)


def export_html(output_path: str) -> str:
    """Export current result to HTML."""
    if _current_result is None:
        return "No analysis to export. Run analysis first."

    if not output_path.strip():
        output_path = "sharingan_output.html"

    if not output_path.endswith(".html"):
        output_path += ".html"

    try:
        _current_result.to_html(output_path)
        return f"✓ Exported to {output_path}"
    except Exception as e:
        return f"✗ Export failed: {str(e)}"


def create_dashboard() -> gr.Blocks:
    """Create the Gradio dashboard interface."""
    with gr.Blocks(
        title="Sharingan - Attention Visualization",
        theme=gr.themes.Base(
            primary_hue="red",
            secondary_hue="gray",
            neutral_hue="gray",
        ),
        css=THEME_CSS,
    ) as demo:
        # Header
        gr.HTML("""
            <div class="sharingan-header">
                <h1>Sharingan</h1>
                <p style="color: rgba(255,255,255,0.8); margin: 0;">
                    Attention Visualization for Transformers
                </p>
            </div>
        """)

        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                gr.Markdown("### Model")
                model_input = gr.Textbox(
                    label="Model Name",
                    placeholder="Qwen/Qwen3-0.6B",
                    value="Qwen/Qwen3-0.6B",
                )
                load_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Markdown("")

                gr.Markdown("### Prompt")
                prompt_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to analyze...",
                    lines=3,
                )

                with gr.Row():
                    generate_check = gr.Checkbox(label="Generate", value=False)
                    max_tokens_input = gr.Slider(
                        label="Max Tokens",
                        minimum=1,
                        maximum=200,
                        value=50,
                        step=1,
                    )

                analyze_btn = gr.Button("Analyze", variant="primary")

                gr.Markdown("### Export")
                export_path = gr.Textbox(
                    label="Output Path",
                    placeholder="output.html",
                )
                export_btn = gr.Button("Export HTML")
                export_status = gr.Markdown("")

            # Right column: Visualizations
            with gr.Column(scale=2):
                summary_output = gr.Markdown("")

                with gr.Tabs():
                    with gr.Tab("Attention Map"):
                        attention_plot = gr.Plot(label="Attention Heatmap")

                    with gr.Tab("By Layer"):
                        with gr.Row():
                            layer_select = gr.Slider(
                                label="Layer",
                                minimum=0,
                                maximum=31,
                                value=0,
                                step=1,
                            )
                            head_select = gr.Dropdown(
                                label="Head",
                                choices=["All (mean)"] + [f"Head {i}" for i in range(32)],
                                value="All (mean)",
                            )
                        layer_plot = gr.Plot(label="Layer Attention")

                    with gr.Tab("Metrics"):
                        entropy_plot = gr.Plot(label="Entropy")

                    with gr.Tab("Tokens"):
                        tokens_output = gr.Markdown("")

        # Event handlers
        load_btn.click(
            load_model,
            inputs=[model_input],
            outputs=[model_status],
        )

        analyze_btn.click(
            analyze_prompt,
            inputs=[prompt_input, generate_check, max_tokens_input],
            outputs=[summary_output, attention_plot, entropy_plot, tokens_output],
        )

        layer_select.change(
            update_layer_view,
            inputs=[layer_select, head_select],
            outputs=[layer_plot],
        )

        head_select.change(
            update_layer_view,
            inputs=[layer_select, head_select],
            outputs=[layer_plot],
        )

        export_btn.click(
            export_html,
            inputs=[export_path],
            outputs=[export_status],
        )

    return demo


def launch_dashboard(
    share: bool = False,
    port: int = 7860,
    **kwargs,
) -> None:
    """Launch the Gradio dashboard.

    Args:
        share: Whether to create a public link
        port: Port to run on
        **kwargs: Additional arguments passed to Gradio launch
    """
    demo = create_dashboard()
    demo.launch(share=share, server_port=port, **kwargs)
