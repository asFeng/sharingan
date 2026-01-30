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


def load_file(file_path: str | None) -> str:
    """Load text from uploaded file."""
    if file_path is None:
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content
    except Exception as e:
        return f"Error loading file: {e}"


def analyze_prompt(
    prompt: str,
    file_path: str | None,
    generate: bool,
    max_tokens: int,
    progress=gr.Progress(),
) -> tuple[str, go.Figure, go.Figure, go.Figure, str]:
    """Analyze a prompt and return visualizations."""
    global _current_result

    # Use file content if provided
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        except Exception as e:
            empty_fig = go.Figure()
            empty_fig.update_layout(**SHARINGAN_TEMPLATE)
            return f"Error reading file: {e}", empty_fig, empty_fig, empty_fig, ""

    if _current_analyzer is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(**SHARINGAN_TEMPLATE)
        return "Please load a model first", empty_fig, empty_fig, empty_fig, ""

    if not prompt.strip():
        empty_fig = go.Figure()
        empty_fig.update_layout(**SHARINGAN_TEMPLATE)
        return "Please enter a prompt", empty_fig, empty_fig, empty_fig, ""

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

        # Create generation attention figure
        gen_fig = create_generation_figure(_current_result)

        # Summary text
        summary = _current_result.summary()
        summary_text = (
            f"**Layers:** {summary['num_layers']} | "
            f"**Heads:** {summary['num_heads']} | "
            f"**Sequence:** {summary['seq_len']} tokens\n\n"
            f"**Prompt:** {_current_result.prompt_length} tokens | "
            f"**Generated:** {_current_result.generated_length} tokens\n\n"
            f"**Mean Entropy:** {summary['mean_entropy']:.3f} | "
            f"**Attention Sinks:** {summary['num_sinks']}"
        )

        # Tokens display with prompt/generated marking
        tokens = _current_result.tokens
        prompt_len = _current_result.prompt_length
        tokens_display = "**Prompt:** " + " ".join(
            f"`{t}`" for t in tokens[:min(prompt_len, 50)]
        )
        if prompt_len > 50:
            tokens_display += f" ... (+{prompt_len - 50} more)"

        if _current_result.generated_length > 0:
            tokens_display += "\n\n**Generated:** " + " ".join(
                f"`{t}`" for t in tokens[prompt_len:prompt_len + 50]
            )
            if _current_result.generated_length > 50:
                tokens_display += f" ... (+{_current_result.generated_length - 50} more)"

        progress(1.0, desc="Done!")
        return summary_text, attn_fig, entropy_fig, gen_fig, tokens_display

    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(**SHARINGAN_TEMPLATE)
        return f"Error: {str(e)}", empty_fig, empty_fig, empty_fig, ""


def create_attention_figure(
    result: AttentionResult,
    layer: int | None = None,
    head: int | None = None,
) -> go.Figure:
    """Create Plotly attention heatmap with token labels and generation boundary."""
    attention = result.get_attention(layer=layer, head=head, aggregate="mean")
    tokens = result.tokens
    prompt_len = result.prompt_length

    # Downsample if needed
    downsampled = False
    if attention.shape[0] > 256:
        from sharingan.attention.downsampler import downsample_attention

        attention = downsample_attention(attention, target_size=256)
        downsampled = True

    # Create hover text with token info
    if not downsampled and len(tokens) <= 100:
        hover_text = []
        for i in range(len(tokens)):
            row = []
            q_type = "P" if i < prompt_len else "G"
            for j in range(len(tokens)):
                k_type = "P" if j < prompt_len else "G"
                row.append(
                    f"Query [{i}] ({q_type}): {tokens[i]!r}<br>"
                    f"Key [{j}] ({k_type}): {tokens[j]!r}<br>"
                    f"Attention: {attention[i, j]:.4f}"
                )
            hover_text.append(row)

        x_labels = [f"{i}:{t[:6]}" for i, t in enumerate(tokens)]
        y_labels = x_labels
    else:
        hover_text = None
        x_labels = None
        y_labels = None

    fig = go.Figure(
        data=go.Heatmap(
            z=attention,
            colorscale=SHARINGAN_COLORSCALE,
            hoverinfo="text" if hover_text else "z",
            text=hover_text,
            x=x_labels,
            y=y_labels,
        )
    )

    # Add generation boundary
    if result.generated_length > 0 and not downsampled:
        boundary = prompt_len - 0.5
        fig.add_hline(y=boundary, line_dash="dash", line_color="#EF4444", line_width=2)
        fig.add_vline(x=boundary, line_dash="dash", line_color="#EF4444", line_width=2)

    title = "Attention (Aggregated)"
    if layer is not None:
        title = f"Layer {layer}" + (f", Head {head}" if head is not None else "")
    if result.generated_length > 0:
        title += f" | P:{prompt_len} G:{result.generated_length}"

    fig.update_layout(
        title=title,
        xaxis_title="Key Position (attending to)",
        yaxis_title="Query Position (attending from)",
        height=550,
        **SHARINGAN_TEMPLATE,
    )
    fig.update_yaxes(autorange="reversed")

    return fig


def create_generation_figure(result: AttentionResult) -> go.Figure:
    """Create generation attention visualization."""
    from plotly.subplots import make_subplots

    if result.generated_length == 0:
        # Return placeholder if no generation
        fig = go.Figure()
        fig.add_annotation(
            text="No generated tokens.<br>Enable 'Generate' to see generation attention.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#9CA3AF")
        )
        fig.update_layout(height=400, **SHARINGAN_TEMPLATE)
        return fig

    attention = result.get_attention(aggregate="mean")
    prompt_len = result.prompt_length
    gen_len = result.generated_length
    tokens = result.tokens

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Generated → Prompt", "Generated → Generated"],
        horizontal_spacing=0.12,
    )

    # Left: Generated attending to prompt
    gen_to_prompt = attention[prompt_len:, :prompt_len]
    prompt_labels = [f"{i}:{t[:6]}" for i, t in enumerate(tokens[:prompt_len])] if prompt_len <= 40 else None
    gen_labels = [f"{i}:{t[:6]}" for i, t in enumerate(tokens[prompt_len:], start=prompt_len)] if gen_len <= 40 else None

    fig.add_trace(
        go.Heatmap(
            z=gen_to_prompt,
            x=prompt_labels,
            y=gen_labels,
            colorscale=SHARINGAN_COLORSCALE,
            showscale=False,
            hovertemplate="Gen→Prompt<br>Attention: %{z:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Right: Generated attending to generated
    gen_to_gen = attention[prompt_len:, prompt_len:]

    fig.add_trace(
        go.Heatmap(
            z=gen_to_gen,
            x=gen_labels,
            y=gen_labels,
            colorscale=SHARINGAN_COLORSCALE,
            showscale=True,
            hovertemplate="Gen→Gen<br>Attention: %{z:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Generation Attention Flow (prompt: {prompt_len}, gen: {gen_len})",
        height=450,
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

    # Add generation boundary
    if result.generated_length > 0:
        fig.add_vline(
            x=result.prompt_length - 0.5,
            line_dash="dash", line_color="#EF4444",
            annotation_text="Gen Start"
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
                file_input = gr.File(
                    label="Or upload text file",
                    file_types=[".txt", ".md"],
                    type="filepath",
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

                    with gr.Tab("Generation"):
                        gr.Markdown("*Shows how generated tokens attend to prompt and each other*")
                        generation_plot = gr.Plot(label="Generation Attention")

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
            inputs=[prompt_input, file_input, generate_check, max_tokens_input],
            outputs=[summary_output, attention_plot, entropy_plot, generation_plot, tokens_output],
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
