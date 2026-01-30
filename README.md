# Sharingan

Attention visualization for transformers. Built on HuggingFace Transformers, with first-class support for Qwen3 models.

## Installation

```bash
pip install sharingan
```

Or install from source:
```bash
git clone https://github.com/sharingan-viz/sharingan
cd sharingan
pip install -e .
```

## Quick Start

### One-liner visualization

```python
from sharingan import visualize

visualize("Qwen/Qwen3-0.6B", "The capital of France is")
```

### Detailed analysis

```python
from sharingan import Sharingan

# Load model
analyzer = Sharingan("Qwen/Qwen3-0.6B")

# Analyze prompt
result = analyzer.analyze("The quick brown fox jumps over the lazy dog.")

# Explore attention patterns
result.attention_entropy()      # Attention distribution analysis
result.attention_sinks()        # Find attention sink tokens
result.token_importance()       # Token importance scores
result.hotspots()               # High-attention token pairs

# Visualize
result.plot()                           # Matplotlib heatmap
result.plot(interactive=True)           # Interactive Plotly
result.plot(layer=5, head=3)            # Specific layer/head
result.to_html("output.html")           # Standalone HTML export
```

### With generation

```python
result = analyzer.analyze(
    "Once upon a time",
    generate=True,
    max_new_tokens=50,
)
```

## CLI Usage

```bash
# Analyze and save visualization
sharingan analyze "The capital of France is" --model Qwen/Qwen3-0.6B -o viz.html

# Launch interactive dashboard
sharingan dashboard --port 7860

# Get model info
sharingan info Qwen/Qwen3-0.6B
```

## Features

- **Multi-scale visualization**: Global overview, segment-level, and local full-resolution views
- **GQA support**: Proper handling of Grouped Query Attention (Qwen3, Llama2, etc.)
- **Analysis metrics**: Entropy, attention sinks, token importance, hotspots
- **Interactive dashboard**: Gradio-based web interface
- **Export**: Standalone HTML files with embedded visualizations

## Long Context Support

For sequences longer than 256 tokens, Sharingan automatically provides:

1. **Global View** (256×256 max): Downsampled overview showing overall patterns
2. **Segment View**: Attention between logical segments
3. **Local View**: Full resolution for selected windows

## Supported Models

- Qwen3 (0.6B, 1.8B, 4B, etc.) - with GQA handling
- Any HuggingFace transformer model that outputs attention weights

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.37.0+

## License

MIT
