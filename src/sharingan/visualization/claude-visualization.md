# Claude Module Doc: visualization/

> AI-facing documentation for the visualization module

## Purpose
Attention visualization using Matplotlib (static) and Plotly (interactive), plus HTML export.

## Current Stage: ✅ Complete

## Files

### heatmap.py
**Functions**:
- `plot_heatmap(result, layer, head, level, ...)` - Main matplotlib heatmap
- `plot_layer_summary(result, ...)` - Grid of all layers
- `plot_entropy(result, ...)` - Entropy line plot

**Theme colors** (Sharingan-inspired):
```python
SHARINGAN_COLORS = {
    "primary": "#B91C1C",     # Deep red
    "secondary": "#1F2937",   # Dark gray
    "accent": "#EF4444",      # Bright red
    "background": "#111827",  # Near black
    "text": "#F9FAFB",        # Off white
}
```

**Custom colormap**: `get_sharingan_cmap()` - Dark to red gradient

### interactive.py
**Functions**:
- `plot_interactive(result, ...)` - Plotly heatmap with hover
- `plot_layer_head_grid(result, ...)` - Subplots grid
- `plot_metrics_dashboard(result, ...)` - Multi-panel metrics

**Plotly colorscale**:
```python
SHARINGAN_COLORSCALE = [
    [0.0, '#111827'],  # Dark
    [0.2, '#1F2937'],
    [0.4, '#7F1D1D'],
    [0.6, '#B91C1C'],
    [0.8, '#EF4444'],
    [1.0, '#FCA5A5'],  # Light red
]
```

### html_export.py
**Functions**:
- `export_html(result, path, include_metrics)` - Standalone HTML file

**HTML template features**:
- Embedded Plotly.js from CDN
- Tabbed interface (Overview, By Layer, Metrics, Tokens)
- Interactive layer/head selection
- Metrics display (entropy, importance)
- Attention sinks listing
- No external dependencies after generation

## Usage in Other Modules

```python
# In core/result.py (AttentionResult.plot)
from sharingan.visualization.heatmap import plot_heatmap
from sharingan.visualization.interactive import plot_interactive

if interactive:
    return plot_interactive(self, layer, head, level)
else:
    return plot_heatmap(self, layer, head, level)

# In core/result.py (AttentionResult.to_html)
from sharingan.visualization.html_export import export_html
export_html(self, path, include_metrics)
```

## Auto Level Selection

```python
if level == "auto":
    level = "global" if seq_len > 256 else "local"
```

- **local**: Full resolution, token labels if ≤50 tokens
- **global**: Downsampled to 256×256, no token labels

## Potential Improvements

- [ ] Animation for layer-by-layer attention
- [ ] Attention flow visualization (sankey diagram)
- [ ] Token highlighting on hover
- [ ] Dark/light theme toggle in HTML export
- [ ] PDF export option
