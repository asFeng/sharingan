# Claude Module Doc: ui/

> AI-facing documentation for the UI module

## Purpose
Gradio-based interactive web dashboard for attention visualization.

## Current Stage: ✅ Complete (basic functionality)

## Files

### dashboard.py
**Functions**:
- `create_dashboard()` - Build Gradio Blocks interface
- `launch_dashboard(share, port)` - Launch the dashboard

**Internal functions**:
- `load_model(model_name)` - Load model into global state
- `load_file(file_path)` - Load text from uploaded file
- `analyze_prompt(prompt, file_path, generate, max_tokens, scale)` - Run analysis
- `update_layer_view(layer, head)` - Update layer/head plot (uses current scale)
- `create_attention_figure(result, layer, head, scale)` - Create Plotly heatmap
- `create_generation_figure(result, scale)` - Show generated→prompt/generated attention
- `create_entropy_figure(result)` - Entropy line plot
- `export_html(output_path)` - Export current result

**Global state** (not ideal, but works):
```python
_current_analyzer: Sharingan | None = None
_current_result: AttentionResult | None = None
_current_scale: str = "sqrt"  # For layer view updates
```

## Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│ [Sharingan Header - Red gradient]                   │
├──────────────┬──────────────────────────────────────┤
│ Model Input  │  Summary: Layers | Heads | Seq len   │
│ [Load]       │  ┌─────────────────────────────────┐ │
│              │  │ Tabs: Attention | Generation |  │ │
│ Prompt Input │  │       By Layer | Metrics | Tokens│ │
│ [File Upload]│  ├─────────────────────────────────┤ │
│ [Generate ✓] │  │                                 │ │
│ [Max Tokens] │  │    [Visualization Area]         │ │
│ [Scale ▼]    │  │    (with generation boundary)   │ │
│ [Analyze]    │  │                                 │ │
│              │  └─────────────────────────────────┘ │
│ Export Path  │                                      │
│ [Export]     │                                      │
└──────────────┴──────────────────────────────────────┘
```

**Scale dropdown options**:
- Square Root (recommended) - `sqrt`
- Logarithmic - `log`
- Row Normalized - `row`
- Percentile Clip - `percentile`
- Rank Based - `rank`
- Raw (no scaling) - `none`

**Generation tab**: Shows 2 heatmaps side-by-side:
- Generated → Prompt (how gen tokens attend to prompt)
- Generated → Generated (how gen tokens attend to each other)

## Theme CSS

Custom CSS for Sharingan dark theme:
- Background: #111827
- Accent: #B91C1C
- Button hover: #EF4444

## Usage from CLI

```python
# In cli/main.py
from sharingan.ui.dashboard import launch_dashboard

@app.command()
def dashboard(port: int = 7860, share: bool = False):
    launch_dashboard(port=port, share=share)
```

## Event Flow

1. User enters model name → `load_model()` → Sets `_current_analyzer`
2. User enters prompt → `analyze_prompt()` → Sets `_current_result`
3. User changes layer/head → `update_layer_view()` → Uses `_current_result`
4. User clicks export → `export_html()` → Uses `_current_result.to_html()`

## Potential Improvements

- [ ] Use Gradio State instead of globals
- [ ] Add progress bar for model loading
- [ ] Add model comparison view
- [ ] Add attention animation playback
- [ ] Add token search/highlight
- [ ] Better error handling with user feedback
- [ ] Add session persistence
