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
- `analyze_prompt(prompt, generate, max_tokens)` - Run analysis
- `update_layer_view(layer, head)` - Update layer/head plot
- `export_html(output_path)` - Export current result

**Global state** (not ideal, but works):
```python
_current_analyzer: Sharingan | None = None
_current_result: AttentionResult | None = None
```

## Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│ [Sharingan Header - Red gradient]                   │
├──────────────┬──────────────────────────────────────┤
│ Model Input  │                                      │
│ [Load]       │  ┌─────────────────────────────────┐ │
│              │  │ Tabs: Attention | Layer | Metrics│ │
│ Prompt Input │  ├─────────────────────────────────┤ │
│ [Generate ✓] │  │                                 │ │
│ [Max Tokens] │  │    [Visualization Area]         │ │
│ [Analyze]    │  │                                 │ │
│              │  │                                 │ │
│ Export Path  │  └─────────────────────────────────┘ │
│ [Export]     │                                      │
└──────────────┴──────────────────────────────────────┘
```

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
