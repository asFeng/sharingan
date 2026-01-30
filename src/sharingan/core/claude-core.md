# Claude Module Doc: core/

> AI-facing documentation for the core module

## Purpose
Contains the main user-facing classes: `Sharingan` analyzer and `AttentionResult` container.

## Current Stage: ✅ Complete

## Files

### analyzer.py
**Main class**: `Sharingan`
- Loads HuggingFace models with eager attention (required for attention weights)
- Uses model adapters for architecture-specific handling
- Entry point for attention analysis

**Key methods**:
- `load()` - Lazy model loading
- `analyze(prompt, generate=False)` - Main analysis entry point
- `analyze_batch(prompts)` - Batch processing (currently sequential)

**Dependencies**:
- `models/registry.py` - Gets appropriate adapter
- `attention/extractor.py` - Extracts attention weights

### result.py
**Main class**: `AttentionResult`
- Container for attention data with analysis methods
- Caches computed metrics (`_entropy`, `_sinks`, `_importance`)

**Key methods**:
- `get_attention(layer, head, aggregate)` - Get attention with optional selection
- `attention_entropy()` - Entropy analysis (cached)
- `attention_sinks()` - Find sink tokens (cached)
- `token_importance()` - Score tokens (cached)
- `hotspots()` - Find high-attention pairs
- `plot()` - Create visualization (delegates to visualization/)
- `to_html()` - Export (delegates to visualization/html_export.py)
- `summary()` - Get summary dict

### config.py
**Main class**: `SharinganConfig`
- Dataclass for configuration
- Device/dtype resolution
- Theme colors

## Usage in Other Modules

```python
# In visualization/
from sharingan.core.result import AttentionResult
def plot_heatmap(result: AttentionResult, ...):
    attention = result.get_attention(...)

# In cli/
from sharingan.core.analyzer import Sharingan
analyzer = Sharingan(model_name)
result = analyzer.analyze(prompt)
```

## Key Design Decisions

1. **Lazy loading**: Model loads on first `analyze()` call, not constructor
2. **Caching**: Metrics cached on AttentionResult to avoid recomputation
3. **Delegation**: `plot()` and `to_html()` delegate to visualization module
4. **Adapter pattern**: Analyzer uses adapters for model-specific handling

## Potential Improvements

- [ ] Add streaming/chunked analysis for very long sequences
- [ ] Add async model loading
- [ ] Better batch processing (parallel tokenization)
