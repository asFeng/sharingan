# Claude Dev Guide - Sharingan Project

> This document is for AI coding agents (Claude, etc.) to quickly understand the project structure, current state, and development patterns. Human developers should read README.md instead.

## Quick Context

**Project**: Sharingan - Attention visualization package for transformers
**Stage**: v0.1.0 (Phase 1-2 complete, Phase 3-5 in progress)
**Primary Model**: Qwen3-0.6B (with GQA handling)
**Framework**: HuggingFace Transformers + Gradio + Plotly

## File Naming Conventions

- `claude-*.md` - AI-facing documentation (this guide, module docs)
- `README.md` - Human-facing documentation
- Files in `src/sharingan/` follow standard Python package conventions

## Project Structure Quick Reference

```
sharingan/
├── claude-dev-guide.md          # THIS FILE - start here
├── pyproject.toml               # Package config, dependencies
├── README.md                    # User documentation
├── src/sharingan/
│   ├── __init__.py              # Public API: Sharingan, visualize()
│   ├── core/                    # Core classes (see claude-core.md)
│   ├── models/                  # Model adapters (see claude-models.md)
│   ├── attention/               # Extraction & analysis (see claude-attention.md)
│   ├── visualization/           # Plotting (see claude-visualization.md)
│   ├── ui/                      # Gradio dashboard (see claude-ui.md)
│   └── cli/                     # CLI commands (see claude-cli.md)
├── tests/                       # Unit tests
└── examples/                    # Usage examples
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core analyzer | ✅ Complete | `Sharingan` class, model loading |
| AttentionResult | ✅ Complete | Container with analysis methods |
| Qwen adapter | ✅ Complete | GQA expansion working |
| Attention extraction | ✅ Complete | Eager attention forcing |
| Downsampler | ✅ Complete | Multi-scale views |
| Metrics | ✅ Complete | Entropy, sinks, importance |
| Matplotlib heatmaps | ✅ Complete | Sharingan theme |
| Plotly interactive | ✅ Complete | Hover info, zoom |
| HTML export | ✅ Complete | Standalone with embedded data |
| Gradio dashboard | ✅ Complete | All tabs working |
| CLI | ✅ Complete | analyze, dashboard, info |
| Tests | ⚠️ Partial | Core tests done, need integration |

## Key Technical Decisions

### 1. GQA (Grouped Query Attention) Handling
- Qwen3-0.6B: 16 Q heads, 8 KV heads (2:1 ratio)
- Solution: `expand_gqa_attention()` repeats KV heads to match Q heads
- Location: `src/sharingan/models/base.py:50-60`

### 2. Flash Attention Workaround
- Flash Attention doesn't return weights by default
- Solution: Force eager attention via `attn_implementation="eager"`
- Location: `src/sharingan/core/analyzer.py:55`

### 3. Memory for Long Sequences
- 32K sequence = 100+ GB attention
- Solution: Streaming downsampling, CPU offloading
- Location: `src/sharingan/attention/downsampler.py`

### 4. Color Theme
- Sharingan-inspired: dark background (#111827) + red accents (#B91C1C, #EF4444)
- Defined in: `visualization/heatmap.py:SHARINGAN_COLORS`
- Plotly version: `visualization/interactive.py:SHARINGAN_COLORSCALE`

## Common Development Tasks

### Adding a new model adapter
1. Create `src/sharingan/models/newmodel.py`
2. Inherit from `ModelAdapter`
3. Use `@register_adapter("pattern")` decorator
4. Implement `num_layers`, `num_heads`, `num_kv_heads` properties

### Adding a new metric
1. Add function to `src/sharingan/attention/metrics.py`
2. Optionally add method to `AttentionResult` in `core/result.py`
3. Add test in `tests/test_metrics.py`

### Adding a visualization
1. Add to appropriate file in `visualization/`
2. Export in `visualization/__init__.py`
3. Consider adding to HTML export template

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=sharingan
```

## Dependencies Graph

```
User Code
    └── sharingan (public API)
            ├── core/analyzer.py (Sharingan class)
            │       ├── models/registry.py (get_adapter)
            │       └── attention/extractor.py (extract_attention)
            │
            ├── core/result.py (AttentionResult)
            │       ├── attention/metrics.py (entropy, sinks, etc.)
            │       ├── attention/downsampler.py (multi-scale)
            │       └── visualization/* (plot methods)
            │
            └── ui/dashboard.py (Gradio)
                    └── visualization/interactive.py (Plotly)
```

## Known Issues / TODOs

1. **Integration tests needed**: Full flow test with actual Qwen3 model
2. **Batch processing**: `analyze_batch()` is sequential, could parallelize
3. **Memory profiling**: Need to verify memory usage on 32K sequences
4. **Dashboard state**: Uses globals, could be cleaner with Gradio State

## Version History

- **v0.1.0** (current): Core functionality complete
  - Sharingan class with model loading
  - Attention extraction with GQA support
  - Multi-scale downsampling
  - Analysis metrics (entropy, sinks, importance)
  - Matplotlib and Plotly visualizations
  - HTML export
  - Gradio dashboard
  - CLI with typer

## Contact / Resources

- Main plan: See transcript at `/fsx/ubuntu/.claude/projects/.../070ea2ba-....jsonl`
- Original design: 5 phases (Foundation → Viz → Analysis → Dashboard → Polish)
