# Claude Module Doc: attention/

> AI-facing documentation for the attention module

## Purpose
Attention extraction from models, downsampling for long sequences, and analysis metrics.

## Current Stage: ✅ Complete

## Files

### extractor.py
**Functions**:
- `extract_attention(model, inputs, adapter)` - Main extraction
- `extract_attention_streaming(...)` - Memory-efficient for long sequences (WIP)

**Key logic**:
```python
outputs = model(**inputs, output_attentions=True)
# outputs.attentions is tuple of [batch, heads, seq, seq] per layer
for layer_attn in outputs.attentions:
    processed = adapter.process_attention(layer_attn)  # GQA expansion
```

### downsampler.py
**Main class**: `MultiScaleAttention`
- Container for global, segment, and local views

**Functions**:
- `downsample_attention(attention, target_size, method)` - Core downsampling
  - Methods: "mean" (average pool), "max" (max pool), "sample" (uniform)
- `create_segment_view(attention, tokens, segment_size)` - Segment aggregation
- `create_local_view(attention, center, window_size)` - Local window
- `create_multiscale_attention(...)` - Create all views

**Downsampling strategy**:
```
seq_len > 256  → global view (256×256)
seq_len > 512  → segment view + local views at hotspots
seq_len ≤ 256  → no downsampling needed
```

### metrics.py
**Functions**:
- `compute_entropy(attention)` - Information entropy (-Σ p log p)
- `find_attention_sinks(attention, threshold)` - Tokens receiving high attention
- `compute_token_importance(attention, method)` - Score tokens
  - Methods: "received", "given", "both", "pagerank"
- `compute_attention_distance(attention)` - Local vs distant attention
- `layer_head_similarity(attention)` - Cosine similarity between patterns

## Usage in Other Modules

```python
# In core/analyzer.py
from sharingan.attention.extractor import extract_attention
attention = extract_attention(model, inputs, adapter=adapter)

# In core/result.py
from sharingan.attention.metrics import compute_entropy
entropy = compute_entropy(self.attention)

# In visualization/
from sharingan.attention.downsampler import downsample_attention
if seq_len > 256:
    attention = downsample_attention(attention, 256)
```

## Memory Considerations

For a 32K sequence:
- Full attention: 32K × 32K × 4 bytes × layers × heads ≈ 100+ GB
- Global view (256×256): ~256 KB per layer-head
- Segment view: ~1 KB

Strategy: Downsample immediately after extraction, offload to CPU.

## Potential Improvements

- [ ] Streaming extraction (process one layer at a time)
- [ ] Smart segmentation (paragraph/sentence boundaries)
- [ ] GPU-accelerated downsampling
- [ ] Attention flow analysis (information propagation)
