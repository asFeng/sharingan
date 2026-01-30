# Claude Module Doc: models/

> AI-facing documentation for the models module

## Purpose
Model adapters for handling architecture-specific details (GQA, layer naming, etc.)

## Current Stage: ✅ Complete (Qwen adapter done)

## Files

### base.py
**Main class**: `ModelAdapter` (abstract base)
- Properties: `num_layers`, `num_heads`, `num_kv_heads`, `gqa_ratio`, `uses_gqa`
- Methods: `expand_gqa_attention()`, `process_attention()`

**Concrete class**: `DefaultAdapter`
- Fallback for unknown models
- Reads standard config attributes

### registry.py
**Functions**:
- `register_adapter(pattern)` - Decorator for registering adapters
- `get_adapter(model, model_name)` - Get appropriate adapter

**Pattern**: Uses string matching on model name or architecture

### qwen.py
**Class**: `QwenAdapter`
- Registered with `@register_adapter("qwen")`
- Handles Qwen3 GQA (16 Q heads, 8 KV heads)

## GQA Expansion Algorithm

```python
# Input: attention [batch, kv_heads, seq, seq]
# Output: attention [batch, q_heads, seq, seq]

def expand_gqa_attention(attention):
    # Each KV head is repeated for its group of Q heads
    # For 2:1 ratio: KV head 0 -> Q heads 0,1
    #                KV head 1 -> Q heads 2,3
    return attention.repeat_interleave(gqa_ratio, dim=1)
```

## Usage in Other Modules

```python
# In core/analyzer.py
from sharingan.models.registry import get_adapter
adapter = get_adapter(model, model_name)

# In attention/extractor.py
processed = adapter.process_attention(raw_attention)
```

## Adding a New Model Adapter

```python
# src/sharingan/models/newmodel.py
from sharingan.models.base import ModelAdapter
from sharingan.models.registry import register_adapter

@register_adapter("newmodel")  # Matches "newmodel" in name
class NewModelAdapter(ModelAdapter):
    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return getattr(self.config, "num_key_value_heads", self.num_heads)
```

Then import in `registry.py`:
```python
from sharingan.models import newmodel  # noqa
```

## Potential Improvements

- [ ] Add Llama adapter (similar GQA)
- [ ] Add Mistral adapter
- [ ] Add adapter for multi-query attention (MQA)
- [ ] Better pattern matching (regex instead of substring)
