"""Tests for GQA (Grouped Query Attention) handling."""

import numpy as np
import pytest
import torch

from sharingan.models.base import ModelAdapter, DefaultAdapter


class MockConfig:
    """Mock model config for testing."""

    def __init__(
        self,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads


class MockModel:
    """Mock model for testing adapters."""

    def __init__(self, config: MockConfig):
        self.config = config


class TestDefaultAdapter:
    """Tests for DefaultAdapter GQA handling."""

    def test_no_gqa(self):
        """Test model without GQA."""
        config = MockConfig(num_attention_heads=16, num_key_value_heads=None)
        model = MockModel(config)
        adapter = DefaultAdapter(model, "test-model")

        assert adapter.num_heads == 16
        assert adapter.num_kv_heads == 16
        assert adapter.gqa_ratio == 1
        assert not adapter.uses_gqa

    def test_gqa_2_to_1(self):
        """Test model with 2:1 GQA ratio."""
        config = MockConfig(num_attention_heads=16, num_key_value_heads=8)
        model = MockModel(config)
        adapter = DefaultAdapter(model, "test-model")

        assert adapter.num_heads == 16
        assert adapter.num_kv_heads == 8
        assert adapter.gqa_ratio == 2
        assert adapter.uses_gqa

    def test_gqa_4_to_1(self):
        """Test model with 4:1 GQA ratio."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
        model = MockModel(config)
        adapter = DefaultAdapter(model, "test-model")

        assert adapter.num_heads == 32
        assert adapter.num_kv_heads == 8
        assert adapter.gqa_ratio == 4
        assert adapter.uses_gqa


class TestGQAExpansion:
    """Tests for GQA attention expansion."""

    def test_expand_gqa_no_expansion_needed(self):
        """Test that non-GQA attention is unchanged."""
        config = MockConfig(num_attention_heads=8, num_key_value_heads=8)
        model = MockModel(config)
        adapter = DefaultAdapter(model, "test-model")

        attention = torch.rand(1, 8, 64, 64)  # batch, heads, seq, seq
        expanded = adapter.expand_gqa_attention(attention)

        assert expanded.shape == (1, 8, 64, 64)
        torch.testing.assert_close(expanded, attention)

    def test_expand_gqa_2_to_1(self):
        """Test 2:1 GQA expansion."""
        config = MockConfig(num_attention_heads=16, num_key_value_heads=8)
        model = MockModel(config)
        adapter = DefaultAdapter(model, "test-model")

        attention = torch.rand(1, 8, 64, 64)  # 8 KV heads
        expanded = adapter.expand_gqa_attention(attention)

        assert expanded.shape == (1, 16, 64, 64)  # 16 Q heads

        # Check that each KV head is repeated twice
        for kv_head in range(8):
            q_head_1 = kv_head * 2
            q_head_2 = kv_head * 2 + 1
            torch.testing.assert_close(
                expanded[0, q_head_1], attention[0, kv_head]
            )
            torch.testing.assert_close(
                expanded[0, q_head_2], attention[0, kv_head]
            )

    def test_expand_gqa_4_to_1(self):
        """Test 4:1 GQA expansion."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
        model = MockModel(config)
        adapter = DefaultAdapter(model, "test-model")

        attention = torch.rand(1, 8, 64, 64)  # 8 KV heads
        expanded = adapter.expand_gqa_attention(attention)

        assert expanded.shape == (1, 32, 64, 64)  # 32 Q heads

        # Check that each KV head is repeated 4 times
        for kv_head in range(8):
            for i in range(4):
                q_head = kv_head * 4 + i
                torch.testing.assert_close(
                    expanded[0, q_head], attention[0, kv_head]
                )


class TestProcessAttention:
    """Tests for attention processing to numpy."""

    def test_process_non_gqa(self):
        """Test processing without GQA."""
        config = MockConfig(num_attention_heads=8, num_key_value_heads=8)
        model = MockModel(config)
        adapter = DefaultAdapter(model, "test-model")

        attention = torch.rand(1, 8, 64, 64)
        processed = adapter.process_attention(attention)

        assert isinstance(processed, np.ndarray)
        assert processed.shape == (8, 64, 64)

    def test_process_with_gqa(self):
        """Test processing with GQA expansion."""
        config = MockConfig(num_attention_heads=16, num_key_value_heads=8)
        model = MockModel(config)
        adapter = DefaultAdapter(model, "test-model")

        attention = torch.rand(1, 8, 64, 64)  # 8 KV heads
        processed = adapter.process_attention(attention)

        assert isinstance(processed, np.ndarray)
        assert processed.shape == (16, 64, 64)  # Expanded to 16 Q heads
