"""Tests for AttentionResult class."""

import numpy as np
import pytest

from sharingan.core.result import AttentionResult


@pytest.fixture
def sample_result():
    """Create a sample AttentionResult for testing."""
    attention = np.random.rand(4, 8, 64, 64)  # 4 layers, 8 heads, 64 seq
    # Normalize to valid attention weights
    attention = attention / attention.sum(axis=-1, keepdims=True)

    return AttentionResult(
        attention=attention,
        tokens=["tok" + str(i) for i in range(64)],
        prompt_length=50,
        generated_length=14,
        model_name="test-model",
    )


class TestAttentionResultProperties:
    """Tests for AttentionResult properties."""

    def test_num_layers(self, sample_result):
        assert sample_result.num_layers == 4

    def test_num_heads(self, sample_result):
        assert sample_result.num_heads == 8

    def test_seq_len(self, sample_result):
        assert sample_result.seq_len == 64

    def test_total_length(self, sample_result):
        assert sample_result.total_length == 64  # prompt + generated


class TestGetAttention:
    """Tests for get_attention method."""

    def test_get_all(self, sample_result):
        """Test getting aggregated attention."""
        attn = sample_result.get_attention()
        assert attn.shape == (64, 64)

    def test_get_specific_layer(self, sample_result):
        """Test getting specific layer attention."""
        attn = sample_result.get_attention(layer=2)
        assert attn.shape == (64, 64)

    def test_get_specific_head(self, sample_result):
        """Test getting specific head attention."""
        attn = sample_result.get_attention(head=3)
        assert attn.shape == (64, 64)

    def test_get_layer_and_head(self, sample_result):
        """Test getting specific layer and head."""
        attn = sample_result.get_attention(layer=1, head=2)
        assert attn.shape == (64, 64)

    def test_get_no_aggregate(self, sample_result):
        """Test getting without aggregation."""
        attn = sample_result.get_attention(aggregate="none")
        assert attn.shape == (4, 8, 64, 64)

    def test_get_max_aggregate(self, sample_result):
        """Test max aggregation."""
        attn = sample_result.get_attention(aggregate="max")
        assert attn.shape == (64, 64)


class TestAttentionEntropy:
    """Tests for attention_entropy method."""

    def test_entropy_shape_default(self, sample_result):
        """Test entropy shape with default params."""
        entropy = sample_result.attention_entropy()
        assert entropy.shape == (64,)

    def test_entropy_specific_layer(self, sample_result):
        """Test entropy for specific layer."""
        entropy = sample_result.attention_entropy(layer=1)
        # Returns [heads, seq_len] when only layer is specified
        assert entropy.ndim == 2
        assert entropy.shape == (8, 64)  # 8 heads, 64 seq

    def test_entropy_caching(self, sample_result):
        """Test that entropy is cached."""
        _ = sample_result.attention_entropy()
        assert sample_result._entropy is not None


class TestAttentionSinks:
    """Tests for attention_sinks method."""

    def test_sinks_returns_list(self, sample_result):
        sinks = sample_result.attention_sinks()
        assert isinstance(sinks, list)

    def test_sinks_caching(self, sample_result):
        _ = sample_result.attention_sinks()
        assert sample_result._sinks is not None


class TestTokenImportance:
    """Tests for token_importance method."""

    def test_importance_shape(self, sample_result):
        importance = sample_result.token_importance()
        assert importance.shape == (64,)

    def test_importance_normalized(self, sample_result):
        importance = sample_result.token_importance()
        assert importance.min() >= 0
        assert importance.max() <= 1


class TestHotspots:
    """Tests for hotspots method."""

    def test_hotspots_returns_list(self, sample_result):
        hotspots = sample_result.hotspots()
        assert isinstance(hotspots, list)

    def test_hotspots_top_k(self, sample_result):
        hotspots = sample_result.hotspots(top_k=5)
        assert len(hotspots) == 5

    def test_hotspots_properties(self, sample_result):
        hotspots = sample_result.hotspots(top_k=1)
        assert "from_position" in hotspots[0]
        assert "to_position" in hotspots[0]
        assert "attention" in hotspots[0]


class TestSummary:
    """Tests for summary method."""

    def test_summary_keys(self, sample_result):
        summary = sample_result.summary()
        expected_keys = [
            "model",
            "num_layers",
            "num_heads",
            "seq_len",
            "prompt_length",
            "generated_length",
            "mean_entropy",
            "min_entropy",
            "max_entropy",
            "num_sinks",
        ]
        for key in expected_keys:
            assert key in summary


class TestRepr:
    """Tests for string representation."""

    def test_repr(self, sample_result):
        repr_str = repr(sample_result)
        assert "AttentionResult" in repr_str
        assert "layers=4" in repr_str
        assert "heads=8" in repr_str
