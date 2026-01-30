"""Tests for attention metrics."""

import numpy as np
import pytest

from sharingan.attention.metrics import (
    compute_entropy,
    find_attention_sinks,
    compute_token_importance,
    compute_attention_distance,
    layer_head_similarity,
)


class TestComputeEntropy:
    """Tests for entropy computation."""

    def test_uniform_distribution(self):
        """Test entropy of uniform distribution (should be high)."""
        # Uniform distribution over 10 tokens
        uniform = np.ones((10, 10)) / 10
        entropy = compute_entropy(uniform)
        # Max entropy for 10 tokens is log(10) ≈ 2.3
        assert np.all(entropy > 2.0)

    def test_peaked_distribution(self):
        """Test entropy of peaked distribution (should be low)."""
        # One-hot attention (all attention to one token)
        peaked = np.zeros((10, 10))
        peaked[:, 0] = 1.0
        entropy = compute_entropy(peaked)
        # Near-zero entropy
        assert np.all(entropy < 0.1)

    def test_entropy_shape(self):
        """Test that entropy has correct shape."""
        attention = np.random.rand(5, 10)
        attention = attention / attention.sum(axis=1, keepdims=True)
        entropy = compute_entropy(attention)
        assert entropy.shape == (5,)


class TestFindAttentionSinks:
    """Tests for attention sink detection."""

    def test_find_obvious_sink(self):
        """Test detection of obvious attention sink."""
        attention = np.ones((100, 100)) / 100  # Uniform baseline
        attention[:, 0] = 0.5  # First token is a sink
        attention = attention / attention.sum(axis=1, keepdims=True)

        sinks = find_attention_sinks(attention, threshold=0.05)
        assert len(sinks) > 0
        assert sinks[0]["position"] == 0

    def test_no_sinks(self):
        """Test that uniform attention has no sinks."""
        attention = np.ones((100, 100)) / 100
        sinks = find_attention_sinks(attention, threshold=0.1)
        # Uniform attention shouldn't have strong sinks
        assert len(sinks) == 0

    def test_sink_properties(self):
        """Test that sink dict has expected properties."""
        attention = np.ones((100, 100)) / 100
        attention[:, 0] = 0.5
        attention = attention / attention.sum(axis=1, keepdims=True)

        sinks = find_attention_sinks(attention, threshold=0.01)
        if sinks:
            sink = sinks[0]
            assert "position" in sink
            assert "attention_received" in sink
            assert "attending_positions" in sink
            assert "attending_ratio" in sink


class TestComputeTokenImportance:
    """Tests for token importance scoring."""

    def test_importance_shape(self):
        """Test that importance has correct shape."""
        attention = np.random.rand(100, 100)
        attention = attention / attention.sum(axis=1, keepdims=True)
        importance = compute_token_importance(attention)
        assert importance.shape == (100,)

    def test_importance_normalized(self):
        """Test that importance is normalized to [0, 1]."""
        attention = np.random.rand(100, 100)
        attention = attention / attention.sum(axis=1, keepdims=True)
        importance = compute_token_importance(attention)
        assert importance.min() >= 0
        assert importance.max() <= 1

    def test_importance_methods(self):
        """Test different importance methods."""
        attention = np.random.rand(50, 50)
        attention = attention / attention.sum(axis=1, keepdims=True)

        for method in ["received", "given", "both", "pagerank"]:
            importance = compute_token_importance(attention, method=method)
            assert importance.shape == (50,)
            assert importance.min() >= 0
            assert importance.max() <= 1

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        attention = np.random.rand(50, 50)
        with pytest.raises(ValueError, match="Unknown method"):
            compute_token_importance(attention, method="invalid")


class TestComputeAttentionDistance:
    """Tests for attention distance metrics."""

    def test_distance_keys(self):
        """Test that distance dict has expected keys."""
        attention = np.random.rand(100, 100)
        attention = attention / attention.sum(axis=1, keepdims=True)
        dist = compute_attention_distance(attention)

        assert "mean_distance" in dist
        assert "local_attention_ratio" in dist
        assert "distant_attention_ratio" in dist

    def test_local_distant_sum(self):
        """Test that local + distant ratios sum to 1."""
        attention = np.random.rand(100, 100)
        attention = attention / attention.sum(axis=1, keepdims=True)
        dist = compute_attention_distance(attention)

        assert abs(dist["local_attention_ratio"] + dist["distant_attention_ratio"] - 1.0) < 1e-6


class TestLayerHeadSimilarity:
    """Tests for layer-head similarity computation."""

    def test_similarity_shape(self):
        """Test similarity matrix shape."""
        attention = np.random.rand(4, 8, 64, 64)  # 4 layers, 8 heads
        similarity = layer_head_similarity(attention)
        assert similarity.shape == (32, 32)  # 4*8 = 32

    def test_self_similarity(self):
        """Test that diagonal is all 1s (self-similarity)."""
        attention = np.random.rand(4, 8, 64, 64)
        similarity = layer_head_similarity(attention)
        np.testing.assert_array_almost_equal(np.diag(similarity), np.ones(32))

    def test_similarity_symmetric(self):
        """Test that similarity matrix is symmetric."""
        attention = np.random.rand(4, 8, 64, 64)
        similarity = layer_head_similarity(attention)
        np.testing.assert_array_almost_equal(similarity, similarity.T)
