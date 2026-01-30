"""Tests for attention downsampling."""

import numpy as np
import pytest

from sharingan.attention.downsampler import (
    downsample_attention,
    create_segment_view,
    create_local_view,
    create_multiscale_attention,
)


class TestDownsampleAttention:
    """Tests for downsample_attention function."""

    def test_no_downsampling_needed(self):
        """Test that small matrices are returned unchanged."""
        attention = np.random.rand(64, 64)
        result = downsample_attention(attention, target_size=128)
        assert result.shape == (64, 64)
        np.testing.assert_array_equal(result, attention)

    def test_downsample_2d_mean(self):
        """Test mean downsampling of 2D matrix."""
        attention = np.random.rand(512, 512)
        result = downsample_attention(attention, target_size=64, method="mean")
        assert result.shape[0] <= 64
        assert result.shape[1] <= 64

    def test_downsample_2d_max(self):
        """Test max downsampling of 2D matrix."""
        attention = np.random.rand(512, 512)
        result = downsample_attention(attention, target_size=64, method="max")
        assert result.shape[0] <= 64
        assert result.shape[1] <= 64

    def test_downsample_2d_sample(self):
        """Test sampling downsampling of 2D matrix."""
        attention = np.random.rand(512, 512)
        result = downsample_attention(attention, target_size=64, method="sample")
        assert result.shape == (64, 64)

    def test_downsample_3d(self):
        """Test downsampling of 3D matrix (heads, seq, seq)."""
        attention = np.random.rand(8, 512, 512)
        result = downsample_attention(attention, target_size=64)
        assert result.shape[0] == 8  # Heads preserved
        assert result.shape[1] <= 64
        assert result.shape[2] <= 64

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        attention = np.random.rand(64, 64)
        with pytest.raises(ValueError, match="Unknown method"):
            downsample_attention(attention, target_size=32, method="invalid")

    def test_invalid_dimensions(self):
        """Test that 1D or 4D+ arrays raise error."""
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            downsample_attention(np.random.rand(64), target_size=32)


class TestCreateSegmentView:
    """Tests for segment view creation."""

    def test_segment_view_shape(self):
        """Test that segment view has correct shape."""
        attention = np.random.rand(512, 512)
        tokens = ["tok"] * 512
        segment_attn, boundaries = create_segment_view(
            attention, tokens, segment_size=128
        )
        assert segment_attn.shape[0] == segment_attn.shape[1]
        assert len(boundaries) == segment_attn.shape[0] + 1

    def test_segment_boundaries(self):
        """Test that boundaries cover full sequence."""
        attention = np.random.rand(512, 512)
        tokens = ["tok"] * 512
        _, boundaries = create_segment_view(attention, tokens, segment_size=128)
        assert boundaries[0] == 0
        assert boundaries[-1] == 512


class TestCreateLocalView:
    """Tests for local view creation."""

    def test_local_view_center(self):
        """Test local view centered correctly."""
        attention = np.random.rand(1000, 1000)
        local, start, end = create_local_view(attention, center=500, window_size=200)
        assert local.shape == (200, 200)
        assert start == 400
        assert end == 600

    def test_local_view_boundary_start(self):
        """Test local view near start boundary."""
        attention = np.random.rand(1000, 1000)
        local, start, end = create_local_view(attention, center=50, window_size=200)
        assert start == 0
        assert end == 200

    def test_local_view_boundary_end(self):
        """Test local view near end boundary."""
        attention = np.random.rand(1000, 1000)
        local, start, end = create_local_view(attention, center=950, window_size=200)
        assert start == 800
        assert end == 1000


class TestCreateMultiscaleAttention:
    """Tests for multi-scale attention creation."""

    def test_multiscale_components(self):
        """Test that multiscale attention has all components."""
        attention = np.random.rand(4, 8, 512, 512)  # layers, heads, seq, seq
        tokens = ["tok"] * 512
        result = create_multiscale_attention(attention, tokens)

        assert result.global_view is not None
        assert result.global_view.shape[0] <= 256
        assert result.original_shape == attention.shape
        assert result.tokens == tokens

    def test_multiscale_short_sequence(self):
        """Test multiscale for sequences that don't need downsampling."""
        attention = np.random.rand(4, 8, 64, 64)
        tokens = ["tok"] * 64
        result = create_multiscale_attention(attention, tokens)

        # For short sequences, global view should be similar size
        assert result.global_view.shape[0] == 64
        # No local views needed for short sequences
        assert len(result.local_views) == 0
