"""Unit tests for pipeline helper functions."""

from __future__ import annotations

import numpy as np

from card_reco.detector.constants import CARD_HEIGHT, CARD_WIDTH
from card_reco.pipeline import (
    _denoise_clahe,
    _expand_corners,
    _make_whole_image_card,
    _rewarp,
)


class TestMakeWholeImageCard:
    """Tests for _make_whole_image_card."""

    def test_portrait_card_shape(self):
        """A card-shaped portrait image returns a DetectedCard."""
        # 5:7 aspect ratio (500×700)
        img = np.zeros((700, 500, 3), dtype=np.uint8)
        result = _make_whole_image_card(img)
        assert result is not None
        assert result.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
        assert result.confidence == 1.0

    def test_landscape_card_shape(self):
        """A card-shaped landscape (7:5) image is accepted and rotated."""
        img = np.zeros((500, 700, 3), dtype=np.uint8)
        result = _make_whole_image_card(img)
        assert result is not None
        assert result.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)

    def test_non_card_aspect_rejected(self):
        """A very wide image is not card-shaped."""
        img = np.zeros((100, 800, 3), dtype=np.uint8)
        result = _make_whole_image_card(img)
        assert result is None

    def test_square_image_rejected(self):
        """A square image is not close enough to 5:7."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = _make_whole_image_card(img)
        assert result is None

    def test_zero_dimension_rejected(self):
        """A zero-height image is rejected."""
        img = np.zeros((0, 500, 3), dtype=np.uint8)
        result = _make_whole_image_card(img)
        assert result is None

    def test_corners_match_image_bounds(self):
        """Corners form the image boundary rectangle."""
        img = np.zeros((700, 500, 3), dtype=np.uint8)
        result = _make_whole_image_card(img)
        assert result is not None
        expected = np.array([[0, 0], [499, 0], [499, 699], [0, 699]], dtype=np.float32)
        np.testing.assert_array_equal(result.corners, expected)


class TestExpandCorners:
    """Tests for _expand_corners."""

    def test_expand_outward(self):
        """Positive percentage expands corners from centroid."""
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        expanded = _expand_corners(corners, 10.0)
        # Centroid is (50, 50). Each corner moves 10% farther from centroid.
        expected = np.array(
            [[-5, -5], [105, -5], [105, 105], [-5, 105]], dtype=np.float32
        )
        np.testing.assert_allclose(expanded, expected, atol=1e-5)

    def test_contract_inward(self):
        """Negative percentage contracts corners toward centroid."""
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        contracted = _expand_corners(corners, -10.0)
        expected = np.array([[5, 5], [95, 5], [95, 95], [5, 95]], dtype=np.float32)
        np.testing.assert_allclose(contracted, expected, atol=1e-5)

    def test_zero_pct_unchanged(self):
        """Zero percentage leaves corners unchanged."""
        corners = np.array([[10, 20], [90, 20], [90, 80], [10, 80]], dtype=np.float32)
        result = _expand_corners(corners, 0.0)
        np.testing.assert_allclose(result, corners, atol=1e-5)


class TestRewarp:
    """Tests for _rewarp."""

    def test_output_dimensions(self):
        """Output is always (CARD_HEIGHT, CARD_WIDTH)."""
        img = np.zeros((600, 400, 3), dtype=np.uint8)
        corners = np.array(
            [[10, 10], [390, 10], [390, 590], [10, 590]], dtype=np.float32
        )
        result = _rewarp(img, corners)
        assert result.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
        assert result.dtype == np.uint8

    def test_corners_clamped_to_image(self):
        """Corners outside the image are clamped; no crash."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        corners = np.array(
            [[-50, -50], [150, -50], [150, 150], [-50, 150]], dtype=np.float32
        )
        result = _rewarp(img, corners)
        assert result.shape == (CARD_HEIGHT, CARD_WIDTH, 3)

    def test_preserves_content(self):
        """A white region warped from the image produces non-zero output."""
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        corners = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], dtype=np.float32)
        result = _rewarp(img, corners)
        assert np.mean(result) > 200  # Most pixels should be white

    def test_denoise_returns_same_shape(self):
        """Smoke test for _denoise_clahe."""
        img = np.random.default_rng(42).integers(0, 255, (100, 80, 3), dtype=np.uint8)
        result = _denoise_clahe(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
