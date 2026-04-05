"""Unit tests for detect_cards orchestrator and _four_point_transform."""

from __future__ import annotations

import numpy as np

from card_reco.detector import _four_point_transform
from card_reco.detector.constants import CARD_HEIGHT, CARD_WIDTH


class TestFourPointTransform:
    """Tests for _four_point_transform."""

    def test_output_dimensions_portrait(self):
        """Portrait card → (CARD_HEIGHT, CARD_WIDTH)."""
        img = np.zeros((800, 600, 3), dtype=np.uint8)
        corners = np.array(
            [[50, 50], [550, 50], [550, 750], [50, 750]], dtype=np.float32
        )
        result = _four_point_transform(img, corners)
        assert result.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
        assert result.dtype == np.uint8

    def test_output_dimensions_landscape(self):
        """Landscape card → rotated to portrait (CARD_HEIGHT, CARD_WIDTH)."""
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        # Wider than tall by > 1.2x → landscape
        corners = np.array(
            [[50, 50], [750, 50], [750, 350], [50, 350]], dtype=np.float32
        )
        result = _four_point_transform(img, corners)
        assert result.shape[0] > result.shape[1]  # portrait after rotation

    def test_preserves_content(self):
        """A white rectangle warped produces non-zero pixels."""
        img = np.full((400, 300, 3), 200, dtype=np.uint8)
        corners = np.array(
            [[10, 10], [290, 10], [290, 390], [10, 390]], dtype=np.float32
        )
        result = _four_point_transform(img, corners)
        assert np.mean(result) > 100

    def test_perspective_correction(self):
        """Slightly trapezoidal corners still produce rectangular output."""
        img = np.full((600, 500, 3), 128, dtype=np.uint8)
        corners = np.array(
            [[20, 10], [480, 20], [470, 580], [30, 590]], dtype=np.float32
        )
        result = _four_point_transform(img, corners)
        assert result.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
