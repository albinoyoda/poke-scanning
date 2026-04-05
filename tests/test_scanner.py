"""Tests for the scanner module and pipeline matcher-reuse refactor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_reco.scanner import Scanner, _draw_detections, _scale_image

# ---------------------------------------------------------------------------
# _scale_image
# ---------------------------------------------------------------------------


class TestScaleImage:
    def test_no_resize_when_fits(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _scale_image(img, 200, 100)
        assert result.shape == (100, 200, 3)

    def test_scales_down_landscape(self) -> None:
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        result = _scale_image(img, 400, 300)
        assert result.shape[1] <= 400
        assert result.shape[0] <= 300

    def test_scales_down_portrait(self) -> None:
        img = np.zeros((800, 400, 3), dtype=np.uint8)
        result = _scale_image(img, 200, 400)
        assert result.shape[1] <= 200
        assert result.shape[0] <= 400

    def test_preserves_three_channels(self) -> None:
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        result = _scale_image(img, 500, 500)
        assert result.ndim == 3
        assert result.shape[2] == 3


# ---------------------------------------------------------------------------
# _draw_detections
# ---------------------------------------------------------------------------


class TestDrawDetections:
    def test_no_detections(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _draw_detections(img, [], [])
        assert result.shape == img.shape
        # Original should not be mutated.
        assert not np.array_equal(result, img) or np.all(img == 0)

    def test_with_detection_and_match(self) -> None:
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        det = MagicMock()
        det.corners = np.array(
            [[10, 10], [100, 10], [100, 150], [10, 150]],
            dtype=np.float32,
        )

        match = MagicMock()
        match.card.name = "Pikachu"
        match.distance = 5.0

        result = _draw_detections(img, [det], [[match]])
        # Should have drawn something (not all black).
        assert result.shape == img.shape
        assert np.any(result != 0)

    def test_detection_without_match(self) -> None:
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        det = MagicMock()
        det.corners = np.array(
            [[10, 10], [100, 10], [100, 150], [10, 150]],
            dtype=np.float32,
        )
        result = _draw_detections(img, [det], [])
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# Scanner init / cleanup
# ---------------------------------------------------------------------------


class TestScannerLifecycle:
    def test_init_defaults(self) -> None:
        scanner = Scanner()
        assert scanner._db_path is None
        assert scanner._monitor_index == 1
        assert scanner._region is None
        assert scanner._matcher is None
        assert scanner._sct is None

    def test_init_with_params(self) -> None:
        scanner = Scanner(
            db_path="test.db",
            monitor=2,
            region=(10, 20, 800, 600),
            top_n=3,
            threshold=30.0,
        )
        assert scanner._db_path == "test.db"
        assert scanner._monitor_index == 2
        assert scanner._region == (10, 20, 800, 600)
        assert scanner._top_n == 3
        assert scanner._threshold == 30.0


# ---------------------------------------------------------------------------
# Pipeline refactor: identify_cards_from_array with pre-created matcher
# ---------------------------------------------------------------------------

DB_PATH = Path("data") / "card_hashes.db"
TEST_DATA_DIR = Path("data") / "tests"

has_test_data = TEST_DATA_DIR.exists() and DB_PATH.exists()
skip_no_data = pytest.mark.skipif(
    not has_test_data,
    reason="Test data or hash DB not available",
)


@skip_no_data
class TestPreCreatedMatcher:
    def test_same_results_with_external_matcher(self) -> None:
        """Passing a pre-created matcher should produce identical results."""
        import cv2  # pylint: disable=import-outside-toplevel

        from card_reco.matcher import (
            CardMatcher,  # pylint: disable=import-outside-toplevel
        )
        from card_reco.pipeline import (  # pylint: disable=import-outside-toplevel
            identify_cards_from_array,
        )

        # Find a test image.
        single_dir = TEST_DATA_DIR / "single_cards" / "axis_aligned"
        if not single_dir.exists():
            pytest.skip("axis_aligned test images not available")
        images = list(single_dir.glob("*.png")) + list(single_dir.glob("*.jpg"))
        if not images:
            pytest.skip("No test images found")
        image = cv2.imread(str(images[0]))
        assert image is not None

        # Run without pre-created matcher (original path).
        results_default = identify_cards_from_array(
            image,
            db_path=str(DB_PATH),
            top_n=3,
            threshold=40.0,
        )

        # Run with pre-created matcher.
        with CardMatcher(str(DB_PATH)) as matcher:
            results_external = identify_cards_from_array(
                image,
                top_n=3,
                threshold=40.0,
                matcher=matcher,
            )

        # Both should produce the same top match.
        assert len(results_default) == len(results_external)
        for default, external in zip(results_default, results_external, strict=True):
            if default and external:
                assert default[0].card.id == external[0].card.id
                assert abs(default[0].distance - external[0].distance) < 1e-6
