"""Tests for the scanner module, CardTracker, and identify_detections."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_reco.models import CardRecord, DetectedCard, MatchResult
from card_reco.scanner import CardTracker, Scanner, _draw_detections, _scale_image

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
        assert scanner._db_path is None  # pylint: disable=protected-access
        assert scanner._monitor_index == 1  # pylint: disable=protected-access
        assert scanner._region is None  # pylint: disable=protected-access
        assert scanner._sct is None  # pylint: disable=protected-access
        assert scanner._backend == "cnn"  # pylint: disable=protected-access

    def test_init_with_params(self) -> None:
        scanner = Scanner(
            db_path="test.db",
            monitor=2,
            region=(10, 20, 800, 600),
            top_n=3,
            threshold=30.0,
            backend="cnn",
        )
        assert scanner._db_path == "test.db"  # pylint: disable=protected-access
        assert scanner._monitor_index == 2  # pylint: disable=protected-access
        assert scanner._region == (10, 20, 800, 600)  # pylint: disable=protected-access
        assert scanner._top_n == 3  # pylint: disable=protected-access
        assert scanner._threshold == 30.0  # pylint: disable=protected-access
        assert scanner._backend == "cnn"  # pylint: disable=protected-access


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
class TestPreCreatedMatcher:  # pylint: disable=too-few-public-methods
    def test_same_results_with_external_matcher(self) -> None:
        """Passing a pre-created matcher should produce identical results."""
        import cv2  # pylint: disable=import-outside-toplevel

        from card_reco.matcher import (  # pylint: disable=import-outside-toplevel
            CardMatcher,
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


# ---------------------------------------------------------------------------
# CardTracker
# ---------------------------------------------------------------------------


def _make_card_record(card_id: str = "card-1", name: str = "Pikachu") -> CardRecord:
    return CardRecord(
        id=card_id,
        name=name,
        set_id="base1",
        set_name="Base",
        number="58",
        rarity="Common",
        image_path="base1/58.png",
    )


def _make_match(
    card_id: str = "card-1",
    name: str = "Pikachu",
    distance: float = 0.85,
) -> MatchResult:
    return MatchResult(
        card=_make_card_record(card_id, name),
        distance=distance,
        rank=1,
    )


def _make_detection(
    x: int = 10, y: int = 10, w: int = 100, h: int = 140
) -> DetectedCard:
    corners = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )
    image = np.zeros((h, w, 3), dtype=np.uint8)
    return DetectedCard(image=image, corners=corners, confidence=1.0)


class TestCardTracker:
    def test_empty_update(self) -> None:
        tracker = CardTracker()
        tracker.update([])
        assert tracker.count == 0
        assert not tracker.get_display_data()

    def test_single_card_tracked(self) -> None:
        tracker = CardTracker()
        det = _make_detection()
        match = _make_match()
        tracker.update([(det, [match])])

        assert tracker.count == 1
        display = tracker.get_display_data()
        assert len(display) == 1
        corners, matches = display[0]
        assert np.array_equal(corners, det.corners)
        assert matches[0].card.name == "Pikachu"

    def test_returns_voted_match_after_multiple_frames(self) -> None:
        tracker = CardTracker()
        det = _make_detection()
        m1 = _make_match("card-1", "Pikachu", 0.85)
        m2 = _make_match("card-2", "Raichu", 0.80)

        # 3 frames with Pikachu, 1 with Raichu → Pikachu wins.
        tracker.update([(det, [m1])])
        tracker.update([(det, [m1])])
        tracker.update([(det, [m1])])
        tracker.update([(det, [m2])])

        display = tracker.get_display_data()
        assert len(display) == 1
        assert display[0][1][0].card.name == "Pikachu"

    def test_new_card_creates_second_track(self) -> None:
        tracker = CardTracker()
        det_a = _make_detection(10, 10, 100, 140)
        det_b = _make_detection(300, 10, 100, 140)
        match_a = _make_match("a", "Pikachu")
        match_b = _make_match("b", "Charizard")

        tracker.update([(det_a, [match_a])])
        assert tracker.count == 1

        tracker.update([(det_a, [match_a]), (det_b, [match_b])])
        assert tracker.count == 2

    def test_stale_tracks_removed(self) -> None:
        tracker = CardTracker()
        det = _make_detection()
        match = _make_match()

        tracker.update([(det, [match])])
        assert tracker.count == 1

        # Simulate the card disappearing for many frames.
        for _ in range(6):
            tracker.update([])
        assert tracker.count == 0

    def test_clear(self) -> None:
        tracker = CardTracker()
        det = _make_detection()
        match = _make_match()
        tracker.update([(det, [match])])
        assert tracker.count == 1

        tracker.clear()
        assert tracker.count == 0

    def test_iou_matching_updates_existing_track(self) -> None:
        tracker = CardTracker()
        det1 = _make_detection(10, 10, 100, 140)
        det2 = _make_detection(12, 12, 100, 140)  # Slightly shifted
        m1 = _make_match("card-1", "Pikachu", 0.8)
        m2 = _make_match("card-1", "Pikachu", 0.9)

        tracker.update([(det1, [m1])])
        tracker.update([(det2, [m2])])

        # Should still be 1 tracked card, not 2.
        assert tracker.count == 1
        display = tracker.get_display_data()
        # Best score should be 0.9 (the higher one stored in best_per_id).
        assert display[0][1][0].distance == 0.9

    def test_no_match_card(self) -> None:
        tracker = CardTracker()
        det = _make_detection()
        tracker.update([(det, [])])

        assert tracker.count == 1
        display = tracker.get_display_data()
        assert display[0][1] == []


# ---------------------------------------------------------------------------
# identify_detections integration
# ---------------------------------------------------------------------------

FAISS_INDEX = Path("data") / "card_embeddings.faiss"
has_cnn_data = FAISS_INDEX.exists()
skip_no_cnn = pytest.mark.skipif(
    not has_cnn_data,
    reason="CNN model or FAISS index not available",
)


@skip_no_cnn
class TestIdentifyDetections:  # pylint: disable=too-few-public-methods
    def test_returns_paired_results(self) -> None:
        """identify_detections returns (DetectedCard, matches) tuples."""
        import cv2  # pylint: disable=import-outside-toplevel

        from card_reco.detector import (  # pylint: disable=import-outside-toplevel
            detect_cards,
        )
        from card_reco.pipeline import (  # pylint: disable=import-outside-toplevel
            identify_detections,
        )

        img_dir = TEST_DATA_DIR / "single_cards" / "axis_aligned"
        if not img_dir.exists():
            pytest.skip("axis_aligned test images not available")
        images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        if not images:
            pytest.skip("No test images found")

        image = cv2.imread(str(images[0]))
        assert image is not None

        detections = detect_cards(image, max_detect_dim=1024, fast=True)
        assert len(detections) > 0

        pairs = identify_detections(detections)
        assert len(pairs) > 0
        for card, matches in pairs:
            assert hasattr(card, "corners")
            assert hasattr(card, "image")
            if matches:
                assert matches[0].card.name  # has a name
