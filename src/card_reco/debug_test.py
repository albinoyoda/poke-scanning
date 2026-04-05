"""Unit tests for DebugWriter save methods."""

from __future__ import annotations

import numpy as np

from card_reco.debug import DebugWriter
from card_reco.models import CardRecord, DetectedCard, MatchResult


def _make_detected(size: int = 100) -> DetectedCard:
    """Create a minimal DetectedCard for testing."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    return DetectedCard(image=img, corners=corners, confidence=0.8)


def _make_match(rank: int = 1, distance: float = 10.0) -> MatchResult:
    """Create a minimal MatchResult for testing."""
    card = CardRecord(
        id="test-1",
        name="Pikachu",
        set_id="test",
        set_name="Test Set",
        number="1",
        rarity="Common",
        image_path="pikachu.png",
        ahash="0" * 64,
        phash="0" * 64,
        dhash="0" * 64,
    )
    return MatchResult(card=card, distance=distance, rank=rank)


class TestDebugWriter:
    """Tests for individual DebugWriter save methods."""

    def test_save_candidates_empty(self, tmp_path):
        """Empty contour list produces a candidates image without crash."""
        dw = DebugWriter(tmp_path / "debug")
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        dw.save_candidates(img, [], min_area=100.0)
        files = list(dw.output_dir.glob("*candidates*"))
        assert len(files) == 1

    def test_save_candidates_with_contours(self, tmp_path):
        """Contours are drawn on the image."""
        dw = DebugWriter(tmp_path / "debug")
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        contour = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.int32)
        contour = contour.reshape(  # pylint: disable=too-many-function-args
            4, 1, 2
        )
        dw.save_candidates(img, [contour], min_area=50.0)
        files = list(dw.output_dir.glob("*candidates*"))
        assert len(files) == 1

    def test_save_corners(self, tmp_path):
        """Corner overlay is written."""
        dw = DebugWriter(tmp_path / "debug")
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        corners = [
            np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
        ]
        dw.save_corners(img, corners, labels=["det0"])
        files = list(dw.output_dir.glob("*corners*"))
        assert len(files) == 1

    def test_save_corners_multiple(self, tmp_path):
        """Multiple corner sets are all drawn."""
        dw = DebugWriter(tmp_path / "debug")
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        corners = [
            np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32),
            np.array(
                [[200, 200], [350, 200], [350, 350], [200, 350]], dtype=np.float32
            ),
        ]
        dw.save_corners(img, corners)
        files = list(dw.output_dir.glob("*corners*"))
        assert len(files) == 1

    def test_save_nms_result(self, tmp_path):
        """NMS result overlay is written."""
        dw = DebugWriter(tmp_path / "debug")
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        det = _make_detected()
        dw.save_nms_result(img, 3, [det])
        files = list(dw.output_dir.glob("*nms*"))
        assert len(files) == 1

    def test_save_nms_result_empty(self, tmp_path):
        """Empty detection list after NMS doesn't crash."""
        dw = DebugWriter(tmp_path / "debug")
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        dw.save_nms_result(img, 0, [])
        files = list(dw.output_dir.glob("*nms*"))
        assert len(files) == 1

    def test_save_match_summary(self, tmp_path):
        """Match summary is written with card image and text."""
        dw = DebugWriter(tmp_path / "debug")
        card_img = np.zeros((100, 80, 3), dtype=np.uint8)
        matches = [_make_match(1, 5.0), _make_match(2, 15.0)]
        dw.save_match_summary(0, card_img, matches)
        files = list(dw.output_dir.glob("*match*"))
        assert len(files) == 1

    def test_save_match_summary_empty(self, tmp_path):
        """Empty match list shows 'No matches found'."""
        dw = DebugWriter(tmp_path / "debug")
        card_img = np.zeros((100, 80, 3), dtype=np.uint8)
        dw.save_match_summary(0, card_img, [])
        files = list(dw.output_dir.glob("*match*"))
        assert len(files) == 1

    def test_save_match_with_distances(self, tmp_path):
        """Match with per-hash distances doesn't crash."""
        dw = DebugWriter(tmp_path / "debug")
        card_img = np.zeros((100, 80, 3), dtype=np.uint8)
        m = _make_match(1, 10.0)
        m.distances = {"ahash": 5, "phash": 3, "dhash": 1}
        dw.save_match_summary(0, card_img, [m])
        files = list(dw.output_dir.glob("*match*"))
        assert len(files) == 1

    def test_save_warped(self, tmp_path):
        """Warped card image is saved."""
        dw = DebugWriter(tmp_path / "debug")
        warped = np.zeros((100, 80, 3), dtype=np.uint8)
        dw.save_warped(0, warped)
        files = list(dw.output_dir.glob("*warped*"))
        assert len(files) == 1

    def test_clean_removes_old(self, tmp_path):
        """clean=True removes prior debug output."""
        out = tmp_path / "debug"
        out.mkdir()
        (out / "old_file.png").write_bytes(b"x")
        DebugWriter(out, clean=True)
        assert not (out / "old_file.png").exists()
        assert out.exists()

    def test_edge_map(self, tmp_path):
        """save_edge_map writes a file."""
        dw = DebugWriter(tmp_path / "debug")
        edge = np.zeros((100, 100), dtype=np.uint8)
        dw.save_edge_map("test_edges", edge)
        files = list(dw.output_dir.glob("*test_edges*"))
        assert len(files) == 1
