"""Unit tests for detector quality scoring and edge verification."""

from __future__ import annotations

import cv2
import numpy as np

from card_reco.detector.quality import contour_quality, corner_edge_fraction


class TestContourQuality:
    """Tests for contour_quality."""

    def _make_rect_contour(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Build a rectangular contour."""
        pts = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.int32,
        )
        return pts.reshape(4, 1, 2)  # pylint: disable=too-many-function-args

    def test_perfect_card_rectangle(self):
        """A clean 5:7 rectangle should score high."""
        contour = self._make_rect_contour(0, 0, 500, 700)
        corners = np.array([[0, 0], [500, 0], [500, 700], [0, 700]], dtype=np.float32)
        score = contour_quality(contour, corners, image_area=1_000_000)
        assert score > 0.7

    def test_zero_area_contour(self):
        """A degenerate contour with zero area returns 0."""
        pts = np.array([[0, 0], [10, 0], [10, 0], [0, 0]], dtype=np.int32)
        contour = pts.reshape(  # pylint: disable=too-many-function-args
            4, 1, 2
        )
        corners = np.array([[0, 0], [10, 0], [10, 0], [0, 0]], dtype=np.float32)
        score = contour_quality(contour, corners, image_area=1_000_000)
        assert score == 0.0

    def test_non_card_aspect_ratio(self):
        """A very elongated rectangle scores lower due to bad aspect ratio."""
        contour = self._make_rect_contour(0, 0, 100, 1000)
        corners = np.array([[0, 0], [100, 0], [100, 1000], [0, 1000]], dtype=np.float32)
        score = contour_quality(contour, corners, image_area=1_000_000)
        # Still gets compactness/rectangularity credit but aspect is off
        assert score < 0.8

    def test_small_contour_penalised(self):
        """A tiny contour gets a lower score than a larger one."""
        small_contour = self._make_rect_contour(0, 0, 10, 14)
        small_corners = np.array([[0, 0], [10, 0], [10, 14], [0, 14]], dtype=np.float32)
        large_contour = self._make_rect_contour(0, 0, 500, 700)
        large_corners = np.array(
            [[0, 0], [500, 0], [500, 700], [0, 700]], dtype=np.float32
        )
        small_score = contour_quality(
            small_contour, small_corners, image_area=10_000_000
        )
        large_score = contour_quality(
            large_contour, large_corners, image_area=10_000_000
        )
        assert small_score < large_score


class TestCornerEdgeFraction:
    """Tests for corner_edge_fraction."""

    def test_all_corners_on_edges(self):
        """All corners near strong edges → fraction 1.0."""
        edge_map = np.zeros((200, 200), dtype=np.uint8)
        corners = np.array(
            [[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32
        )
        # Draw edges at each corner location
        for pt in corners:
            cv2.circle(edge_map, (int(pt[0]), int(pt[1])), 3, 255, -1)
        assert corner_edge_fraction(corners, edge_map) == 1.0

    def test_no_corners_on_edges(self):
        """No edges anywhere → fraction 0.0."""
        edge_map = np.zeros((200, 200), dtype=np.uint8)
        corners = np.array(
            [[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32
        )
        assert corner_edge_fraction(corners, edge_map) == 0.0

    def test_half_corners_on_edges(self):
        """Two of four corners near edges → fraction 0.5."""
        edge_map = np.zeros((200, 200), dtype=np.uint8)
        corners = np.array(
            [[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32
        )
        # Only draw edges at first two corners
        cv2.circle(edge_map, (50, 50), 3, 255, -1)
        cv2.circle(edge_map, (150, 50), 3, 255, -1)
        assert corner_edge_fraction(corners, edge_map) == 0.5

    def test_corner_at_image_boundary(self):
        """Corners near image edges don't crash (clamp to bounds)."""
        edge_map = np.full((100, 100), 255, dtype=np.uint8)
        corners = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32)
        assert corner_edge_fraction(corners, edge_map) == 1.0
