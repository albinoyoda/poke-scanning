"""Unit tests for corner extraction and refinement."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from card_reco.detector.corners import (
    _intersect_param_lines,
    corner_geometry,
    extract_corners,
    has_card_aspect_ratio,
    order_corners,
    refine_corners_from_hull,
)


class TestCornerGeometry:
    """Tests for corner_geometry."""

    def test_square(self):
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        w, h, ratio = corner_geometry(corners)
        assert w == pytest.approx(100.0)
        assert h == pytest.approx(100.0)
        assert ratio == pytest.approx(1.0)

    def test_card_aspect_ratio(self):
        corners = np.array([[0, 0], [500, 0], [500, 700], [0, 700]], dtype=np.float32)
        w, h, ratio = corner_geometry(corners)
        assert w == pytest.approx(500.0)
        assert h == pytest.approx(700.0)
        assert ratio == pytest.approx(5.0 / 7.0, abs=0.01)

    def test_zero_dimension(self):
        corners = np.array([[0, 0], [0, 0], [0, 100], [0, 100]], dtype=np.float32)
        _, _, ratio = corner_geometry(corners)
        assert ratio == 0.0


class TestOrderCorners:
    """Tests for order_corners."""

    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        result = order_corners(pts)
        np.testing.assert_array_equal(result[0], [0, 0])  # TL
        np.testing.assert_array_equal(result[1], [100, 0])  # TR
        np.testing.assert_array_equal(result[2], [100, 100])  # BR
        np.testing.assert_array_equal(result[3], [0, 100])  # BL

    def test_shuffled(self):
        pts = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
        result = order_corners(pts)
        np.testing.assert_array_equal(result[0], [0, 0])
        np.testing.assert_array_equal(result[1], [100, 0])
        np.testing.assert_array_equal(result[2], [100, 100])
        np.testing.assert_array_equal(result[3], [0, 100])

    def test_perspective_distorted(self):
        """Non-rectangular points are still correctly ordered."""
        pts = np.array([[10, 5], [95, 10], [90, 105], [5, 95]], dtype=np.float32)
        result = order_corners(pts)
        # TL should be smallest sum, BR should be largest sum
        assert result[0].sum() < result[2].sum()
        # All four corners should be present
        assert result.shape == (4, 2)


class TestHasCardAspectRatio:
    """Tests for has_card_aspect_ratio."""

    def test_card_shaped(self):
        corners = np.array([[0, 0], [500, 0], [500, 700], [0, 700]], dtype=np.float32)
        assert has_card_aspect_ratio(corners) is True

    def test_square_rejected(self):
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        assert has_card_aspect_ratio(corners) is False

    def test_very_narrow_rejected(self):
        corners = np.array([[0, 0], [10, 0], [10, 700], [0, 700]], dtype=np.float32)
        assert has_card_aspect_ratio(corners) is False

    def test_landscape_card(self):
        """A 7:5 landscape card has the same ratio as 5:7 portrait."""
        corners = np.array([[0, 0], [700, 0], [700, 500], [0, 500]], dtype=np.float32)
        assert has_card_aspect_ratio(corners) is True


class TestExtractCorners:
    """Tests for extract_corners."""

    def test_rectangular_contour(self):
        """A clean rectangle yields 4 ordered corners."""
        pts = np.array(
            [[10, 10], [200, 10], [200, 290], [10, 290]],
            dtype=np.int32,
        )
        contour = pts.reshape(  # pylint: disable=too-many-function-args
            4, 1, 2
        )
        corners = extract_corners(contour)
        assert corners is not None
        assert corners.shape == (4, 2)

    def test_too_few_points_returns_none(self):
        """A line segment can't yield 4 corners."""
        pts = np.array([[0, 0], [100, 0]], dtype=np.int32)
        contour = pts.reshape(  # pylint: disable=too-many-function-args
            2, 1, 2
        )
        assert extract_corners(contour) is None

    def test_circle_uses_minrect_fallback(self):
        """A dense circular contour may use the minAreaRect fallback."""
        # Generate a dense ellipse contour (roughly rectangular after minAreaRect)
        img = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (280, 280), 255, 2)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0
        corners = extract_corners(contours[0])  # ty: ignore[invalid-argument-type]
        # Should get corners from the rectangle outline
        assert corners is not None
        assert corners.shape == (4, 2)


class TestRefineFromHull:
    """Tests for refine_corners_from_hull."""

    def test_basic_refinement(self):
        """Hull refinement produces valid 4-point output."""
        # A rounded-corner rectangle contour
        img = np.zeros((400, 300), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (280, 380), 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        refined = refine_corners_from_hull(contour, box)  # ty: ignore[invalid-argument-type]
        assert refined.shape == (4, 2)
        # All corners should be inside/near the image
        assert np.all(refined >= -10)
        assert np.all(refined <= 410)

    def test_degenerate_hull_falls_back(self):
        """A line-like contour with <4 hull points returns original box."""
        pts = np.array([[0, 0], [100, 0], [100, 1]], dtype=np.int32)
        contour = pts.reshape(  # pylint: disable=too-many-function-args
            3, 1, 2
        )
        box = np.array([[0, 0], [100, 0], [100, 1], [0, 1]], dtype=np.float32)
        result = refine_corners_from_hull(contour, box)
        np.testing.assert_array_equal(result, box)


class TestIntersectParamLines:
    """Tests for _intersect_param_lines."""

    def test_perpendicular_lines(self):
        """Two perpendicular lines through the origin intersect at origin."""
        # Horizontal: direction (1, 0), point (0, 0)
        line_a = (1.0, 0.0, 0.0, 0.0)
        # Vertical: direction (0, 1), point (0, 0)
        line_b = (0.0, 1.0, 0.0, 0.0)
        pt = _intersect_param_lines(line_a, line_b)
        assert pt is not None
        assert pt[0] == pytest.approx(0.0, abs=1e-6)
        assert pt[1] == pytest.approx(0.0, abs=1e-6)

    def test_offset_lines(self):
        """Two perpendicular lines at offsets intersect at crossing point."""
        # Horizontal through y=50
        line_a = (1.0, 0.0, 0.0, 50.0)
        # Vertical through x=30
        line_b = (0.0, 1.0, 30.0, 0.0)
        pt = _intersect_param_lines(line_a, line_b)
        assert pt is not None
        assert pt[0] == pytest.approx(30.0, abs=1e-6)
        assert pt[1] == pytest.approx(50.0, abs=1e-6)

    def test_parallel_lines_return_none(self):
        """Parallel lines have no intersection."""
        line_a = (1.0, 0.0, 0.0, 0.0)
        line_b = (1.0, 0.0, 0.0, 50.0)
        assert _intersect_param_lines(line_a, line_b) is None
