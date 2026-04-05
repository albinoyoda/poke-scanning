"""Unit tests for detection strategies helpers."""

from __future__ import annotations

import pytest

from card_reco.detector.strategies import (
    _angle_dist,
    _line_intersection,
    _pick_two_lines,
)


class TestAngleDist:
    """Tests for _angle_dist."""

    def test_same_angle(self):
        assert _angle_dist(45.0, 45.0) == pytest.approx(0.0)

    def test_opposite_directions(self):
        """0 and 180 are the same line direction."""
        assert _angle_dist(0.0, 180.0) == pytest.approx(0.0)

    def test_perpendicular(self):
        assert _angle_dist(0.0, 90.0) == pytest.approx(90.0)

    def test_wrap_around(self):
        """170 and 10 are 20 apart (wrapping around 180)."""
        assert _angle_dist(170.0, 10.0) == pytest.approx(20.0)

    def test_small_difference(self):
        assert _angle_dist(5.0, 175.0) == pytest.approx(10.0)


class TestLineIntersection:
    """Tests for _line_intersection."""

    def _seg(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> tuple[float, float, int, int, int, int]:
        """Build a segment tuple with dummy angle=0 and length=0."""
        return (0.0, 0.0, x1, y1, x2, y2)

    def test_perpendicular_at_origin(self):
        s1 = self._seg(0, 0, 100, 0)  # horizontal
        s2 = self._seg(0, 0, 0, 100)  # vertical
        pt = _line_intersection(s1, s2)
        assert pt is not None
        assert pt[0] == pytest.approx(0.0, abs=1e-3)
        assert pt[1] == pytest.approx(0.0, abs=1e-3)

    def test_offset_cross(self):
        s1 = self._seg(0, 50, 100, 50)  # horizontal at y=50
        s2 = self._seg(30, 0, 30, 100)  # vertical at x=30
        pt = _line_intersection(s1, s2)
        assert pt is not None
        assert pt[0] == pytest.approx(30.0, abs=1e-3)
        assert pt[1] == pytest.approx(50.0, abs=1e-3)

    def test_parallel_returns_none(self):
        s1 = self._seg(0, 0, 100, 0)
        s2 = self._seg(0, 50, 100, 50)
        assert _line_intersection(s1, s2) is None


class TestPickTwoLines:
    """Tests for _pick_two_lines."""

    def _hseg(
        self, y: int, length: float = 100.0
    ) -> tuple[float, float, int, int, int, int]:
        """Build a horizontal segment at height *y*."""
        return (0.0, length, 0, y, 100, y)

    def test_two_separated_lines(self):
        """Two far-apart horizontal lines are both selected."""
        segs = [self._hseg(10), self._hseg(200)]
        result = _pick_two_lines(segs, 0.0)
        assert result is not None
        assert len(result) == 2

    def test_too_close_returns_none(self):
        """Lines with gap < 20 are rejected."""
        segs = [self._hseg(50), self._hseg(55)]
        result = _pick_two_lines(segs, 0.0)
        assert result is None

    def test_single_line_returns_none(self):
        result = _pick_two_lines([self._hseg(50)], 0.0)
        assert result is None

    def test_selects_longest(self):
        """From each side of the midpoint the longest segment is picked."""
        hi = (0.0, 200.0, 0, 10, 200, 10)  # long, near top
        hi_short = (0.0, 50.0, 0, 15, 50, 15)
        lo = (0.0, 150.0, 0, 300, 150, 300)  # long, near bottom
        lo_short = (0.0, 30.0, 0, 305, 30, 305)
        result = _pick_two_lines([hi, hi_short, lo, lo_short], 0.0)
        assert result is not None
        # Two lines selected, both should be the longest from their group
        selected_lengths = sorted([result[0][1], result[1][1]])
        assert selected_lengths == [150.0, 200.0]
