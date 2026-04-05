"""Unit tests for NMS and centroid deduplication."""

from __future__ import annotations

import numpy as np
import pytest

from card_reco.detector.nms import (
    centroid_dedup,
    compute_overlap,
    non_max_suppression,
)
from card_reco.models import DetectedCard


def _make_det(x: float, y: float, size: float, confidence: float = 0.5) -> DetectedCard:
    """Create a DetectedCard with corners at (x, y) to (x+size, y+size)."""
    corners = np.array(
        [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
        dtype=np.float32,
    )
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    return DetectedCard(image=img, corners=corners, confidence=confidence)


class TestComputeOverlap:
    """Tests for compute_overlap (bounding-rect IoU)."""

    def test_identical_corners(self):
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        assert compute_overlap(corners, corners) == pytest.approx(1.0)

    def test_no_overlap(self):
        c1 = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float32)
        c2 = np.array(
            [[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.float32
        )
        assert compute_overlap(c1, c2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        c1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        c2 = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)
        iou = compute_overlap(c1, c2)
        # Intersection: 50×50=2500, Union: 10000+10000-2500=17500
        assert iou == pytest.approx(2500 / 17500, abs=0.01)

    def test_zero_area(self):
        c1 = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
        c2 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        assert compute_overlap(c1, c2) == pytest.approx(0.0, abs=1e-3)


class TestNonMaxSuppression:
    """Tests for non_max_suppression."""

    def test_no_detections(self):
        assert not non_max_suppression([])

    def test_single_detection(self):
        det = _make_det(0, 0, 100)
        result = non_max_suppression([det])
        assert len(result) == 1

    def test_overlapping_removes_lower_confidence(self):
        """Two highly overlapping detections — lower conf is suppressed."""
        d1 = _make_det(0, 0, 100, confidence=0.9)
        d2 = _make_det(5, 5, 100, confidence=0.3)
        result = non_max_suppression([d1, d2], overlap_thresh=0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_non_overlapping_kept(self):
        """Well-separated detections are both kept."""
        d1 = _make_det(0, 0, 50, confidence=0.9)
        d2 = _make_det(200, 200, 50, confidence=0.8)
        result = non_max_suppression([d1, d2])
        assert len(result) == 2

    def test_different_size_kept(self):
        """Overlapping but very different-sized detections are kept."""
        d1 = _make_det(0, 0, 200, confidence=0.9)
        d2 = _make_det(10, 10, 50, confidence=0.8)
        result = non_max_suppression([d1, d2], overlap_thresh=0.3)
        assert len(result) == 2


class TestCentroidDedup:
    """Tests for centroid_dedup."""

    def test_no_detections(self):
        assert not centroid_dedup([])

    def test_single_detection(self):
        det = _make_det(0, 0, 100)
        result = centroid_dedup([det])
        assert len(result) == 1

    def test_nearby_centroids_deduped(self):
        """Two detections with very close centroids → only the best kept."""
        d1 = _make_det(0, 0, 100, confidence=0.9)
        d2 = _make_det(2, 2, 100, confidence=0.5)
        result = centroid_dedup([d1, d2], min_dist_frac=0.15)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_distant_centroids_kept(self):
        """Detections with distant centroids are both kept."""
        d1 = _make_det(0, 0, 50, confidence=0.9)
        d2 = _make_det(300, 300, 50, confidence=0.8)
        result = centroid_dedup([d1, d2])
        assert len(result) == 2

    def test_different_area_near_centroids_kept(self):
        """Different-sized detections at same location are both kept."""
        d1 = _make_det(0, 0, 200, confidence=0.9)
        d2 = _make_det(10, 10, 50, confidence=0.8)
        result = centroid_dedup([d1, d2], min_dist_frac=0.5)
        assert len(result) == 2
