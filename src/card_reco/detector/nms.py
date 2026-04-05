"""Non-maximum suppression and centroid-based deduplication."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from card_reco.detector.constants import _AREA_DIVERSITY_THRESH
from card_reco.models import DetectedCard


def _detection_area(det: DetectedCard) -> float:
    """Compute the contour area of a detection's corner quad."""
    return float(cv2.contourArea(det.corners.reshape(4, 1, 2).astype(np.int32)))


def non_max_suppression(
    detections: list[DetectedCard],
    overlap_thresh: float = 0.5,
) -> list[DetectedCard]:
    """Remove overlapping detections, keeping the one with higher confidence.

    When two detections overlap but differ significantly in area (ratio below
    *_AREA_DIVERSITY_THRESH*), both are kept so that the matcher can evaluate
    both the broad and tight crops.  This helps rotated and perspective-
    distorted cards where a tighter crop often yields a better hash match.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    detections.sort(key=lambda d: d.confidence, reverse=True)

    # Precompute areas
    areas = [_detection_area(d) for d in detections]
    keep: list[DetectedCard] = []
    keep_areas: list[float] = []

    for det, det_area in zip(detections, areas, strict=True):
        suppressed = False
        for kept, kept_area in zip(keep, keep_areas, strict=True):
            if compute_overlap(det.corners, kept.corners) > overlap_thresh:
                # Allow both when areas differ enough — smaller crops
                # may be tighter and produce better hash matches.
                max_area = max(det_area, kept_area)
                if max_area > 0:
                    area_ratio = min(det_area, kept_area) / max_area
                    if area_ratio < _AREA_DIVERSITY_THRESH:
                        continue  # Don't suppress — different-sized crop
                suppressed = True
                break
        if not suppressed:
            keep.append(det)
            keep_areas.append(det_area)

    return keep


def centroid_dedup(
    detections: list[DetectedCard],
    min_dist_frac: float = 0.15,
) -> list[DetectedCard]:
    """Remove near-duplicate detections whose centroids are close.

    Multiple edge-detection strategies often find slightly different
    contours for the same physical card.  When two detections have
    centroids within *min_dist_frac* of the smaller detection's
    diagonal **and** similar area, the lower-confidence one is dropped.

    Detections that differ significantly in area (ratio below
    *_AREA_DIVERSITY_THRESH*) are always kept, preserving
    different-scale crops such as a card inside a graded slab.
    """
    if len(detections) <= 1:
        return detections

    detections.sort(key=lambda d: d.confidence, reverse=True)

    # Precompute areas
    areas = [_detection_area(d) for d in detections]
    keep: list[DetectedCard] = []
    keep_areas: list[float] = []

    for det, det_area in zip(detections, areas, strict=True):
        cx = float(det.corners[:, 0].mean())
        cy = float(det.corners[:, 1].mean())
        rect = cv2.boundingRect(det.corners.astype(np.int32))
        diag = float(np.sqrt(rect[2] ** 2 + rect[3] ** 2))

        suppressed = False
        for kept, kept_area in zip(keep, keep_areas, strict=True):
            # Skip suppression when areas differ significantly.
            max_area = max(det_area, kept_area)
            if max_area > 0:
                area_ratio = min(det_area, kept_area) / max_area
                if area_ratio < _AREA_DIVERSITY_THRESH:
                    continue

            kcx = float(kept.corners[:, 0].mean())
            kcy = float(kept.corners[:, 1].mean())
            dist = float(np.sqrt((cx - kcx) ** 2 + (cy - kcy) ** 2))
            if dist < diag * min_dist_frac:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)
            keep_areas.append(det_area)

    return keep


def compute_overlap(
    corners1: NDArray[np.float32], corners2: NDArray[np.float32]
) -> float:
    """Compute IoU between two sets of corners using bounding rects."""
    rect1 = cv2.boundingRect(corners1.astype(np.int32))
    rect2 = cv2.boundingRect(corners2.astype(np.int32))

    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = rect1[2] * rect1[3]
    area2 = rect2[2] * rect2[3]
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union
