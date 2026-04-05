"""Contour quality scoring and edge verification."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from card_reco.detector.constants import (
    _EDGE_VERIFY_RADIUS,
    ASPECT_RATIO_TOLERANCE,
    CARD_ASPECT_RATIO,
)
from card_reco.detector.corners import corner_geometry


def contour_quality(
    contour: NDArray[np.uint8],
    corners: NDArray[np.float32],
    image_area: float,
) -> float:
    """Score a candidate contour by shape quality (0-1).

    Combines three metrics:
    - **Compactness**: contour area / convex-hull area.  A clean card
      contour should be >= 0.85; merged card + ground blobs score lower.
    - **Rectangularity**: contour area / minAreaRect area.  Perfect
      rectangles score 1.0; irregular shapes score worse.
    - **Aspect ratio closeness**: 1 - normalised distance from the ideal
      5 : 7 card ratio.

    All three are averaged to a 0-1 score.
    """
    contour_area = cv2.contourArea(contour)
    if contour_area <= 0:
        return 0.0

    # Compactness
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    compactness = contour_area / hull_area if hull_area > 0 else 0.0

    # Rectangularity
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    rect_area = cv2.contourArea(box.reshape(4, 1, 2))
    rectangularity = contour_area / rect_area if rect_area > 0 else 0.0

    # Aspect ratio closeness (from the extracted corners, not the contour)
    _, _, ratio = corner_geometry(corners)
    if ratio == 0.0:
        return 0.0
    ar_delta = abs(ratio - CARD_ASPECT_RATIO)
    aspect_score = max(0.0, 1.0 - ar_delta / ASPECT_RATIO_TOLERANCE)

    # Size factor — prefer contours that are a reasonable fraction of the
    # image (not tiny noise, not the entire frame).
    area_frac = contour_area / image_area
    size_score = min(1.0, area_frac / 0.05)  # saturates at 5 % of image

    return (compactness + rectangularity + aspect_score + size_score) / 4.0


def corner_edge_fraction(
    corners: NDArray[np.float32],
    edge_map: np.ndarray,
) -> float:
    """Return the fraction of corners that sit near a strong Canny edge.

    For each of the four corners, sample a square patch of radius
    *_EDGE_VERIFY_RADIUS* in *edge_map*.  A corner is "edge-backed" if
    any pixel in that patch is non-zero.

    Detections whose corners land on featureless areas (ground, sky) will
    return a low fraction, signalling a bad crop.
    """
    h, w = edge_map.shape[:2]
    backed = 0
    for pt in corners:
        cx, cy = round(pt[0]), round(pt[1])
        x0 = max(0, cx - _EDGE_VERIFY_RADIUS)
        y0 = max(0, cy - _EDGE_VERIFY_RADIUS)
        x1 = min(w, cx + _EDGE_VERIFY_RADIUS + 1)
        y1 = min(h, cy + _EDGE_VERIFY_RADIUS + 1)
        patch = edge_map[y0:y1, x0:x1]
        if patch.size > 0 and np.any(patch > 0):
            backed += 1
    return backed / 4.0
