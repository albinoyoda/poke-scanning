"""Card detection: detect_cards() and perspective transform.

This package splits the detection pipeline into focused submodules:

- ``constants`` — shared numeric thresholds and factory helpers
- ``corners`` — corner extraction and refinement
- ``quality`` — contour scoring and edge verification
- ``strategies`` — multiple edge-detection approaches
- ``nms`` — non-maximum suppression and deduplication
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from card_reco.detector.constants import (
    _MIN_CORNER_EDGE_FRAC,
    CARD_DST_PORTRAIT,
    CARD_HEIGHT,
    CARD_WIDTH,
    MAX_CARD_AREA_RATIO,
    MIN_CARD_AREA_RATIO,
    make_clahe,
)
from card_reco.detector.corners import (
    extract_corners,
    has_card_aspect_ratio,
    order_corners,
    refine_corners_edge_intersect,
    refine_corners_from_hull,
)
from card_reco.detector.nms import centroid_dedup, non_max_suppression
from card_reco.detector.quality import contour_quality, corner_edge_fraction
from card_reco.detector.strategies import find_card_contours
from card_reco.models import DetectedCard

if TYPE_CHECKING:
    from card_reco.debug import DebugWriter

# Backward-compatible aliases for code that imported private names.
_order_corners = order_corners
_refine_corners_from_hull = refine_corners_from_hull
_refine_corners_edge_intersect = refine_corners_edge_intersect

__all__ = [
    "CARD_DST_PORTRAIT",
    "CARD_HEIGHT",
    "CARD_WIDTH",
    "_order_corners",
    "_refine_corners_edge_intersect",
    "_refine_corners_from_hull",
    "detect_cards",
    "order_corners",
    "refine_corners_edge_intersect",
    "refine_corners_from_hull",
]


def detect_cards(
    image: NDArray[np.uint8],
    debug: DebugWriter | None = None,
) -> list[DetectedCard]:
    """Detect and extract Pokemon cards from an image.

    Finds rectangular card-shaped regions using edge detection and contour
    analysis, then applies perspective transforms to produce normalized
    top-down views of each card.

    When *debug* is provided, intermediate images are written to its
    output directory.
    """
    if debug:
        debug.save_input(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if debug:
        debug.save_preprocessed(gray, blurred)

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * MIN_CARD_AREA_RATIO
    max_area = image_area * MAX_CARD_AREA_RATIO

    detected: list[DetectedCard] = []

    # Build primary edge map for corner verification (step 3).
    clahe_verify = make_clahe()
    enhanced_verify = clahe_verify.apply(blurred)
    primary_edges = cv2.Canny(enhanced_verify, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    primary_edges = cv2.dilate(primary_edges, kernel, iterations=1)

    # Try multiple edge detection strategies and merge results
    candidates = find_card_contours(
        blurred, min_area, max_area, original_bgr=image, debug=debug
    )

    if debug:
        debug.save_candidates(image, candidates, min_area)

    all_corners: list[NDArray[np.float32]] = []
    corner_labels: list[str] = []

    for contour in candidates:
        corners = extract_corners(contour)
        if corners is None:
            continue

        if not has_card_aspect_ratio(corners):
            continue

        contour_pts = corners.reshape((4, 1, 2)).astype(np.int32)
        area = cv2.contourArea(contour_pts)
        if area < min_area:
            continue

        # Verify corners sit near Canny edges.
        edge_frac = corner_edge_fraction(corners, primary_edges)
        if edge_frac < _MIN_CORNER_EDGE_FRAC:
            continue

        # Quality-based confidence instead of raw area.
        quality = contour_quality(contour, corners, image_area)
        confidence = quality * edge_frac

        warped = _four_point_transform(image, corners)
        detected.append(
            DetectedCard(
                image=warped,
                corners=corners,
                confidence=confidence,
                contour=contour,
            )
        )
        all_corners.append(corners)
        pct = area / image_area * 100
        corner_labels.append(f"{len(detected) - 1} {pct:.0f}% q={confidence:.2f}")

    if debug and all_corners:
        debug.save_corners(image, all_corners, corner_labels)

    # Remove overlapping detections
    before_nms = len(detected)
    detected = non_max_suppression(detected, overlap_thresh=0.5)

    # Second pass: centroid-distance dedup.
    detected = centroid_dedup(detected)

    if debug:
        debug.save_nms_result(image, before_nms, detected)
        for i, det in enumerate(detected):
            debug.save_warped(i, det.image)

    # Sort by quality (highest confidence first)
    detected.sort(key=lambda d: d.confidence, reverse=True)

    return detected


def _four_point_transform(
    image: NDArray[np.uint8], corners: NDArray[np.float32]
) -> NDArray[np.uint8]:
    """Apply a perspective transform to extract a top-down view of a card."""
    dst = CARD_DST_PORTRAIT.copy()

    # Determine if the card is in landscape orientation
    tl, tr, br, bl = corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2

    # If wider than tall, the card is landscape — use rotated destination
    if avg_width > avg_height * 1.2:
        dst = np.array(
            [
                [0, 0],
                [CARD_HEIGHT - 1, 0],
                [CARD_HEIGHT - 1, CARD_WIDTH - 1],
                [0, CARD_WIDTH - 1],
            ],
            dtype=np.float32,
        )
        w, h = CARD_HEIGHT, CARD_WIDTH
    else:
        w, h = CARD_WIDTH, CARD_HEIGHT

    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = np.asarray(cv2.warpPerspective(image, matrix, (w, h)), dtype=np.uint8)

    # If we warped to landscape, rotate to portrait
    if avg_width > avg_height * 1.2:
        warped = np.asarray(
            cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE), dtype=np.uint8
        )

    return warped
