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


def _downscale_for_detection(
    image: NDArray[np.uint8], max_dim: int
) -> tuple[NDArray[np.uint8], float]:
    """Return *(detect_image, scale)* — downscale if *max_dim* > 0."""
    h, w = image.shape[:2]
    if 0 < max_dim < max(h, w):
        scale = max_dim / max(h, w)
        resized = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        return np.asarray(resized, dtype=np.uint8), scale
    return image, 1.0


def _validate_contour(
    contour: NDArray[np.uint8],
    min_area: float,
    primary_edges: NDArray[np.uint8],
    detect_area: int,
    scale: float,
    image: NDArray[np.uint8],
) -> DetectedCard | None:
    """Extract and validate a single contour, returning a DetectedCard or None."""
    corners = extract_corners(contour)
    if corners is None:
        return None
    if not has_card_aspect_ratio(corners):
        return None

    contour_pts = corners.reshape((4, 1, 2)).astype(np.int32)
    area = cv2.contourArea(contour_pts)
    if area < min_area:
        return None

    edge_frac = corner_edge_fraction(corners, primary_edges)
    if edge_frac < _MIN_CORNER_EDGE_FRAC:
        return None

    quality = contour_quality(contour, corners, detect_area)
    confidence = quality * edge_frac
    orig_corners = corners / scale if scale != 1.0 else corners
    warped = _four_point_transform(image, orig_corners)
    return DetectedCard(
        image=warped,
        corners=orig_corners,
        confidence=confidence,
        contour=contour,
    )


def detect_cards(
    image: NDArray[np.uint8],
    debug: DebugWriter | None = None,
    *,
    max_detect_dim: int = 0,
    fast: bool = False,
) -> list[DetectedCard]:
    """Detect and extract Pokemon cards from an image.

    Finds rectangular card-shaped regions using edge detection and contour
    analysis, then applies perspective transforms to produce normalized
    top-down views of each card.

    When *max_detect_dim* is set (> 0) the image is downscaled so its
    longest edge does not exceed *max_detect_dim* pixels.  Detection
    runs on the smaller image for speed; the perspective warp still
    uses the original full-resolution image.

    When *fast* is ``True``, only the two cheapest detection strategies
    (Canny + adaptive threshold) are used, skipping HSV colour
    segmentation, morphological closing, and Hough-line detection.

    When *debug* is provided, intermediate images are written to its
    output directory.
    """
    if debug:
        debug.save_input(image)

    detect_image, scale = _downscale_for_detection(image, max_detect_dim)

    gray = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if debug:
        debug.save_preprocessed(gray, blurred)

    detect_area = detect_image.shape[0] * detect_image.shape[1]
    min_area = detect_area * MIN_CARD_AREA_RATIO
    max_area = detect_area * MAX_CARD_AREA_RATIO

    detected: list[DetectedCard] = []

    # Build primary edge map for corner verification (step 3).
    clahe_verify = make_clahe()
    enhanced_verify = clahe_verify.apply(blurred)
    primary_edges = cv2.Canny(enhanced_verify, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    primary_edges = cv2.dilate(primary_edges, kernel, iterations=1)

    # Try multiple edge detection strategies and merge results
    candidates = find_card_contours(
        blurred,
        min_area,
        max_area,
        original_bgr=detect_image,
        debug=debug,
        fast=fast,
    )

    if debug:
        debug.save_candidates(detect_image, candidates, min_area)

    all_corners: list[NDArray[np.float32]] = []
    corner_labels: list[str] = []

    for contour in candidates:
        det = _validate_contour(
            contour, min_area, primary_edges, detect_area, scale, image
        )
        if det is None:
            continue

        detected.append(det)
        # Use detect-resolution corners for debug overlays.
        det_corners = det.corners * scale if scale != 1.0 else det.corners
        all_corners.append(det_corners)
        area = cv2.contourArea(det_corners.reshape((4, 1, 2)).astype(np.int32))
        pct = area / detect_area * 100
        corner_labels.append(f"{len(detected) - 1} {pct:.0f}% q={det.confidence:.2f}")

    if debug and all_corners:
        debug.save_corners(detect_image, all_corners, corner_labels)

    # Remove overlapping detections
    before_nms = len(detected)
    detected = non_max_suppression(detected, overlap_thresh=0.5)

    # Second pass: centroid-distance dedup.
    detected = centroid_dedup(detected)

    if debug:
        debug.save_nms_result(detect_image, before_nms, detected)
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
