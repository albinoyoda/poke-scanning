from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from card_reco.models import DetectedCard

# Standard Pokemon card aspect ratio: 2.5" x 3.5" → 5:7
CARD_WIDTH = 734
CARD_HEIGHT = 1024
MIN_CARD_AREA_RATIO = 0.015  # Card must be at least 1.5% of image area
MAX_CARD_AREA_RATIO = 0.95  # Skip contours that are basically the whole image
CARD_ASPECT_RATIO = 5.0 / 7.0  # width / height ≈ 0.714
ASPECT_RATIO_TOLERANCE = 0.25  # Allow 0.46 – 0.96
MIN_RECT_COMPACTNESS = 0.65  # Contour must fill at least 65% of its minAreaRect
_AREA_DIVERSITY_THRESH = 0.85  # NMS keeps both detections when area ratio < this


def detect_cards(image: NDArray[np.uint8]) -> list[DetectedCard]:
    """Detect and extract Pokemon cards from an image.

    Finds rectangular card-shaped regions using edge detection and contour
    analysis, then applies perspective transforms to produce normalized
    top-down views of each card.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * MIN_CARD_AREA_RATIO
    max_area = image_area * MAX_CARD_AREA_RATIO

    detected: list[DetectedCard] = []

    # Try multiple edge detection strategies and merge results
    candidates = _find_card_contours(blurred, min_area, max_area, original_bgr=image)

    for contour in candidates:
        corners = _extract_corners(contour)
        if corners is None:
            continue

        if not _has_card_aspect_ratio(corners):
            continue

        contour_pts = corners.reshape((4, 1, 2)).astype(np.int32)
        area = cv2.contourArea(contour_pts)
        if area < min_area:
            continue

        warped = _four_point_transform(image, corners)
        confidence = min(1.0, area / (image_area * 0.1))
        detected.append(
            DetectedCard(image=warped, corners=corners, confidence=confidence)
        )

    # Remove overlapping detections
    detected = _non_max_suppression(detected, overlap_thresh=0.5)

    # Sort by area (largest first)
    detected.sort(
        key=lambda d: cv2.contourArea(d.corners.reshape(4, 1, 2).astype(np.int32)),
        reverse=True,
    )

    return detected


def _find_card_contours(
    blurred: np.ndarray,
    min_area: float,
    max_area: float,
    original_bgr: NDArray[np.uint8] | None = None,
) -> list[NDArray[np.uint8]]:
    """Find contours that could be cards using multiple strategies."""
    candidates: list[NDArray[np.uint8]] = []
    seen: set[tuple[int, int, int]] = set()

    def _collect(contours: Sequence) -> None:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            key = (round(cx / 50), round(cy / 50), round(area / 1000))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(contour)

    # Strategy 1: Canny edge detection with multiple thresholds and modes
    for low, high in [(50, 150), (30, 100)]:
        edged = cv2.Canny(blurred, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=1)

        for retr_mode in (cv2.RETR_EXTERNAL, cv2.RETR_TREE):
            contours, _ = cv2.findContours(edged, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
            _collect(contours)

    # Strategy 2: Adaptive thresholding (better for low-contrast card borders)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    for retr_mode in (cv2.RETR_EXTERNAL, cv2.RETR_TREE):
        contours, _ = cv2.findContours(thresh, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
        _collect(contours)

    # Strategy 3: HSV color segmentation for colored card borders
    if original_bgr is not None:
        _collect_hsv_contours(original_bgr, _collect)

    return candidates


def _collect_hsv_contours(
    image: NDArray[np.uint8],
    collect_fn: Callable,
) -> None:
    """Find card contours using HSV color ranges for card borders."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # Color ranges covering common Pokemon card border colors
    color_groups = [
        # Red/orange borders (fire cards, Charizard, etc.)
        [
            ([0, 50, 100], [10, 255, 255]),
            ([170, 50, 100], [180, 255, 255]),
            ([5, 40, 100], [20, 255, 255]),
        ],
        # Yellow/gold borders
        [([15, 40, 100], [35, 255, 255]), ([10, 30, 120], [25, 255, 255])],
    ]

    for ranges in color_groups:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        collect_fn(contours)


def _extract_corners(
    contour: NDArray[np.uint8],
) -> NDArray[np.float32] | None:
    """Extract four corners from a contour.

    Tries polygon approximation first. Falls back to minAreaRect for
    compact contours (e.g. cards with rounded corners).
    """
    peri = cv2.arcLength(contour, True)

    # Try polygon approximation with varying epsilon
    for eps_factor in (0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2).astype(np.float32))

    # Fallback: minAreaRect, but only for compact contours (avoids noise)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    rect_area = cv2.contourArea(box.reshape(4, 1, 2))
    if rect_area > 0:
        contour_area = cv2.contourArea(contour)
        compactness = contour_area / rect_area
        if compactness >= MIN_RECT_COMPACTNESS:
            corners = _refine_corners_from_hull(contour, box.astype(np.float32))
            return _order_corners(corners)

    return None


def _refine_corners_from_hull(
    contour: NDArray[np.uint8],
    box: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Find actual card corners from convex hull nearest to minAreaRect corners.

    minAreaRect returns a perfect rotated rectangle, discarding perspective
    distortion.  This function finds the convex-hull points closest to each
    box corner, preserving the real trapezoidal shape of tilted cards.

    Falls back to the original box corners when the hull-based result would
    be degenerate (e.g. two hull points collapsing to the same location).
    """
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    if len(hull) < 4:
        return box.astype(np.float32)

    corners = np.zeros((4, 2), dtype=np.float32)
    used_indices: set[int] = set()
    for i, box_corner in enumerate(box):
        distances = np.linalg.norm(hull - box_corner, axis=1)
        order = np.argsort(distances)
        for idx in order:
            if int(idx) not in used_indices:
                corners[i] = hull[idx]
                used_indices.add(int(idx))
                break

    # Validate: all 4 corners should be sufficiently separated.
    # If any pair is too close, the hull didn't capture all four card
    # corners properly — fall back to the original box.
    min_edge = float("inf")
    for i in range(4):
        edge_len = float(np.linalg.norm(corners[i] - corners[(i + 1) % 4]))
        min_edge = min(min_edge, edge_len)
    if min_edge < 10.0:
        return box.astype(np.float32)

    return corners


def _has_card_aspect_ratio(corners: NDArray[np.float32]) -> bool:
    """Check if four corners form a rectangle with card-like aspect ratio."""
    tl, tr, br, bl = corners
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))

    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2

    if avg_height == 0 or avg_width == 0:
        return False

    # Normalize so shorter side is numerator
    short = min(avg_width, avg_height)
    long = max(avg_width, avg_height)
    ratio = short / long

    return abs(ratio - CARD_ASPECT_RATIO) < ASPECT_RATIO_TOLERANCE


def _non_max_suppression(
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
    keep: list[DetectedCard] = []

    for det in detections:
        suppressed = False
        det_area = float(cv2.contourArea(det.corners.reshape(4, 1, 2).astype(np.int32)))
        for kept in keep:
            if _compute_overlap(det.corners, kept.corners) > overlap_thresh:
                # Allow both when areas differ enough — smaller crops
                # may be tighter and produce better hash matches.
                kept_area = float(
                    cv2.contourArea(kept.corners.reshape(4, 1, 2).astype(np.int32))
                )
                max_area = max(det_area, kept_area)
                if max_area > 0:
                    area_ratio = min(det_area, kept_area) / max_area
                    if area_ratio < _AREA_DIVERSITY_THRESH:
                        continue  # Don't suppress — different-sized crop
                suppressed = True
                break
        if not suppressed:
            keep.append(det)

    return keep


def _compute_overlap(
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


def _order_corners(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """Order four corner points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]  # bottom-left has largest difference

    return rect


def _four_point_transform(
    image: NDArray[np.uint8], corners: NDArray[np.float32]
) -> NDArray[np.uint8]:
    """Apply a perspective transform to extract a top-down view of a card."""
    dst = np.array(
        [
            [0, 0],
            [CARD_WIDTH - 1, 0],
            [CARD_WIDTH - 1, CARD_HEIGHT - 1],
            [0, CARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

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
