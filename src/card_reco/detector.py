from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from card_reco.models import DetectedCard

if TYPE_CHECKING:
    from card_reco.debug import DebugWriter

# Standard Pokemon card aspect ratio: 2.5" x 3.5" → 5:7
CARD_WIDTH = 734
CARD_HEIGHT = 1024
MIN_CARD_AREA_RATIO = 0.015  # Card must be at least 1.5% of image area
MAX_CARD_AREA_RATIO = 0.95  # Skip contours that are basically the whole image
CARD_ASPECT_RATIO = 5.0 / 7.0  # width / height ≈ 0.714
ASPECT_RATIO_TOLERANCE = 0.25  # Allow 0.46 – 0.96
MIN_RECT_COMPACTNESS = 0.65  # Contour must fill at least 65% of its minAreaRect
_AREA_DIVERSITY_THRESH = 0.85  # NMS keeps both detections when area ratio < this
_EDGE_VERIFY_RADIUS = 15  # Pixel radius for corner edge verification
_MIN_CORNER_EDGE_FRAC = 0.5  # At least half the corners must sit near an edge

# Destination corners for the standard portrait perspective warp.
CARD_DST_PORTRAIT = np.array(
    [
        [0, 0],
        [CARD_WIDTH - 1, 0],
        [CARD_WIDTH - 1, CARD_HEIGHT - 1],
        [0, CARD_HEIGHT - 1],
    ],
    dtype=np.float32,
)


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
    # CLAHE enhancement ensures weak card-border edges on bright or
    # low-contrast backgrounds are preserved for corner validation.
    clahe_verify = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_verify = clahe_verify.apply(blurred)
    primary_edges = cv2.Canny(enhanced_verify, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    primary_edges = cv2.dilate(primary_edges, kernel, iterations=1)

    # Try multiple edge detection strategies and merge results
    candidates = _find_card_contours(
        blurred, min_area, max_area, original_bgr=image, debug=debug
    )

    if debug:
        debug.save_candidates(image, candidates, min_area)

    all_corners: list[NDArray[np.float32]] = []
    corner_labels: list[str] = []

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

        # Step 3: Verify corners sit near Canny edges.
        # Detections whose corners land on blank areas (e.g. featureless
        # ground) are penalised so they lose to tighter, edge-backed quads.
        edge_frac = _corner_edge_fraction(corners, primary_edges)
        if edge_frac < _MIN_CORNER_EDGE_FRAC:
            continue

        # Step 2: Quality-based confidence instead of raw area.
        quality = _contour_quality(contour, corners, image_area)
        confidence = quality * edge_frac

        warped = _four_point_transform(image, corners)
        detected.append(
            DetectedCard(image=warped, corners=corners, confidence=confidence)
        )
        all_corners.append(corners)
        pct = area / image_area * 100
        corner_labels.append(f"{len(detected) - 1} {pct:.0f}% q={confidence:.2f}")

    if debug and all_corners:
        debug.save_corners(image, all_corners, corner_labels)

    # Remove overlapping detections
    before_nms = len(detected)
    detected = _non_max_suppression(detected, overlap_thresh=0.5)

    # Second pass: centroid-distance dedup.  Multiple edge strategies
    # often find slightly different contours for the same card, leading
    # to near-duplicate detections that the IoU-based NMS misses.
    detected = _centroid_dedup(detected)

    if debug:
        debug.save_nms_result(image, before_nms, detected)
        for i, det in enumerate(detected):
            debug.save_warped(i, det.image)

    # Sort by quality (highest confidence first)
    detected.sort(key=lambda d: d.confidence, reverse=True)

    return detected


def _find_card_contours(
    blurred: np.ndarray,
    min_area: float,
    max_area: float,
    original_bgr: NDArray[np.uint8] | None = None,
    debug: DebugWriter | None = None,
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
    for low, high in [(50, 150), (80, 200)]:
        edged = cv2.Canny(blurred, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=1)

        if debug:
            debug.save_edge_map(f"canny_{low}_{high}", edged)

        for retr_mode in (cv2.RETR_EXTERNAL, cv2.RETR_TREE):
            contours, _ = cv2.findContours(edged, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
            _collect(contours)

    # Strategy 2: Adaptive thresholding (better for low-contrast card borders)
    # Apply CLAHE to enhance local contrast before thresholding, so that
    # card borders on similar-brightness surfaces stay distinct.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    if debug:
        debug.save_edge_map("clahe", enhanced)

    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    if debug:
        debug.save_edge_map("adaptive_thresh", thresh)

    for retr_mode in (cv2.RETR_EXTERNAL, cv2.RETR_TREE):
        contours, _ = cv2.findContours(thresh, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
        _collect(contours)

    # Strategy 3: HSV color segmentation for colored card borders
    if original_bgr is not None:
        _collect_hsv_contours(original_bgr, _collect, debug=debug)

    # Strategy 4: Morphological closing on Canny to bridge edge gaps
    edged_base = cv2.Canny(blurred, 50, 150)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged_base, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    if debug:
        debug.save_edge_map("morph_close", closed)

    for retr_mode in (cv2.RETR_EXTERNAL, cv2.RETR_TREE):
        contours, _ = cv2.findContours(closed, retr_mode, cv2.CHAIN_APPROX_SIMPLE)
        _collect(contours)

    # Strategy 5: Hough-line quad detection
    hough_contour = _hough_quad(blurred, min_area, debug=debug)
    if hough_contour is not None:
        _collect([hough_contour])

    return candidates


def _collect_hsv_contours(
    image: NDArray[np.uint8],
    collect_fn: Callable,
    debug: DebugWriter | None = None,
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
        # Green borders (grass cards)
        [([35, 40, 80], [85, 255, 255])],
        # Blue/purple borders (water, psychic, dragon cards)
        [([85, 40, 80], [135, 255, 255])],
        # Silver/gray borders (steel, colorless, modern holos)
        # Low saturation, moderate-to-high value — separates from dark bg
        [([0, 0, 130], [180, 50, 230])],
    ]

    group_names = [
        "hsv_red_orange",
        "hsv_yellow_gold",
        "hsv_green",
        "hsv_blue_purple",
        "hsv_silver_gray",
    ]
    for name, ranges in zip(group_names, color_groups, strict=True):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if debug:
            debug.save_edge_map(name, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        collect_fn(contours)


def _hough_quad(
    blurred: np.ndarray,
    min_area: float,
    debug: DebugWriter | None = None,
) -> NDArray[np.uint8] | None:
    """Detect a card quadrilateral from Hough lines in the Canny edge map.

    Finds line segments, clusters them into two roughly perpendicular
    groups via their angles, selects the strongest pair from each group,
    and computes the four intersection points to form a quadrilateral.

    Returns a 4-point contour array compatible with *_collect*, or *None*
    when no valid quad can be formed.
    """
    edges = cv2.Canny(blurred, 50, 150)
    h, w = edges.shape[:2]
    min_line_len = int(min(h, w) * 0.15)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=min_line_len,
        maxLineGap=20,
    )
    if lines is None or len(lines) < 4:
        return None

    # Compute angle (0-180) for each segment
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        segments.append((angle, length, x1, y1, x2, y2))

    # Cluster into two perpendicular groups using a 90-degree split.
    # Find the angle that best separates lines into two balanced groups.
    best_split = 0.0
    best_score = -1.0
    for pivot in range(0, 180, 5):
        g1 = [s for s in segments if _angle_dist(s[0], pivot) < 30]
        g2 = [s for s in segments if _angle_dist(s[0], pivot + 90) < 30]
        score = min(len(g1), len(g2))
        if score > best_score:
            best_score = score
            best_split = pivot

    group_a = [s for s in segments if _angle_dist(s[0], best_split) < 30]
    group_b = [s for s in segments if _angle_dist(s[0], best_split + 90) < 30]

    if len(group_a) < 2 or len(group_b) < 2:
        return None

    # Pick the two most separated lines from each group (by perpendicular
    # offset from origin) to get the rectangle's opposing edges.
    edges_a = _pick_two_lines(group_a, best_split)
    edges_b = _pick_two_lines(group_b, best_split + 90)
    if edges_a is None or edges_b is None:
        return None

    # Compute four corner intersections
    corners = []
    for la in edges_a:
        for lb in edges_b:
            pt = _line_intersection(la, lb)
            if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
                corners.append(pt)

    if len(corners) != 4:
        return None

    pts = np.array(corners, dtype=np.float32)
    ordered = _order_corners(pts)
    contour = np.reshape(ordered, (4, 1, 2)).astype(np.int32)

    # Compute quad area via the Shoelace formula (avoids a pylint
    # false-positive on cv2.contourArea with C-binding stubs).
    xs = ordered[:, 0]
    ys = ordered[:, 1]
    area = 0.5 * abs(float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))
    if area < min_area:
        return None

    if debug:
        _save_hough_debug(debug, edges, group_a, group_b, edges_a, edges_b, contour)

    return contour


def _save_hough_debug(
    debug: DebugWriter,
    edges: np.ndarray,
    group_a: list[tuple[float, float, int, int, int, int]],
    group_b: list[tuple[float, float, int, int, int, int]],
    edges_a: list[tuple[float, float, int, int, int, int]],
    edges_b: list[tuple[float, float, int, int, int, int]],
    contour: NDArray[np.int32],
) -> None:
    """Write Hough line debug visualisation."""
    vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for seg in group_a:
        cv2.line(vis, (seg[2], seg[3]), (seg[4], seg[5]), (0, 0, 255), 1)
    for seg in group_b:
        cv2.line(vis, (seg[2], seg[3]), (seg[4], seg[5]), (255, 0, 0), 1)
    for la in edges_a:
        cv2.line(vis, (la[2], la[3]), (la[4], la[5]), (0, 255, 255), 2)
    for lb in edges_b:
        cv2.line(vis, (lb[2], lb[3]), (lb[4], lb[5]), (255, 255, 0), 2)
    # pylint: disable-next=too-many-function-args
    cv2.polylines(vis, [contour], True, (0, 255, 0), 2)
    debug.save_edge_map("hough_quad", vis)


def _angle_dist(a: float, b: float) -> float:
    """Shortest angular distance between two angles in [0, 180) space."""
    d = abs(a - (b % 180))
    return min(d, 180 - d)


def _pick_two_lines(
    group: list[tuple[float, float, int, int, int, int]],
    ref_angle: float,
) -> list[tuple[float, float, int, int, int, int]] | None:
    """Select the two most-separated parallel lines from a group.

    Projects each segment's midpoint onto the axis perpendicular to
    *ref_angle* and picks the pair with the largest offset gap.
    """
    # Perpendicular direction for projection
    rad = np.radians(ref_angle)
    perp = np.array([-np.sin(rad), np.cos(rad)])

    # Project midpoints
    scored = []
    for seg in group:
        mx = (seg[2] + seg[4]) / 2.0
        my = (seg[3] + seg[5]) / 2.0
        proj = mx * perp[0] + my * perp[1]
        scored.append((proj, seg))

    scored.sort(key=lambda x: x[0])

    # The two most separated are the first and last
    if len(scored) < 2:
        return None

    # Pick the longest line near the min-projection end and the
    # longest line near the max-projection end.
    gap = scored[-1][0] - scored[0][0]
    if gap < 20:
        return None

    mid = (scored[0][0] + scored[-1][0]) / 2.0
    lo_group = [s for s in scored if s[0] < mid]
    hi_group = [s for s in scored if s[0] >= mid]
    if not lo_group or not hi_group:
        return None

    best_lo = max(lo_group, key=lambda s: s[1][1])[1]
    best_hi = max(hi_group, key=lambda s: s[1][1])[1]
    return [best_lo, best_hi]


def _line_intersection(
    seg1: tuple[float, float, int, int, int, int],
    seg2: tuple[float, float, int, int, int, int],
) -> tuple[float, float] | None:
    """Compute the intersection of two infinite lines defined by segments."""
    x1, y1, x2, y2 = seg1[2], seg1[3], seg1[4], seg1[5]
    x3, y3, x4, y4 = seg2[2], seg2[3], seg2[4], seg2[5]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


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


def _contour_quality(
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
    tl, tr, br, bl = corners
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2
    if avg_height == 0 or avg_width == 0:
        return 0.0
    short = min(avg_width, avg_height)
    long = max(avg_width, avg_height)
    ratio = short / long
    ar_delta = abs(ratio - CARD_ASPECT_RATIO)
    aspect_score = max(0.0, 1.0 - ar_delta / ASPECT_RATIO_TOLERANCE)

    # Size factor — prefer contours that are a reasonable fraction of the
    # image (not tiny noise, not the entire frame).
    area_frac = contour_area / image_area
    size_score = min(1.0, area_frac / 0.05)  # saturates at 5 % of image

    return (compactness + rectangularity + aspect_score + size_score) / 4.0


def _corner_edge_fraction(
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


def _centroid_dedup(
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
    keep: list[DetectedCard] = []

    for det in detections:
        cx = float(det.corners[:, 0].mean())
        cy = float(det.corners[:, 1].mean())
        det_area = float(cv2.contourArea(det.corners.reshape(4, 1, 2).astype(np.int32)))
        rect = cv2.boundingRect(det.corners.astype(np.int32))
        diag = float(np.sqrt(rect[2] ** 2 + rect[3] ** 2))

        suppressed = False
        for kept in keep:
            # Skip suppression when areas differ significantly.
            kept_area = float(
                cv2.contourArea(kept.corners.reshape(4, 1, 2).astype(np.int32))
            )
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
