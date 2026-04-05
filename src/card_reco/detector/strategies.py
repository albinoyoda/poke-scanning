"""Detection strategies: multiple edge-detection and segmentation approaches."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from card_reco.detector.constants import (
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    CANNY_PAIRS,
    HSV_COLOR_GROUPS,
    HSV_GROUP_NAMES,
    make_clahe,
)
from card_reco.detector.corners import order_corners

if TYPE_CHECKING:
    from card_reco.debug import DebugWriter


def find_card_contours(
    blurred: np.ndarray,
    min_area: float,
    max_area: float,
    original_bgr: NDArray[np.uint8] | None = None,
    debug: DebugWriter | None = None,
) -> list[NDArray[np.uint8]]:
    """Find contours that could be cards using multiple strategies."""
    candidates: list[NDArray[np.uint8]] = []
    seen: set[tuple[int, int, int]] = set()

    def _collect(contours: Sequence) -> None:  # type: ignore[type-arg]
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
    for low, high in CANNY_PAIRS:
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
    clahe = make_clahe()
    enhanced = clahe.apply(blurred)

    if debug:
        debug.save_edge_map("clahe", enhanced)

    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
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
    collect_fn: Callable,  # type: ignore[type-arg]
    debug: DebugWriter | None = None,
) -> None:
    """Find card contours using HSV color ranges for card borders."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for name, ranges in zip(HSV_GROUP_NAMES, HSV_COLOR_GROUPS, strict=True):
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

    Returns a 4-point contour array compatible with ``find_card_contours``,
    or *None* when no valid quad can be formed.
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
    segments: list[tuple[float, float, int, int, int, int]] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        segments.append((angle, float(length), int(x1), int(y1), int(x2), int(y2)))

    # Cluster into two perpendicular groups using a 90-degree split.
    best_split = 0.0
    best_score = -1.0
    for pivot in range(0, 180, 5):
        g1 = [s for s in segments if _angle_dist(s[0], pivot) < 30]
        g2 = [s for s in segments if _angle_dist(s[0], pivot + 90) < 30]
        score = min(len(g1), len(g2))
        if score > best_score:
            best_score = score
            best_split = float(pivot)

    group_a = [s for s in segments if _angle_dist(s[0], best_split) < 30]
    group_b = [s for s in segments if _angle_dist(s[0], best_split + 90) < 30]

    if len(group_a) < 2 or len(group_b) < 2:
        return None

    # Pick the two most separated lines from each group
    edges_a = _pick_two_lines(group_a, best_split)
    edges_b = _pick_two_lines(group_b, best_split + 90)
    if edges_a is None or edges_b is None:
        return None

    # Compute four corner intersections
    corners: list[tuple[float, float]] = []
    for la in edges_a:
        for lb in edges_b:
            pt = _line_intersection(la, lb)
            if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
                corners.append(pt)

    if len(corners) != 4:
        return None

    pts = np.array(corners, dtype=np.float32)
    ordered = order_corners(pts)
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
    rad = np.radians(ref_angle)
    perp = np.array([-np.sin(rad), np.cos(rad)])

    scored: list[tuple[float, tuple[float, float, int, int, int, int]]] = []
    for seg in group:
        mx = (seg[2] + seg[4]) / 2.0
        my = (seg[3] + seg[5]) / 2.0
        proj = mx * perp[0] + my * perp[1]
        scored.append((proj, seg))

    scored.sort(key=lambda x: x[0])

    if len(scored) < 2:
        return None

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
    return (float(ix), float(iy))
