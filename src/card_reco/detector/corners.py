"""Corner extraction and refinement for detected card contours."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from card_reco.detector.constants import (
    _CORNER_REFINE_MAX_RMS,
    _CORNER_REFINE_MAX_SHIFT,
    _CORNER_REFINE_MIN_PTS,
    _CORNER_REFINE_PERP_MIN,
    _CORNER_REFINE_PERP_SCALE,
    _CORNER_REFINE_TRIM,
    ASPECT_RATIO_TOLERANCE,
    CARD_ASPECT_RATIO,
    MIN_RECT_COMPACTNESS,
)


def corner_geometry(
    corners: NDArray[np.float32],
) -> tuple[float, float, float]:
    """Compute average width, height, and normalised aspect ratio.

    Returns ``(avg_width, avg_height, ratio)`` where *ratio* is
    ``short / long`` (always <= 1.0).
    """
    tl, tr, br, bl = corners
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))

    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2

    if avg_height == 0 or avg_width == 0:
        return avg_width, avg_height, 0.0

    short = min(avg_width, avg_height)
    long = max(avg_width, avg_height)
    return avg_width, avg_height, short / long


def order_corners(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """Order four corner points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]  # bottom-left has largest difference

    return rect


def has_card_aspect_ratio(corners: NDArray[np.float32]) -> bool:
    """Check if four corners form a rectangle with card-like aspect ratio."""
    _, _, ratio = corner_geometry(corners)
    if ratio == 0.0:
        return False
    return abs(ratio - CARD_ASPECT_RATIO) < ASPECT_RATIO_TOLERANCE


def extract_corners(
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
            return order_corners(approx.reshape(4, 2).astype(np.float32))

    # Fallback: minAreaRect, but only for compact contours (avoids noise)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    rect_area = cv2.contourArea(box.reshape(4, 1, 2))
    if rect_area > 0:
        contour_area = cv2.contourArea(contour)
        compactness = contour_area / rect_area
        if compactness >= MIN_RECT_COMPACTNESS:
            corners = refine_corners_from_hull(contour, box.astype(np.float32))
            return order_corners(corners)

    return None


def refine_corners_from_hull(
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


def refine_corners_edge_intersect(
    contour: NDArray[np.uint8],
    corners: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Refine corners by fitting lines to straight edge segments.

    ``approxPolyDP`` places corners where the rounded corner arc meets a
    straight edge.  This function fits lines to the middle portion of
    each edge (skipping the rounded ends) and intersects adjacent lines
    to find the true geometric corner — typically a few pixels outside
    the card, giving a cleaner perspective warp.

    Falls back to *corners* unchanged when the contour has too few
    points for a reliable fit or the result looks wrong.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)

    fitted_lines: list[tuple[float, float, float, float]] = []
    min_edge_len = float("inf")

    for i in range(4):
        c1 = corners[i]
        c2 = corners[(i + 1) % 4]
        edge_vec = c2 - c1
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 10:
            return corners
        min_edge_len = min(min_edge_len, edge_len)

        edge_dir = edge_vec / edge_len
        edge_normal = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float32)

        # Project every contour point onto this edge's coordinate frame.
        rel = pts - c1
        along = rel @ edge_dir  # distance along the edge
        perp = rel @ edge_normal  # perpendicular distance from edge

        # Keep points close to the line and away from the rounded corners.
        perp_thresh = max(_CORNER_REFINE_PERP_MIN, edge_len * _CORNER_REFINE_PERP_SCALE)
        trim = edge_len * _CORNER_REFINE_TRIM
        mask = (np.abs(perp) < perp_thresh) & (along > trim) & (along < edge_len - trim)

        edge_pts = pts[mask]
        if len(edge_pts) < _CORNER_REFINE_MIN_PTS:
            return corners

        line = cv2.fitLine(edge_pts.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()

        # Verify fit quality: compute RMS perpendicular residual.
        line_norm = np.array([-float(vy), float(vx)], dtype=np.float32)
        residuals = (edge_pts - np.array([float(x0), float(y0)])) @ line_norm
        rms = float(np.sqrt(np.mean(residuals**2)))
        if rms > _CORNER_REFINE_MAX_RMS:
            return corners  # Noisy edge — line fit is unreliable.

        fitted_lines.append((float(vx), float(vy), float(x0), float(y0)))

    # Intersect adjacent fitted lines to produce refined corners.
    max_shift = min_edge_len * _CORNER_REFINE_MAX_SHIFT
    refined = np.zeros((4, 2), dtype=np.float32)

    for i in range(4):
        # Corner[i] sits at the junction of edge (i-1) and edge i.
        line_a = fitted_lines[(i - 1) % 4]
        line_b = fitted_lines[i]

        pt = _intersect_param_lines(line_a, line_b)
        if pt is None:
            return corners  # Nearly parallel — shouldn't happen for a card.

        shift = np.array(pt) - corners[i]
        if float(np.linalg.norm(shift)) > max_shift:
            return corners  # Displacement too large; bail out.

        refined[i] = pt

    return refined


def _intersect_param_lines(
    line_a: tuple[float, float, float, float],
    line_b: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Intersect two lines in parametric form ``(vx, vy, x0, y0)``."""
    vx1, vy1, x01, y01 = line_a
    vx2, vy2, x02, y02 = line_b

    denom = vx1 * vy2 - vy1 * vx2
    if abs(denom) < 1e-8:
        return None

    dx = x02 - x01
    dy = y02 - y01
    t = (dx * vy2 - dy * vx2) / denom
    return (x01 + t * vx1, y01 + t * vy1)
