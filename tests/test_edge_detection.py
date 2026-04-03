"""Edge detection accuracy tests using annotated ground-truth images.

Each ``*_edges.png`` image in the test data contains a pure-green rectangle
(~BGR 0,255,0) drawn around the card border.  These tests extract the green
rectangle corners as ground truth, run ``detect_cards()`` on the original
image, and verify that the detected corners are within a pixel tolerance of
the expected ones.

No hash database or card matching is required.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from card_reco.detector import detect_cards

TEST_DATA_DIR = Path("data") / "tests" / "single_cards" / "axis_aligned"

# Maximum average corner distance (pixels) to consider a detection correct.
CORNER_TOLERANCE_PX = 30


def _extract_green_corners(edges_image: np.ndarray) -> np.ndarray:
    """Extract the four corners of a green rectangle drawn on *edges_image*.

    Masks for bright-green pixels (high G, low R and B), finds the largest
    contour, and approximates it to four corners.

    Returns an ordered (4, 2) float32 array: TL, TR, BR, BL.
    """
    hsv = cv2.cvtColor(edges_image, cv2.COLOR_BGR2HSV)
    # Pure green: H ~60° (in OpenCV 0-180: ~40-80), high S, high V
    lower = np.array([35, 100, 100])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert contours, "No green contour found in edges image"

    # Take the largest contour
    largest = max(contours, key=lambda c: cv2.contourArea(c))
    peri = cv2.arcLength(largest, True)

    # Approximate to a polygon — should yield 4 corners for a rectangle
    for eps_factor in (0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(largest, eps_factor * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return _order_corners(pts)

    # Fallback: use minAreaRect
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    return _order_corners(box)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _avg_corner_distance(detected: np.ndarray, expected: np.ndarray) -> float:
    """Average Euclidean distance between corresponding ordered corners."""
    return float(np.mean(np.linalg.norm(detected - expected, axis=1)))


def _find_edge_test_cases() -> list[tuple[str, Path, Path]]:
    """Discover all original/edges image pairs in the test directory."""
    if not TEST_DATA_DIR.exists():
        return []
    pairs: list[tuple[str, Path, Path]] = []
    for edges_path in sorted(TEST_DATA_DIR.glob("*_edges.png")):
        stem = edges_path.stem.replace("_edges", "")
        original_path = edges_path.parent / f"{stem}.png"
        if original_path.exists():
            pairs.append((stem, original_path, edges_path))
    return pairs


_test_cases = _find_edge_test_cases()
has_test_data = len(_test_cases) > 0
skip_no_data = pytest.mark.skipif(
    not has_test_data, reason="Edge detection test data not available"
)

# Images where detection is known to be inaccurate (corner distance > tolerance).
# These are marked xfail so the suite stays green while we iterate on the algorithm.
_KNOWN_INACCURATE: set[str] = set()


@skip_no_data
@pytest.mark.parametrize(
    "name,original_path,edges_path",
    _test_cases,
    ids=[c[0] for c in _test_cases],
)
def test_detected_corners_match_ground_truth(
    name: str,
    original_path: Path,
    edges_path: Path,
) -> None:
    """Detected card corners should be close to the annotated green box."""
    if name in _KNOWN_INACCURATE:
        pytest.xfail(f"{name} is a known inaccurate detection")

    original = cv2.imread(str(original_path))
    assert original is not None, f"Failed to load {original_path}"

    edges_img = cv2.imread(str(edges_path))
    assert edges_img is not None, f"Failed to load {edges_path}"

    expected_corners = _extract_green_corners(edges_img)
    detections = detect_cards(np.asarray(original, dtype=np.uint8))

    assert len(detections) >= 1, f"No cards detected in {name}"

    # Find the detection whose corners are closest to the expected ones
    best_dist = float("inf")
    best_corners = None
    for det in detections:
        dist = _avg_corner_distance(det.corners, expected_corners)
        if dist < best_dist:
            best_dist = dist
            best_corners = det.corners

    assert best_corners is not None
    assert best_dist < CORNER_TOLERANCE_PX, (
        f"{name}: best average corner distance {best_dist:.1f}px "
        f"exceeds tolerance {CORNER_TOLERANCE_PX}px\n"
        f"  Expected: {expected_corners.tolist()}\n"
        f"  Detected: {best_corners.tolist()}"
    )


@skip_no_data
def test_all_axis_aligned_cards_detected() -> None:
    """Every axis-aligned test image should produce at least one detection."""
    failures: list[str] = []
    for name, original_path, _ in _test_cases:
        original = cv2.imread(str(original_path))
        if original is None:
            failures.append(f"{name}: failed to load image")
            continue
        detections = detect_cards(np.asarray(original, dtype=np.uint8))
        if len(detections) == 0:
            failures.append(f"{name}: no cards detected")
    assert not failures, "Detection failures:\n" + "\n".join(failures)
