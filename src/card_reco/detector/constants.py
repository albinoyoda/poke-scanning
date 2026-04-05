"""Shared constants for the card detection pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cv2

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

# Corner refinement: fit lines to straight edge segments, intersect to find
# the true corner (outside the rounded arc).
_CORNER_REFINE_TRIM = 0.15  # Fraction of edge length to skip at each end
_CORNER_REFINE_PERP_MIN = 3.0  # Min perpendicular distance threshold (pixels)
_CORNER_REFINE_PERP_SCALE = 0.02  # Perp threshold = max(min, scale * edge_len)
_CORNER_REFINE_MIN_PTS = 15  # Minimum contour points per edge for a valid fit
_CORNER_REFINE_MAX_SHIFT = 0.10  # Max displacement as fraction of min edge len
_CORNER_REFINE_MAX_RMS = 5.0  # Max RMS residual (px) for a valid line fit

# Detection strategy parameters
CANNY_PAIRS: list[tuple[int, int]] = [(50, 150), (80, 200)]
ADAPTIVE_BLOCK_SIZE = 15
ADAPTIVE_C = 3

# CLAHE parameters shared across the pipeline
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# HSV color ranges for card border detection
HSV_COLOR_GROUPS: list[list[tuple[list[int], list[int]]]] = [
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
    [([0, 0, 130], [180, 50, 230])],
]

HSV_GROUP_NAMES: list[str] = [
    "hsv_red_orange",
    "hsv_yellow_gold",
    "hsv_green",
    "hsv_blue_purple",
    "hsv_silver_gray",
]

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


def make_clahe() -> cv2.CLAHE:
    """Create a CLAHE instance with the standard pipeline parameters."""
    # Import cv2 here to keep constant-only imports lightweight.
    import cv2  # pylint: disable=import-outside-toplevel

    return cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
