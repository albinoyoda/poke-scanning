from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from card_reco.models import DetectedCard

# Standard Pokemon card aspect ratio: 2.5" x 3.5" → 5:7
CARD_WIDTH = 734
CARD_HEIGHT = 1024
MIN_CARD_AREA_RATIO = 0.02  # Card must be at least 2% of image area


def detect_cards(image: NDArray[np.uint8]) -> list[DetectedCard]:
    """Detect and extract Pokemon cards from an image.

    Finds rectangular card-shaped regions using edge detection and contour
    analysis, then applies perspective transforms to produce normalized
    top-down views of each card.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * MIN_CARD_AREA_RATIO

    detected: list[DetectedCard] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        corners = _order_corners(approx.reshape(4, 2).astype(np.float32))
        warped = _four_point_transform(image, corners)

        confidence = min(1.0, area / (image_area * 0.1))
        detected.append(
            DetectedCard(image=warped, corners=corners, confidence=confidence)
        )

    # Sort by area (largest first)
    detected.sort(
        key=lambda d: cv2.contourArea(d.corners.reshape(4, 1, 2).astype(np.int32)),
        reverse=True,
    )

    return detected


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
