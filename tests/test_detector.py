from __future__ import annotations

import cv2
import numpy as np

from card_reco.detector import _order_corners, detect_cards


def _make_card_image(
    bg_width: int = 1200,
    bg_height: int = 900,
    card_rect: tuple[int, int, int, int] = (200, 100, 500, 660),
    card_color: tuple[int, int, int] = (0, 120, 255),
) -> np.ndarray:
    """Create a synthetic image with a colored rectangle representing a card."""
    image = np.full((bg_height, bg_width, 3), 40, dtype=np.uint8)
    x, y, w, h = card_rect
    cv2.rectangle(image, (x, y), (x + w, y + h), card_color, -1)
    return image


class TestDetectCards:
    def test_detects_single_card(self):
        image = _make_card_image()
        cards = detect_cards(image)
        assert len(cards) >= 1
        assert cards[0].image.shape[0] > 0
        assert cards[0].image.shape[1] > 0

    def test_detects_multiple_cards(self):
        image = np.full((1000, 1600, 3), 30, dtype=np.uint8)
        # Card 1
        cv2.rectangle(image, (50, 50), (400, 550), (200, 100, 50), -1)
        # Card 2
        cv2.rectangle(image, (500, 100), (850, 600), (50, 200, 100), -1)
        cards = detect_cards(image)
        assert len(cards) >= 2

    def test_no_cards_in_blank_image(self):
        image = np.full((800, 600, 3), 128, dtype=np.uint8)
        cards = detect_cards(image)
        assert len(cards) == 0

    def test_card_too_small_is_ignored(self):
        """A very small rectangle should be below the area threshold."""
        image = np.full((2000, 2000, 3), 30, dtype=np.uint8)
        cv2.rectangle(image, (900, 900), (920, 920), (255, 255, 255), -1)
        cards = detect_cards(image)
        assert len(cards) == 0

    def test_output_shape_is_portrait(self):
        image = _make_card_image()
        cards = detect_cards(image)
        assert len(cards) >= 1
        h, w = cards[0].image.shape[:2]
        assert h >= w, f"Expected portrait orientation, got {w}x{h}"


class TestOrderCorners:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 140], [0, 140]], dtype=np.float32)
        ordered = _order_corners(pts)
        assert ordered[0][0] == 0 and ordered[0][1] == 0  # top-left
        assert ordered[2][0] == 100 and ordered[2][1] == 140  # bottom-right

    def test_shuffled_order(self):
        pts = np.array([[100, 140], [0, 0], [0, 140], [100, 0]], dtype=np.float32)
        ordered = _order_corners(pts)
        assert ordered[0][0] == 0 and ordered[0][1] == 0
        assert ordered[1][0] == 100 and ordered[1][1] == 0
