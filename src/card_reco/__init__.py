"""card_reco — Pokemon card recognition from images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from card_reco.detector import detect_cards
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher
from card_reco.models import MatchResult


def identify_cards(
    image_path: str | Path,
    db_path: str | Path | None = None,
    top_n: int = 5,
    threshold: float = 40.0,
) -> list[list[MatchResult]]:
    """Identify Pokemon cards in an image.

    Args:
        image_path: Path to the input image.
        db_path: Path to the hash database. Uses default if None.
        top_n: Maximum number of match candidates per detected card.
        threshold: Maximum combined hash distance to consider a match.

    Returns:
        A list of match lists — one per detected card in the image.
        Each inner list contains up to top_n MatchResult objects sorted
        by distance (best match first).
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return identify_cards_from_array(
        image, db_path=db_path, top_n=top_n, threshold=threshold
    )


def identify_cards_from_array(
    image: np.ndarray,
    db_path: str | Path | None = None,
    top_n: int = 5,
    threshold: float = 40.0,
) -> list[list[MatchResult]]:
    """Identify Pokemon cards from a BGR numpy array.

    Same as identify_cards but accepts a pre-loaded image array.
    """
    detected = detect_cards(image)

    if not detected:
        return []

    all_results: list[list[MatchResult]] = []

    with CardMatcher(db_path) as matcher:
        for card in detected:
            hashes = compute_hashes(card.image)
            matches = matcher.find_matches(hashes, top_n=top_n, threshold=threshold)
            all_results.append(matches)

    return all_results
