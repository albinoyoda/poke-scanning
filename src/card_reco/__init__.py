"""card_reco — Pokemon card recognition from images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from card_reco.detector import detect_cards
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher
from card_reco.models import MatchResult

if TYPE_CHECKING:
    from card_reco.debug import DebugWriter


def identify_cards(
    image_path: str | Path,
    db_path: str | Path | None = None,
    top_n: int = 5,
    threshold: float = 40.0,
    debug: DebugWriter | None = None,
) -> list[list[MatchResult]]:
    """Identify Pokemon cards in an image.

    Args:
        image_path: Path to the input image.
        db_path: Path to the hash database. Uses default if None.
        top_n: Maximum number of match candidates per detected card.
        threshold: Maximum combined hash distance to consider a match.
        debug: Optional DebugWriter for saving intermediate images.

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
        image, db_path=db_path, top_n=top_n, threshold=threshold, debug=debug
    )


def identify_cards_from_array(
    image: np.ndarray,
    db_path: str | Path | None = None,
    top_n: int = 5,
    threshold: float = 40.0,
    confident_threshold: float = 25.0,
    debug: DebugWriter | None = None,
) -> list[list[MatchResult]]:
    """Identify Pokemon cards from a BGR numpy array.

    Same as identify_cards but accepts a pre-loaded image array.
    Tries at most two orientations (0° and 180°) per detected card.
    If the first orientation produces a confident match (distance below
    *confident_threshold*), the 180° flip is skipped entirely.
    """
    detected = detect_cards(image, debug=debug)

    if not detected:
        return []

    all_results: list[list[MatchResult]] = []

    with CardMatcher(db_path) as matcher:
        for i, card in enumerate(detected):
            # Try the card as-is first (0° orientation).
            hashes = compute_hashes(card.image)
            best_matches = matcher.find_matches(
                hashes, top_n=top_n, threshold=threshold
            )

            # Skip 180° flip when the first orientation is already confident.
            if best_matches and best_matches[0].distance < confident_threshold:
                if debug:
                    debug.save_match_summary(i, card.image, best_matches)
                all_results.append(best_matches)
                continue

            # Try 180° rotation and keep whichever orientation matched better.
            rotated = np.asarray(cv2.rotate(card.image, cv2.ROTATE_180), dtype=np.uint8)
            hashes_180 = compute_hashes(rotated)
            matches_180 = matcher.find_matches(
                hashes_180, top_n=top_n, threshold=threshold
            )
            if matches_180 and (
                not best_matches or matches_180[0].distance < best_matches[0].distance
            ):
                best_matches = matches_180

            if debug:
                debug.save_match_summary(i, card.image, best_matches)
            all_results.append(best_matches)

    return all_results
