"""card_reco — Pokemon card recognition from images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from card_reco.detector import CARD_HEIGHT, CARD_WIDTH, detect_cards
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher
from card_reco.models import DetectedCard, MatchResult

if TYPE_CHECKING:
    from card_reco.debug import DebugWriter

_RELAXED_FALLBACK_MIN_CONFIDENCE = 0.85
_RELAXED_FALLBACK_HEADROOM = 25.0
_RELAXED_FALLBACK_SEPARATION = 15.0
_RELAXED_FALLBACK_MIN_CONSENSUS = 2

# Tolerance for considering the input image a single card (aspect ratio).
_CARD_AR = CARD_WIDTH / CARD_HEIGHT  # ~0.714
_AR_TOLERANCE = 0.12


def _make_whole_image_card(image: np.ndarray) -> DetectedCard | None:
    """Create a DetectedCard from the whole image if it looks card-shaped.

    Returns ``None`` when the image aspect ratio is too far from a
    standard Pokemon card (5:7).  The image is resized to the canonical
    card dimensions so that hashes are comparable to references.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    ar = w / h
    # Also accept landscape (rotated 90°) — will be handled by 180° flip.
    if abs(ar - _CARD_AR) > _AR_TOLERANCE and abs(ar - 1.0 / _CARD_AR) > _AR_TOLERANCE:
        return None

    if ar > 1.0:
        # Landscape → rotate to portrait first
        image = np.asarray(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), dtype=np.uint8)

    resized = np.asarray(
        cv2.resize(image, (CARD_WIDTH, CARD_HEIGHT), interpolation=cv2.INTER_AREA),
        dtype=np.uint8,
    )
    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    return DetectedCard(image=resized, corners=corners, confidence=1.0)


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

    When the input image has roughly card-shaped aspect ratio, it is
    also hashed directly (without detection) so that pre-cropped card
    photos can be identified even when the detector re-warps them
    poorly.
    """
    detected = detect_cards(image, debug=debug)

    # When the input itself looks like a single card, inject a
    # "whole-image" candidate so that pre-cropped photos can match
    # directly without relying on the detector.
    whole_image_card = _make_whole_image_card(image)
    if whole_image_card is not None:
        detected = [whole_image_card, *list(detected)]

    if not detected:
        return []

    all_results: list[list[MatchResult]] = []

    with CardMatcher(db_path) as matcher:
        for i, card in enumerate(detected):
            enable_relaxed_fallback = (
                card.confidence >= _RELAXED_FALLBACK_MIN_CONFIDENCE
            )

            # Try the card as-is first (0° orientation).
            hashes = compute_hashes(card.image)
            best_matches = matcher.find_matches(
                hashes,
                top_n=top_n,
                threshold=threshold,
                enable_relaxed_fallback=enable_relaxed_fallback,
                relaxed_headroom=_RELAXED_FALLBACK_HEADROOM,
                min_separation=_RELAXED_FALLBACK_SEPARATION,
                min_consensus=_RELAXED_FALLBACK_MIN_CONSENSUS,
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
                hashes_180,
                top_n=top_n,
                threshold=threshold,
                enable_relaxed_fallback=enable_relaxed_fallback,
                relaxed_headroom=_RELAXED_FALLBACK_HEADROOM,
                min_separation=_RELAXED_FALLBACK_SEPARATION,
                min_consensus=_RELAXED_FALLBACK_MIN_CONSENSUS,
            )
            if matches_180 and (
                not best_matches or matches_180[0].distance < best_matches[0].distance
            ):
                best_matches = matches_180

            if debug:
                debug.save_match_summary(i, card.image, best_matches)
            all_results.append(best_matches)

    return all_results
