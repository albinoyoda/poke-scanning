"""card_reco — Pokemon card recognition from images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from card_reco.detector import CARD_DST_PORTRAIT, CARD_HEIGHT, CARD_WIDTH, detect_cards
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher
from card_reco.models import DetectedCard, MatchResult

if TYPE_CHECKING:
    from card_reco.debug import DebugWriter

_RELAXED_FALLBACK_MIN_CONFIDENCE = 0.85
_RELAXED_FALLBACK_HEADROOM = 25.0
_RELAXED_FALLBACK_SEPARATION = 15.0
_RELAXED_FALLBACK_MIN_CONSENSUS = 2

# Crop exploration uses more aggressive fallback parameters because
# the preprocessing (denoise + CLAHE) narrows the gap between
# photographed cards and clean reference scans.
_CROP_EXPLORE_HEADROOM = 25.0
_CROP_EXPLORE_SEPARATION = 3.0
_CROP_EXPLORE_MIN_CONSENSUS = 2

# Crop-exploration paddings: percentage to expand (+) or contract (-)
# the detected corners from the centroid.  Multiple crops improve
# hash matching for photos with imprecise contour boundaries.
_CROP_PADDINGS: tuple[float, ...] = (-3.0, 2.0)

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


def _denoise_clahe(image: np.ndarray) -> np.ndarray:
    """Apply non-local means denoising + CLAHE contrast enhancement.

    Reduces camera sensor noise and normalises local contrast so that
    photos of holographic / reflective cards hash closer to clean
    reference scans.
    """
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _expand_corners(corners: np.ndarray, pct: float) -> np.ndarray:
    """Expand (or contract) corner points by *pct* % from their centroid."""
    centroid = corners.mean(axis=0)
    return centroid + (corners - centroid) * (1.0 + pct / 100.0)


def _rewarp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-warp a region defined by *corners* to card dimensions."""
    h, w = image.shape[:2]
    clamped = corners.copy().astype(np.float32)
    clamped[:, 0] = np.clip(clamped[:, 0], 0, w - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, h - 1)
    matrix = cv2.getPerspectiveTransform(clamped, CARD_DST_PORTRAIT)
    return np.asarray(
        cv2.warpPerspective(image, matrix, (CARD_WIDTH, CARD_HEIGHT)),
        dtype=np.uint8,
    )


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

            # Crop exploration: when the basic match is still not
            # confident, try denoised/CLAHE-enhanced variants and
            # alternative crop paddings.  Photos of holographic or
            # reflective cards often need this extra effort.
            if not best_matches or best_matches[0].distance > threshold:
                best_matches = _explore_crops(
                    image,
                    card,
                    matcher,
                    best_matches,
                    top_n=top_n,
                    threshold=threshold,
                    enable_relaxed_fallback=enable_relaxed_fallback,
                )

            if debug:
                debug.save_match_summary(i, card.image, best_matches)
            all_results.append(best_matches)

    return all_results


def _explore_crops(
    source_image: np.ndarray,
    card: DetectedCard,
    matcher: CardMatcher,
    current_best: list[MatchResult],
    *,
    top_n: int,
    threshold: float,
    enable_relaxed_fallback: bool,
) -> list[MatchResult]:
    """Try alternative crop paddings and preprocessing to improve matching.

    For photos with imprecise contour boundaries or holographic
    reflections, the default single crop may not hash close enough
    to the reference.  This function re-warps the card region at
    several padding levels and applies denoise + CLAHE preprocessing,
    keeping whichever configuration produces the lowest match distance.

    Uses the default matching threshold (40) to avoid false positives
    when the caller specifies a higher threshold, and applies more
    permissive fallback parameters since the preprocessing narrows the
    gap between photographed cards and clean reference scans.
    """
    best = current_best

    # Always use the standard threshold so the crop exploration
    # doesn't inherit an abnormally high caller-supplied value.
    effective_threshold = min(threshold, 40.0)

    def _try_image(img: np.ndarray) -> None:
        nonlocal best
        for rot, _code in [(False, None), (True, cv2.ROTATE_180)]:
            oriented = img
            if rot:
                oriented = np.asarray(cv2.rotate(img, cv2.ROTATE_180), dtype=np.uint8)
            hashes = compute_hashes(oriented)
            matches = matcher.find_matches(
                hashes,
                top_n=top_n,
                threshold=effective_threshold,
                enable_relaxed_fallback=enable_relaxed_fallback,
                relaxed_headroom=_CROP_EXPLORE_HEADROOM,
                min_separation=_CROP_EXPLORE_SEPARATION,
                min_consensus=_CROP_EXPLORE_MIN_CONSENSUS,
            )
            if matches and (not best or matches[0].distance < best[0].distance):
                best = matches

    # Re-try the original crop with the more permissive fallback
    # parameters.  The basic matching phase uses stricter separation;
    # this lets borderline cards pass.
    _try_image(card.image)

    # Try denoise + CLAHE on the original crop.
    _try_image(_denoise_clahe(card.image))

    # Try alternative crop paddings with preprocessing.
    for pct in _CROP_PADDINGS:
        expanded = _expand_corners(card.corners, pct)
        warped = _rewarp(source_image, expanded)
        _try_image(warped)
        _try_image(_denoise_clahe(warped))

    return best
