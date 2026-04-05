"""Pipeline orchestration: identify cards from images or arrays."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from card_reco.detector import (
    CARD_DST_PORTRAIT,
    CARD_HEIGHT,
    CARD_WIDTH,
    detect_cards,
    refine_corners_edge_intersect,
)
from card_reco.detector.constants import make_clahe
from card_reco.detector.nms import compute_overlap
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher
from card_reco.models import CardHashes, DetectedCard, MatchResult

if TYPE_CHECKING:
    from card_reco.debug import DebugWriter
    from card_reco.embedder import CardEmbedder
    from card_reco.faiss_index import CardIndex

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

# IoU threshold for skipping a detection that overlaps a confidently-
# matched card.  0.5 means at least half the bounding boxes overlap.
_CLAIMED_OVERLAP_THRESH = 0.5


def _is_claimed(corners: np.ndarray, claimed: list[np.ndarray]) -> bool:
    """Return True if *corners* overlaps any already-claimed region."""
    for claimed_corners in claimed:
        if compute_overlap(corners, claimed_corners) > _CLAIMED_OVERLAP_THRESH:
            return True
    return False


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
    """Apply bilateral filter denoising + CLAHE contrast enhancement.

    Reduces camera sensor noise and normalises local contrast so that
    photos of holographic / reflective cards hash closer to clean
    reference scans.  Uses bilateral filtering instead of non-local
    means for a ~100x speed improvement with comparable edge
    preservation.
    """
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    clahe = make_clahe()
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
    backend: str = "hash",
    max_detect_dim: int = 0,
    fast: bool = False,
) -> list[list[MatchResult]]:
    """Identify Pokemon cards in an image.

    Args:
        image_path: Path to the input image.
        db_path: Path to the hash database. Uses default if None.
        top_n: Maximum number of match candidates per detected card.
        threshold: Maximum combined hash distance to consider a match
            (hash backend) or minimum cosine similarity (cnn backend).
        debug: Optional DebugWriter for saving intermediate images.
        backend: ``"hash"`` for perceptual hashing, ``"cnn"`` for CNN
            embedding + FAISS search.
        max_detect_dim: When > 0, downscale the image so its longest
            edge does not exceed this many pixels during detection.
            Perspective warps still use the full-resolution image.
        fast: When ``True``, use only the fastest detection strategies
            (Canny + adaptive threshold), skipping HSV, morph-close,
            and Hough line detection.

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
        image,
        db_path=db_path,
        top_n=top_n,
        threshold=threshold,
        debug=debug,
        backend=backend,
        max_detect_dim=max_detect_dim,
        fast=fast,
    )


def identify_cards_from_array(
    image: np.ndarray,
    db_path: str | Path | None = None,
    top_n: int = 5,
    threshold: float = 40.0,
    confident_threshold: float = 25.0,
    debug: DebugWriter | None = None,
    matcher: CardMatcher | None = None,
    backend: str = "hash",
    embedder: CardEmbedder | None = None,
    card_index: CardIndex | None = None,
    max_detect_dim: int = 0,
    fast: bool = False,
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

    When *matcher* is provided it is used directly (and **not** closed
    by this function).  This allows callers that process many images to
    amortise the one-time database-load cost.

    Set *backend* to ``"cnn"`` to use CNN embeddings + FAISS search.
    Provide *embedder* and *card_index* to reuse pre-loaded resources.

    *max_detect_dim* and *fast* are forwarded to
    :func:`~card_reco.detector.detect_cards`.
    """
    detected = detect_cards(
        image, debug=debug, max_detect_dim=max_detect_dim, fast=fast
    )

    # When the input itself looks like a single card, inject a
    # "whole-image" candidate so that pre-cropped photos can match
    # directly without relying on the detector.
    whole_image_card = _make_whole_image_card(image)
    if whole_image_card is not None:
        detected = [whole_image_card, *list(detected)]

    if not detected:
        return []

    if backend == "cnn":
        return _run_cnn_pipeline(
            detected,
            top_n=top_n,
            threshold=threshold,
            confident_threshold=confident_threshold,
            debug=debug,
            embedder=embedder,
            card_index=card_index,
        )

    all_results: list[list[MatchResult]] = []

    # Use the caller-supplied matcher or create (and close) a fresh one.
    owns_matcher = matcher is None
    if matcher is None:
        matcher = CardMatcher(db_path)

    try:
        _run_matching(
            matcher,
            detected,
            image,
            all_results,
            top_n=top_n,
            threshold=threshold,
            confident_threshold=confident_threshold,
            debug=debug,
        )
    finally:
        if owns_matcher:
            matcher.close()

    return all_results


def _run_matching(
    matcher: CardMatcher,
    detected: list[DetectedCard],
    image: np.ndarray,
    all_results: list[list[MatchResult]],
    *,
    top_n: int,
    threshold: float,
    confident_threshold: float,
    debug: DebugWriter | None,
) -> None:
    """Match each detected card using *matcher* (extraction of inner loop).

    Once a detection matches confidently (distance below
    *confident_threshold*), subsequent detections that overlap it by
    more than 50 % IoU are skipped.  This avoids redundant hashing and
    matching of duplicate cutouts for the same physical card.
    """
    # Corners of confidently-matched detections — used to skip
    # overlapping cutouts later in the list.
    claimed_corners: list[np.ndarray] = []

    for i, card in enumerate(detected):
        # Skip if this detection overlaps a previously matched card.
        if _is_claimed(card.corners, claimed_corners):
            continue

        enable_relaxed_fallback = card.confidence >= _RELAXED_FALLBACK_MIN_CONFIDENCE

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
            claimed_corners.append(card.corners)
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
        # alternative crop paddings.
        if not best_matches or best_matches[0].distance > threshold:
            best_matches = _explore_crops(
                image,
                card,
                matcher,
                best_matches,
                top_n=top_n,
                threshold=threshold,
                confident_threshold=confident_threshold,
                enable_relaxed_fallback=enable_relaxed_fallback,
                card_hashes_0=hashes,
                card_hashes_180=hashes_180,
            )

        if debug:
            debug.save_match_summary(i, card.image, best_matches)
        all_results.append(best_matches)

        # Claim the region so overlapping detections are skipped.
        if best_matches and best_matches[0].distance < confident_threshold:
            claimed_corners.append(card.corners)


def _explore_crops(
    source_image: np.ndarray,
    card: DetectedCard,
    matcher: CardMatcher,
    current_best: list[MatchResult],
    *,
    top_n: int,
    threshold: float,
    confident_threshold: float = 25.0,
    enable_relaxed_fallback: bool,
    card_hashes_0: CardHashes,
    card_hashes_180: CardHashes,
) -> list[MatchResult]:
    """Try alternative crop paddings and preprocessing to improve matching.

    For photos with imprecise contour boundaries or holographic
    reflections, the default single crop may not hash close enough
    to the reference.  This function re-warps the card region at
    several padding levels and applies denoise + CLAHE preprocessing,
    keeping whichever configuration produces the lowest match distance.

    Exits early when a match below *confident_threshold* is found.

    *card_hashes_0* and *card_hashes_180* are the already-computed
    hashes for the card at 0° and 180° orientations.  They are reused
    with the more permissive crop-exploration matcher parameters,
    avoiding two redundant ``compute_hashes`` calls (~260 ms each).
    """
    # Always use the standard threshold so the crop exploration
    # doesn't inherit an abnormally high caller-supplied value.
    effective_threshold = min(threshold, 40.0)

    best = current_best

    # Re-try the original crop hashes with relaxed fallback params.
    # The hashes are already computed by the caller, so we only pay
    # for the (cheap) matching step here.
    for precomputed in (card_hashes_0, card_hashes_180):
        matches = matcher.find_matches(
            precomputed,
            top_n=top_n,
            threshold=effective_threshold,
            enable_relaxed_fallback=enable_relaxed_fallback,
            relaxed_headroom=_CROP_EXPLORE_HEADROOM,
            min_separation=_CROP_EXPLORE_SEPARATION,
            min_consensus=_CROP_EXPLORE_MIN_CONSENSUS,
        )
        if matches and (not best or matches[0].distance < best[0].distance):
            best = matches
            if best[0].distance < confident_threshold:
                return best

    # Build remaining candidate images (denoised, refined, padded).
    candidate_images: list[np.ndarray] = []
    candidate_images.append(_denoise_clahe(card.image))

    # Try edge-intersection refined corners.
    if card.contour is not None:
        refined = refine_corners_edge_intersect(card.contour, card.corners)
        if not np.array_equal(refined, card.corners):
            warped_refined = _rewarp(source_image, refined)
            candidate_images.append(warped_refined)
            candidate_images.append(_denoise_clahe(warped_refined))

    # Try alternative crop paddings with preprocessing.
    for pct in _CROP_PADDINGS:
        expanded = _expand_corners(card.corners, pct)
        warped = _rewarp(source_image, expanded)
        candidate_images.append(warped)
        candidate_images.append(_denoise_clahe(warped))

    # Evaluate each candidate image in both orientations.
    for img in candidate_images:
        for rotate in (False, True):
            oriented = img
            if rotate:
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
                if best[0].distance < confident_threshold:
                    return best

    return best


# ───────────────────────────────────────────────────────────────────
#  CNN Embedding + FAISS pipeline
# ───────────────────────────────────────────────────────────────────

# Default CNN thresholds (cosine similarity scale, higher = better).
_CNN_CONFIDENT_THRESHOLD = 0.70
_CNN_MATCH_THRESHOLD = 0.40


_CNN_CENTER_CROP = 0.85


def _center_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    """Return a center crop of *image* at the given fraction (0-1)."""
    h, w = image.shape[:2]
    ch, cw = int(h * fraction), int(w * fraction)
    y, x = (h - ch) // 2, (w - cw) // 2
    return np.asarray(image[y : y + ch, x : x + cw], dtype=np.uint8)


def _cnn_fallback_variants(
    detected: list[DetectedCard],
    need_fallback: list[int],
    primary_embs: np.ndarray,
    embedder: CardEmbedder,
    card_index: CardIndex,
    top_n: int,
    threshold: float,
) -> list[tuple[int, DetectedCard, list[MatchResult]]]:
    """Batch-embed and search fallback variants for non-confident cards."""
    fallback_images: list[np.ndarray] = []
    for idx in need_fallback:
        card = detected[idx]
        img_180 = np.asarray(cv2.rotate(card.image, cv2.ROTATE_180), dtype=np.uint8)
        crop = _center_crop(card.image, _CNN_CENTER_CROP)
        crop_180 = np.asarray(cv2.rotate(crop, cv2.ROTATE_180), dtype=np.uint8)
        fallback_images.extend([img_180, crop, crop_180])

    fallback_embs = embedder.embed_batch(fallback_images)
    results: list[tuple[int, DetectedCard, list[MatchResult]]] = []

    for j, idx in enumerate(need_fallback):
        card = detected[idx]
        best = card_index.search(primary_embs[idx], top_k=top_n, threshold=threshold)
        for k in range(3):
            emb = fallback_embs[j * 3 + k]
            matches = card_index.search(emb, top_k=top_n, threshold=threshold)
            if matches and (not best or matches[0].distance > best[0].distance):
                best = matches
        results.append((idx, card, best))

    return results


def _run_cnn_pipeline(  # pylint: disable=too-many-locals
    detected: list[DetectedCard],
    *,
    top_n: int,
    threshold: float,
    confident_threshold: float,
    debug: DebugWriter | None,
    embedder: CardEmbedder | None,
    card_index: CardIndex | None,
) -> list[list[MatchResult]]:
    """CNN embedding + FAISS search pipeline.

    Uses a two-phase early-exit strategy with batch inference:

    1. Batch-embed all detections at 0° full-frame.  Cards that match
       confidently are done; the rest are queued for fallback variants.
    2. For remaining cards, batch the fallback variants (180 deg, center-
       crop x 0 deg, center-crop x 180 deg) and search.
    3. Deduplicate — sort all candidates by best score (descending),
       then emit results skipping overlapping regions (IoU > 0.5).
    """
    # Lazy imports to avoid loading ONNX Runtime when using hash backend.
    # pylint: disable=import-outside-toplevel
    from card_reco.embedder import CardEmbedder as _Embedder
    from card_reco.faiss_index import CardIndex as _Index

    owns_embedder = embedder is None
    owns_index = card_index is None
    if embedder is None:
        embedder = _Embedder()
    if card_index is None:
        card_index = _Index()

    # Use CNN-appropriate thresholds unless caller overrode them.
    if threshold == 40.0:
        threshold = _CNN_MATCH_THRESHOLD
    if confident_threshold == 25.0:
        confident_threshold = _CNN_CONFIDENT_THRESHOLD

    candidates: list[tuple[int, DetectedCard, list[MatchResult]]] = []

    try:
        # --- Phase 1: batch-embed 0° full-frame, early-exit confident ---
        primary_embs = embedder.embed_batch([c.image for c in detected])
        need_fallback: list[int] = []

        for i, (card, emb) in enumerate(zip(detected, primary_embs, strict=True)):
            matches = card_index.search(emb, top_k=top_n, threshold=threshold)
            if matches and matches[0].distance >= confident_threshold:
                candidates.append((i, card, matches))
            else:
                need_fallback.append(i)

        # --- Phase 2: batch fallback variants for remaining cards ---
        if need_fallback:
            candidates.extend(
                _cnn_fallback_variants(
                    detected,
                    need_fallback,
                    primary_embs,
                    embedder,
                    card_index,
                    top_n,
                    threshold,
                )
            )

        # --- Phase 3: deduplicate — highest scoring first ---
        candidates.sort(key=lambda c: c[2][0].distance if c[2] else -1.0, reverse=True)

        all_results: list[list[MatchResult]] = []
        claimed_corners: list[np.ndarray] = []

        for idx, card, best in candidates:
            if _is_claimed(card.corners, claimed_corners):
                continue
            if debug:
                debug.save_match_summary(idx, card.image, best)
            all_results.append(best)
            if best and best[0].distance >= confident_threshold:
                claimed_corners.append(card.corners)
    finally:
        if owns_embedder:
            del embedder
        if owns_index:
            del card_index

    return all_results
