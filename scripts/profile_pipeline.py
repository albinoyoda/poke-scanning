#!/usr/bin/env python3
"""Profile the full card recognition pipeline with detailed timing breakdown."""

import cProfile
import io
import pstats
import time
from pathlib import Path

import cv2
import numpy as np

# --- Stage-level profiling ---


def profile_pipeline(image_path: str) -> None:
    """Profile each pipeline stage independently."""
    from card_reco.detector import detect_cards
    from card_reco.detector.constants import MAX_CARD_AREA_RATIO, MIN_CARD_AREA_RATIO
    from card_reco.detector.strategies import find_card_contours
    from card_reco.hasher import compute_hashes
    from card_reco.matcher import CardMatcher
    from card_reco.pipeline import (
        _denoise_clahe,
        _explore_crops,
        _make_whole_image_card,
    )

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    print(f"Image: {image_path}")
    print(f"Resolution: {w}x{h} ({w * h:,} pixels)")
    print(f"{'=' * 60}")

    # --- 1. Detection stage ---
    print("\n--- DETECTION ---")

    # Preprocessing
    t0 = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    t_preprocess = time.perf_counter() - t0
    print(f"  Grayscale + GaussianBlur: {t_preprocess * 1000:.1f} ms")

    image_area = h * w
    min_area = image_area * MIN_CARD_AREA_RATIO
    max_area = image_area * MAX_CARD_AREA_RATIO

    # Strategy timing
    t0 = time.perf_counter()
    candidates = find_card_contours(blurred, min_area, max_area, original_bgr=image)
    t_strategies = time.perf_counter() - t0
    print(
        f"  find_card_contours (all strategies): {t_strategies * 1000:.1f} ms  ({len(candidates)} candidates)"
    )

    # Full detection
    t0 = time.perf_counter()
    detected = detect_cards(image)
    t_detect_total = time.perf_counter() - t0
    print(
        f"  detect_cards() total: {t_detect_total * 1000:.1f} ms  ({len(detected)} cards detected)"
    )

    # --- 2. Hashing stage ---
    print("\n--- HASHING ---")
    if detected:
        # Single card hashing
        t0 = time.perf_counter()
        h0 = compute_hashes(detected[0].image)
        t_hash_one = time.perf_counter() - t0
        print(f"  compute_hashes (1 card): {t_hash_one * 1000:.1f} ms")

        # All cards hashing
        hash_times = []
        for card in detected:
            t0 = time.perf_counter()
            compute_hashes(card.image)
            hash_times.append(time.perf_counter() - t0)
        total_hash = sum(hash_times)
        print(
            f"  compute_hashes ({len(detected)} cards): {total_hash * 1000:.1f} ms total"
        )
        print(f"    avg per card: {total_hash / len(detected) * 1000:.1f} ms")
        print(
            f"    min/max: {min(hash_times) * 1000:.1f} / {max(hash_times) * 1000:.1f} ms"
        )

        # Hash both orientations (what the pipeline actually does)
        t0 = time.perf_counter()
        for card in detected:
            compute_hashes(card.image)
            rotated = np.asarray(cv2.rotate(card.image, cv2.ROTATE_180), dtype=np.uint8)
            compute_hashes(rotated)
        t_hash_both = time.perf_counter() - t0
        print(
            f"  compute_hashes (both orientations, {len(detected)} cards): {t_hash_both * 1000:.1f} ms"
        )

    # --- 3. Matching stage ---
    print("\n--- MATCHING ---")

    # DB load
    t0 = time.perf_counter()
    matcher = CardMatcher()
    matcher.preload()
    t_db_load = time.perf_counter() - t0
    print(f"  DB load + matrix build: {t_db_load * 1000:.1f} ms")
    print(f"  Reference cards: {len(matcher._cards) if matcher._cards else 0}")

    if detected:
        hashes = compute_hashes(detected[0].image)

        # Single match
        t0 = time.perf_counter()
        results = matcher.find_matches(hashes, top_n=5, threshold=40.0)
        t_match_one = time.perf_counter() - t0
        print(f"  find_matches (1 query): {t_match_one * 1000:.1f} ms")

        # All cards matching
        match_times = []
        for card in detected:
            h_card = compute_hashes(card.image)
            t0 = time.perf_counter()
            matcher.find_matches(h_card, top_n=5, threshold=40.0)
            match_times.append(time.perf_counter() - t0)
        total_match = sum(match_times)
        print(
            f"  find_matches ({len(detected)} queries): {total_match * 1000:.1f} ms total"
        )
        print(f"    avg per query: {total_match / len(detected) * 1000:.1f} ms")

    # --- 4. Denoise + CLAHE (crop exploration) ---
    print("\n--- PREPROCESSING (denoise + CLAHE) ---")
    if detected:
        t0 = time.perf_counter()
        _denoise_clahe(detected[0].image)
        t_denoise = time.perf_counter() - t0
        print(f"  _denoise_clahe (1 card): {t_denoise * 1000:.1f} ms")

    # --- 5. Full pipeline ---
    print("\n--- FULL PIPELINE ---")
    from card_reco.pipeline import identify_cards_from_array

    # Warm up matcher
    matcher2 = CardMatcher()
    matcher2.preload()

    t0 = time.perf_counter()
    all_results = identify_cards_from_array(image, matcher=matcher2)
    t_full = time.perf_counter() - t0
    print(f"  identify_cards_from_array(): {t_full * 1000:.1f} ms")
    print(f"  Cards identified: {len(all_results)}")
    for i, results in enumerate(all_results):
        if results:
            print(
                f"    Card {i}: {results[0].card.name} (dist={results[0].distance:.1f})"
            )
        else:
            print(f"    Card {i}: no match")

    matcher.close()
    matcher2.close()

    # --- 6. cProfile for detailed breakdown ---
    print(f"\n{'=' * 60}")
    print("--- cProfile (full pipeline, top 30 by cumulative time) ---")

    matcher3 = CardMatcher()
    matcher3.preload()

    pr = cProfile.Profile()
    pr.enable()
    identify_cards_from_array(image, matcher=matcher3)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    # Also sort by tottime
    print("--- cProfile (top 30 by total time) ---")
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(30)
    print(s2.getvalue())

    matcher3.close()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY - Time budget for real-time (5 Hz = 200ms/frame):")
    print(f"  Detection:  {t_detect_total * 1000:.1f} ms")
    if detected:
        print(f"  Hashing:    {total_hash * 1000:.1f} ms ({len(detected)} cards)")
        print(f"  Matching:   {total_match * 1000:.1f} ms ({len(detected)} queries)")
    print(f"  Full pipeline: {t_full * 1000:.1f} ms")
    print(f"  Target: 200 ms (5 Hz)")
    if t_full > 0:
        print(f"  Current rate: {1.0 / t_full:.1f} Hz")
        print(f"  Speedup needed: {t_full / 0.2:.1f}x")


if __name__ == "__main__":
    profile_pipeline("data/tests/multiple_cards/7_cards.png")
