"""Check what relaxed params would match Electivire."""

import cv2
import numpy as np

from card_reco.detector import detect_cards
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher


def main():
    img = cv2.imread("data/tests/single_cards/axis_aligned/electivire_ex_69_182.png")
    cards = detect_cards(img)

    matcher = CardMatcher()
    matcher.preload()

    card = cards[0]  # highest confidence
    h = compute_hashes(card.image)

    # Try different separation values
    for sep in [15.0, 12.0, 10.0, 8.0, 5.0]:
        results = matcher.find_matches(
            h,
            top_n=3,
            threshold=60.0,
            enable_relaxed_fallback=True,
            relaxed_headroom=25.0,
            min_separation=sep,
            min_consensus=2,
        )
        if results:
            print(f"  sep={sep}: {results[0].card.id} dist={results[0].distance:.1f}")
        else:
            print(f"  sep={sep}: no match")

    # Try higher headroom
    for headroom in [25.0, 30.0, 35.0]:
        results = matcher.find_matches(
            h,
            top_n=3,
            threshold=60.0,
            enable_relaxed_fallback=True,
            relaxed_headroom=headroom,
            min_separation=15.0,
            min_consensus=2,
        )
        if results:
            print(
                f"  headroom={headroom}: {results[0].card.id} dist={results[0].distance:.1f}"
            )
        else:
            print(f"  headroom={headroom}: no match")

    # Check what the best distances look like for Electivire ex name group
    combined_all = []
    idxs = matcher._name_to_indices.get("Electivire ex", np.array([]))
    # We need to compute combined distances to check
    from card_reco.hasher import hex_to_bits
    from card_reco.matcher import _TOTAL_WEIGHT, _WEIGHT_ORDER, _WEIGHTS_ARRAY

    _, matrix = matcher._ensure_loaded()
    n_bits = matrix.shape[2]
    query = np.empty((4, n_bits), dtype=np.uint8)
    for j, key in enumerate(_WEIGHT_ORDER):
        query[j] = hex_to_bits(getattr(h, key))
    diff = matrix ^ query[np.newaxis, :, :]
    per_hash = diff.sum(axis=2).astype(np.float64)
    combined = per_hash @ _WEIGHTS_ARRAY / _TOTAL_WEIGHT

    # Show Electivire distances
    for idx in idxs:
        card_rec = matcher._cards[idx]
        print(f"  DB card: {card_rec.id} {card_rec.name} dist={combined[idx]:.1f}")

    # Show top-5 closest names
    top_idx = np.argsort(combined)[:10]
    for idx in top_idx:
        card_rec = matcher._cards[idx]
        print(f"  Top match: {card_rec.id} {card_rec.name} dist={combined[idx]:.1f}")

    matcher.close()


if __name__ == "__main__":
    main()
