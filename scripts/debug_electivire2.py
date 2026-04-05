"""Diagnose why Electivire fails with the new code."""

import cv2
import numpy as np

from card_reco.detector import detect_cards
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher


def main():
    img = cv2.imread("data/tests/single_cards/axis_aligned/electivire_ex_69_182.png")
    cards = detect_cards(img)
    print(
        f"{len(cards)} detections, confidences: {[f'{c.confidence:.2f}' for c in cards]}"
    )

    matcher = CardMatcher()
    matcher.preload()

    # Try each detection at both orientations with relaxed params
    for i, card in enumerate(cards):
        for rot_label, do_rot in [("0°", False), ("180°", True)]:
            oriented = card.image
            if do_rot:
                oriented = np.asarray(
                    cv2.rotate(card.image, cv2.ROTATE_180), dtype=np.uint8
                )
            h = compute_hashes(oriented)

            # Match with threshold=60, relaxed fallback
            results = matcher.find_matches(
                h,
                top_n=3,
                threshold=60.0,
                enable_relaxed_fallback=True,
                relaxed_headroom=25.0,
                min_separation=15.0,
                min_consensus=2,
            )
            if results:
                print(
                    f"  Det {i} {rot_label} conf={card.confidence:.2f}: "
                    f"{results[0].card.id} dist={results[0].distance:.1f}"
                )
            else:
                # Try without relaxed to see raw best distance
                raw = matcher.find_matches(h, top_n=1, threshold=100.0)
                best_d = raw[0].distance if raw else float("inf")
                print(
                    f"  Det {i} {rot_label} conf={card.confidence:.2f}: "
                    f"no match (best raw dist={best_d:.1f})"
                )

    # Check how many "Electivire ex" cards exist in the DB
    if matcher._name_to_indices and "Electivire ex" in matcher._name_to_indices:
        idxs = matcher._name_to_indices["Electivire ex"]
        print(f"\nReference DB has {len(idxs)} 'Electivire ex' cards")
    else:
        print("\nNo 'Electivire ex' in name_to_indices!")

    matcher.close()


if __name__ == "__main__":
    main()
