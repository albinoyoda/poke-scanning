"""Quick test to diagnose the Electivire regression."""

import cv2

from card_reco.matcher import CardMatcher
from card_reco.pipeline import identify_cards_from_array

img = cv2.imread("data/tests/single_cards/axis_aligned/electivire_ex_69_182.png")
matcher = CardMatcher()
matcher.preload()

for thresh in [60.0, 90.0]:
    results = identify_cards_from_array(
        img, matcher=matcher, threshold=thresh, top_n=5, confident_threshold=25.0
    )
    print(f"threshold={thresh}:")
    for i, r_list in enumerate(results):
        for r in r_list[:2]:
            print(f"  Card {i}: {r.card.id} {r.card.name} dist={r.distance:.1f}")
        if not r_list:
            print(f"  Card {i}: no match")
    print()

matcher.close()
