"""Brightness normalization experiment: does stretching help match quality?"""

from pathlib import Path

import cv2
import numpy as np

from card_reco.detector import detect_cards
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher


def normalize_stretch(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


db_path = Path("data/card_hashes.db")
if not db_path.exists():
    print("No hash DB, skipping match comparison")
    exit()

test_dir = Path("data/tests/single_cards/axis_aligned")
with CardMatcher(db_path) as matcher:
    for p in sorted(test_dir.glob("*.png")):
        if "_edges" in p.name:
            continue
        img = cv2.imread(str(p))
        dets = detect_cards(np.asarray(img, dtype=np.uint8))
        if not dets:
            print(f"{p.stem}: no detection")
            continue
        card = dets[0].image
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()

        # Original match
        h_orig = compute_hashes(card)
        m_orig = matcher.find_matches(h_orig, top_n=1, threshold=100)

        # Stretched match
        h_norm = compute_hashes(normalize_stretch(card))
        m_norm = matcher.find_matches(h_norm, top_n=1, threshold=100)

        d_orig = m_orig[0].distance if m_orig else 999
        d_norm = m_norm[0].distance if m_norm else 999
        id_orig = m_orig[0].card.id if m_orig else "n/a"
        id_norm = m_norm[0].card.id if m_norm else "n/a"
        delta = d_orig - d_norm
        marker = "<< IMPROVED" if delta > 2 else (">> WORSE" if delta < -2 else "")

        print(
            f"{p.stem:35s}  bright={brightness:5.1f}  "
            f"orig={d_orig:5.1f} ({id_orig})  "
            f"norm={d_norm:5.1f} ({id_norm})  "
            f"delta={delta:+.1f} {marker}"
        )
