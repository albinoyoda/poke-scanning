"""Debug the improved detector on test images."""
import cv2
import numpy as np
from pathlib import Path
from card_reco.detector import (
    detect_cards, _find_card_contours, _extract_corners, 
    _has_card_aspect_ratio, MIN_CARD_AREA_RATIO, MAX_CARD_AREA_RATIO,
    CARD_ASPECT_RATIO, ASPECT_RATIO_TOLERANCE
)
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher

TEST_DIR = Path("data/tests")
DB_PATH = Path("data/card_hashes.db")

for img_name in ["moltres_151.png", "image_psa_graded_charizard.png", "image_3x3.png"]:
    img_path = TEST_DIR / img_name
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    h, w = image.shape[:2]
    image_area = h * w
    min_area = image_area * MIN_CARD_AREA_RATIO
    max_area = image_area * MAX_CARD_AREA_RATIO
    
    print(f"\n{'='*60}")
    print(f"{img_name}: {w}x{h}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    candidates = _find_card_contours(blurred, min_area, max_area)
    print(f"Candidates from _find_card_contours: {len(candidates)}")
    
    for i, contour in enumerate(candidates):
        area = cv2.contourArea(contour)
        corners = _extract_corners(contour)
        has_ratio = _has_card_aspect_ratio(corners) if corners is not None else False
        
        if corners is not None:
            tl, tr, br, bl = corners
            w_top = float(np.linalg.norm(tr - tl))
            h_left = float(np.linalg.norm(bl - tl))
            short = min(w_top, h_left)
            long = max(w_top, h_left)
            ratio = short / long if long > 0 else 0
        else:
            ratio = 0
        
        print(f"  #{i}: area={area:.0f} ({area/image_area*100:.1f}%), "
              f"corners={'Y' if corners is not None else 'N'}, "
              f"ratio={ratio:.3f} (target={CARD_ASPECT_RATIO:.3f}±{ASPECT_RATIO_TOLERANCE}), "
              f"card_ratio={'Y' if has_ratio else 'N'}")
    
    # Run actual detector
    cards = detect_cards(image)
    print(f"\nDetected cards: {len(cards)}")
    
    # If cards detected, try matching
    if cards:
        with CardMatcher(DB_PATH) as matcher:
            for j, card in enumerate(cards):
                hashes = compute_hashes(card.image)
                matches = matcher.find_matches(hashes, top_n=3, threshold=60.0)
                if matches:
                    print(f"  Card {j}: top match = {matches[0].card.name} "
                          f"(id={matches[0].card.id}, dist={matches[0].distance:.1f})")
                else:
                    # Try with higher threshold to see what's close
                    matches = matcher.find_matches(hashes, top_n=3, threshold=200.0)
                    if matches:
                        print(f"  Card {j}: no match at threshold=60. "
                              f"Top at 200: {matches[0].card.name} "
                              f"(id={matches[0].card.id}, dist={matches[0].distance:.1f})")
                    else:
                        print(f"  Card {j}: no match even at threshold=200")
