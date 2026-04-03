"""Check match distances for all test images at high threshold to understand matching quality."""
import cv2
import numpy as np
from pathlib import Path
from card_reco.detector import detect_cards
from card_reco.hasher import compute_hashes
from card_reco.matcher import CardMatcher

TEST_DIR = Path("data/tests")
DB_PATH = Path("data/card_hashes.db")

rotations = [
    ("0°", None),
    ("90°CW", cv2.ROTATE_90_CLOCKWISE),
    ("180°", cv2.ROTATE_180),
    ("90°CCW", cv2.ROTATE_90_COUNTERCLOCKWISE),
]

for img_name in ["moltres_151.png", "image_psa_graded_charizard.png", "image_3x3.png"]:
    img_path = TEST_DIR / img_name
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    print(f"\n{'='*70}")
    print(f"{img_name}")
    
    cards = detect_cards(image)
    print(f"  Detected: {len(cards)} cards")
    
    with CardMatcher(DB_PATH) as matcher:
        for j, card in enumerate(cards):
            print(f"\n  Card {j} (conf={card.confidence:.2f}):")
            best_rot = None
            best_matches = []
            for rot_name, rot_code in rotations:
                if rot_code is None:
                    rotated = card.image
                else:
                    rotated = np.asarray(cv2.rotate(card.image, rot_code), dtype=np.uint8)
                hashes = compute_hashes(rotated)
                matches = matcher.find_matches(hashes, top_n=3, threshold=200.0)
                if matches and (not best_matches or matches[0].distance < best_matches[0].distance):
                    best_matches = matches
                    best_rot = rot_name
            
            if best_matches:
                print(f"    Best rotation: {best_rot}")
                for m in best_matches:
                    print(f"    #{m.rank}: {m.card.name} ({m.card.id}) dist={m.distance:.1f} "
                          f"[a={m.distances.get('ahash',0)} p={m.distances.get('phash',0)} "
                          f"d={m.distances.get('dhash',0)} w={m.distances.get('whash',0)}]")
            else:
                print(f"    No match at threshold=200 in any rotation")
