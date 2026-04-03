"""Save detected card crops for visual inspection."""
import cv2
from pathlib import Path
from card_reco.detector import detect_cards

TEST_DIR = Path("data/tests")
OUT_DIR = Path("data/tests/debug_crops")
OUT_DIR.mkdir(exist_ok=True)

for img_name in ["moltres_151.png", "image_psa_graded_charizard.png", "image_3x3.png"]:
    img_path = TEST_DIR / img_name
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    
    cards = detect_cards(image)
    stem = img_path.stem
    print(f"{img_name}: {len(cards)} cards detected")
    for i, card in enumerate(cards):
        out_path = OUT_DIR / f"{stem}_card{i}.png"
        cv2.imwrite(str(out_path), card.image)
        print(f"  Saved {out_path} ({card.image.shape[1]}x{card.image.shape[0]})")
