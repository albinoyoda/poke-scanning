"""Debug script to understand detector behavior on test images."""
import cv2
import numpy as np
from pathlib import Path

from card_reco.detector import detect_cards, MIN_CARD_AREA_RATIO

TEST_DIR = Path("data/tests")

for img_name in ["moltres_151.png", "image_psa_graded_charizard.png", "image_3x3.png"]:
    img_path = TEST_DIR / img_name
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"SKIP {img_name}: cannot load")
        continue

    h, w = image.shape[:2]
    print(f"\n{'='*60}")
    print(f"{img_name}: {w}x{h}, area={w*h}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_area = h * w
    min_area = image_area * MIN_CARD_AREA_RATIO
    
    print(f"Total contours: {len(contours)}")
    print(f"Min area threshold (2%): {min_area:.0f}")
    
    large_contours = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) >= min_area]
    large_contours.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Contours above min area: {len(large_contours)}")
    for i, (area, c) in enumerate(large_contours[:15]):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(f"  #{i}: area={area:.0f} ({area/image_area*100:.1f}%), vertices={len(approx)}")
    
    # Try with RETR_TREE
    contours_tree, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    large_tree = [(cv2.contourArea(c), c) for c in contours_tree if cv2.contourArea(c) >= min_area]
    large_tree.sort(key=lambda x: x[0], reverse=True)
    print(f"\nWith RETR_TREE: {len(large_tree)} contours above min area")
    for i, (area, c) in enumerate(large_tree[:15]):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(f"  #{i}: area={area:.0f} ({area/image_area*100:.1f}%), vertices={len(approx)}")
    
    # Try different Canny thresholds
    for low, high in [(30, 100), (20, 80), (75, 200)]:
        edged2 = cv2.Canny(blurred, low, high)
        edged2 = cv2.dilate(edged2, kernel, iterations=1)
        contours2, _ = cv2.findContours(edged2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for c in contours2:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                quads.append((area, approx))
        print(f"\nCanny({low},{high}): {len(quads)} quads found")
        for i, (a, q) in enumerate(quads[:5]):
            print(f"  area={a:.0f} ({a/image_area*100:.1f}%)")

    # Try adaptive threshold approach
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours3, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large3 = [(cv2.contourArea(c), c) for c in contours3 if cv2.contourArea(c) >= min_area]
    print(f"\nAdaptive threshold: {len(large3)} large contours")
    quads3 = []
    for a, c in large3:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quads3.append((a, approx))
    print(f"  Quads: {len(quads3)}")
    
    # Now run actual detector
    cards = detect_cards(image)
    print(f"\nDetector result: {len(cards)} cards detected")
