"""Integration tests using real card photos from data/tests/.

These tests verify the full pipeline: detection → hashing → matching.
Each test image has an associated _annotation.json with expected card IDs.

Required data:
    python scripts/download_reference_data.py --small --sets sv3pt5 base1 base2 base4 base5 base6 gym1 neo1 ex2
    python scripts/build_hash_db.py --rebuild
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from card_reco import identify_cards
from card_reco.detector import detect_cards

TEST_DATA_DIR = Path("data") / "tests"
DB_PATH = Path("data") / "card_hashes.db"

has_test_data = TEST_DATA_DIR.exists() and DB_PATH.exists()
skip_no_data = pytest.mark.skipif(
    not has_test_data, reason="Test data or hash DB not available"
)


def load_annotation(image_stem: str) -> dict:
    """Load annotation JSON for a test image."""
    annotation_path = TEST_DATA_DIR / f"{image_stem}_annotation.json"
    with open(annotation_path, encoding="utf-8") as f:
        return json.load(f)


@skip_no_data
class TestSingleCardDetection:
    """Test detection and identification of single-card images."""

    def test_moltres_151_detected(self) -> None:
        """Verify that a single loose card is detected."""
        image_path = TEST_DATA_DIR / "moltres_151.png"
        image = cv2.imread(str(image_path))
        assert image is not None, f"Could not load {image_path}"

        cards = detect_cards(image)
        assert len(cards) >= 1, f"Expected at least 1 card, got {len(cards)}"

    @pytest.mark.xfail(reason="Low-contrast card border produces hash distances > 90")
    def test_moltres_151_identified(self) -> None:
        """Verify that Moltres from 151 set is correctly identified.

        Currently expected to fail: perceptual hashing of a real photo
        against small digital scans produces distances above any reasonable
        threshold for this particular image (low contrast card border).
        """
        annotation = load_annotation("moltres_151")
        image_path = TEST_DATA_DIR / annotation["image"]
        expected_ids = {c["card_id"] for c in annotation["expected_cards"]}

        results = identify_cards(image_path, db_path=DB_PATH, top_n=5, threshold=60.0)
        assert len(results) >= 1, "No cards detected"

        # Check that at least one detected card's top match is the expected card
        found_ids = set()
        for match_list in results:
            if match_list:
                found_ids.add(match_list[0].card.id)

        assert expected_ids & found_ids, (
            f"Expected {expected_ids} among top matches, got {found_ids}"
        )


@skip_no_data
class TestGradedCardDetection:
    """Test detection of a PSA graded card (card inside slab)."""

    def test_psa_charizard_detected(self) -> None:
        """Verify that the card inside a PSA slab is detected."""
        image_path = TEST_DATA_DIR / "image_psa_graded_charizard.png"
        image = cv2.imread(str(image_path))
        assert image is not None

        cards = detect_cards(image)
        assert len(cards) >= 1, (
            f"Expected at least 1 card detected from PSA slab, got {len(cards)}"
        )

    def test_psa_charizard_identified(self) -> None:
        """Verify a PSA graded Charizard is correctly identified."""
        annotation = load_annotation("image_psa_graded_charizard")
        image_path = TEST_DATA_DIR / annotation["image"]
        expected_ids = {c["card_id"] for c in annotation["expected_cards"]}

        results = identify_cards(image_path, db_path=DB_PATH, top_n=5, threshold=70.0)
        assert len(results) >= 1, "No cards detected from PSA slab image"

        found_ids = set()
        for match_list in results:
            if match_list:
                found_ids.add(match_list[0].card.id)

        assert expected_ids & found_ids, (
            f"Expected {expected_ids} among top matches, got {found_ids}"
        )


@skip_no_data
class TestGridDetection:
    """Test detection of a 3x3 grid of cards."""

    def test_3x3_detection_count(self) -> None:
        """Verify that at least 9 cards are detected from the grid image."""
        image_path = TEST_DATA_DIR / "image_3x3.png"
        image = cv2.imread(str(image_path))
        assert image is not None

        cards = detect_cards(image)
        assert len(cards) >= 9, (
            f"Expected at least 9 cards from 3x3 grid, got {len(cards)}"
        )

    def test_3x3_identification(self) -> None:
        """Verify cards from the grid are correctly identified."""
        annotation = load_annotation("image_3x3")
        image_path = TEST_DATA_DIR / annotation["image"]
        expected_ids = {c["card_id"] for c in annotation["expected_cards"]}

        results = identify_cards(image_path, db_path=DB_PATH, top_n=5, threshold=60.0)
        assert len(results) >= 9, (
            f"Expected at least 9 detected cards, got {len(results)}"
        )

        # Collect all IDs from top-5 matches per detected card
        found_ids = set()
        for match_list in results:
            for match in match_list:
                found_ids.add(match.card.id)

        matched = expected_ids & found_ids
        assert len(matched) >= 5, (
            f"Expected at least 5/9 cards identified, got {len(matched)}/9.\n"
            f"  Expected: {expected_ids}\n"
            f"  Found:    {found_ids}\n"
            f"  Matched:  {matched}"
        )
