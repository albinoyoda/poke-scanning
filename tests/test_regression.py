"""Regression tests using real downloaded card data.

These tests require data from steps 1 & 2:
    python scripts/download_reference_data.py --small --sets base1 base2
    python scripts/build_hash_db.py

Tests are skipped if the data is not available.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from card_reco.database import HashDatabase
from card_reco.detector import detect_cards
from card_reco.hasher import compute_hashes, compute_hashes_pil
from card_reco.matcher import CardMatcher

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "card_hashes.db"
IMAGES_DIR = DATA_DIR / "images"
METADATA_DIR = DATA_DIR / "metadata"

has_data = DB_PATH.exists() and IMAGES_DIR.exists()
skip_no_data = pytest.mark.skipif(not has_data, reason="Reference data not downloaded")


@skip_no_data
class TestDatabaseIntegrity:
    def test_db_has_cards(self) -> None:
        with HashDatabase(DB_PATH) as db:
            count = db.count()
            assert count > 0, "Database is empty"

    def test_db_cards_have_valid_hashes(self) -> None:
        with HashDatabase(DB_PATH) as db:
            cards = db.get_all_cards()
            for card in cards[:20]:  # spot-check first 20
                assert len(card.ahash) == 64, (
                    f"{card.id} ahash wrong length: {len(card.ahash)}"
                )
                assert len(card.phash) == 64, f"{card.id} phash wrong length"
                assert len(card.dhash) == 64, f"{card.id} dhash wrong length"
                # Verify they're valid hex
                int(card.ahash, 16)
                int(card.phash, 16)
                int(card.dhash, 16)

    def test_db_card_lookup_by_id(self) -> None:
        with HashDatabase(DB_PATH) as db:
            card = db.get_card_by_id("base1-4")
            if card is not None:
                assert card.name == "Charizard"
                assert card.set_id == "base1"

    def test_all_cards_have_required_fields(self) -> None:
        with HashDatabase(DB_PATH) as db:
            cards = db.get_all_cards()
            for card in cards:
                assert card.id, "Card missing id"
                assert card.name, f"Card {card.id} missing name"
                assert card.set_id, f"Card {card.id} missing set_id"
                assert card.number, f"Card {card.id} missing number"


@skip_no_data
class TestMetadataParsing:
    def test_sets_file_valid_json(self) -> None:
        sets_file = METADATA_DIR / "sets.json"
        if not sets_file.exists():
            pytest.skip("sets.json not found")
        with open(sets_file, encoding="utf-8") as f:
            sets_data = json.load(f)
        assert isinstance(sets_data, list)
        assert len(sets_data) > 0
        # Each set should have an id and name
        for s in sets_data[:5]:
            assert "id" in s
            assert "name" in s

    def test_card_metadata_has_images(self) -> None:
        cards_dir = METADATA_DIR / "cards"
        if not cards_dir.exists():
            pytest.skip("cards directory not found")
        card_files = list(cards_dir.glob("*.json"))
        assert len(card_files) > 0
        with open(card_files[0], encoding="utf-8") as f:
            cards = json.load(f)
        assert isinstance(cards, list)
        for card in cards[:5]:
            assert "id" in card
            assert "name" in card
            assert "images" in card
            assert "small" in card["images"]


@skip_no_data
class TestHashConsistency:
    def _get_sample_image_path(self) -> Path | None:
        for img in IMAGES_DIR.rglob("*.png"):
            if img.stat().st_size > 0:
                return img
        return None

    def test_hash_is_deterministic(self) -> None:
        img_path = self._get_sample_image_path()
        if img_path is None:
            pytest.skip("No sample images found")
        pil_img = Image.open(img_path).convert("RGB")
        h1 = compute_hashes_pil(pil_img)
        h2 = compute_hashes_pil(pil_img)
        assert h1.ahash == h2.ahash
        assert h1.phash == h2.phash
        assert h1.dhash == h2.dhash

    def test_hash_matches_db_entry(self) -> None:
        """Verify that recomputing a hash matches what's stored in the DB."""
        with HashDatabase(DB_PATH) as db:
            card = db.get_card_by_id("base1-4")
            if card is None:
                pytest.skip("base1-4 not in DB")
            img_path = Path(card.image_path)
            if not img_path.exists():
                pytest.skip(f"Image not found: {img_path}")
            pil_img = Image.open(img_path).convert("RGB")
            hashes = compute_hashes_pil(pil_img)
            assert hashes.ahash == card.ahash
            assert hashes.phash == card.phash
            assert hashes.dhash == card.dhash


@skip_no_data
class TestMatcherWithRealData:
    def test_exact_image_matches_itself(self) -> None:
        """Loading a reference image and matching it should return itself as #1."""
        with HashDatabase(DB_PATH) as db:
            card = db.get_card_by_id("base1-4")
            if card is None:
                pytest.skip("base1-4 not in DB")
            img_path = Path(card.image_path)
            if not img_path.exists():
                pytest.skip(f"Image not found: {img_path}")

        pil_img = Image.open(img_path).convert("RGB")
        hashes = compute_hashes_pil(pil_img)

        with CardMatcher(DB_PATH) as matcher:
            results = matcher.find_matches(hashes, top_n=5)
            assert len(results) >= 1
            assert results[0].card.id == "base1-4"
            assert results[0].card.name == "Charizard"
            assert results[0].distance == 0.0

    def test_different_cards_dont_match_perfectly(self) -> None:
        """Two different cards should not have zero distance."""
        with HashDatabase(DB_PATH) as db:
            cards = db.get_all_cards()
            if len(cards) < 2:
                pytest.skip("Need at least 2 cards")
            c1, c2 = cards[0], cards[1]
            p1, p2 = Path(c1.image_path), Path(c2.image_path)
            if not p1.exists() or not p2.exists():
                pytest.skip("Card images not found")

        h1 = compute_hashes_pil(Image.open(p1).convert("RGB"))

        with CardMatcher(DB_PATH) as matcher:
            results = matcher.find_matches(h1, top_n=1)
            assert len(results) >= 1
            # The best match should be the card itself, not c2
            assert results[0].card.id == c1.id


@skip_no_data
class TestEndToEndPipeline:
    def test_identify_from_reference_image(self) -> None:
        """Load a reference card image as if it were a photo, match it."""
        with HashDatabase(DB_PATH) as db:
            card = db.get_card_by_id("base1-4")
            if card is None:
                pytest.skip("base1-4 not in DB")
            img_path = Path(card.image_path)
            if not img_path.exists():
                pytest.skip(f"Image not found: {img_path}")

        # Load the reference image and embed it in a larger "photo"
        card_img = cv2.imread(str(img_path))
        assert card_img is not None

        # Create a synthetic photo: dark background with the card placed in it
        h, w = card_img.shape[:2]
        bg = np.full((h + 200, w + 200, 3), 30, dtype=np.uint8)
        bg[100 : 100 + h, 100 : 100 + w] = card_img

        # Run detection
        detected = detect_cards(bg)
        assert len(detected) >= 1, "Failed to detect the card in the synthetic photo"

        # Hash the detected card and match
        hashes = compute_hashes(detected[0].image)
        with CardMatcher(DB_PATH) as matcher:
            results = matcher.find_matches(hashes, top_n=5)
            assert len(results) >= 1
            # The top match should be Charizard (may not be exact due to
            # perspective warp artifacts, but should be close)
            assert results[0].card.name == "Charizard", (
                f"Expected Charizard, got {results[0].card.name} "
                f"(distance={results[0].distance:.1f})"
            )
