"""Integration tests using real card photos from data/tests/.

These tests verify the full pipeline: detection → hashing → matching.
Each subfolder has an _annotations.json with expected card IDs.

Required data:
    python scripts/download_reference_data.py --small \
        --sets sv3pt5 base1 base2 base4 base5 base6 gym1 neo1 ex2 \
               basep ex1 ex6 neo4 cel25 swshp
    python scripts/build_hash_db.py --rebuild
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from card_reco import identify_cards
from card_reco.detector import detect_cards

TEST_DATA_DIR = Path("data") / "tests"
DB_PATH = Path("data") / "card_hashes.db"

SINGLE_AXIS_ALIGNED_DIR = TEST_DATA_DIR / "single_cards" / "axis_aligned"
SINGLE_TILTED_DIR = TEST_DATA_DIR / "single_cards" / "tilted"
GRADED_DIR = TEST_DATA_DIR / "graded_cards"
MULTIPLE_DIR = TEST_DATA_DIR / "multiple_cards"

has_test_data = TEST_DATA_DIR.exists() and DB_PATH.exists()
skip_no_data = pytest.mark.skipif(
    not has_test_data, reason="Test data or hash DB not available"
)


def load_folder_annotations(folder: Path) -> dict:
    """Load _annotations.json from a test subfolder."""
    annotation_path = folder / "_annotations.json"
    with open(annotation_path, encoding="utf-8") as f:
        return json.load(f)


def get_card_id(folder: Path, image_name: str) -> str | None:
    """Look up the card_id for an image from the folder annotations."""
    data = load_folder_annotations(folder)
    for entry in data.get("annotations", data.get("expected_cards", [])):
        if entry["image"] == image_name:
            return entry["card_id"]
    return None


def _collect_match_ids(results: list) -> set[str]:
    """Collect all card IDs from nested match results."""
    ids: set[str] = set()
    for match_list in results:
        for match in match_list:
            ids.add(match.card.id)
    return ids


@skip_no_data
class TestSingleCardDetection:
    """Test detection and identification of single-card images."""

    def test_moltres_151_detected(self) -> None:
        """Verify that a single loose card is detected."""
        image_path = SINGLE_TILTED_DIR / "moltres_151.png"
        image = cv2.imread(str(image_path))
        assert image is not None, f"Could not load {image_path}"

        cards = detect_cards(np.asarray(image, dtype=np.uint8))
        assert len(cards) >= 1, f"Expected at least 1 card, got {len(cards)}"

    @pytest.mark.xfail(reason="Low-contrast card border produces hash distances > 90")
    def test_moltres_151_identified(self) -> None:
        """Verify that Moltres from 151 set is correctly identified.

        Currently expected to fail: perceptual hashing of a real photo
        against small digital scans produces distances above any reasonable
        threshold for this particular image (low contrast card border).
        """
        expected_id = get_card_id(SINGLE_TILTED_DIR, "moltres_151.png")
        assert expected_id is not None
        image_path = SINGLE_TILTED_DIR / "moltres_151.png"
        expected_ids = {expected_id}

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
        image_path = GRADED_DIR / "charizard_sl_graded.png"
        image = cv2.imread(str(image_path))
        assert image is not None

        cards = detect_cards(np.asarray(image, dtype=np.uint8))
        assert len(cards) >= 1, (
            f"Expected at least 1 card detected from PSA slab, got {len(cards)}"
        )

    def test_psa_charizard_identified(self) -> None:
        """Verify a PSA graded Charizard is correctly identified."""
        annotations = load_folder_annotations(GRADED_DIR)
        expected_ids = {c["card_id"] for c in annotations["expected_cards"]}
        image_path = GRADED_DIR / "charizard_sl_graded.png"

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
        image_path = MULTIPLE_DIR / "3x3_top_loaders.png"
        image = cv2.imread(str(image_path))
        assert image is not None

        cards = detect_cards(np.asarray(image, dtype=np.uint8))
        assert len(cards) >= 9, (
            f"Expected at least 9 cards from 3x3 grid, got {len(cards)}"
        )

    def test_3x3_identification(self) -> None:
        """Verify cards from the grid are correctly identified."""
        annotations = load_folder_annotations(MULTIPLE_DIR)
        image_entry = annotations["images"][0]
        image_path = MULTIPLE_DIR / image_entry["image"]
        expected_ids = {c["card_id"] for c in image_entry["cards"]}

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


@skip_no_data
class TestAxisAlignedSingleCards:
    """Test detection and identification of axis-aligned single-card images.

    These are the simplest case: cards photographed straight-on with minimal
    perspective distortion or rotation.

    Current identification status (combined hash distance):
      - pikachu-v-swsh061:        ~6   (excellent)
      - birthday-pikachu-24:      ~50  (good)
      - shining_kabutops_108_105: ~57  (good)
      - treecko_75_109:           ~61  (marginal — best detection is card #1)
      - shining_raichu:           ~73  (poor — photo-vs-scan gap)
      - xerneas_12_25:            ~84  (poor — photo-vs-scan gap)
      - pidgeot_12_112:           ~107 (fail — detector picks wrong region)
    """

    @pytest.fixture()
    def annotations(self) -> list[dict]:
        """Load annotations for axis-aligned test cards."""
        data = load_folder_annotations(SINGLE_AXIS_ALIGNED_DIR)
        return data["annotations"]

    @pytest.mark.parametrize(
        "image_name",
        [
            "birthday-pikachu-24.png",
            "pidgeot_12_112.png",
            "pikachu-v-swsh061.png",
            "shining_kabutops_108_105.png",
            "shining_raichu.png",
            "treecko_75_109.png",
            "xerneas_12_25.png",
        ],
    )
    def test_card_detected(self, image_name: str) -> None:
        """Verify that a single axis-aligned card is detected."""
        image_path = SINGLE_AXIS_ALIGNED_DIR / image_name
        image = cv2.imread(str(image_path))
        assert image is not None, f"Could not load {image_path}"

        cards = detect_cards(np.asarray(image, dtype=np.uint8))
        assert len(cards) >= 1, (
            f"Expected at least 1 card in {image_name}, got {len(cards)}"
        )

    @pytest.mark.parametrize(
        "image_name",
        [
            "birthday-pikachu-24.png",
            "pikachu-v-swsh061.png",
            "shining_kabutops_108_105.png",
        ],
    )
    def test_card_identified(self, image_name: str) -> None:
        """Cards that reliably match within threshold 60."""
        expected_id = get_card_id(SINGLE_AXIS_ALIGNED_DIR, image_name)
        assert expected_id is not None

        image_path = SINGLE_AXIS_ALIGNED_DIR / image_name
        results = identify_cards(image_path, db_path=DB_PATH, top_n=5, threshold=60.0)
        assert len(results) >= 1, f"No cards detected in {image_name}"

        found_ids = _collect_match_ids(results)
        assert expected_id in found_ids, (
            f"{image_name}: expected {expected_id} in top-5 matches, got {found_ids}"
        )

    @pytest.mark.parametrize(
        ("image_name", "reason"),
        [
            pytest.param(
                "treecko_75_109.png",
                "Combined distance ~61 just above default threshold",
            ),
            pytest.param(
                "shining_raichu.png",
                "Combined distance ~73, photo-vs-scan gap too large",
            ),
            pytest.param(
                "xerneas_12_25.png",
                "Combined distance ~84, photo-vs-scan gap too large",
            ),
            pytest.param(
                "pidgeot_12_112.png",
                "Combined distance ~107, detector favours wrong region",
            ),
        ],
    )
    @pytest.mark.xfail(
        reason="Hash distance exceeds threshold for photo-vs-scan comparison"
    )
    def test_card_identified_xfail(self, image_name: str, reason: str) -> None:
        """Cards that currently fail identification — documents known gaps."""
        expected_id = get_card_id(SINGLE_AXIS_ALIGNED_DIR, image_name)
        assert expected_id is not None, reason

        image_path = SINGLE_AXIS_ALIGNED_DIR / image_name
        results = identify_cards(image_path, db_path=DB_PATH, top_n=5, threshold=60.0)
        assert len(results) >= 1, f"No cards detected in {image_name}"

        found_ids = _collect_match_ids(results)
        assert expected_id in found_ids, (
            f"{image_name}: expected {expected_id} in top-5, got {found_ids}. "
            f"Reason: {reason}"
        )

    def test_all_axis_aligned_detected(self, annotations: list[dict]) -> None:
        """Verify all axis-aligned images produce at least one detection."""
        failures: list[str] = []
        for entry in annotations:
            image_path = SINGLE_AXIS_ALIGNED_DIR / entry["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                failures.append(f"{entry['image']}: could not load")
                continue
            cards = detect_cards(np.asarray(image, dtype=np.uint8))
            if len(cards) < 1:
                failures.append(f"{entry['image']}: no cards detected")
        assert not failures, "Detection failures:\n" + "\n".join(failures)

    def test_axis_aligned_identification_rate(self, annotations: list[dict]) -> None:
        """At least 3 of 7 axis-aligned cards are identified (current baseline)."""
        matched = 0
        for entry in annotations:
            image_path = SINGLE_AXIS_ALIGNED_DIR / entry["image"]
            results = identify_cards(
                image_path, db_path=DB_PATH, top_n=5, threshold=60.0
            )
            found_ids = _collect_match_ids(results)
            if entry["card_id"] in found_ids:
                matched += 1

        total = len(annotations)
        assert matched >= 3, (
            f"Identified {matched}/{total} axis-aligned cards, expected >= 3"
        )
