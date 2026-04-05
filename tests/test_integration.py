"""Integration tests using real card photos from data/tests/.

These tests verify the full pipeline: detection → hashing → matching.
Each subfolder has an _annotations.json with expected card IDs.

Required data:
    python scripts/download_reference_data.py --small \
        --sets sv3pt5 base1 base2 base3 base4 base5 base6 gym1 neo1 ex2 \
               basep ex1 ex6 neo4 cel25 swshp sv10 ecard3 ex12 ex16
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
SINGLE_ROTATED_DIR = TEST_DATA_DIR / "single_cards" / "rotated"
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

    def test_moltres_151_identified(self) -> None:
        """Verify that Moltres from 151 set is correctly identified.

        The hash backend cannot match this card (distance > 90), but the
        fine-tuned CNN backend with center-crop exploration succeeds.
        """
        expected_id = get_card_id(SINGLE_TILTED_DIR, "moltres_151.png")
        assert expected_id is not None
        image_path = SINGLE_TILTED_DIR / "moltres_151.png"
        expected_ids = {expected_id}

        results = identify_cards(image_path, backend="cnn", top_n=5)
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
        assert len(matched) >= 8, (
            f"Expected at least 8/9 cards identified by exact ID, "
            f"got {len(matched)}/9.\n"
            f"  Expected: {expected_ids}\n"
            f"  Found:    {found_ids}\n"
            f"  Matched:  {matched}"
        )

    def test_3x3_identification_by_name(self) -> None:
        """Verify all 9 grid cards are identified by name.

        Some cards may match a different set variant (e.g. Jungle vs
        Base Set 2 Wigglytuff), which is acceptable when the matched
        name is correct.  The name-consensus fallback enables this for
        cards with high photo-vs-scan hash distances.
        """
        annotations = load_folder_annotations(MULTIPLE_DIR)
        image_entry = annotations["images"][0]
        image_path = MULTIPLE_DIR / image_entry["image"]
        expected_names = {c["name"] for c in image_entry["cards"]}

        results = identify_cards(image_path, db_path=DB_PATH, top_n=5, threshold=60.0)

        found_names: set[str] = set()
        for match_list in results:
            if match_list:
                found_names.add(match_list[0].card.name)

        matched_names = expected_names & found_names
        assert len(matched_names) >= 9, (
            f"Expected 9/9 cards identified by name, got {len(matched_names)}/9.\n"
            f"  Expected: {expected_names}\n"
            f"  Found:    {found_names}\n"
            f"  Matched:  {matched_names}"
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
      - pidgeot_12_112:           ~54  (good — improved by area-diverse NMS)
      - shining_raichu:           ~58  (good — improved by area-diverse NMS)
      - treecko_75_109:           ~55  (good — improved by quality scoring)
      - electivire_ex_69_182:     ~78  (via relaxed + whole-image fallback)
      - wiggly_base_2:            ~82  (via consensus + whole-image fallback)
      - xerneas_12_25:            ~84  (poor — photo-vs-scan gap)
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
            "electivire_ex_69_182.png",
            "pidgeot_12_112.png",
            "pikachu-v-swsh061.png",
            "shining_kabutops_108_105.png",
            "shining_raichu.png",
            "treecko_75_109.png",
            "wiggly_base_2.png",
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
            "electivire_ex_69_182.png",
            "pidgeot_12_112.png",
            "pikachu-v-swsh061.png",
            "shining_kabutops_108_105.png",
            "shining_raichu.png",
            "treecko_75_109.png",
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
                "xerneas_12_25.png",
                "Hash distance ~84 too large; CNN backend succeeds",
            ),
        ],
    )
    def test_card_identified_cnn_fallback(self, image_name: str, reason: str) -> None:
        """Cards where hash matching fails but CNN backend succeeds."""
        expected_id = get_card_id(SINGLE_AXIS_ALIGNED_DIR, image_name)
        assert expected_id is not None, reason

        image_path = SINGLE_AXIS_ALIGNED_DIR / image_name
        results = identify_cards(image_path, backend="cnn", top_n=5)
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
        """At least 6 of 9 axis-aligned cards are identified (current baseline)."""
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
        assert matched >= 7, (
            f"Identified {matched}/{total} axis-aligned cards, expected >= 7"
        )

    @pytest.mark.parametrize(
        ("image_name", "expected_name"),
        [
            ("electivire_ex_69_182.png", "Electivire ex"),
            ("wiggly_base_2.png", "Wigglytuff"),
        ],
    )
    def test_cropout_identified_by_name(
        self, image_name: str, expected_name: str
    ) -> None:
        """Pre-cropped card photos match the correct card by name.

        These are debug-pipeline crop-outs with heavy holo glare.
        They are identified via the whole-image fallback combined with
        relaxed matching (clear-winner separation or name consensus).
        The exact set variant may differ from the annotation since the
        holo pattern shifts hash distances between reprints.
        """
        image_path = SINGLE_AXIS_ALIGNED_DIR / image_name
        results = identify_cards(image_path, db_path=DB_PATH, top_n=5, threshold=60.0)
        assert len(results) >= 1, f"No results for {image_name}"

        found_names: set[str] = set()
        for match_list in results:
            if match_list:
                found_names.add(match_list[0].card.name)

        assert expected_name in found_names, (
            f"{image_name}: expected '{expected_name}' among top matches, "
            f"got {found_names}"
        )


@skip_no_data
class TestRotatedSingleCards:
    """Test detection and identification of rotated single-card images.

    These cards are photographed at non-trivial in-plane rotation angles.
    The detector must find the card contour despite rotation and produce
    a correctly warped portrait image.

        Current status:
            - sandslash_skyridge_non_holo_93_144: ~25  (excellent)
            - electivire_ex_69_182:              ~78  (passes via clear-winner fallback)
    """

    @pytest.fixture()
    def annotations(self) -> list[dict]:
        """Load annotations for rotated test cards."""
        data = load_folder_annotations(SINGLE_ROTATED_DIR)
        return data["annotations"]

    @pytest.mark.parametrize(
        "image_name",
        [
            "electivire_ex_69_182.png",
            "sandslash_skyridge_non_holo_93_144.png",
        ],
    )
    def test_card_detected(self, image_name: str) -> None:
        """Verify that at least one region is detected in a rotated card image."""
        image_path = SINGLE_ROTATED_DIR / image_name
        image = cv2.imread(str(image_path))
        assert image is not None, f"Could not load {image_path}"

        cards = detect_cards(np.asarray(image, dtype=np.uint8))
        assert len(cards) >= 1, (
            f"Expected at least 1 card in {image_name}, got {len(cards)}"
        )

    def test_sandslash_identified(self) -> None:
        """Sandslash rotated card matches within threshold 60."""
        image_name = "sandslash_skyridge_non_holo_93_144.png"
        expected_id = get_card_id(SINGLE_ROTATED_DIR, image_name)
        assert expected_id is not None

        image_path = SINGLE_ROTATED_DIR / image_name
        results = identify_cards(image_path, db_path=DB_PATH, top_n=5, threshold=60.0)
        assert len(results) >= 1, f"No cards detected in {image_name}"

        found_ids = _collect_match_ids(results)
        assert expected_id in found_ids, (
            f"{image_name}: expected {expected_id} in top-5 matches, got {found_ids}"
        )

    def test_electivire_identified(self) -> None:
        """Electivire rotated card is identified despite textured background."""
        image_name = "electivire_ex_69_182.png"
        expected_id = get_card_id(SINGLE_ROTATED_DIR, image_name)
        assert expected_id is not None

        image_path = SINGLE_ROTATED_DIR / image_name
        results = identify_cards(image_path, db_path=DB_PATH, top_n=5, threshold=60.0)
        assert len(results) >= 1, f"No cards detected in {image_name}"

        found_ids = _collect_match_ids(results)
        assert expected_id in found_ids, (
            f"{image_name}: expected {expected_id} in top-5, got {found_ids}"
        )


@skip_no_data
class TestTiltedSingleCards:
    """Test detection and identification of tilted (perspective distorted) cards.

    These cards are photographed at an angle, producing trapezoidal
    projections. The detector must find the card contour and apply a
    perspective transform to normalise the card.

    Current identification status (combined hash distance):
      - kingdra_neo:           ~15  (excellent)
      - alakazam_lc:           ~20  (excellent)
      - bayleef_neo_1st_ed:    ~36  (good)
      - exeggcute_102_165:     ~55  (good — improved by area-diverse NMS)
      - moltres_151:           ~91  (fail — low-contrast border, wrong region)
    """

    @pytest.fixture()
    def annotations(self) -> list[dict]:
        """Load annotations for tilted test cards."""
        data = load_folder_annotations(SINGLE_TILTED_DIR)
        return data["annotations"]

    @pytest.mark.parametrize(
        "image_name",
        [
            "alakazam_lc.png",
            "bayleef_neo_1st_ed.png",
            "exeggcute_102_165.png",
            "kingdra_neo.png",
            "moltres_151.png",
        ],
    )
    def test_card_detected(self, image_name: str) -> None:
        """Verify that a tilted card is detected."""
        image_path = SINGLE_TILTED_DIR / image_name
        image = cv2.imread(str(image_path))
        assert image is not None, f"Could not load {image_path}"

        cards = detect_cards(np.asarray(image, dtype=np.uint8))
        assert len(cards) >= 1, (
            f"Expected at least 1 card in {image_name}, got {len(cards)}"
        )

    @pytest.mark.parametrize(
        "image_name",
        [
            "alakazam_lc.png",
            "bayleef_neo_1st_ed.png",
            "exeggcute_102_165.png",
            "kingdra_neo.png",
        ],
    )
    def test_card_identified(self, image_name: str) -> None:
        """Tilted cards that reliably match within threshold 60."""
        expected_id = get_card_id(SINGLE_TILTED_DIR, image_name)
        assert expected_id is not None

        image_path = SINGLE_TILTED_DIR / image_name
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
                "moltres_151.png",
                "Hash distance ~91 too large; CNN backend succeeds",
            ),
        ],
    )
    def test_card_identified_cnn_fallback(self, image_name: str, reason: str) -> None:
        """Tilted cards where hash matching fails but CNN succeeds."""
        expected_id = get_card_id(SINGLE_TILTED_DIR, image_name)
        assert expected_id is not None, reason

        image_path = SINGLE_TILTED_DIR / image_name
        results = identify_cards(image_path, backend="cnn", top_n=5)
        assert len(results) >= 1, f"No cards detected in {image_name}"

        found_ids = _collect_match_ids(results)
        assert expected_id in found_ids, (
            f"{image_name}: expected {expected_id} in top-5, got {found_ids}. "
            f"Reason: {reason}"
        )

    def test_all_tilted_detected(self, annotations: list[dict]) -> None:
        """Verify all tilted images produce at least one detection."""
        failures: list[str] = []
        for entry in annotations:
            image_path = SINGLE_TILTED_DIR / entry["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                failures.append(f"{entry['image']}: could not load")
                continue
            cards = detect_cards(np.asarray(image, dtype=np.uint8))
            if len(cards) < 1:
                failures.append(f"{entry['image']}: no cards detected")
        assert not failures, "Detection failures:\n" + "\n".join(failures)

    def test_tilted_identification_rate(self, annotations: list[dict]) -> None:
        """At least 3 of 5 tilted cards are identified (current baseline)."""
        matched = 0
        for entry in annotations:
            image_path = SINGLE_TILTED_DIR / entry["image"]
            results = identify_cards(
                image_path, db_path=DB_PATH, top_n=5, threshold=60.0
            )
            found_ids = _collect_match_ids(results)
            if entry["card_id"] in found_ids:
                matched += 1

        total = len(annotations)
        assert matched >= 4, f"Identified {matched}/{total} tilted cards, expected >= 4"
