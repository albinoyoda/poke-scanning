"""Integration tests for the CNN embedding backend.

These tests mirror the structure of test_integration.py but use the CNN
backend (MobileNetV3-Small + FAISS cosine similarity).

Required data:
    python scripts/export_model.py
    python scripts/build_embedding_db.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from card_reco import identify_cards
from card_reco.detector import detect_cards
from tests.test_integration import (
    GRADED_DIR,
    MULTIPLE_DIR,
    SINGLE_AXIS_ALIGNED_DIR,
    SINGLE_ROTATED_DIR,
    SINGLE_TILTED_DIR,
    _collect_match_ids,
    get_card_id,
    load_folder_annotations,
)

INDEX_PATH = Path("data") / "card_embeddings.faiss"
MODEL_PATH = Path("data") / "mobilenet_v3_small.onnx"

has_cnn_data = (
    INDEX_PATH.exists() and MODEL_PATH.exists() and (Path("data") / "tests").exists()
)
skip_no_cnn = pytest.mark.skipif(
    not has_cnn_data, reason="CNN model/index or test data not available"
)


# ------------------------------------------------------------------
# Axis-aligned single cards
# ------------------------------------------------------------------


@skip_no_cnn
class TestCNNAxisAligned:  # pylint: disable=too-few-public-methods
    """CNN identification of axis-aligned single-card images.

    All axis-aligned single-card images are correctly identified
    after fine-tuning with contrastive learning.
    """

    @pytest.mark.parametrize(
        "image_name",
        [
            "pikachu-v-swsh061.png",
            "xerneas_12_25.png",
            "electivire_ex_69_182.png",
            "jolteon_gold_star.png",
            "regirock_gold_star.png",
            "birthday-pikachu-24.png",
            "gengar_skyridge_h9.png",
            "treecko_75_109.png",
            "shining_kabutops_108_105.png",
            "shining_raichu.png",
            "pidgeot_12_112.png",
            "wiggly_base_2.png",
        ],
    )
    def test_card_identified(self, image_name: str) -> None:
        """All axis-aligned cards match via CNN backend."""
        expected_id = get_card_id(SINGLE_AXIS_ALIGNED_DIR, image_name)
        assert expected_id is not None

        image_path = SINGLE_AXIS_ALIGNED_DIR / image_name
        results = identify_cards(str(image_path), backend="cnn", top_n=5)
        assert len(results) >= 1, f"No cards detected in {image_name}"

        found_ids = _collect_match_ids(results)
        assert expected_id in found_ids, (
            f"{image_name}: expected {expected_id} in CNN top-5, got {found_ids}"
        )


# ------------------------------------------------------------------
# Rotated single cards
# ------------------------------------------------------------------


@skip_no_cnn
class TestCNNRotated:  # pylint: disable=too-few-public-methods
    """CNN identification of rotated single-card images.

    Calibrated:
      - sandslash_skyridge_non_holo_93_144: 0.943 (OK)
      - electivire_ex_69_182:              0.929 (OK)
    """

    @pytest.mark.parametrize(
        "image_name",
        [
            "sandslash_skyridge_non_holo_93_144.png",
            "electivire_ex_69_182.png",
        ],
    )
    def test_card_identified(self, image_name: str) -> None:
        expected_id = get_card_id(SINGLE_ROTATED_DIR, image_name)
        assert expected_id is not None

        image_path = SINGLE_ROTATED_DIR / image_name
        results = identify_cards(str(image_path), backend="cnn", top_n=5)
        assert len(results) >= 1

        found_ids = _collect_match_ids(results)
        assert expected_id in found_ids, (
            f"{image_name}: expected {expected_id} in CNN top-5, got {found_ids}"
        )


# ------------------------------------------------------------------
# Tilted single cards
# ------------------------------------------------------------------


@skip_no_cnn
class TestCNNTilted:  # pylint: disable=too-few-public-methods
    """CNN identification of tilted single-card images.

    All tilted single-card images are correctly identified
    after fine-tuning with center-crop exploration.
    """

    @pytest.mark.parametrize(
        "image_name",
        [
            "bayleef_neo_1st_ed.png",
            "exeggcute_102_165.png",
            "kingdra_neo.png",
            "alakazam_lc.png",
            "moltres_151.png",
        ],
    )
    def test_card_identified(self, image_name: str) -> None:
        expected_id = get_card_id(SINGLE_TILTED_DIR, image_name)
        assert expected_id is not None

        image_path = SINGLE_TILTED_DIR / image_name
        results = identify_cards(str(image_path), backend="cnn", top_n=5)
        assert len(results) >= 1

        found_ids = _collect_match_ids(results)
        assert expected_id in found_ids, (
            f"{image_name}: expected {expected_id} in CNN top-5, got {found_ids}"
        )


# ------------------------------------------------------------------
# Graded cards
# ------------------------------------------------------------------


@skip_no_cnn
class TestCNNGraded:  # pylint: disable=too-few-public-methods
    """CNN identification of PSA graded card."""

    def test_psa_charizard_identified(self) -> None:
        annotations = load_folder_annotations(GRADED_DIR)
        expected_ids = {c["card_id"] for c in annotations["expected_cards"]}
        image_path = GRADED_DIR / "charizard_sl_graded.png"

        results = identify_cards(str(image_path), backend="cnn", top_n=5)
        assert len(results) >= 1

        found_ids = _collect_match_ids(results)
        assert expected_ids & found_ids, (
            f"Expected {expected_ids} among CNN top-5, got {found_ids}"
        )


# ------------------------------------------------------------------
# Multi-card images
# ------------------------------------------------------------------


@skip_no_cnn
class TestCNN3x3Grid:
    """CNN identification of 3x3 grid of cards."""

    def test_3x3_detection_count(self) -> None:
        """Verify at least 9 cards are detected (detector is backend-agnostic)."""
        image_path = MULTIPLE_DIR / "3x3_top_loaders.png"
        image = cv2.imread(str(image_path))
        assert image is not None

        cards = detect_cards(np.asarray(image, dtype=np.uint8))
        assert len(cards) >= 9

    def test_3x3_cnn_identification(self) -> None:
        """Verify CNN backend finds all 9 cards by exact ID."""
        annotations = load_folder_annotations(MULTIPLE_DIR)
        image_entry = annotations["images"][0]
        image_path = MULTIPLE_DIR / image_entry["image"]
        expected_ids = {c["card_id"] for c in image_entry["cards"]}

        results = identify_cards(str(image_path), backend="cnn", top_n=5)
        found_ids = _collect_match_ids(results)
        matched = expected_ids & found_ids
        assert len(matched) >= 9, (
            f"CNN: expected 9/9 exact ID matches, "
            f"got {len(matched)}/9.\n"
            f"  Expected: {expected_ids}\n"
            f"  Matched:  {matched}"
        )
