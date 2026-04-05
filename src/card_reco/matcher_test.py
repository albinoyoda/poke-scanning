"""Unit tests for matcher helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from card_reco.matcher import _accept_by_consensus_or_separation, _build_name_groups
from card_reco.models import CardRecord


def _card(name: str, card_id: str = "test-1") -> CardRecord:
    return CardRecord(
        id=card_id,
        name=name,
        set_id="test",
        set_name="Test",
        number="1",
        rarity="Common",
        image_path="x.png",
        ahash="0" * 64,
        phash="0" * 64,
        dhash="0" * 64,
    )


class TestBuildNameGroups:
    """Tests for _build_name_groups."""

    def test_single_card(self):
        cards = [_card("Pikachu")]
        combined = np.array([10.0])
        groups = _build_name_groups(cards, combined, headroom_limit=50.0)
        assert "Pikachu" in groups
        best_dist, best_idx, n_close = groups["Pikachu"]
        assert best_dist == pytest.approx(10.0)
        assert best_idx == 0
        assert n_close == 1

    def test_two_names(self):
        cards = [_card("Pikachu", "p1"), _card("Charizard", "c1")]
        combined = np.array([5.0, 15.0])
        groups = _build_name_groups(cards, combined, headroom_limit=50.0)
        assert len(groups) == 2
        assert groups["Pikachu"][0] == pytest.approx(5.0)
        assert groups["Charizard"][0] == pytest.approx(15.0)

    def test_same_name_variants(self):
        """Multiple cards with the same name are grouped."""
        cards = [
            _card("Electivire ex", "e1"),
            _card("Electivire ex", "e2"),
            _card("Pikachu", "p1"),
        ]
        combined = np.array([10.0, 20.0, 50.0])
        groups = _build_name_groups(cards, combined, headroom_limit=30.0)
        assert groups["Electivire ex"][0] == pytest.approx(10.0)
        assert groups["Electivire ex"][2] == 2  # both within headroom

    def test_variant_outside_headroom(self):
        """Variants above headroom_limit are not counted."""
        cards = [
            _card("Electivire ex", "e1"),
            _card("Electivire ex", "e2"),
        ]
        combined = np.array([10.0, 100.0])
        groups = _build_name_groups(cards, combined, headroom_limit=50.0)
        assert groups["Electivire ex"][2] == 1  # only one within headroom

    def test_best_index_tracks_minimum(self):
        cards = [_card("A", "a1"), _card("A", "a2"), _card("A", "a3")]
        combined = np.array([30.0, 10.0, 20.0])
        groups = _build_name_groups(cards, combined, headroom_limit=50.0)
        assert groups["A"][1] == 1  # index of card with dist=10.0


class TestAcceptByConsensusOrSeparation:
    """Tests for _accept_by_consensus_or_separation."""

    def test_consensus_accepts(self):
        """Enough close variants → accept."""
        assert (
            _accept_by_consensus_or_separation(
                n_close=3,
                min_consensus=2,
                best_dist=10.0,
                second_best_dist=15.0,
                min_separation=20.0,
            )
            is True
        )

    def test_separation_accepts(self):
        """Large gap to runner-up → accept."""
        assert (
            _accept_by_consensus_or_separation(
                n_close=1,
                min_consensus=2,
                best_dist=10.0,
                second_best_dist=40.0,
                min_separation=15.0,
            )
            is True
        )

    def test_neither_rejects(self):
        """No consensus and small gap → reject."""
        assert (
            _accept_by_consensus_or_separation(
                n_close=1,
                min_consensus=2,
                best_dist=10.0,
                second_best_dist=12.0,
                min_separation=15.0,
            )
            is False
        )

    def test_no_runner_up_no_consensus(self):
        """Single name in DB, below consensus → reject."""
        assert (
            _accept_by_consensus_or_separation(
                n_close=1,
                min_consensus=2,
                best_dist=10.0,
                second_best_dist=None,
                min_separation=15.0,
            )
            is False
        )

    def test_no_runner_up_with_consensus(self):
        """Single name in DB, meets consensus → accept."""
        assert (
            _accept_by_consensus_or_separation(
                n_close=3,
                min_consensus=2,
                best_dist=10.0,
                second_best_dist=None,
                min_separation=15.0,
            )
            is True
        )
