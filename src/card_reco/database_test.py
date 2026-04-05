"""Unit tests for database helpers added in Phase 1."""

from __future__ import annotations

import tempfile
from pathlib import Path

from card_reco.database import HashDatabase
from card_reco.models import CardHashes


def _sample_hashes() -> CardHashes:
    return CardHashes(
        ahash="ff00" * 16,
        phash="aa55" * 16,
        dhash="1234" * 16,
    )


class TestGetAllIds:
    """Tests for HashDatabase.get_all_ids."""

    def test_empty_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with HashDatabase(db_path) as db:
                assert db.get_all_ids() == set()

    def test_returns_all_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with HashDatabase(db_path) as db:
                for i in range(5):
                    db.insert_card(
                        card_id=f"set-{i}",
                        name=f"Card {i}",
                        set_id="set",
                        set_name="Test Set",
                        number=str(i),
                        rarity="Common",
                        image_path=f"data/images/set/set-{i}.png",
                        hashes=_sample_hashes(),
                    )
                db.commit()
                ids = db.get_all_ids()
                assert ids == {"set-0", "set-1", "set-2", "set-3", "set-4"}

    def test_ids_match_get_all_cards(self):
        """get_all_ids returns the same IDs as get_all_cards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with HashDatabase(db_path) as db:
                for i in range(3):
                    db.insert_card(
                        card_id=f"xy-{i}",
                        name=f"Card {i}",
                        set_id="xy",
                        set_name="XY",
                        number=str(i),
                        rarity="Common",
                        image_path=f"card-{i}.png",
                        hashes=_sample_hashes(),
                    )
                db.commit()
                ids_method = db.get_all_ids()
                ids_cards = {c.id for c in db.get_all_cards()}
                assert ids_method == ids_cards
