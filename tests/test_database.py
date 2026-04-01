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
        whash="fedc" * 16,
    )


class TestHashDatabase:
    def test_insert_and_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with HashDatabase(db_path) as db:
                db.insert_card(
                    card_id="xy1-1",
                    name="Venusaur-EX",
                    set_id="xy1",
                    set_name="XY",
                    number="1",
                    rarity="Rare Holo EX",
                    image_path="data/images/xy1/xy1-1.png",
                    hashes=_sample_hashes(),
                )
                db.commit()
                assert db.count() == 1

    def test_get_card_by_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with HashDatabase(db_path) as db:
                db.insert_card(
                    card_id="base1-4",
                    name="Charizard",
                    set_id="base1",
                    set_name="Base",
                    number="4",
                    rarity="Rare Holo",
                    image_path="data/images/base1/base1-4.png",
                    hashes=_sample_hashes(),
                )
                db.commit()

                card = db.get_card_by_id("base1-4")
                assert card is not None
                assert card.name == "Charizard"
                assert card.set_name == "Base"

    def test_get_card_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with HashDatabase(db_path) as db:
                assert db.get_card_by_id("nonexistent") is None

    def test_get_all_cards(self):
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

                cards = db.get_all_cards()
                assert len(cards) == 5

    def test_upsert_replaces_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with HashDatabase(db_path) as db:
                db.insert_card(
                    card_id="xy1-1",
                    name="Venusaur-EX",
                    set_id="xy1",
                    set_name="XY",
                    number="1",
                    rarity="Rare Holo EX",
                    image_path="old_path.png",
                    hashes=_sample_hashes(),
                )
                db.insert_card(
                    card_id="xy1-1",
                    name="Venusaur-EX Updated",
                    set_id="xy1",
                    set_name="XY",
                    number="1",
                    rarity="Rare Holo EX",
                    image_path="new_path.png",
                    hashes=_sample_hashes(),
                )
                db.commit()
                assert db.count() == 1
                card = db.get_card_by_id("xy1-1")
                assert card is not None
                assert card.name == "Venusaur-EX Updated"
