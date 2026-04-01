from __future__ import annotations

import tempfile
from pathlib import Path

from card_reco.database import HashDatabase
from card_reco.matcher import CardMatcher
from card_reco.models import CardHashes


class TestCardMatcher:
    def _build_db(self, tmpdir: str) -> Path:
        """Create a small test DB with a few cards."""
        db_path = Path(tmpdir) / "test.db"
        with HashDatabase(db_path) as db:
            # Card 1: Charizard
            db.insert_card(
                card_id="base1-4",
                name="Charizard",
                set_id="base1",
                set_name="Base",
                number="4",
                rarity="Rare Holo",
                image_path="charizard.png",
                hashes=CardHashes(
                    ahash="ff00" * 16,
                    phash="aa55" * 16,
                    dhash="1234" * 16,
                    whash="fedc" * 16,
                ),
            )
            # Card 2: Pikachu (different hashes)
            db.insert_card(
                card_id="base1-58",
                name="Pikachu",
                set_id="base1",
                set_name="Base",
                number="58",
                rarity="Common",
                image_path="pikachu.png",
                hashes=CardHashes(
                    ahash="00ff" * 16,
                    phash="55aa" * 16,
                    dhash="4321" * 16,
                    whash="cdef" * 16,
                ),
            )
            db.commit()
        return db_path

    def test_exact_match_returns_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._build_db(tmpdir)
            with CardMatcher(db_path) as matcher:
                query = CardHashes(
                    ahash="ff00" * 16,
                    phash="aa55" * 16,
                    dhash="1234" * 16,
                    whash="fedc" * 16,
                )
                results = matcher.find_matches(query, top_n=5)
                assert len(results) >= 1
                assert results[0].card.name == "Charizard"
                assert results[0].distance == 0.0

    def test_close_match_ranks_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._build_db(tmpdir)
            with CardMatcher(db_path) as matcher:
                # Query is close to Charizard (1 bit difference in ahash)
                query = CardHashes(
                    ahash="ff00" * 15 + "ff01",
                    phash="aa55" * 16,
                    dhash="1234" * 16,
                    whash="fedc" * 16,
                )
                results = matcher.find_matches(query, top_n=5)
                assert len(results) >= 1
                assert results[0].card.name == "Charizard"
                assert results[0].distance < 5.0

    def test_no_match_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._build_db(tmpdir)
            with CardMatcher(db_path) as matcher:
                # Very different hashes — all zeros (64 hex chars)
                query = CardHashes(
                    ahash="0" * 64,
                    phash="0" * 64,
                    dhash="0" * 64,
                    whash="0" * 64,
                )
                results = matcher.find_matches(query, top_n=5, threshold=5.0)
                # With a tight threshold, we might not match
                for r in results:
                    assert r.distance <= 5.0

    def test_results_are_ranked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._build_db(tmpdir)
            with CardMatcher(db_path) as matcher:
                query = CardHashes(
                    ahash="ff00" * 16,
                    phash="aa55" * 16,
                    dhash="1234" * 16,
                    whash="fedc" * 16,
                )
                results = matcher.find_matches(query, top_n=5, threshold=1000)
                if len(results) >= 2:
                    assert results[0].rank == 1
                    assert results[1].rank == 2
                    assert results[0].distance <= results[1].distance
