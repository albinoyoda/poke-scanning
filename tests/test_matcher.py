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

    def test_name_group_separation_ignores_same_name_variants(self):
        """Name-group separation treats same-name variants as one group.

        Two Electivire variants (distances 0.22, 0.44) and one Pikachu
        (distance 1.78).  Individual separation would fail because the
        runner-up is the second Electivire variant (separation 0.22),
        but name-group separation correctly compares Electivire vs
        Pikachu (separation 1.56) and accepts.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "name_group.db"
            base = "0" * 64
            with HashDatabase(db_path) as db:
                db.insert_card(
                    card_id="sv10-69",
                    name="Electivire ex",
                    set_id="sv10",
                    set_name="Destined Rivals",
                    number="69",
                    rarity="Rare",
                    image_path="electivire1.png",
                    hashes=CardHashes(
                        ahash=base[:-1] + "1",
                        phash=base,
                        dhash=base,
                        whash=base,
                    ),
                )
                db.insert_card(
                    card_id="sv10-182",
                    name="Electivire ex",
                    set_id="sv10",
                    set_name="Destined Rivals",
                    number="182",
                    rarity="Ultra Rare",
                    image_path="electivire2.png",
                    hashes=CardHashes(
                        ahash=base[:-1] + "3",
                        phash=base,
                        dhash=base,
                        whash=base,
                    ),
                )
                db.insert_card(
                    card_id="base1-58",
                    name="Pikachu",
                    set_id="base1",
                    set_name="Base",
                    number="58",
                    rarity="Common",
                    image_path="pikachu.png",
                    hashes=CardHashes(
                        ahash=base[:-2] + "ff",
                        phash=base,
                        dhash=base,
                        whash=base,
                    ),
                )
                db.commit()

            with CardMatcher(db_path) as matcher:
                query = CardHashes(
                    ahash="0" * 64,
                    phash="0" * 64,
                    dhash="0" * 64,
                    whash="0" * 64,
                )
                # Name-group separation succeeds: Electivire group at
                # ~0.22 vs Pikachu group at ~1.78, separation ~1.56.
                results = matcher.find_matches(
                    query,
                    top_n=5,
                    threshold=0.0,
                    enable_relaxed_fallback=True,
                    relaxed_headroom=5.0,
                    min_separation=1.0,
                    min_consensus=99,  # disable consensus
                )
                assert len(results) == 2
                assert results[0].card.name == "Electivire ex"
                assert results[1].card.name == "Electivire ex"
                assert results[0].distance <= results[1].distance

    def test_relaxed_fallback_returns_clear_best(self):
        """Fallback can return one clear winner above a strict threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._build_db(tmpdir)
            with CardMatcher(db_path) as matcher:
                # One-bit deviation from Charizard: above threshold=0 but far
                # better than Pikachu, so fallback should allow it.
                query = CardHashes(
                    ahash="ff00" * 15 + "ff01",
                    phash="aa55" * 16,
                    dhash="1234" * 16,
                    whash="fedc" * 16,
                )
                strict = matcher.find_matches(query, top_n=5, threshold=0.0)
                assert strict == []

                relaxed = matcher.find_matches(
                    query,
                    top_n=5,
                    threshold=0.0,
                    enable_relaxed_fallback=True,
                    relaxed_headroom=5.0,
                    min_separation=1.0,
                )
                assert len(relaxed) == 1
                assert relaxed[0].card.id == "base1-4"
                assert relaxed[0].distance > 0.0

    def test_relaxed_fallback_requires_separation(self):
        """Fallback should not trigger if top two candidates are too close
        and they have different names (no consensus)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "close.db"
            with HashDatabase(db_path) as db:
                base_hash = "0" * 64
                db.insert_card(
                    card_id="card-a",
                    name="Card A",
                    set_id="test",
                    set_name="Test",
                    number="1",
                    rarity="Common",
                    image_path="a.png",
                    hashes=CardHashes(
                        ahash=base_hash[:-1] + "1",  # 1-bit difference from query
                        phash=base_hash,
                        dhash=base_hash,
                        whash=base_hash,
                    ),
                )
                db.insert_card(
                    card_id="card-b",
                    name="Card B",
                    set_id="test",
                    set_name="Test",
                    number="2",
                    rarity="Common",
                    image_path="b.png",
                    hashes=CardHashes(
                        ahash=base_hash[:-1] + "3",  # 2-bit difference from query
                        phash=base_hash,
                        dhash=base_hash,
                        whash=base_hash,
                    ),
                )
                db.commit()

            with CardMatcher(db_path) as matcher:
                query = CardHashes(
                    ahash="0" * 64,
                    phash="0" * 64,
                    dhash="0" * 64,
                    whash="0" * 64,
                )
                results = matcher.find_matches(
                    query,
                    top_n=5,
                    threshold=0.0,
                    enable_relaxed_fallback=True,
                    relaxed_headroom=5.0,
                    min_separation=1.0,
                )
                assert results == []

    def test_relaxed_fallback_respects_headroom(self):
        """Fallback should not trigger when the best distance is too large."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = self._build_db(tmpdir)
            with CardMatcher(db_path) as matcher:
                query = CardHashes(
                    ahash="0" * 64,
                    phash="0" * 64,
                    dhash="0" * 64,
                    whash="0" * 64,
                )
                results = matcher.find_matches(
                    query,
                    top_n=5,
                    threshold=0.0,
                    enable_relaxed_fallback=True,
                    relaxed_headroom=5.0,
                    min_separation=0.0,
                )
                assert results == []

    def test_consensus_fallback_same_name_cluster(self):
        """Consensus fallback triggers when 2+ cards of the same name
        cluster at the top, even without large separation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "consensus.db"
            # Two Wigglytuff variants (same name) and one Clefable.
            # Hash values chosen so both Wigglytuffs are close to the query
            # and Clefable is slightly farther.
            base = "0" * 64
            with HashDatabase(db_path) as db:
                db.insert_card(
                    card_id="base2-16",
                    name="Wigglytuff",
                    set_id="base2",
                    set_name="Jungle",
                    number="16",
                    rarity="Rare Holo",
                    image_path="wiggly1.png",
                    hashes=CardHashes(
                        ahash=base[:-1] + "1",
                        phash=base,
                        dhash=base,
                        whash=base,
                    ),
                )
                db.insert_card(
                    card_id="base4-19",
                    name="Wigglytuff",
                    set_id="base4",
                    set_name="Base Set 2",
                    number="19",
                    rarity="Rare Holo",
                    image_path="wiggly2.png",
                    hashes=CardHashes(
                        ahash=base[:-1] + "3",
                        phash=base,
                        dhash=base,
                        whash=base,
                    ),
                )
                db.insert_card(
                    card_id="base2-17",
                    name="Clefable",
                    set_id="base2",
                    set_name="Jungle",
                    number="17",
                    rarity="Rare Holo",
                    image_path="clefable.png",
                    hashes=CardHashes(
                        ahash=base[:-1] + "7",
                        phash=base,
                        dhash=base,
                        whash=base,
                    ),
                )
                db.commit()

            with CardMatcher(db_path) as matcher:
                query = CardHashes(
                    ahash="0" * 64,
                    phash="0" * 64,
                    dhash="0" * 64,
                    whash="0" * 64,
                )
                # Strict threshold excludes all; separation-based fails
                # because the two Wigglytuffs are very close.
                strict = matcher.find_matches(query, top_n=5, threshold=0.0)
                assert strict == []

                # Consensus fallback should find 2 Wigglytuffs and accept.
                # Two-stage matching returns all variants of the winning
                # name within headroom.
                results = matcher.find_matches(
                    query,
                    top_n=5,
                    threshold=0.0,
                    enable_relaxed_fallback=True,
                    relaxed_headroom=5.0,
                    min_separation=50.0,  # impossibly high — blocks separation
                    min_consensus=2,
                )
                assert len(results) == 2
                assert results[0].card.name == "Wigglytuff"
                assert results[1].card.name == "Wigglytuff"
                assert results[0].distance <= results[1].distance

    def test_consensus_fallback_requires_min_count(self):
        """Consensus should not trigger if only one card of that name exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "no_consensus.db"
            base = "0" * 64
            with HashDatabase(db_path) as db:
                db.insert_card(
                    card_id="sv10-69",
                    name="Electivire ex",
                    set_id="sv10",
                    set_name="Test",
                    number="69",
                    rarity="Rare",
                    image_path="elec.png",
                    hashes=CardHashes(
                        ahash=base[:-1] + "1",
                        phash=base,
                        dhash=base,
                        whash=base,
                    ),
                )
                db.insert_card(
                    card_id="sv10-999",
                    name="Pikachu",
                    set_id="sv10",
                    set_name="Test",
                    number="999",
                    rarity="Common",
                    image_path="pika.png",
                    hashes=CardHashes(
                        ahash=base[:-1] + "3",
                        phash=base,
                        dhash=base,
                        whash=base,
                    ),
                )
                db.commit()

            with CardMatcher(db_path) as matcher:
                query = CardHashes(
                    ahash="0" * 64,
                    phash="0" * 64,
                    dhash="0" * 64,
                    whash="0" * 64,
                )
                # Only 1 Electivire, not enough for consensus.
                # Separation is ~0.33, much less than min_separation=50.
                results = matcher.find_matches(
                    query,
                    top_n=5,
                    threshold=0.0,
                    enable_relaxed_fallback=True,
                    relaxed_headroom=5.0,
                    min_separation=50.0,
                    min_consensus=2,
                )
                assert results == []
