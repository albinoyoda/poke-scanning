from __future__ import annotations

from pathlib import Path

from card_reco.database import HashDatabase
from card_reco.hasher import hamming_distance
from card_reco.models import CardHashes, CardRecord, MatchResult

# Weights for combining hash distances (higher = more important)
HASH_WEIGHTS: dict[str, float] = {
    "phash": 1.0,
    "dhash": 1.0,
    "ahash": 0.8,
    "whash": 0.8,
}

# Maximum combined weighted distance to consider a match
DEFAULT_THRESHOLD = 40.0


class CardMatcher:
    """Matches input card hashes against reference database."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db = HashDatabase(db_path) if db_path else HashDatabase()
        self._cards: list[CardRecord] | None = None

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> CardMatcher:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _load_cards(self) -> list[CardRecord]:
        if self._cards is None:
            self._cards = self._db.get_all_cards()
        return self._cards

    def find_matches(
        self,
        hashes: CardHashes,
        top_n: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> list[MatchResult]:
        """Find the best matching cards for a set of input hashes.

        Returns up to top_n matches, sorted by weighted distance (lower = better).
        Only includes matches with combined distance below threshold.
        """
        cards = self._load_cards()
        results: list[MatchResult] = []

        for card in cards:
            distances = {
                "ahash": hamming_distance(hashes.ahash, card.ahash),
                "phash": hamming_distance(hashes.phash, card.phash),
                "dhash": hamming_distance(hashes.dhash, card.dhash),
                "whash": hamming_distance(hashes.whash, card.whash),
            }

            weighted_sum = sum(distances[k] * HASH_WEIGHTS[k] for k in HASH_WEIGHTS)
            total_weight = sum(HASH_WEIGHTS.values())
            combined_distance = weighted_sum / total_weight

            if combined_distance <= threshold:
                results.append(
                    MatchResult(
                        card=card,
                        distance=combined_distance,
                        distances=distances,
                    )
                )

        results.sort(key=lambda r: r.distance)
        for i, result in enumerate(results[:top_n]):
            result.rank = i + 1

        return results[:top_n]
