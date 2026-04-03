from __future__ import annotations

from pathlib import Path

import numpy as np

from card_reco.database import HashDatabase
from card_reco.hasher import hex_to_bits
from card_reco.models import CardHashes, CardRecord, MatchResult

# Weights for combining hash distances (higher = more important)
HASH_WEIGHTS: dict[str, float] = {
    "phash": 1.0,
    "dhash": 1.0,
    "ahash": 0.8,
    "whash": 0.8,
}

_WEIGHT_ORDER = ("ahash", "phash", "dhash", "whash")
_WEIGHTS_ARRAY = np.array([HASH_WEIGHTS[k] for k in _WEIGHT_ORDER], dtype=np.float64)
_TOTAL_WEIGHT = float(_WEIGHTS_ARRAY.sum())

# Maximum combined weighted distance to consider a match
DEFAULT_THRESHOLD = 40.0


class CardMatcher:
    """Matches input card hashes against reference database.

    On first query the reference hashes are loaded from SQLite and
    converted to a packed NumPy bit-matrix so that Hamming distances
    for the entire database can be computed in one vectorised operation.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db = HashDatabase(db_path) if db_path else HashDatabase()
        self._cards: list[CardRecord] | None = None
        # shape (N, 4, hash_bits) — 4 hash types per card, each as bit array
        self._hash_matrix: np.ndarray | None = None

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> CardMatcher:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _ensure_loaded(self) -> tuple[list[CardRecord], np.ndarray]:
        """Load cards and build the vectorised hash matrix once."""
        if self._cards is not None and self._hash_matrix is not None:
            return self._cards, self._hash_matrix

        cards = self._db.get_all_cards()
        if not cards:
            self._cards = []
            self._hash_matrix = np.empty((0, 4, 0), dtype=np.uint8)
            return self._cards, self._hash_matrix

        # Build (N, 4, hash_bits) matrix from hex strings
        sample_bits = hex_to_bits(cards[0].ahash)
        n_bits = len(sample_bits)
        matrix = np.empty((len(cards), 4, n_bits), dtype=np.uint8)

        for i, card in enumerate(cards):
            for j, key in enumerate(_WEIGHT_ORDER):
                matrix[i, j] = hex_to_bits(getattr(card, key))

        self._cards = cards
        self._hash_matrix = matrix
        return self._cards, self._hash_matrix

    def find_matches(
        self,
        hashes: CardHashes,
        top_n: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> list[MatchResult]:
        """Find the best matching cards for a set of input hashes.

        Returns up to top_n matches, sorted by weighted distance
        (lower = better). Only includes matches with combined distance
        below threshold.
        """
        cards, matrix = self._ensure_loaded()
        if not cards:
            return []

        # Build query vector (4, hash_bits)
        n_bits = matrix.shape[2]
        query = np.empty((4, n_bits), dtype=np.uint8)
        for j, key in enumerate(_WEIGHT_ORDER):
            query[j] = hex_to_bits(getattr(hashes, key))

        # Vectorised hamming: XOR then count non-zero bits
        # matrix shape (N, 4, bits), query shape (4, bits) → diff (N, 4, bits)
        diff = matrix ^ query[np.newaxis, :, :]
        # per_hash_dist shape (N, 4)
        per_hash_dist = diff.sum(axis=2).astype(np.float64)

        # Weighted combined distance (N,)
        combined = per_hash_dist @ _WEIGHTS_ARRAY / _TOTAL_WEIGHT

        # Filter to candidates below threshold
        mask = combined <= threshold
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return []

        # Get top_n by distance
        candidate_dists = combined[indices]
        if len(indices) > top_n:
            top_idx = np.argpartition(candidate_dists, top_n)[:top_n]
            indices = indices[top_idx]
            candidate_dists = candidate_dists[top_idx]

        # Sort the final candidates
        order = np.argsort(candidate_dists)
        indices = indices[order]

        results: list[MatchResult] = []
        for rank, idx in enumerate(indices, start=1):
            card = cards[idx]
            dists = per_hash_dist[idx]
            results.append(
                MatchResult(
                    card=card,
                    distance=float(combined[idx]),
                    distances={k: int(dists[j]) for j, k in enumerate(_WEIGHT_ORDER)},
                    rank=rank,
                )
            )

        return results
