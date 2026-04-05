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


def _build_name_groups(
    cards: list[CardRecord],
    combined: np.ndarray,
    headroom_limit: float,
) -> dict[str, tuple[float, int, int]]:
    """Build per-name summary.

    Returns ``{name: (best_distance, best_index, n_within_headroom)}``.
    """
    name_groups: dict[str, tuple[float, int, int]] = {}
    for k, card in enumerate(cards):
        dist_k = float(combined[k])
        name = card.name
        if name in name_groups:
            prev_best, prev_idx, prev_count = name_groups[name]
            new_count = prev_count + (1 if dist_k <= headroom_limit else 0)
            if dist_k < prev_best:
                name_groups[name] = (dist_k, k, new_count)
            else:
                name_groups[name] = (prev_best, prev_idx, new_count)
        else:
            name_groups[name] = (
                dist_k,
                k,
                1 if dist_k <= headroom_limit else 0,
            )
    return name_groups


def _accept_by_consensus_or_separation(
    n_close: int,
    min_consensus: int,
    best_dist: float,
    second_best_dist: float | None,
    min_separation: float,
) -> bool:
    """Decide whether to accept the winning name group.

    Returns ``True`` when either:
    - **Consensus**: at least *min_consensus* variants cluster within headroom.
    - **Separation**: the best variant is at least *min_separation* better
      than the runner-up name group's best variant.
    """
    if n_close >= min_consensus:
        return True
    return (
        second_best_dist is not None
        and (second_best_dist - best_dist) >= min_separation
    )


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

    def _name_group_fallback(
        self,
        cards: list[CardRecord],
        combined: np.ndarray,
        headroom_limit: float,
        min_consensus: int,
        min_separation: float,
    ) -> np.ndarray:
        """Name-group two-stage fallback for relaxed matching.

        Groups all cards by name, selects the name whose best variant
        is closest, then accepts via consensus or separation.  Returns
        indices of the winning name's variants within headroom, or an
        empty array when no name qualifies.
        """
        name_groups = _build_name_groups(cards, combined, headroom_limit)

        # Rank name groups by their best variant's distance.
        ranked_names = sorted(name_groups, key=lambda n: name_groups[n][0])
        if not ranked_names:
            return np.empty(0, dtype=np.int64)

        best_name = ranked_names[0]
        best_dist, _, n_close = name_groups[best_name]

        if best_dist > headroom_limit:
            return np.empty(0, dtype=np.int64)

        second_best_dist = (
            name_groups[ranked_names[1]][0] if len(ranked_names) > 1 else None
        )
        if not _accept_by_consensus_or_separation(
            n_close, min_consensus, best_dist, second_best_dist, min_separation
        ):
            return np.empty(0, dtype=np.int64)

        # Return all variants of the winning name within headroom
        # (stage-2 variant refinement).
        return np.array(
            [
                k
                for k, card in enumerate(cards)
                if card.name == best_name and float(combined[k]) <= headroom_limit
            ],
            dtype=np.int64,
        )

    def find_matches(
        self,
        hashes: CardHashes,
        top_n: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
        enable_relaxed_fallback: bool = False,
        relaxed_headroom: float = 25.0,
        min_separation: float = 15.0,
        min_consensus: int = 2,
    ) -> list[MatchResult]:
        """Find the best matching cards for a set of input hashes.

        Returns up to top_n matches, sorted by weighted distance
        (lower = better). Only includes matches with combined distance
        below threshold.

        When *enable_relaxed_fallback* is true and no match passes
        *threshold*, a **name-group two-stage** fallback is attempted:

        *Stage 1 — identify by name*: reference cards are grouped by
        name and the group whose best variant has the lowest distance
        is selected.

        *Stage 2 — accept & refine*: the winning group is accepted
        when at least one of two criteria is met:

        1. **Name consensus** — the group has at least *min_consensus*
           variants within *relaxed_headroom* of *threshold*.
        2. **Name-group separation** — the group's best variant is at
           least *min_separation* lower than the best variant of any
           other name group.

        When accepted, all variants of the winning name within
        headroom are returned (up to *top_n*), enabling the caller
        to see the variant landscape.
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

        # Relaxed fallback via name-group two-stage matching.
        if len(indices) == 0 and enable_relaxed_fallback and len(combined) > 0:
            indices = self._name_group_fallback(
                cards,
                combined,
                headroom_limit=threshold + relaxed_headroom,
                min_consensus=min_consensus,
                min_separation=min_separation,
            )

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
