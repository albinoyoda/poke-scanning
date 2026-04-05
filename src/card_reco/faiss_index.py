"""FAISS-based vector index for card embedding search.

Wraps a FAISS ``IndexFlatIP`` (inner product on L2-normalised vectors,
equivalent to cosine similarity) and maps result indices back to card
metadata stored in a separate JSON sidecar file.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from card_reco.models import CardRecord, MatchResult

DEFAULT_INDEX_PATH = Path("data") / "card_embeddings.faiss"
DEFAULT_META_PATH = Path("data") / "card_embeddings_meta.json"


class CardIndex:
    """FAISS-backed nearest-neighbour search over card embeddings."""

    def __init__(
        self,
        index_path: Path | str | None = None,
        meta_path: Path | str | None = None,
    ) -> None:
        idx_p = Path(index_path) if index_path else DEFAULT_INDEX_PATH
        meta_p = Path(meta_path) if meta_path else DEFAULT_META_PATH

        if not idx_p.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {idx_p}\n"
                "Run: uv run python scripts/build_embedding_db.py"
            )
        if not meta_p.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {meta_p}\n"
                "Run: uv run python scripts/build_embedding_db.py"
            )

        self._index = faiss.read_index(str(idx_p))
        with open(meta_p, encoding="utf-8") as f:
            self._cards: list[CardRecord] = [
                CardRecord(**entry) for entry in json.load(f)
            ]

    @property
    def ntotal(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal

    def search(
        self,
        embedding: NDArray[np.float32],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[MatchResult]:
        """Search for the closest cards to *embedding*.

        *threshold* is a **minimum cosine similarity** (0-1).  Results
        below this similarity are excluded.

        Returns up to *top_k* ``MatchResult`` objects sorted by
        descending similarity (best first).  ``MatchResult.distance``
        stores the cosine similarity (higher = better).
        """
        query = np.ascontiguousarray(embedding.reshape(1, -1).astype(np.float32))
        similarities, indices = self._index.search(query, top_k)

        results: list[MatchResult] = []
        for rank, (sim, idx) in enumerate(
            zip(similarities[0], indices[0], strict=True), start=1
        ):
            if idx < 0:
                continue
            if float(sim) < threshold:
                continue
            results.append(
                MatchResult(
                    card=self._cards[idx],
                    distance=float(sim),
                    distances={},
                    rank=rank,
                )
            )
        return results

    @staticmethod
    def build(
        embeddings: NDArray[np.float32],
        cards: list[CardRecord],
        index_path: Path | str,
        meta_path: Path | str,
    ) -> None:
        """Build and save a FAISS index with metadata sidecar.

        *embeddings* must be L2-normalised, shape ``(N, dim)``.
        *cards* must have the same length and order as *embeddings*.
        """
        idx_p = Path(index_path)
        meta_p = Path(meta_path)
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # pylint: disable=no-value-for-parameter
        data = np.ascontiguousarray(embeddings.astype(np.float32))
        # FAISS Python wrapper replaces SWIG add(n, x) with add(x);
        # ty stubs still show the old signature — use noqa to bypass.
        index.add(data)  # ty: ignore[missing-argument]  # pylint: disable=no-value-for-parameter
        faiss.write_index(index, str(idx_p))

        meta = []
        for card in cards:
            meta.append(
                {
                    "id": card.id,
                    "name": card.name,
                    "set_id": card.set_id,
                    "set_name": card.set_name,
                    "number": card.number,
                    "rarity": card.rarity,
                    "image_path": card.image_path,
                }
            )
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta, f)
