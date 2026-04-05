"""Unit tests for card_reco.faiss_index (FAISS-based card search)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from card_reco.embedder import EMBEDDING_DIM
from card_reco.faiss_index import CardIndex
from card_reco.models import CardRecord

INDEX_PATH = Path("data") / "card_embeddings.faiss"
META_PATH = Path("data") / "card_embeddings_meta.json"

has_index = INDEX_PATH.exists() and META_PATH.exists()
skip_no_index = pytest.mark.skipif(not has_index, reason="FAISS index not available")


def _random_embeddings(n: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-10)


def _make_cards(n: int) -> list[CardRecord]:
    return [
        CardRecord(
            id=f"test-{i}",
            name=f"Card {i}",
            set_id="test",
            set_name="Test Set",
            number=str(i),
            rarity="Common",
            image_path=f"card_{i}.png",
        )
        for i in range(n)
    ]


class TestCardIndexBuildAndSearch:
    """Test building, saving, loading, and searching a FAISS index."""

    def test_build_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            meta_path = Path(tmpdir) / "test_meta.json"
            n = 10
            embeddings = _random_embeddings(n)
            cards = _make_cards(n)

            CardIndex.build(embeddings, cards, idx_path, meta_path)

            index = CardIndex(idx_path, meta_path)
            assert index.ntotal == n

    def test_search_returns_match_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            meta_path = Path(tmpdir) / "test_meta.json"
            n = 20
            embeddings = _random_embeddings(n)
            cards = _make_cards(n)
            CardIndex.build(embeddings, cards, idx_path, meta_path)

            index = CardIndex(idx_path, meta_path)
            results = index.search(embeddings[0], top_k=5)
            assert len(results) >= 1
            assert results[0].card.id == "test-0"
            assert results[0].rank == 1

    def test_self_match_has_similarity_near_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            meta_path = Path(tmpdir) / "test_meta.json"
            embeddings = _random_embeddings(10)
            cards = _make_cards(10)
            CardIndex.build(embeddings, cards, idx_path, meta_path)

            index = CardIndex(idx_path, meta_path)
            results = index.search(embeddings[3], top_k=1)
            assert len(results) == 1
            assert results[0].card.id == "test-3"
            assert results[0].distance > 0.99

    def test_threshold_filters_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            meta_path = Path(tmpdir) / "test_meta.json"
            embeddings = _random_embeddings(10)
            cards = _make_cards(10)
            CardIndex.build(embeddings, cards, idx_path, meta_path)

            index = CardIndex(idx_path, meta_path)
            # Very high threshold should filter most results
            results = index.search(embeddings[0], top_k=10, threshold=0.99)
            # Only the self-match should survive
            assert len(results) == 1
            assert results[0].card.id == "test-0"

    def test_top_k_limits_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            meta_path = Path(tmpdir) / "test_meta.json"
            embeddings = _random_embeddings(20)
            cards = _make_cards(20)
            CardIndex.build(embeddings, cards, idx_path, meta_path)

            index = CardIndex(idx_path, meta_path)
            results = index.search(embeddings[0], top_k=3)
            assert len(results) <= 3

    def test_results_sorted_by_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            meta_path = Path(tmpdir) / "test_meta.json"
            embeddings = _random_embeddings(20)
            cards = _make_cards(20)
            CardIndex.build(embeddings, cards, idx_path, meta_path)

            index = CardIndex(idx_path, meta_path)
            results = index.search(embeddings[0], top_k=5)
            for i in range(len(results) - 1):
                assert results[i].distance >= results[i + 1].distance

    def test_missing_index_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            CardIndex("nonexistent.faiss", "nonexistent.json")

    def test_missing_meta_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            meta_path = Path(tmpdir) / "test_meta.json"
            embeddings = _random_embeddings(5)
            cards = _make_cards(5)
            CardIndex.build(embeddings, cards, idx_path, meta_path)
            # Delete metadata file
            meta_path.unlink()
            with pytest.raises(FileNotFoundError, match="Metadata file not found"):
                CardIndex(idx_path, meta_path)


@skip_no_index
class TestCardIndexProduction:
    """Tests using the production FAISS index."""

    def test_production_index_loads(self) -> None:
        index = CardIndex()
        assert index.ntotal > 0

    def test_production_search_returns_results(self) -> None:
        index = CardIndex()
        query = _random_embeddings(1)[0]
        results = index.search(query, top_k=5)
        assert len(results) >= 1
