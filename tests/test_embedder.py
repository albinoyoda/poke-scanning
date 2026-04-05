"""Unit tests for card_reco.embedder (CNN embedding extraction)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from card_reco.embedder import EMBEDDING_DIM, CardEmbedder

MODEL_PATH = Path("data") / "mobilenet_v3_small.onnx"


has_model = MODEL_PATH.exists()
skip_no_model = pytest.mark.skipif(not has_model, reason="ONNX model not available")


def _make_solid_image(color: tuple[int, int, int] = (100, 150, 200)) -> np.ndarray:
    """Create a solid-color BGR image."""
    img = np.zeros((200, 140, 3), dtype=np.uint8)
    img[:] = color
    return img


@skip_no_model
class TestCardEmbedder:
    @pytest.fixture()
    def embedder(self) -> CardEmbedder:
        return CardEmbedder(MODEL_PATH)

    def test_embedding_shape(self, embedder: CardEmbedder) -> None:
        image = _make_solid_image()
        emb = embedder.embed(image)
        assert emb.shape == (EMBEDDING_DIM,)

    def test_embedding_dtype(self, embedder: CardEmbedder) -> None:
        image = _make_solid_image()
        emb = embedder.embed(image)
        assert emb.dtype == np.float32

    def test_embedding_is_l2_normalised(self, embedder: CardEmbedder) -> None:
        image = _make_solid_image()
        emb = embedder.embed(image)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self, embedder: CardEmbedder) -> None:
        image = _make_solid_image()
        emb1 = embedder.embed(image)
        emb2 = embedder.embed(image)
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_images_produce_different_embeddings(
        self, embedder: CardEmbedder
    ) -> None:
        img1 = _make_solid_image((255, 0, 0))
        img2 = _make_solid_image((0, 0, 255))
        emb1 = embedder.embed(img1)
        emb2 = embedder.embed(img2)
        assert not np.allclose(emb1, emb2)

    def test_embed_pil(self, embedder: CardEmbedder) -> None:
        pil_img = Image.fromarray(np.zeros((200, 140, 3), dtype=np.uint8))
        emb = embedder.embed_pil(pil_img)
        assert emb.shape == (EMBEDDING_DIM,)
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

    def test_embed_batch_shapes(self, embedder: CardEmbedder) -> None:
        images = [_make_solid_image((i * 50, 100, 200)) for i in range(3)]
        batch_emb = embedder.embed_batch(images)
        assert batch_emb.shape == (3, EMBEDDING_DIM)

    def test_embed_batch_matches_individual(self, embedder: CardEmbedder) -> None:
        images = [_make_solid_image((i * 50, 100, 200)) for i in range(3)]
        batch_emb = embedder.embed_batch(images)
        for i, img in enumerate(images):
            single = embedder.embed(img)
            np.testing.assert_allclose(batch_emb[i], single, atol=1e-5)

    def test_embed_batch_empty(self, embedder: CardEmbedder) -> None:
        result = embedder.embed_batch([])
        assert result.shape == (0, EMBEDDING_DIM)

    def test_model_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            CardEmbedder("nonexistent.onnx")
