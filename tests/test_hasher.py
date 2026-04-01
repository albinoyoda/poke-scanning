from __future__ import annotations

import numpy as np
from PIL import Image

from card_reco.hasher import compute_hashes, compute_hashes_pil, hamming_distance


def _make_solid_image(color: tuple[int, int, int] = (100, 150, 200)) -> np.ndarray:
    """Create a solid-color BGR image."""
    img = np.zeros((200, 140, 3), dtype=np.uint8)
    img[:] = color
    return img


class TestComputeHashes:
    def test_returns_all_hash_types(self):
        image = _make_solid_image()
        hashes = compute_hashes(image)
        assert hashes.ahash
        assert hashes.phash
        assert hashes.dhash
        assert hashes.whash

    def test_hashes_are_hex_strings(self):
        image = _make_solid_image()
        hashes = compute_hashes(image)
        for h in [hashes.ahash, hashes.phash, hashes.dhash, hashes.whash]:
            int(h, 16)  # should not raise

    def test_identical_images_have_zero_distance(self):
        image = _make_solid_image()
        h1 = compute_hashes(image)
        h2 = compute_hashes(image)
        assert hamming_distance(h1.phash, h2.phash) == 0

    def test_different_images_have_nonzero_distance(self):
        # Use patterned images — solid colors produce identical luminance hashes
        img1 = np.zeros((200, 140, 3), dtype=np.uint8)
        img1[:100, :, :] = 255  # top half white
        img2 = np.zeros((200, 140, 3), dtype=np.uint8)
        img2[100:, :, :] = 255  # bottom half white
        h1 = compute_hashes(img1)
        h2 = compute_hashes(img2)
        # At least one hash type should differ
        distances = [
            hamming_distance(h1.ahash, h2.ahash),
            hamming_distance(h1.phash, h2.phash),
            hamming_distance(h1.dhash, h2.dhash),
            hamming_distance(h1.whash, h2.whash),
        ]
        assert any(d > 0 for d in distances)

    def test_pil_and_numpy_produce_same_hashes(self):
        bgr = _make_solid_image((50, 100, 200))
        rgb = bgr[..., ::-1]
        pil_img = Image.fromarray(rgb)

        h_np = compute_hashes(bgr)
        h_pil = compute_hashes_pil(pil_img)

        assert h_np.ahash == h_pil.ahash
        assert h_np.phash == h_pil.phash
        assert h_np.dhash == h_pil.dhash
        assert h_np.whash == h_pil.whash


class TestHammingDistance:
    def test_same_hash_is_zero(self):
        # hash_size=16 produces 64-char hex strings
        h = "ff00" * 16  # 64 hex chars
        assert hamming_distance(h, h) == 0

    def test_single_bit_difference(self):
        h1 = "ff00" * 16
        h2 = "ff00" * 15 + "ff01"  # 1 bit difference in last segment
        assert hamming_distance(h1, h2) == 1
