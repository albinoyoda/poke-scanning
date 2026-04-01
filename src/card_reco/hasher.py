from __future__ import annotations

import imagehash
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from card_reco.models import CardHashes

HASH_SIZE = 16


def compute_hashes(image: NDArray[np.uint8]) -> CardHashes:
    """Compute perceptual hashes for a card image (BGR numpy array)."""
    pil_image = Image.fromarray(image[..., ::-1])  # BGR -> RGB
    return compute_hashes_pil(pil_image)


def compute_hashes_pil(pil_image: Image.Image) -> CardHashes:
    """Compute perceptual hashes for a PIL Image."""
    return CardHashes(
        ahash=str(imagehash.average_hash(pil_image, hash_size=HASH_SIZE)),
        phash=str(imagehash.phash(pil_image, hash_size=HASH_SIZE)),
        dhash=str(imagehash.dhash(pil_image, hash_size=HASH_SIZE)),
        whash=str(imagehash.whash(pil_image, hash_size=HASH_SIZE)),
    )


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute the hamming distance between two hex-encoded hashes."""
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return int(h1 - h2)
