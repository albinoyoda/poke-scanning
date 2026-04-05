from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class DetectedCard:
    """A card region extracted from an input image."""

    image: NDArray[np.uint8]
    corners: NDArray[np.float32]
    confidence: float = 0.0
    contour: NDArray[np.uint8] | None = field(default=None, repr=False)


@dataclass
class CardHashes:
    """Perceptual hashes for a single card image."""

    ahash: str
    phash: str
    dhash: str
    whash: str


@dataclass
class CardRecord:
    """A card entry from the reference database."""

    id: str
    name: str
    set_id: str
    set_name: str
    number: str
    rarity: str
    image_path: str
    ahash: str
    phash: str
    dhash: str
    whash: str


@dataclass
class MatchResult:
    """Result of matching an input card against the reference database."""

    card: CardRecord
    distance: float
    distances: dict[str, int] = field(default_factory=dict)
    rank: int = 0
