# Architecture

## Overview

`card_reco` is a Pokemon card recognition pipeline. Given a photo containing
one or more Pokemon cards, it detects each card region, normalises it to a
standard orientation, computes perceptual hashes, and matches the hashes
against a pre-built SQLite reference database.

```
Image → detect_cards() → [DetectedCard, …]
  └→ for each detection × 4 rotations:
       compute_hashes() → CardHashes
       CardMatcher.find_matches() → [MatchResult, …]
```

## Module Map

| Module | Responsibility |
|---|---|
| `card_reco/__init__.py` | Public API: `identify_cards()`, `identify_cards_from_array()` |
| `card_reco/detector.py` | OpenCV contour-based card detection and perspective normalisation |
| `card_reco/hasher.py` | Perceptual hashing via `imagehash` (ahash, phash, dhash, whash) |
| `card_reco/matcher.py` | Brute-force linear scan matching with weighted Hamming distance |
| `card_reco/database.py` | SQLite wrapper for the reference hash database |
| `card_reco/models.py` | Dataclasses: `DetectedCard`, `CardHashes`, `CardRecord`, `MatchResult` |
| `card_reco/cli.py` | CLI entry point (`card-reco identify <image>`) |

## Detection Pipeline (`detector.py`)

Three strategies run in parallel and merge candidates via centroid dedup:

1. **Canny edge detection** — two threshold pairs (50/150, 30/100) × two
   retrieval modes (EXTERNAL, TREE).
2. **Adaptive thresholding** — Gaussian, block size 15, for low-contrast
   card borders.
3. **HSV colour segmentation** — masks for red/orange and yellow/gold
   border colours.

Each candidate contour is validated:
- Area between 1.5 %–95 % of image area.
- Polygon approximation to 4 corners (fallback: `minAreaRect` with
  ≥ 65 % compactness).
- Aspect ratio within ±0.25 of 5:7 (standard card ratio).

Accepted contours are perspective-warped to a 734 × 1024 portrait image.
Landscape-oriented cards (width > 1.2× height) are warped to landscape first,
then rotated to portrait. Non-max suppression (IoU > 0.5) removes
overlapping detections.

## Hashing (`hasher.py`)

Each detected card image is converted to a PIL Image and hashed four ways
at `hash_size=16` (256-bit hashes, 64-char hex strings):

| Hash | Algorithm | Strength |
|---|---|---|
| `ahash` | Average hash | Fast, tolerant of minor changes |
| `phash` | Perceptual hash (DCT) | Best overall texture matching |
| `dhash` | Difference hash | Good edge/gradient sensitivity |
| `whash` | Wavelet hash | Multi-scale frequency analysis |

## Matching (`matcher.py`)

Brute-force linear scan over all reference cards. For each reference card,
four Hamming distances are computed and combined with weights:

```
combined = (0.8·ahash + 1.0·phash + 1.0·dhash + 0.8·whash) / 3.6
```

Results below the threshold (default 40.0) are sorted by distance.

The top-level API tries **all 4 cardinal rotations** (0°, 90°, 180°, 270°)
for each detected card and keeps the rotation with the lowest best-match
distance.

## Reference Database (`database.py`)

SQLite table `cards` with columns: `id`, `name`, `set_id`, `set_name`,
`number`, `rarity`, `image_path`, `ahash`, `phash`, `dhash`, `whash`.

Built by `scripts/build_hash_db.py` from card images in
`data/images/{set_id}/` and metadata in `data/metadata/cards/{set_id}.json`.

## Data Layout

```
data/
├── card_hashes.db              # Reference hash database
├── metadata/
│   ├── sets.json               # Set index (from PokemonTCG/pokemon-tcg-data)
│   └── cards/{set_id}.json     # Per-set card metadata
├── images/{set_id}/            # Reference card images
└── tests/                      # Test photos
    ├── single_cards/
    │   ├── axis_aligned/       # Straight-on photos (simplest)
    │   ├── rotated/            # Cards rotated in-plane
    │   └── tilted/             # Perspective-distorted photos
    ├── graded_cards/           # PSA-slabbed cards
    ├── multiple_cards/         # Multi-card grid photos
    └── card_backs/             # Card back detection tests
```

Each test subfolder contains `_annotations.json` with expected card IDs.

## Scripts

| Script | Purpose |
|---|---|
| `download_reference_data.py` | Download card metadata + images from PokemonTCG GitHub |
| `build_hash_db.py` | Build/rebuild the SQLite hash database from local images |
| `find_cards.py` | Quick CLI card finder |
| `find_sets.py` | Search available sets |
| `check_cards.py` | Verify card data integrity |
| `save_crops.py` | Save detected card crops for debugging |
| `debug_detector.py` | Visualise detection pipeline |
| `debug_detector2.py` | Additional detection debugging |
| `debug_matching.py` | Visualise matching distances |

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Build | hatchling + uv |
| Computer vision | OpenCV (`opencv-python ≥ 4.8`) |
| Image processing | Pillow (≥ 10.0), NumPy (≥ 1.24) |
| Perceptual hashing | `imagehash` (≥ 4.3) |
| Reference DB | SQLite 3 (stdlib) |
| Data download | requests, tqdm |
| Linting | ruff (lint + format), pylint (score 10.00 required) |
| Type checking | ty |
| Testing | pytest |

## Performance Profile

Measured on a single axis-aligned card image with ~3 500 reference cards:

| Stage | Time | Notes |
|---|---|---|
| Detection | ~35 ms | 3 strategies, fast |
| Hashing (1 card) | ~260 ms | PIL + imagehash library overhead |
| DB load + matrix build | ~40 ms | SQLite → Python objects → NumPy bit-matrix (one-time) |
| Matching (1 query) | ~4 ms | Vectorised NumPy Hamming distance |
| **Full pipeline** | **~0.4 s** | 3 detected × 4 rotations × (hash + match) |

### Matching Implementation

Reference hashes are converted from hex strings to a packed NumPy bit-matrix
of shape `(N, 4, 256)` on first query. Each `find_matches()` call then
computes all Hamming distances in one vectorised operation:

```python
diff = matrix ^ query[np.newaxis, :, :]   # (N, 4, 256)
per_hash_dist = diff.sum(axis=2)          # (N, 4)
combined = per_hash_dist @ weights / total_weight  # (N,)
```

This replaced a Python loop with `imagehash.hex_to_hash()` calls, yielding
a **~29× end-to-end speedup** (12 s → 0.4 s) and ~**240× matching speedup**
(950 ms → 4 ms per query).

### Remaining Bottleneck

The dominant cost is now **hashing** (~260 ms per card × 12 hash operations
= ~3.1 s). The `imagehash` library converts each image to PIL, resizes, and
computes DCT/wavelet transforms in pure Python/NumPy.

### Further Improvement Opportunities

1. **Eliminate rotation brute-force** — The perspective transform already
   knows the card orientation from the corner ordering. By inferring
   orientation during detection (e.g. top-heavy content → top-up), the 4×
   rotation cost is eliminated.

2. **Pre-filter by set or colour** — Use card border colour or set symbols
   to narrow the search space before hashing.

3. **BK-trees or VP-trees** — Index hashes in a metric tree for sub-linear
   nearest-neighbour search on Hamming distance.

4. **Multi-index hashing** — Split each 256-bit hash into 4 × 64-bit
   chunks and use hash-table lookups for candidate generation, then verify
   with full distance.

5. **Smaller hash size** — `hash_size=8` (64 bits) is faster to compute
   and compare, and more tolerant of photo-vs-scan differences. The trade-off
   is lower discrimination between visually similar cards.

6. **CNN embeddings** — Modern systems often use a small neural network
   (MobileNet, EfficientNet) to produce a compact embedding vector, then
   match via cosine similarity with FAISS or Annoy. This handles
   perspective, lighting, and rotation far better than perceptual hashes.

### Estimated Impact of Further Improvements

For a 3 500-card database with 3 detections per image:

| Improvement | Expected speedup | Effort |
|---|---|---|
| Eliminate rotation brute-force | ~4× on hash+match | Medium |
| Reduce hash_size from 16 to 8 | ~2× on hashing | Low (needs DB rebuild) |
| BK-tree indexing | ~2–5× on matching | Medium |
| CNN embeddings + FAISS | ~50–100× + better accuracy | High |
