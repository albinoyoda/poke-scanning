# Architecture

## Overview

`card_reco` is a Pokemon card recognition pipeline. Given a photo containing
one or more Pokemon cards, it detects each card region, normalises it to a
standard orientation, and identifies each card against a reference database.

Two matching backends are available:

### Hash backend (default: `--backend hash`)

```
Image → detect_cards() → [DetectedCard, …]
  └→ for each detection:
       compute_hashes(0°) → CardHashes → find_matches()
       if distance < confident_threshold: done
       else: compute_hashes(180°) → find_matches() → keep best
```

Uses perceptual hashes (ahash, phash, dhash) compared by weighted Hamming
distance against a SQLite reference database.

### CNN backend (`--backend cnn`)

```
Image → detect_cards() → [DetectedCard, …]
  +-> for each detection (Phase 1 - scoring):
       for variant in [full_frame, center_crop(85%)]:
         for orientation in [0 deg, 180 deg]:
           embedder.embed(variant) -> float32[576] -> index.search()
           keep highest cosine similarity across all 4 attempts
  +-> Phase 2 - deduplication:
       sort detections by best score (descending)
       emit results, skipping overlapping regions (IoU > 0.5)
```

Uses a **fine-tuned MobileNetV3-Small** (ONNX Runtime) to produce 576-dim
L2-normalised embeddings, matched by cosine similarity via a FAISS
`IndexFlatIP` index over 20,149 reference cards.

The model was fine-tuned on all reference card images using NT-Xent
(InfoNCE) contrastive learning.  Center-crop exploration (85% of the
warped card) strips border artifacts from perspective transforms and
graded card slabs, significantly improving similarity scores for
challenging images.  Best-first deduplication ensures overlapping
detections of the same physical card keep the highest-scoring match
rather than the first one encountered.

**Accuracy:** 100% top-5 identification across all test images (single
cards, rotated, tilted, graded, multi-card grids).  Surpasses the hash
backend on difficult cases (low-contrast borders, 151-set illustration
style, graded slabs).

## Module Map

| Module | Responsibility |
|---|---|
| `card_reco/__init__.py` | Public API re-exports: `identify_cards()`, `identify_cards_from_array()` |
| `card_reco/pipeline.py` | Pipeline orchestration, crop exploration, preprocessing helpers |
| `card_reco/detector/__init__.py` | Top-level detection orchestrator: `detect_cards()`, `_four_point_transform()` |
| `card_reco/detector/strategies.py` | Detection strategies: Canny, adaptive threshold, HSV segmentation, Hough quads |
| `card_reco/detector/corners.py` | Corner extraction, refinement (`refine_corners_from_hull`, `refine_corners_edge_intersect`), ordering, aspect-ratio checks |
| `card_reco/detector/nms.py` | Non-max suppression and centroid deduplication |
| `card_reco/detector/quality.py` | Contour quality scoring and corner edge-fraction verification |
| `card_reco/detector/constants.py` | Shared constants (`CARD_WIDTH`, `CARD_HEIGHT`, `CANNY_PAIRS`, etc.) and CLAHE factory |
| `card_reco/hasher.py` | Perceptual hashing via `imagehash` (ahash, phash, dhash) |
| `card_reco/matcher.py` | Vectorised NumPy matching with weighted Hamming distance |
| `card_reco/database.py` | SQLite wrapper for the reference hash database |
| `card_reco/embedder.py` | CNN embedding extraction via ONNX Runtime (MobileNetV3-Small) |
| `card_reco/faiss_index.py` | FAISS-based vector index for card embedding search |
| `card_reco/models.py` | Dataclasses: `DetectedCard`, `CardHashes`, `CardRecord`, `MatchResult` |
| `card_reco/debug.py` | Debug image writer for pipeline visualisation |
| `card_reco/cli.py` | CLI entry point (`card-reco identify <image>`) |

## Detection Pipeline (`detector/`)

The detector is a subpackage with focused modules:

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

## CNN Embedding (`embedder.py`, `faiss_index.py`)

### Model

MobileNetV3-Small backbone producing 576-dimensional L2-normalised
embeddings.  The model was fine-tuned from ImageNet-1k pretrained weights
using contrastive learning (`scripts/finetune_model.py`):

| Parameter | Value |
|---|---|
| Architecture | MobileNetV3-Small (backbone only, no classifier) |
| Embedding dim | 576 (global average pool output) |
| Loss | NT-Xent (InfoNCE), temperature 0.07 |
| Training | SimCLR-style: two augmented views per card, all other cards as negatives |
| Augmentation | RandomResizedCrop(224), rotation +/-15 deg, perspective 0.15, colour jitter, Gaussian blur, random erasing |
| Schedule | 2 warmup epochs (backbone frozen) + 10 full fine-tuning epochs |
| Optimiser | AdamW (backbone lr=1e-4, projector lr=1e-3), cosine annealing |
| Reference images | 20,149 cards across 168 sets |
| Training loss | 0.596 -> 0.009 over 12 epochs |
| ONNX model size | ~4 MB (model + external data) |
| Inference | ONNX Runtime, ~2 ms per card on CPU |

A projection head (576 -> 576 -> 128, ReLU) is used during training for
the contrastive objective, then discarded at ONNX export time.

### Index

FAISS `IndexFlatIP` (inner product = cosine similarity for L2-normalised
vectors).  Exact brute-force search over 20,149 reference embeddings.

| File | Content |
|---|---|
| `card_embeddings.faiss` | FAISS index (20,149 x 576 float32) |
| `card_embeddings_meta.json` | Ordered card ID list + metadata sidecar |

### CNN Pipeline Thresholds

| Threshold | Value | Meaning |
|---|---|---|
| `_CNN_CONFIDENT_THRESHOLD` | 0.70 | Skip further orientations/crops |
| `_CNN_MATCH_THRESHOLD` | 0.40 | Minimum similarity to include in results |
| `_CNN_CENTER_CROP` | 0.85 | Center crop fraction for border removal |

## Hashing (`hasher.py`)

Each detected card image is converted to a PIL Image and hashed three ways
at `hash_size=16` (256-bit hashes, 64-char hex strings):

| Hash | Algorithm | Strength |
|---|---|---|
| `ahash` | Average hash | Fast, tolerant of minor changes |
| `phash` | Perceptual hash (DCT) | Best overall texture matching |
| `dhash` | Difference hash | Good edge/gradient sensitivity |

## Matching (`matcher.py`)

Brute-force linear scan over all reference cards. For each reference card,
three Hamming distances are computed and combined with weights:

```
combined = (0.8·ahash + 1.0·phash + 1.0·dhash) / 2.8
```

Results below the threshold (default 40.0) are sorted by distance.

The top-level API tries **at most 2 orientations** (0° and 180°) for each
detected card.  If the 0° orientation yields a confident match (distance
below `confident_threshold`, default 25.0), the 180° flip is skipped.
Empirical analysis showed 90°/270° rotations never produce useful matches
(all "wins" had distances > 84 with wrong cards).

## Reference Database (`database.py`)

SQLite table `cards` with columns: `id`, `name`, `set_id`, `set_name`,
`number`, `rarity`, `image_path`, `ahash`, `phash`, `dhash`.

Built by `scripts/build_hash_db.py` from card images in
`data/images/{set_id}/` and metadata in `data/metadata/cards/{set_id}.json`.

## Data Layout

```
data/
├── card_hashes.db              # Reference hash database (hash backend)
├── card_embeddings.faiss       # FAISS embedding index (20,149 × 576)
├── card_embeddings_meta.json   # Card metadata sidecar for FAISS index
├── mobilenet_v3_small.onnx     # Fine-tuned CNN model (CNN backend)
├── mobilenet_v3_small.onnx.data# ONNX external data (model weights, ~3.7 MB)
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
| `export_model.py` | Export pretrained MobileNetV3-Small to ONNX (one-time) |
| `finetune_model.py` | Fine-tune MobileNetV3-Small with contrastive learning |
| `build_embedding_db.py` | Build FAISS embedding index from reference card images |
| `find_cards.py` | Quick CLI card finder |
| `find_sets.py` | Search available sets |
| `check_cards.py` | Verify card data integrity |

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Build | hatchling + uv |
| Computer vision | OpenCV (`opencv-python ≥ 4.8`) |
| Image processing | Pillow (≥ 10.0), NumPy (≥ 1.24) |
| Perceptual hashing | `imagehash` (≥ 4.3) |
| CNN inference | ONNX Runtime (≥ 1.17) |
| Vector search | FAISS-cpu (≥ 1.7) |
| Reference DB | SQLite 3 (stdlib) |
| Data download | requests, tqdm |
| CNN fine-tuning | PyTorch + torchvision (dev only, for `finetune_model.py`) |
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
| **Full pipeline (single card)** | **~0.2 s** | 1 detection × 1–2 orientations |
| **Full pipeline (3×3 grid)** | **~0.8 s** | 13 detections × 1–2 orientations |

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

The dominant cost is now **hashing** (~260 ms per card × 1–2 orientations).
The `imagehash` library converts each image to PIL, resizes, and computes
DCT/wavelet transforms in pure Python/NumPy.

### Further Improvement Opportunities

1. **Pre-filter by set or colour** — Use card border colour or set symbols
   to narrow the search space before hashing.

2. **BK-trees or VP-trees** — Index hashes in a metric tree for sub-linear
   nearest-neighbour search on Hamming distance.

3. **Multi-index hashing** — Split each 256-bit hash into 4 × 64-bit
   chunks and use hash-table lookups for candidate generation, then verify
   with full distance.

4. **Smaller hash size** — `hash_size=8` (64 bits) is faster to compute
   and compare, and more tolerant of photo-vs-scan differences. The trade-off
   is lower discrimination between visually similar cards.

5. **~~CNN embeddings~~** — Implemented as the `cnn` backend. Uses
   MobileNetV3-Small fine-tuned with NT-Xent contrastive learning on all
   20,149 reference cards. Achieves 100% top-5 identification on all test
   images, surpassing the hash backend on difficult cases. Center-crop
   exploration and best-first deduplication handle border artifacts and
   overlapping detections.

### Estimated Impact of Further Improvements

For a 3 500-card database with 3 detections per image:

| Improvement | Expected speedup | Effort |
|---|---|---|
| ~~Eliminate rotation brute-force~~ | ~~Done (2× on hashing)~~ | ~~Done~~ |
| Reduce hash_size from 16 to 8 | ~2× on hashing | Low (needs DB rebuild) |
| BK-tree indexing | ~2–5× on matching | Medium |
| ~~CNN embeddings + FAISS~~ | ~~Done (cnn backend)~~ | ~~Done~~ |
