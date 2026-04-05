# Phase 3 — CNN Embedding + FAISS Implementation Plan

## Current State (Post Phase 1)

| Stage | Time | Calls |
|---|---|---|
| **Full pipeline** | **8 212 ms** | |
| `_explore_crops` | 7 397 ms | 15 |
| `find_matches` (brute XOR) | 6 771 ms | 213 |
| `compute_hashes` (imagehash) | 1 283 ms | 183 |
| `_name_group_fallback` | 2 942 ms | 139 |
| `detect_cards` | 287 ms | 1 |

**Throughput**: 0.12 Hz.  Target: 5 Hz (200 ms/frame).

### Root cause analysis

1. **Match calls dominate**: 213 calls × ~32 ms each.  Each call XORs
   256-bit × 3 hashes across 20 149 reference cards.  NumPy vectorisation
   helps but O(N) per query with N=20k is fundamentally expensive when
   repeated 213 times.

2. **Crop exploration is a workaround for hash fragility**: Perceptual
   hashes are sensitive to small crop/rotation/lighting changes, so the
   pipeline tries ~12 preprocessing variants per failed card (denoise,
   CLAHE, pad, re-warp).  A robust embedding that tolerates these
   variations would eliminate the need for crop exploration entirely.

3. **Name-group fallback is a matching workaround**: When no single
   reference card passes the distance threshold, the matcher groups cards
   by name and checks if a name "consensus" exists.  A better similarity
   metric eliminates this heuristic layer.

---

## Goal

Replace perceptual hashing + brute-force Hamming matching with:

- A **CNN feature extractor** that produces a compact float-vector
  embedding per card image.
- A **FAISS index** for sub-millisecond nearest-neighbour search over the
  20 149 reference embeddings.

### Expected performance budget

| Stage | Target | Notes |
|---|---|---|
| `detect_cards` | ~300 ms | Unchanged |
| CNN embedding (per card) | ~10–20 ms | ONNX Runtime CPU, MobileNetV3-Small |
| FAISS search (per card) | <1 ms | `IndexFlatIP` on 20k × 512 dims |
| Orientation (×2) | ~25–45 ms | 0° + 180° only |
| **Per card total** | **~35–65 ms** | |
| **7 cards** | **~550–750 ms** | Detection + 7 × embedding + 7 × search |
| **Pipeline overhead** | ~50 ms | Image load, debug output, etc. |
| **Total (7 cards)** | **~600–800 ms** | ~1.3–1.7 Hz |

With ONNX int8 quantisation: embedding drops to ~5–8 ms/card →
**~400–500 ms** total → **2–2.5 Hz**.

Crop exploration elimination saves the most time — going from 213 match
calls to ~14 (7 cards × 2 orientations) is the key structural change.

---

## Architecture Overview

```
Image → detect_cards() → [DetectedCard, …]            # UNCHANGED
  └→ for each detection:
       embed(0°)  → float32[512]  → faiss_search()    # NEW
       if distance < confident_threshold: done
       else: embed(180°) → faiss_search() → keep best  # SIMPLIFIED
```

### New modules

| Module | Responsibility |
|---|---|
| `card_reco/embedder.py` | CNN model loading, ONNX Runtime inference, preprocessing |
| `card_reco/faiss_index.py` | FAISS index build, load, save, search wrapper |

### Modified modules

| Module | Change |
|---|---|
| `pipeline.py` | Replace `compute_hashes` → `embed`, `matcher.find_matches` → `faiss_search`. Remove `_explore_crops`. Simplify to 0°/180° only |
| `matcher.py` | **Deprecated** — replaced by `faiss_index.py`. Keep as fallback behind feature flag during migration |
| `hasher.py` | **Deprecated** — replaced by `embedder.py`. Keep as fallback behind feature flag |
| `models.py` | `CardHashes` → `CardEmbedding`. `CardRecord` drops hash fields, adds embedding index. `MatchResult.distances` simplified |
| `database.py` | Schema drops `ahash/phash/dhash` columns. Metadata-only (id, name, set, rarity, image_path). Embeddings stored in FAISS index file |
| `__init__.py` | Public API unchanged: `identify_cards()`, `identify_cards_from_array()` |
| `cli.py` | No changes to interface.  `--backend hash|cnn` flag for migration period |

### New data files

| File | Content | Size |
|---|---|---|
| `data/mobilenet_v3_small.onnx` | ONNX model (fp32) | ~10 MB |
| `data/card_embeddings.faiss` | FAISS flat index (20 149 × 512 × float32) | ~39 MB |
| `data/card_embeddings_meta.json` | Ordered list of card IDs mapping FAISS row → card ID | ~600 KB |

### Unchanged

- `detector/` subpackage — completely independent.
- `debug.py` — still writes pipeline stage images.
- `scripts/download_reference_data.py` — still downloads card images/metadata.

---

## Model Selection

### Primary: MobileNetV3-Small (torchvision → ONNX)

| Property | Value |
|---|---|
| Architecture | MobileNetV3-Small |
| Input | 224×224 RGB, normalised (ImageNet mean/std) |
| Output | 576-dim feature vector (from `avgpool`, before classifier) |
| Model size | ~2.5 MB (fp32 ONNX) |
| Inference (CPU, ONNX Runtime) | ~5–15 ms per image |
| Pre-trained | ImageNet-1k |
| Why | Smallest/fastest model that produces reasonable visual features. Card images have distinct colour patterns, artwork, borders — ImageNet features capture these well |

### Fallback: EfficientNet-B0

| Property | Value |
|---|---|
| Output | 1280-dim feature vector |
| Model size | ~20 MB |
| Inference (CPU) | ~15–25 ms per image |
| Why | Higher-dimensional embeddings if MobileNetV3-Small accuracy is insufficient |

### Embedding normalisation

All embeddings are L2-normalised before storage and search.  This converts
L2 distance to cosine similarity via inner product:
`cosine_sim = dot(a, b)` when `‖a‖ = ‖b‖ = 1`.

FAISS `IndexFlatIP` (inner product) on normalised vectors is equivalent to
cosine similarity search, which is the standard metric for image retrieval.

---

## Training / Fine-Tuning Strategy

### Phase 3a: Zero-shot (no training)

Use pre-trained ImageNet features directly.  No fine-tuning.

**Rationale**: The 20k reference images are *scans* (clean, high-quality,
standardised).  Query images are *photos* (noisy, perspective-transformed,
variable lighting).  Pre-trained features may struggle with this domain gap.
However, this is the fastest path to a working prototype and gives us a
baseline to evaluate.

**Evaluation**: Run the existing test suite.  If identification rates match
or exceed the hash-based approach (7/9 axis-aligned, multi-card grids,
graded cards), zero-shot is sufficient.

### Phase 3b: Fine-tuning (if needed)

If zero-shot accuracy is insufficient:

1. **Data augmentation pipeline**: For each reference card image, generate
   synthetic "photo-like" variants:
   - Random perspective transform (±15°)
   - Random rotation (±10°)
   - Colour jitter (brightness ±20%, contrast ±15%, saturation ±20%)
   - Gaussian noise (σ = 5–15)
   - Bilateral filter blur
   - CLAHE-like local contrast changes
   - Random crop (±5% border)

2. **Contrastive loss (SimCLR-style)**: Train the feature extractor so that
   augmented variants of the same card are close in embedding space, while
   different cards are far apart.  Use NT-Xent loss with in-batch negatives.

3. **Training data**: 20 149 reference images × 10 augmentations each =
   ~200k training pairs.  No external data needed — the reference DB *is*
   the training data.

4. **Training time**: ~1–2 hours on CPU, ~10 minutes on GPU.  One-time cost.

5. **Output**: Fine-tuned ONNX model replaces the ImageNet-pretrained one.

---

## FAISS Index Design

### Index type: `IndexFlatIP` (brute-force inner product)

At 20k vectors, brute-force is optimal:

| Index type | Search time (20k, 512-d) | Build time | Memory |
|---|---|---|---|
| `IndexFlatIP` | <0.5 ms | Instant | 39 MB |
| `IVF256,Flat` | <0.2 ms | ~2 s | 39 MB + overhead |
| `HNSW32` | <0.1 ms | ~5 s | 39 MB + graph |

The flat index is simpler, has exact results, and <0.5 ms is already well
within budget.  No need for approximate methods at this scale.

### Index operations

```python
import faiss
import numpy as np

# Build
d = 512  # embedding dimension (576 for MobileNetV3-Small after projection)
index = faiss.IndexFlatIP(d)
embeddings = np.array([...], dtype=np.float32)  # (20149, d), L2-normalised
index.add(embeddings)
faiss.write_index(index, "data/card_embeddings.faiss")

# Search
query = np.array([[...]], dtype=np.float32)  # (1, d), L2-normalised
D, I = index.search(query, k=5)  # D = similarities, I = indices
# D[0] is array of cosine similarities (1.0 = identical, 0.0 = orthogonal)
# I[0] is array of reference card indices
```

### Metadata mapping

FAISS indices are sequential integers (0, 1, 2, …).  A separate
`card_embeddings_meta.json` file maps each index to the card's metadata:

```json
[
  {"idx": 0, "id": "base1-1", "name": "Alakazam", "set_id": "base1", ...},
  {"idx": 1, "id": "base1-2", "name": "Blastoise", "set_id": "base1", ...},
  ...
]
```

Alternatively, keep the SQLite DB for metadata and just store the ordered
card ID list.  The FAISS index row number maps to the card ID list index,
which maps to the SQLite row.

---

## Implementation Steps

### Step 1: Add ONNX Runtime + FAISS dependencies

```toml
# pyproject.toml
dependencies = [
    ...
    "onnxruntime>=1.17",
    "faiss-cpu>=1.7",
]
```

Also add `torch` and `torchvision` as dev dependencies (for model export
only, not runtime):

```toml
[dependency-groups]
dev = [
    ...
    "torch>=2.0",
    "torchvision>=0.15",
]
```

**Risk**: `faiss-cpu` has platform-specific builds.  Verify it installs
cleanly on Windows with `uv sync`.

### Step 2: Export MobileNetV3-Small to ONNX (`scripts/export_model.py`)

```python
import torch
import torchvision.models as models

model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
# Remove classifier head — we want the feature extractor output
model.classifier = torch.nn.Identity()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input,
    "data/mobilenet_v3_small.onnx",
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=17,
)
```

This script runs once and produces the ONNX file.  It is a dev-only
dependency (torch/torchvision are not needed at runtime).

**Output dimension**: MobileNetV3-Small `avgpool` output is 576-dim.  We
may optionally add a learned projection head (576→512 or 576→256) during
fine-tuning to reduce dimensionality and speed up FAISS search.  For
zero-shot, we use the raw 576-dim features.

### Step 3: Implement `embedder.py`

```
src/card_reco/embedder.py
```

Key interfaces:

```python
EMBEDDING_DIM: int = 576  # MobileNetV3-Small feature dimension

class CardEmbedder:
    def __init__(self, model_path: Path | str | None = None) -> None:
        """Load ONNX model via onnxruntime.InferenceSession."""

    def embed(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """BGR card image → L2-normalised embedding vector (576,)."""

    def embed_pil(self, pil_image: Image.Image) -> NDArray[np.float32]:
        """PIL RGB Image → L2-normalised embedding vector (576,)."""

    def embed_batch(self, images: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """Multiple BGR images → batch of embeddings (N, 576)."""
```

**Preprocessing** (matching ImageNet conventions):
1. Resize to 224×224 (bilinear).
2. Convert BGR→RGB, uint8→float32, scale to [0, 1].
3. Normalise: `(pixel - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]`.
4. Transpose to NCHW: `(1, 3, 224, 224)`.

**Post-processing**:
1. L2-normalise the output vector.

### Step 4: Implement `faiss_index.py`

```
src/card_reco/faiss_index.py
```

Key interfaces:

```python
class CardIndex:
    def __init__(
        self,
        index_path: Path | str | None = None,
        meta_path: Path | str | None = None,
    ) -> None:
        """Load FAISS index and card ID mapping."""

    def search(
        self,
        embedding: NDArray[np.float32],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[MatchResult]:
        """Single embedding → top-k matches above similarity threshold."""

    def search_batch(
        self,
        embeddings: NDArray[np.float32],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[list[MatchResult]]:
        """Batch of embeddings → per-query top-k matches."""

    @staticmethod
    def build(
        embeddings: NDArray[np.float32],
        card_ids: list[str],
        index_path: Path | str,
        meta_path: Path | str,
    ) -> None:
        """Build and save FAISS index + metadata from embeddings."""
```

**Threshold semantics change**: Hash-based distances are 0–256 (lower =
better).  Cosine similarities are -1 to 1 (higher = better).  The
`threshold` parameter becomes a *minimum similarity* rather than a
*maximum distance*.

### Step 5: Update `models.py`

```python
@dataclass
class CardEmbedding:
    """CNN embedding for a single card image."""
    vector: NDArray[np.float32]  # L2-normalised, shape (embedding_dim,)

@dataclass
class CardRecord:
    id: str
    name: str
    set_id: str
    set_name: str
    number: str
    rarity: str
    image_path: str
    # No hash/embedding fields — embeddings live in FAISS index

@dataclass
class MatchResult:
    card: CardRecord
    distance: float         # Cosine similarity (0–1, higher = better)
    distances: dict[str, float]  # {"cosine": 0.87} or empty — kept for API compat
    rank: int = 0
```

**Migration note**: `CardHashes` is kept for backward compatibility during
the transition period but is unused by the CNN path.

### Step 6: Build embedding database (`scripts/build_embedding_db.py`)

Mirrors `build_hash_db.py` but uses the CNN embedder:

1. Load ONNX model via `CardEmbedder`.
2. For each reference card image: `embed_pil(pil_image)` → 576-dim vector.
3. Collect all embeddings into a `(20149, 576)` float32 matrix.
4. L2-normalise each row.
5. Build FAISS `IndexFlatIP` and save to `data/card_embeddings.faiss`.
6. Save ordered card ID list to `data/card_embeddings_meta.json`.

**Batch optimisation**: Process in batches of 32–64 images.
`embed_batch()` feeds a single large tensor through ONNX Runtime, which
is significantly faster than one-at-a-time inference.

**Estimated build time**: 20 149 cards @ ~10 ms/card = ~200 s (~3.5 min).

### Step 7: Update `pipeline.py`

**Key changes**:

1. Replace `CardMatcher` with `CardEmbedder` + `CardIndex`.
2. Replace `compute_hashes()` calls with `embedder.embed()`.
3. Replace `matcher.find_matches()` calls with `index.search()`.
4. **Remove `_explore_crops()` entirely** — CNN embeddings should be robust
   enough to match without crop exploration.
5. Simplify to: embed(0°) → search → if confident, done → embed(180°) →
   search → keep best.
6. Remove `_name_group_fallback` — FAISS top-k results replace consensus
   logic.

**Threshold recalibration**:

| Hash-based | CNN-based | Semantics |
|---|---|---|
| `threshold = 40.0` (max distance) | `threshold = 0.5` (min similarity) | Accept if above |
| `confident_threshold = 25.0` | `confident_threshold = 0.7` | Skip 180° flip |
| `_RELAXED_FALLBACK_HEADROOM = 25.0` | Removed | No longer needed |
| `_CROP_EXPLORE_*` | Removed | No crop exploration |

**Exact threshold values will be determined empirically** by running the
test suite with the CNN backend and measuring the similarity distribution
for correct vs. incorrect matches.

### Step 8: Update `database.py`

Simplify schema to metadata-only:

```sql
CREATE TABLE cards (
    id TEXT PRIMARY KEY,
    name TEXT,
    set_id TEXT,
    set_name TEXT,
    number TEXT,
    rarity TEXT,
    image_path TEXT
);
```

Hash columns removed.  The database is only used for looking up card
metadata by ID.  Embeddings and search are handled by FAISS.

### Step 9: Add `--backend` flag to CLI

During migration, support both backends:

```
card-reco identify <image> --backend hash   # existing hash-based pipeline
card-reco identify <image> --backend cnn    # new CNN+FAISS pipeline (default)
```

This allows A/B comparison and safe rollback.

### Step 10: Update tests

1. **Unit tests for `embedder.py`**: Verify embedding shape, normalisation,
   deterministic output, batch consistency.
2. **Unit tests for `faiss_index.py`**: Verify build/save/load, search
   returns correct format, threshold filtering.
3. **Integration tests**: Run existing test images through CNN backend.
   Adjust expected results if identification rates change.
4. **Regression tests**: Compare CNN vs hash results on the full test suite.
5. **Performance tests**: Verify per-card embedding time < 20 ms, search
   time < 1 ms.

### Step 11: ONNX Quantisation (optional, Phase 3c)

If inference time needs further reduction:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "data/mobilenet_v3_small.onnx",
    "data/mobilenet_v3_small_int8.onnx",
    weight_type=QuantType.QUInt8,
)
```

Expected speedup: ~1.5–2× on x86-64 with AVX2/VNNI.

---

## Implementation Order

| Step | Description | Depends on | Effort |
|---|---|---|---|
| 1 | Add `onnxruntime`, `faiss-cpu` deps | — | Low |
| 2 | Export MobileNetV3-Small to ONNX | Step 1 (torch dev dep) | Low |
| 3 | Implement `embedder.py` + unit tests | Step 2 (ONNX model) | Medium |
| 4 | Implement `faiss_index.py` + unit tests | Step 1 | Medium |
| 5 | Update `models.py` | — | Low |
| 6 | Build embedding DB script + run it | Steps 3, 4 | Medium |
| 7 | Update `pipeline.py` (CNN path) | Steps 3, 4, 5, 6 | High |
| 8 | Simplify `database.py` | Step 5 | Low |
| 9 | Add `--backend` CLI flag | Step 7 | Low |
| 10 | Update/add tests, threshold calibration | Steps 7, 9 | High |
| 11 | ONNX int8 quantisation (optional) | Step 3 | Low |

### Recommended grouping

**Phase 3a** (Steps 1–4): Foundation.  New modules with tests, no existing
code changes.  Can be merged independently.

**Phase 3b** (Steps 5–8): Integration.  Wire CNN+FAISS into the pipeline.
Build embedding DB.  This is the breaking change.

**Phase 3c** (Steps 9–10): Migration.  CLI flag, test updates, threshold
calibration.  Validate identification rates.

**Phase 3d** (Step 11): Optimisation.  Quantisation, batch inference.

---

## Risk Analysis

| Risk | Impact | Mitigation |
|---|---|---|
| **Zero-shot accuracy too low** | Cards not identified | Fine-tune with augmented data (Phase 3b training strategy) |
| **Photo-vs-scan domain gap** | Low similarity for correct matches | Data augmentation during fine-tuning; add projection head |
| **`faiss-cpu` install issues on Windows** | Build breaks | Pin version, vendor wheel if needed; fallback to pure NumPy cosine |
| **ONNX Runtime version conflicts** | Import errors | Pin onnxruntime version, test in CI |
| **Embedding dimension too large** | FAISS memory/speed | Add PCA or learned projection (576→256) |
| **Threshold calibration difficulty** | False positives/negatives | Plot similarity distributions for correct/incorrect matches; use ROC analysis |
| **Crop exploration still needed** | Less speedup than expected | Keep simplified version; try 2–3 crops max instead of 12 |

### Fallback plan

The `--backend hash` flag preserves the existing pipeline.  If CNN+FAISS
does not meet accuracy targets within reasonable effort, revert to hash
backend and focus on Tier 2 optimisations instead (sub-linear hash
matching with BK-trees, temporal tracking for video mode).

---

## Success Criteria

| Metric | Hash baseline | Target |
|---|---|---|
| **Pipeline time (7 cards)** | 8 212 ms | ≤ 800 ms |
| **Throughput** | 0.12 Hz | ≥ 1.5 Hz |
| **Axis-aligned identification** | 7/9 | ≥ 7/9 |
| **Rotated card identification** | 2/4 | ≥ 2/4 |
| **Multi-card grid rate** | passes | passes |
| **Graded card rate** | passes | passes |
| **Embedding time (per card)** | N/A | ≤ 20 ms |
| **Search time (per card)** | ~32 ms | ≤ 1 ms |
| **Model size** | N/A | ≤ 15 MB |
| **Index size** | 5 MB (hash DB) | ≤ 50 MB |
| **pylint score** | 10.00 | 10.00 |
| **All tests pass** | 209 passed, 3 xfail | same or better |

---

## Dependency Summary

### Runtime (new)

| Package | Version | Size | Purpose |
|---|---|---|---|
| `onnxruntime` | ≥ 1.17 | ~50 MB | CNN inference |
| `faiss-cpu` | ≥ 1.7 | ~15 MB | Vector search |

### Dev-only (new)

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Model export to ONNX (one-time) |
| `torchvision` | ≥ 0.15 | Pre-trained MobileNetV3-Small |

### Removed at end of migration

| Package | Reason |
|---|---|
| `imagehash` | Replaced by CNN embeddings |
| `pywt` (transitive) | Was only used by whash (already removed) |

---

## File Changes Summary

### New files

| File | Type |
|---|---|
| `src/card_reco/embedder.py` | Library module |
| `src/card_reco/faiss_index.py` | Library module |
| `scripts/export_model.py` | Dev script (one-time) |
| `scripts/build_embedding_db.py` | Build script |
| `data/mobilenet_v3_small.onnx` | Model artifact |
| `data/card_embeddings.faiss` | Index artifact |
| `data/card_embeddings_meta.json` | Metadata mapping |
| `tests/test_embedder.py` | Unit tests |
| `tests/test_faiss_index.py` | Unit tests |

### Modified files

| File | Nature of change |
|---|---|
| `pyproject.toml` | Add onnxruntime, faiss-cpu deps |
| `src/card_reco/models.py` | Add `CardEmbedding`, simplify `CardRecord` |
| `src/card_reco/pipeline.py` | CNN path, remove `_explore_crops` |
| `src/card_reco/database.py` | Metadata-only schema |
| `src/card_reco/__init__.py` | Re-export embedder if needed |
| `src/card_reco/cli.py` | `--backend` flag |
| `tests/test_integration.py` | Adapt for CNN backend |
| `tests/test_regression.py` | Adapt for CNN backend |
| `docs/ARCHITECTURE.md` | Document new architecture |
| `.github/copilot-instructions.md` | Update pipeline description |

### Deprecated (kept behind feature flag, removed later)

| File | Status |
|---|---|
| `src/card_reco/hasher.py` | Unused by CNN path |
| `src/card_reco/matcher.py` | Unused by CNN path |
| `data/card_hashes.db` | Unused by CNN path |
