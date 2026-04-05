# Real-Time Performance Plan

**Goal**: ≥1 Hz sustained identification on multi-card images (7 cards),
with a path to real-time video scanning (~5–10 effective FPS).

**Status**: Phase A complete. CNN pipeline improved from 862 ms (1.16 Hz)
to **185 ms (5.4 Hz)** with all optimizations enabled — a **4.7× speedup**.

## Current Profiling Baseline

Test image: `7_cards.png` (2048×1474, 7 physical cards)

### Hash Backend (8 069 ms → 0.12 Hz)

| Stage | Time | Calls | % of total |
|---|---|---|---|
| `detect_cards()` | 272 ms | 1 | 3.4% |
| `compute_hashes` (imagehash) | 1 260 ms | 183 | 15.6% |
| `find_matches` (NumPy XOR+sum) | 6 593 ms | 213 | 81.7% |
| └ `_name_group_fallback` | 2 917 ms | 139 | 36.1% |
| `_denoise_clahe` (bilateral) | 317 ms | 45 | 3.9% |
| `_explore_crops` (orchestrator) | 7 245 ms | 15 | 89.8% |
| **Full pipeline** | **8 069 ms** | | |

**Root cause**: 15 of 16 detections fail confident matching → fall into
`_explore_crops` → 183 hash calls + 213 match queries. The hash backend
is fundamentally unsuitable for real-time use because:
- `imagehash` is slow (~7 ms/hash, PIL-based)
- Brute-force hamming over 20K × 256-bit reference matrix is expensive
- Crop exploration multiplies both costs by ~12× per failed detection

### CNN Backend (862 ms → 1.16 Hz) ← Starting point

| Stage | Time | Calls | % of total |
|---|---|---|---|
| `detect_cards()` | 272 ms | 1 | 31.6% |
| Image prep (rotate, center-crop) | ~23 ms | 64 | 2.7% |
| `embedder.embed()` individual | 480 ms | 64 | 55.7% |
| `card_index.search()` | 145 ms | 64 | 16.8% |
| Phase 2 dedup + overhead | ~(remainder) | | |
| **Full pipeline** | **862 ms** | | |

**Key insight**: The CNN backend is already 9.4× faster than hash and
almost at 1 Hz. Every optimization minute should be spent on CNN.

---

## Measured Optimization Opportunities

These numbers were measured on the same 7_cards.png image:

| Optimization | Before | After | Speedup |
|---|---|---|---|
| Batch embedding (64 images) | 480 ms (individual) | 284 ms (batch) | 1.7× |
| Early-exit on confident match | 480+145 ms (all 64) | 126 ms (16+3 extra) | 4.9× |
| Detection at 1024px | 272 ms | 134 ms | 2.0× |
| Detection at 768px | 272 ms | 101 ms | 2.7× |

### Projected vs Actual Optimized CNN Pipeline

| Configuration | Projected | Actual (avg 5 runs) | Cards found |
|---|---|---|---|
| Baseline (batch + early-exit) | ~300 ms | **426 ms (2.3 Hz)** | 13 |
| + max_detect_dim=1024 | ~230 ms | **332 ms (3.0 Hz)** | 14 |
| + max_detect_dim=1024 + fast=True | ~160 ms | **185 ms (5.4 Hz)** | 11 |

Additional benchmark on `3x3_top_loaders.png` (9 cards):

| Configuration | Time | Cards found |
|---|---|---|
| Default CNN | 350 ms (2.9 Hz) | 16 |
| Optimized (1024px + fast) | 255 ms (3.9 Hz) | 15 |

The fast mode trades ~2 detections for a 2.3× speedup over the new
default. For video use (where temporal voting recovers missed cards),
this is an excellent trade-off.

---

## Improvement Roadmap

### Phase A — CNN Pipeline Optimizations (target: ≥3 Hz static)

These are algorithm-level changes to the existing CNN pipeline.

#### A1. Batch ONNX Inference ✅ IMPLEMENTED

**Problem**: `_run_cnn_pipeline` calls `embedder.embed()` 64 times
individually (16 detections × 4 variants each). Each call has ONNX
session overhead and cannot leverage GPU/CPU parallelism.

**Fix**: Collect all variant images, call `embedder.embed_batch()` once.
Measured: 480 ms → 284 ms for 64 images (1.7× faster).

**Implementation**: `_run_cnn_pipeline` in `pipeline.py` now uses
two batched phases instead of individual embed calls.

**Effort**: Low — `embed_batch` already exists.

#### A2. Early-Exit on Confident Match ✅ IMPLEMENTED

**Problem**: All 4 variants (full×0°, full×180°, crop×0°, crop×180°)
are always computed, even when the first variant matches at >0.70
cosine similarity.

**Fix**: Two-phase approach in `_run_cnn_pipeline`:
1. Phase 1: Batch-embed all detections at 0° full-frame.
   FAISS search each. Cards with similarity ≥ 0.70 skip fallback.
2. Phase 2: Batch the 3 fallback variants (180°, center-crop×0°,
   center-crop×180°) for remaining cards.
3. Phase 3: Sort all results by best score descending,
   deduplicate via IoU (best score wins).

Measured: 625 ms → 126 ms (4.9× faster). 15/16 detections are
confident on the first try.

**Implementation**: Combined with A1 in the rewritten `_run_cnn_pipeline`.
New `_cnn_fallback_variants()` helper generates the 3 fallback images.

**Effort**: Low.

#### A3. Reduced-Resolution Detection ✅ IMPLEMENTED

**Problem**: `detect_cards()` runs 5 strategies (Canny, adaptive,
HSV, morph-close, Hough) at full 2048×1474 resolution. Per-strategy
breakdown:

| Strategy | Time at 2048px |
|---|---|
| Canny (2 pairs × 2 modes) | 53 ms |
| Adaptive threshold | 20 ms |
| HSV segmentation | 55 ms |
| Hough lines | 58 ms |
| Morph close | 15 ms |
| **Total strategies** | **213 ms** |

**Fix**: Downscale to 1024px long edge before detection, then scale
corners back to original coordinates for perspective warp at full
resolution. Measured: 272 ms → 134 ms.

**Implementation**: `detect_cards()` gained keyword-only `max_detect_dim`
parameter. New `_downscale_for_detection()` helper handles scaling.
Corners are computed on the downscaled image and mapped back to
original coordinates for full-resolution warping.

**Risk**: At 1024px, detected 21 cards vs 16 at 2048px (more false
positives, but NMS + CNN dedup handles these). At 768px, same 21.
At 512px, 24 candidates — too many false positives.

**Effort**: Low — add `max_detect_dim` parameter to `detect_cards()`.

#### A4. Skip Overlapping Detections Before Embedding ❌ DROPPED

**Problem**: 16 detections for 7 physical cards means ~9 duplicates
are all embedded and searched. The current Phase 2 dedup is
*after* embedding everything.

**Attempted fix**: Apply IoU-based pre-filtering during Phase 1.
Once a card matches confidently, skip overlapping detections.

**Result**: Caused test regressions. A worse-quality overlapping
detection could claim a region (in processing order) before a
better detection was evaluated. The dedup must happen *after* all
cards are scored so the highest-scoring detection always wins.
The Phase 3 dedup (sort by score descending, then IoU suppress)
achieves this correctly.

**Estimated saving was**: ~40 ms. Not worth the quality loss.

#### A5. Detection Strategy Pruning ✅ IMPLEMENTED

**Problem**: 5 detection strategies is overkill for well-lit photos.
Almost all real cards are found by Canny alone. HSV and Hough are
backup strategies that rarely contribute unique detections.

**Fix**: `find_card_contours()` gained keyword-only `fast` parameter.
When `fast=True`, strategies 3 (HSV), 4 (morph-close), and 5 (Hough)
are skipped. Only Canny + adaptive threshold run.

**Implementation**: `detect_cards()` forwards the `fast` parameter.
The public API (`identify_cards`, `identify_cards_from_array`) expose
it as a keyword argument.

**Measured saving**: Combined with A3, detection drops from 272 ms
to ~50 ms. Trades ~2 detections for significant speed improvement.

**Effort**: Low — add `fast` parameter.

---

### Phase B — Video Pipeline Architecture (target: >=5 effective FPS)

These changes update the Scanner GUI for real-time continuous scanning.

#### B1. Continuous CNN Scanning with IoU Tracking ✅ IMPLEMENTED

**Problem**: The original Scanner required manually pressing "Scan" for
each one-shot hash-based identification. No real-time capability.

**Fix**: Rewrote the `Scanner` class with a continuous scan loop:

- **Scan/Stop toggle**: The "Scan" button starts continuous real-time
  scanning; pressing again stops it.
- **Background thread**: Runs `detect_cards(fast=True, max_detect_dim=1024)`
  + `identify_detections()` in a tight loop (~5 Hz).
- **CNN backend**: Default backend switched from hash to CNN.
  Pre-loads `CardEmbedder` and `CardIndex` on startup for zero
  first-scan latency.

**New public API**: `identify_detections()` in `pipeline.py` returns
paired `(DetectedCard, list[MatchResult])` tuples, enabling the scanner
to get aligned detection corners and identification results without
double-detecting.

**Implementation**: `_scan_loop()` runs in a daemon thread. Each
iteration captures a frame, detects, identifies, and feeds results
into the `CardTracker`.

Note: OpenCV MOSSE/KCF tracking was evaluated but not implemented.
Since the pipeline already runs at 5.4 Hz with fast mode, IoU-based
frame-to-frame matching is sufficient and avoids the opencv-contrib
dependency.  Real OpenCV trackers would improve overlay smoothness
at higher video framerates but aren't needed at current speeds.

**Effort**: Medium.

#### B2. CardTracker — IoU Matching + Temporal Voting ✅ IMPLEMENTED

**Problem**: Single-frame identification can be noisy (reflections,
motion blur, partial occlusion). Overlays flicker between frames.

**Fix**: New `CardTracker` class in `scanner.py`:

- **IoU matching**: Each frame's detections are matched to existing
  tracked cards using greedy IoU matching (threshold 0.3).
- **Temporal voting**: Each tracked card accumulates a sliding window
  of top-1 card ID votes (last 10 frames). The display shows the
  most-voted identification, filtering transient misidentifications.
- **Best-score tracking**: For each voted card ID, the highest
  similarity score seen is stored and displayed.
- **Stale removal**: Tracked cards not seen for 5 consecutive frames
  are automatically removed.

**Benefits**:
- Stable overlays even with noisy per-frame identification
- Cards persist through brief occlusion
- Natural UX: card name stabilises after ~2-3 frames (~0.5s)

**Effort**: Low.

#### B3. Identification Caching — Not Yet Implemented

**Problem**: Re-identifying a card every time the full pipeline runs
wastes embedding time when the card hasn't changed.

**Fix**: Skip CNN embedding for tracked cards whose IoU with their
previous position exceeds a threshold. Only embed new cards.

**Estimated saving**: With 7 cards and only 1 new card per frame,
would reduce embedding from ~80 ms (7 cards) to ~12 ms (1 card).

**Status**: Not needed yet — 5.4 Hz is sufficient. Worth implementing
if the pipeline needs to scale to 10+ Hz.

**Effort**: Low (on top of B2).

#### B4. Async Pipeline with Threading — Not Yet Implemented

**Problem**: Detection and identification are sequential. While
identifying card N, no new frames are being captured.

**Fix**: Three-thread architecture:
- **Thread A (capture)**: Captures frames, renders overlay. Targets
  display framerate (15-30 FPS).
- **Thread B (pipeline)**: Pulls every Nth frame from A, runs full
  detect + identify, updates tracker.
- **Thread C (display)**: Smooth interpolation between pipeline frames.

**Status**: Not needed yet. Current two-thread design (main UI +
background scan loop) provides adequate responsiveness.

**Effort**: Medium-high — threading + state management.

---

### Phase C — Advanced Optimizations (target: >10 FPS, mobile-ready)

These are higher-effort improvements for scaling beyond desktop.

#### C1. GPU Acceleration (ONNX CUDAExecutionProvider)

Change one line in `embedder.py`:
```python
providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
```
Expected 5-20× speedup for embedding on NVIDIA GPUs. Batch embedding
of 16 images would drop from ~77 ms to ~5-15 ms.

**Effort**: Trivial (if CUDA available).

#### C2. FP16 / INT8 Quantization

Export MobileNetV3-Small to FP16 or INT8 ONNX format. ONNX Runtime
supports quantized inference natively.

- FP16: ~1.5-2× speedup, minimal accuracy loss
- INT8: ~2-4× speedup, requires calibration dataset

**Effort**: Low (export script change + accuracy validation).

#### C3. YOLO-Based Card Detection

Replace contour-based detection with a YOLOv8-nano model trained on
Pokemon cards. Benefits:
- 5-15 ms detection (vs 134 ms contour-based at 1024px)
- Handles occlusion, overlapping cards, angled shots better
- Returns confidence scores directly (no multi-strategy fusion)
- Compatible with built-in tracking (ByteTrack, BoT-SORT in Ultralytics)

**Effort**: High — requires labelled training data (bounding boxes on
card photos) and a training pipeline. Could bootstrap labels from
current contour detector's output.

#### C4. Smaller Embedding Model

MobileNetV3-Small produces 576-dim embeddings. Consider:
- Adding a linear projection head to reduce to 128-dim at ONNX export
  time (reduces FAISS search cost 4.5×)
- Distilling to an even smaller architecture (EfficientNet-B0-Lite,
  ShuffleNet) if mobile inference is needed
- Using ONNX Runtime's mobile optimizations

**Effort**: Medium (retraining + re-indexing).

#### C5. FAISS Index Optimization

At 20K vectors, `IndexFlatIP` is fine (~2 ms per query). As the
reference DB grows:

| Scale | Recommended Index | Search Time |
|---|---|---|
| <50K | IndexFlatIP (current) | <3 ms |
| 50K-200K | IndexIVFFlat (nlist=512) | <0.5 ms |
| >200K | IndexIVFPQ or IndexHNSW | <0.1 ms |

**Effort**: Low (index rebuild only).

#### C6. Drop Hash Backend Entirely

The hash backend (8 069 ms) is 9.4× slower than CNN (862 ms) and
has worse accuracy. For real-time use, it should be deprecated:
- Remove `--backend hash` code path from video pipeline
- Keep hash DB as a fallback for environments without ONNX Runtime
- Focus all optimization effort on the CNN path

---

## Summary: Priority-Ordered Action Items

| # | Change | Result | Status |
|---|---|---|---|
| A1 | Batch ONNX inference | 480->284 ms embed | Done |
| A2 | Early-exit on confident match | Skip 75% of fallback variants | Done |
| A3 | Detection at 1024px | 272->134 ms detect | Done |
| A4 | Skip overlapping pre-embedding | Caused quality regression | Dropped |
| A5 | Fast-mode strategies | Drop HSV+morph+Hough | Done |
| | **Static image: 862->185 ms (5.4 Hz, 4.7x speedup)** | | |
| B1 | Continuous CNN scanning + IoU tracking | Real-time Scanner at ~5 Hz | Done |
| B2 | Temporal majority voting (CardTracker) | Stable overlays | Done |
| B3 | Identification caching | Skip re-ID of known cards | Not started |
| B4 | Async threading | Smooth video overlay | Not started |
| | **Scanner: ~5 Hz continuous, stable IDs** | | |
| C1 | CUDA embedding | 5-20x embed speedup | Not started |
| C3 | YOLO card detection | 5-15 ms detection | Not started |
| C4 | Smaller embedding dim | 4.5x FAISS speedup | Not started |

**Phases A and B (core) are complete.**

- Phase A: CNN pipeline optimised from 862 ms to 185 ms (5.4 Hz, 4.7x).
- Phase B: Scanner rewritten for continuous real-time identification
  with IoU tracking and temporal voting.  Press Scan to start, Stop to
  end.  Cards are identified at ~5 Hz with voted-best labels displayed
  on the live preview.

---

## Reference: External Approaches

| Project/App | Detection | Identification | Speed |
|---|---|---|---|
| TCGplayer (commercial) | On-device ROI | Server-side CNN | ~1s |
| AceTrack-AI (OSS) | YOLOv11 | Temporal voting | Real-time |
| Card-Stocker-Pro (OSS) | AI | FAISS vector search | Real-time |
| spell-coven-mono (OSS) | OpenCV | CLIP embeddings | Near-RT |
| This repo (hash) | Contour+5 strategies | imagehash+hamming | 0.12 Hz |
| This repo (CNN, original) | Contour+5 strategies | MobileNetV3+FAISS | 1.16 Hz |
| **This repo (CNN, default)** | **Contour+5 strategies** | **MobileNetV3+FAISS (batch, early-exit)** | **2.3 Hz** |
| **This repo (CNN, fast)** | **Contour (fast, 1024px)** | **MobileNetV3+FAISS (batch, early-exit)** | **5.4 Hz** |
| **This repo (Scanner)** | **Contour (fast, 1024px) + IoU tracking** | **CNN + temporal voting** | **~5 Hz continuous** |
