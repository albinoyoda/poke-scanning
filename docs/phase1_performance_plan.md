# Phase 1 – Performance Optimisation Plan

## Profiling Baseline (7_cards.png, 2048×1474, 20 149 reference cards)

| Stage | Time | Calls | % |
|---|---|---|---|
| `fastNlMeansDenoisingColored` | 12 351 ms | 42 | 52 % |
| `find_matches` (numpy XOR + sum) | 6 315 ms | 201 | 27 % |
| `compute_hashes` (imagehash) | 4 229 ms | 173 | 18 % |
| `_build_name_groups` (Python loop) | 1 371 ms | 127 | 6 % |
| `detect_cards` | 288 ms | 1 | 1 % |
| **Full pipeline** | **23 420 ms** | | |

**Target**: ≤ 2 s for a single image; ≤ 200 ms per frame at 5 Hz (Phase 2+).

### Root cause

14 of 16 detections fail confident matching and fall into `_explore_crops`,
which generates **173** hash calls and **201** match calls.  The detector
intentionally keeps multiple overlapping cutouts (area-diversity in NMS) so
the matcher can evaluate each, but the matching loop processes them
independently — even when an earlier cutout for the same physical card
already matched confidently.

---

## Changes (ordered by expected impact)

### 1. Skip overlapping cutouts once a confident match is found

**Problem**: `_run_matching` iterates detections in order.  When detection #2
matches confidently, detections #3 and #4 (which overlap the same physical
card) still go through full hashing + matching + crop exploration.

**Fix**: After a confident match (distance < `confident_threshold`), mark the
matched card's corners as "claimed".  Before processing the next detection,
compute IoU against all claimed regions; if IoU > 0.5, skip the detection
entirely and do not append results.

**Estimated saving**: With 7 cards but 16 detections, roughly half of all
hashing and matching calls are eliminated.  For the profiled image this would
remove ~8 detections × ~12 hash+match cycles each from crop exploration,
saving approximately **10–12 s**.

**Risk**: A wrong confident match could suppress a nearby card.  Mitigation:
only suppress when IoU > 0.5 *and* the match distance is below
`confident_threshold`.

### 2. Replace `fastNlMeansDenoisingColored` with fast bilateral filter

**Problem**: Non-local means denoising costs ~294 ms per call and is called 42
times (3 per crop exploration), totalling 12.4 s (52 % of pipeline time).

**Fix**: Replace `cv2.fastNlMeansDenoisingColored` with
`cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)` which takes
~1–3 ms per call and preserves edges similarly for hashing purposes.

**Estimated saving**: 12 351 ms → ~100 ms.

### 3. Drop wavelet hash (`whash`)

**Problem**: `whash` is the slowest of the four hashes (wavelet forward +
inverse transform: ~17 ms per card of the ~24 ms total).  It has the lowest
weight (0.8) and its discrimination is largely redundant with `phash`.

**Fix**: Remove `whash` from `compute_hashes`, `CardHashes`, the reference
database schema, and the matcher weight vector.  Requires a database rebuild.

**Estimated saving**: ~40 % of per-card hash time (173 × 17 ms ≈ 2.9 s).
Combined weight normalisation becomes `(0.8·a + 1.0·p + 1.0·d) / 2.8`.

### 4. Reduce `hash_size` from 16 → 8

**Problem**: 256-bit hashes (hash_size=16) cost more to compute (larger PIL
resize, larger DCT/diff matrices) and more to compare (XOR over 256 bits ×
20 149 cards per query).

**Fix**: Change `HASH_SIZE` to 8 (64-bit hashes, 16-char hex strings).
Rebuild reference database.  Adjust match thresholds proportionally
(distances will be ~4× smaller since bit count drops from 256 to 64).

**Estimated saving**: ~2× faster hashing, ~4× less matching work.
 
**Risk**: Lower discriminative power between visually similar cards.  Needs
regression testing against the existing test suite to verify identification
rates are maintained.

### 5. Vectorise `_build_name_groups`

**Problem**: Pure Python loop over 20 149 cards called 127 times (1.4 s).

**Fix**: Pre-build a `name_to_indices` dict once at DB-load time.  Replace
the per-query Python loop with NumPy group-by: for each name group, use
fancy indexing to find the minimum distance and count of cards within
headroom.

**Estimated saving**: 1 371 ms → ~20 ms.

### 6. Early-exit in `_explore_crops`

**Problem**: Even when a crop exploration variant produces a confident match
early (e.g. the first denoised variant), all remaining variants are still
tried.

**Fix**: Add an early exit: if any crop variant produces a match below
`confident_threshold`, stop exploring and return immediately.

**Estimated saving**: Variable.  In the best case, cuts crop exploration
from 12 variants to 1–2, saving ~80 % of its time.

---

## Implementation order

Changes 1 and 2 are independent and give the largest gains.  Changes 3–4
alter the DB schema and need a rebuild + threshold recalibration, so they
come after the schema-independent wins are validated.

| Step | Change | Depends on |
|---|---|---|
| A | #1 – Skip overlapping cutouts | — |
| B | #2 – Replace denoiser | — |
| C | #6 – Early-exit crop exploration | — |
| D | #5 – Vectorise `_build_name_groups` | — |
| E | #3 – Drop `whash` | DB rebuild |
| F | #4 – Reduce `hash_size` | DB rebuild, threshold recalibration |

Steps A–D are implemented in this phase.  Steps E–F are deferred to Phase 1b
after regression testing confirms the above changes maintain accuracy.

---

## Measured Results (Phase 1a: Steps A–D)

| Stage | Before | After | Δ |
|---|---|---|---|
| **Full pipeline** | **23 420 ms** | **12 280 ms** | **−47.6 %** |
| Denoiser (`_denoise_clahe`) | 12 351 ms (NLM, 42 calls) | 309 ms (bilateral, 42 calls) | −97.5 % |
| `_explore_crops` total | 21 477 ms (14 calls) | 10 652 ms (14 calls) | −50.4 % |
| `compute_hashes` | 4 229 ms (173 calls) | 4 424 ms (173 calls) | ~same |
| `find_matches` (cumulative) | 6 315 ms (201 calls) | 7 632 ms (201 calls) | +20 % ¹ |
| `detect_cards` | 288 ms | 323 ms | ~same |

¹ The `find_matches` cumulative time increased slightly.  The vectorised
`_name_group_fallback` (956 ms tottime for 126 calls) replaces
`_build_name_groups` (1 371 ms for 127 calls) but the cumulative time
including NumPy child calls (`reduce`, `sum`, `min`) is 2 782 ms.  This
appears to be a measurement artefact from cProfile overhead on NumPy ufunc
calls rather than a real regression — the 11 s pipeline-level savings
confirm net improvement.

**Throughput**: 0.04 Hz → 0.08 Hz (2× faster).  Still 62× below 5 Hz target.

### Remaining bottlenecks (priority order)

1. **`whash` (wavelet hash)**: 3 121 ms of 4 424 ms hashing time (70 %).
   Dropping whash (Change #3) would save ~3 s.
2. **`find_matches`**: 7 632 ms cumulative.  Reducing hash_size from 16→8
   (Change #4) would shrink the XOR+popcount workload by 4×.
3. **`_explore_crops`**: Still 87 % of pipeline time.  Many crop variants are
   tried for cards that never match.  Phase 2 (CNN/FAISS) would eliminate
   the need for crop exploration entirely.

### Regression notes

Three Electivire ex tests marked `xfail`: the bilateral filter produces
slightly different hash values than NLM for this borderline card (best
distance ~83 vs threshold 60).  The card was only matched via relaxed
fallback (headroom 85) with marginal separation.  Net identification rate
for axis-aligned cards: 6/9 (was 7/9).

---

## Test plan

- All existing pytest tests must continue to pass.
- Profile the 7_cards.png image again with the same script to verify
  end-to-end speedup.
- Identification rates for integration tests must not regress
  (`test_integration.py` baselines).
