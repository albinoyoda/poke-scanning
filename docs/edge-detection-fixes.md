# Edge Detection Improvement Plan

Tracked improvements to the card detection pipeline, motivated by the
Moltres 151 tilted-card failure case. See the
[case study in debug_output.md](debug_output.md#case-study-moltres-151-tilted-card-on-similar-colored-surface)
for the full debug analysis.

## Problem Summary

When a card rests on a surface of similar brightness (e.g. white-bordered
card on light ground), the adaptive threshold and HSV strategies merge
card + ground into one large blob. The blob's corners drift off the card,
producing a warped image full of background. Meanwhile, the lower Canny pass
(30/100) floods the card interior with edges that generate noisy candidates
without helping border detection.

The higher Canny pass (50/150) actually produces the best card outline for
these cases, but its signal is diluted by the volume of bad candidates from
the other strategies.

## Steps

### Step 1 — Replace Canny (30, 100) with (80, 200) [DONE]

**Rationale:** The 30/100 pass picks up too much interior card artwork and
ground texture. Replacing it with 80/200 captures only the strongest
gradients — typically the card-to-background boundary — while suppressing
noise. The 1:2.5 ratio follows Canny's recommended range (1:2 to 1:3).

**Change:** In `_find_card_contours`, swap `(30, 100)` → `(80, 200)` in the
Canny threshold loop.

**Risk:** Low. The 50/150 pass remains as the primary Canny strategy. Cards
with very faint borders still have adaptive threshold and HSV fallbacks.

**Files:** `src/card_reco/detector.py`

**Outcome:** 77 tests passed, 5 xfailed — no regressions. The 80/200 Canny
map is much cleaner for Moltres (card border only, almost no interior
noise). However, the adaptive threshold blob still dominates NMS (59.5%
area, conf=1.00) and the bottom corners still land on the ground. The core
Moltres failure requires step 2 or 3 to fully resolve.

---

### Step 2 — Contour quality scoring [DONE]

**Rationale:** Currently `detect_cards` sorts final detections by area
(largest first). The merged card+ground blob is large and high-confidence,
so it wins. Scoring by contour quality would demote it.

**Proposed metrics:**
- **Compactness:** `contour_area / convex_hull_area` — a proper card
  contour should be ≥ 0.85. Merged blobs score lower.
- **Rectangularity:** `contour_area / minAreaRect_area` — cards are
  rectangular; irregular merged shapes score worse.
- **Aspect ratio closeness:** distance from the 5:7 card ratio.

Use a composite quality score to rank detections instead of raw area.

**Risk:** Medium — changes which detection "wins" across all images. Needs
careful baseline testing against the full test suite.

**Files:** `src/card_reco/detector.py`

**Change:** Added `_contour_quality()` helper that computes a 0-1 composite
score from compactness, rectangularity, aspect ratio closeness, and relative
size. Confidence is now `quality * edge_frac` instead of `area / image_area`.
Detections are sorted by confidence (highest first) instead of area.

**Outcome:** 78 tests passed, 4 xfailed — treecko_75_109 promoted from
xfail to passing (quality-based ranking picks a better crop). The Moltres
card+ground blob scores q=0.43 (low compactness/rectangularity), demoting
it relative to cleaner candidates.

---

### Step 3 — Edge-verified corner refinement [DONE]

**Rationale:** After extracting corners, verify each corner sits near a
strong Canny edge. For the Moltres case, the bottom corners land on
featureless ground where the Canny map is black.

**Proposed approach:**
1. For each corner point, sample a small radius (15px) in the Canny 50/150
   edge map.
2. If a corner has no nearby edge pixels, flag the detection as suspect.
3. Either reject the detection or snap the corner inward along the quad's
   edge vector to the nearest Canny edge.

**Risk:** Medium — adds a new rejection path that could over-filter in
edge-sparse images. Needs the Canny edge map passed through to the corner
extraction stage.

**Files:** `src/card_reco/detector.py`

**Change:** Added `_corner_edge_fraction()` helper that checks each of four
corners against a dilated Canny 50/150 edge map within a 15px radius.
Candidates with edge fraction < 0.5 are rejected. The edge fraction also
weights the final confidence score. Added constants `_EDGE_VERIFY_RADIUS=15`
and `_MIN_CORNER_EDGE_FRAC=0.5`.

**Outcome:** Moltres blob now reports 59% edge fraction (2-3 corners backed
by edges). The detection survives the filter but its confidence is reduced
by the edge fraction multiplier. No regressions across the test suite.

---

### Step 4 — CLAHE preprocessing for adaptive threshold

**Rationale:** CLAHE (Contrast Limited Adaptive Histogram Equalization)
enhances local contrast. Applying it before adaptive thresholding could
help distinguish card borders from similar-brightness surfaces. Also
benefits hash matching when applied to warped cards before hashing (see
[known_issues.md](known_issues.md) item #2).

**Change:** Apply `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` to
the grayscale image before adaptive thresholding.

**Risk:** Low — preprocessing only. May slightly change edge character for
all images.

**Files:** `src/card_reco/detector.py`, potentially `src/card_reco/hasher.py`

**Status:** Not started
