# Debug Output Guide

Run the pipeline with `--debug` to write intermediate images for every stage
of detection and matching:

```sh
card-reco identify photo.jpg --debug            # writes to debug/
card-reco identify photo.jpg --debug my_output  # writes to my_output/
```

The output directory is cleaned on each run. Files are numbered in pipeline
order (`01_`, `02_`, …) so a sorted file listing matches execution order.

---

## Image Stages

### 01 — Input

**Filename:** `01_input.png`

The raw BGR image as loaded by OpenCV. Use this as a visual baseline to
compare against every later stage. If the image looks wrong here (rotated by
the camera EXIF, heavily compressed, very dark) the problem is upstream of
the pipeline.

### 02 — Preprocessing

**Filenames:** `02_preprocess_gray.png`, `02_preprocess_blurred.png`

- **Gray** — grayscale conversion. Cards should be clearly distinguishable
  from the background. If the card border merges with the surface it sits on,
  edge detection will struggle.
- **Blurred** — Gaussian blur (5×5 kernel). Smooths out noise while
  preserving card outlines. If fine card border detail is already lost here
  the card may be too small in the frame.

### 03–06 — Edge / Threshold / Mask Maps

Multiple detection strategies each produce a binary (white-on-black) map.
The exact count depends on the number of strategies enabled.

| Label | Strategy | What to look for |
|---|---|---|
| `canny_50_150` | Canny edges (high thresholds) | Clean thin outlines around the card border. Best for high-contrast photos. |
| `canny_30_100` | Canny edges (low thresholds) | More sensitive variant. Picks up faint borders but also more noise. |
| `adaptive_thresh` | Adaptive Gaussian threshold | Good for uneven lighting. Shows strong edges even when one side of the card is shadowed. |
| `hsv_red_orange` | HSV color mask (red/orange) | Highlights fire-type card borders (Charizard, etc.). Should be mostly blank for non-red cards. |
| `hsv_yellow_gold` | HSV color mask (yellow/gold) | Highlights yellow/gold card borders. |

**What success looks like:** A closed (or nearly closed) white outline tracing
the card perimeter, with minimal noise elsewhere in the image.

**Warning signs:**
- Card border has gaps → contour finder may not close the shape.
- Background clutter produces outlines as large as the card → false candidates.
- Card border is completely absent → the card may have a low-contrast border
  on a similar-colored surface.

### 07 — Candidates

**Filename:** `07_candidates.png` (step number may vary)

All contours that passed the area filter (≥ 1.5% and ≤ 95% of the image)
drawn on the original image. Each contour is labeled with its index and the
percentage of the image area it covers.

- **Green** contours are large (> 5× the minimum area).
- **Blue/cyan** contours are near the minimum area threshold.

**What success looks like:** One or more contours tightly tracing each card's
border. There may be duplicates from different strategies — that is expected
and resolved later by NMS.

**Warning signs:**
- No contours at all → detection failed entirely; check the edge maps above.
- Contours wrap the wrong object (a book, table edge, etc.) → the object has
  a similar rectangular shape to a card.
- Contour is much larger than the card → it captured the card plus surrounding
  area. The perspective warp will include background.

### 08 — Corners

**Filename:** `08_corners.png` (step number may vary)

Quadrilaterals extracted from the candidate contours displayed on the input
image. Each detection has four numbered corner points (0 = top-left,
1 = top-right, 2 = bottom-right, 3 = bottom-left) and a label showing its
index and area percentage.

**What success looks like:** Four corners sitting precisely on the card's
physical corners. For tilted or perspective-distorted cards the quadrilateral
should be a trapezoid matching the visible shape — not a perfect rectangle.

**Warning signs:**
- Corners land inside the card artwork → warp will crop part of the card.
- Corners land outside the card → warp will include background.
- Corner ordering is wrong (e.g. top-left point is actually bottom-right) →
  the warped image will be flipped or rotated, hurting hash matching.

### 09 — NMS (Non-Maximum Suppression)

**Filename:** `09_nms.png` (step number may vary)

Shows which detections survived de-duplication. The header text reads
`NMS: X -> Y`, where X is the count before and Y after suppression. Each
survivor is drawn with its area percentage and confidence score.

Overlapping detections with IoU > 0.5 are merged, keeping the higher-confidence one. Detections whose areas differ by more than 15%
(`_AREA_DIVERSITY_THRESH = 0.85`) are both kept even if they overlap, since
one may be a tight crop and the other a broader view.

**What success looks like:** Exactly one detection per physical card, each
with a high confidence value (closer to 1.0).

**Warning signs:**
- Two detections survive for the same card → one is a tight crop and the other
  is loose. The matcher will try both; the tight crop usually matches better.
- A detection is removed that should have survived → the overlap threshold may
  be too aggressive for that image layout.

### 10+ — Warped Cards

**Filenames:** `10_warped_0.png`, `11_warped_1.png`, …

Each detected card after perspective correction, scaled to 734×1024 (standard
card aspect ratio 5:7). This is the image that gets hashed and compared
against the reference database.

**What success looks like:** The card fills the entire image in portrait
orientation. Artwork, text, and borders are all visible and not clipped. The
card should look like a flat top-down scan.

**Warning signs:**
- Part of the card is missing (cropped) → corners were misplaced.
- Background is visible on one or more sides → contour was too large.
- Image is rotated 90° or 180° → corner ordering was wrong. The pipeline
  automatically tries 180° rotation during matching, but 90° rotation is not
  corrected.
- Image is blurry or heavily skewed → the source photo may be too angled or
  out of focus.

### Final — Match Summary

**Filenames:** `12_match_0.png`, `13_match_1.png`, …

A composite image with the warped card on the left and the top-5 match
results on the right. Each match line shows:

```
#1 Charizard (base1-4) d=12.3
   ahash=3 phash=4 dhash=2 whash=3
```

- **rank** — position in the sorted results.
- **name, set, number** — the reference card identity.
- **d (distance)** — combined weighted Hamming distance across all four hashes.
  Lower is better.
- **per-hash distances** — individual hash distances (ahash, phash, dhash,
  whash). Useful for diagnosing which hash type is causing trouble.

---

## Interpreting Match Quality

| Distance (d) | Interpretation |
|---|---|
| **< 15** | Near-certain correct match. The card is well-detected and closely matches the reference. |
| **15 – 25** | Confident match. Minor differences from lighting, wear, or edition variants. |
| **25 – 35** | Plausible match. Verify visually. Could be a correct match with quality degradation, or a near-miss (e.g. same artwork in a different set). |
| **35 – 40** | Weak match. Likely wrong or the card is severely degraded/misframed. |
| **> 40** | No match (default threshold). The card is either not in the database or detection failed badly. |

### Signs of a Successful Match

1. **Low distance on the #1 result** (d < 25) **with a clear gap to #2.**
   A gap of 5+ between the top two results indicates the match is unambiguous.
2. **All four per-hash distances are low and balanced.** If ahash, phash,
   dhash, and whash all agree (each ≤ 8 or so), the match is robust.
3. **Warped card looks clean.** When you look at the warped image, the card
   fills the frame, is right-side up, and is not clipped.

### Signs of a Poor or Failed Match

1. **Top distance is high** (d > 35) **or there is no match at all.** Look at
   the warped image — if the card is clipped, rotated 90°, or includes
   background, the problem is in detection, not matching.
2. **Top several results have very similar distances.** For example, three cards
   all at d ≈ 30 means the hashes can't differentiate them. This often happens
   with cards that share artwork (promo reprints, regional variants).
3. **One hash disagrees sharply.** If phash = 2 but whash = 20, the wavelet
   hash is picking up something the others aren't (often a border crop issue).
   Check the warped image edges.
4. **Correct card appears at #2 or #3 instead of #1.** Usually caused by the
   card being 180° flipped (the pipeline tries both orientations, but if
   neither is great the wrong one may win) or by a different edition of the
   same card being closer in hash space.

### Common Root Causes by Pipeline Stage

| Symptom | Stage to check | Likely cause |
|---|---|---|
| No cards detected | Edge maps, candidates | Card border blends into background; try better lighting or a contrasting surface. |
| Card partially cropped in warp | Corners | Corners are inward of the true card edge; the contour missed part of the border. |
| Background included in warp | Corners, candidates | Contour was too large; possibly picked up the table edge or a sleeve border. |
| Good warp but wrong match | Match summary | Card may not be in the reference database, or is a regional/edition variant with a different hash. |
| Card rotated 90° in warp | Corners | Corner ordering failed; aspect ratio check may have been tricked by a square-ish contour. |

---

## Case Study: Moltres 151 (Tilted Card on Similar-Colored Surface)

**Image:** `data/tests/single_cards/tilted/moltres_151.png`

A white-bordered card propped at an angle against a dark tiled wall. The card
rests on a light-colored ground surface. The card casts a shadow on the wall
behind it. This is a hard case because the card and the ground share a very
similar brightness.

```
Scene layout (side view, simplified):

          W = dark tiled wall
          C = card (white border, tilted against wall)
          G = light ground (similar color to card border)

              W W W W W
              W W W W W
             /C C C/
            / C C /
      G G G G G G G G G G
```

### Stage-by-stage findings

**03 — Canny 50/150:** Looks good overall. The wall tiles and shadow cast some
contours, but the majority of the edge lines trace the card outline cleanly.
Most of the visible contours are useful candidates.

**04 — Canny 30/100:** More noise from the lower thresholds. The card still
accounts for roughly 95% of the contours, but edges now appear scattered
across the card's interior artwork as well — not just the border.

**05 — Adaptive threshold:** This is where things break down. Adaptive
thresholding sees the card as white *and* the ground as white, producing a
single merged bright region. The boundary between card and ground
disappears. The dark wall behind the card becomes the only strong contrast
edge, but it is not useful for isolating the card.

**06–07 — HSV masks:** Same fundamental problem. The card border and the
ground are both white/light, so color segmentation cannot separate them.
No useful card outline is produced.

**08 — Candidates:** Candidate contours appear around parts of the card edge,
which is encouraging. However, many additional candidates also appear on the
ground surface in front of the card, because the adaptive threshold lumped
card + ground together.

**09 — Corners:** The four computed corners become a combination of the card
and the ground. Instead of a tight quad around the card, the bottom corners
slide outward to encompass the ground area:

```
       x————x          ← top corners sit on the card's upper edge (correct)
       | C  |
———————|    |————————
       |____|

  x         G        x ← bottom corners land on the ground (wrong)
```

The resulting quadrilateral is much wider and taller than the actual card.
The perspective warp therefore includes a large area of ground, producing a
warped image that is mostly background with the card occupying only a
fraction of the frame. This severely damages hash matching — the combined
distance ends up around ~91, well above the 40.0 match threshold.

### Root cause

The core issue is **low contrast between the card border and the adjacent
surface**. When the card rests on a surface that is a similar color to its
border, threshold-based and color-based strategies merge the two regions
into one large blob. Canny edges can still pick up the card outline, but the
candidates from the threshold strategy add noise that pulls the final corners
off the card.

### Potential mitigations (not yet implemented)

- **Per-strategy candidate scoring.** Weight Canny-derived contours higher
  than adaptive-threshold contours when the threshold map shows a large merged
  region.
- **Contour rectangularity check.** The merged card + ground contour has poor
  compactness and a wrong aspect ratio; a stricter shape filter could reject
  it before corner extraction.
- **Edge-guided corner refinement.** After extracting initial corners, verify
  that each corner sits on a strong edge in the Canny map. If bottom corners
  land on flat ground (no edge), snap them inward to the nearest Canny edge.
- **Photo guidance.** Advise users to place cards on a contrasting surface
  (dark mat for light-bordered cards) for best results.

