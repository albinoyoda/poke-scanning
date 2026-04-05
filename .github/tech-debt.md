# Technical Debt Audit

> Generated: 2026-04-05
> Updated: 2026-04-05 — Phase 1 & Phase 2 completed
>
> Scope: `src/card_reco/` (9 modules → 14 modules after restructuring), `tests/` (7 files, ~1 760 lines), `scripts/` (5 files, ~431 lines)

---

## Table of Contents

- [Technical Debt Audit](#technical-debt-audit)
  - [Table of Contents](#table-of-contents)
  - [Summary](#summary)
  - [Findings by Module](#findings-by-module)
    - [1. `__init__.py` — Pipeline orchestration (315 lines)](#1-__init__py--pipeline-orchestration-315-lines)
      - [Remediation](#remediation)
    - [2. `detector.py` — Card detection (916 lines)](#2-detectorpy--card-detection-916-lines)
      - [Remediation](#remediation-1)
    - [3. `database.py` — Hash database (131 lines)](#3-databasepy--hash-database-131-lines)
      - [Remediation](#remediation-2)
    - [4. `debug.py` — Debug writer (269 lines)](#4-debugpy--debug-writer-269-lines)
      - [Remediation](#remediation-3)
    - [5. `matcher.py` — Card matching (244 lines)](#5-matcherpy--card-matching-244-lines)
      - [Remediation](#remediation-4)
    - [6. `hasher.py` — Perceptual hashing (43 lines)](#6-hasherpy--perceptual-hashing-43-lines)
    - [7. `models.py` — Data models (53 lines)](#7-modelspy--data-models-53-lines)
    - [8. `cli.py` — Command-line interface (98 lines)](#8-clipy--command-line-interface-98-lines)
      - [Remediation](#remediation-5)
    - [9. `scripts/` — Utility scripts](#9-scripts--utility-scripts)
      - [Remediation](#remediation-6)
    - [10. `tests/` — Test suite](#10-tests--test-suite)
      - [Remediation](#remediation-7)
    - [11. `docs/` — Documentation](#11-docs--documentation)
      - [Remediation](#remediation-8)
  - [Implementation Roadmap](#implementation-roadmap)
    - [Dependency Graph](#dependency-graph)
    - [Phase 1 — Constants \& Type Safety ✅ COMPLETED](#phase-1--constants--type-safety--completed)
    - [Phase 2 — Module Restructuring ✅ COMPLETED](#phase-2--module-restructuring--completed)
    - [Phase 3 — Test Coverage](#phase-3--test-coverage)
    - [Phase 4 — Performance Optimizations](#phase-4--performance-optimizations)
    - [Phase 5 — Polish \& Documentation](#phase-5--polish--documentation)

---

## Summary

| Severity | Count | Resolved |
|----------|-------|----------|
| High     | 4     | 3 (1.1, 2.1, 2.2 partially via 2.6) |
| Medium   | 12    | 9 (1.3, 1.4, 2.3, 2.4, 2.5, 3.1, 3.2, 4.1, 10.1) |
| Low      | 12    | 10 (1.5, 1.6, 2.6, 2.7, 3.3, 5.1, 9.1, 9.2, 9.3, 9.4) |

The codebase is clean and well-structured overall, but has accumulated debt in
four areas: (1) pipeline logic packed into `__init__.py`, (2) `detector.py`
approaching the 1 000-line threshold, (3) significant unit test gaps for pure
helper functions, and (4) scattered duplication of constants, geometry helpers,
and database mapping code.

**Phase 1 & 2 are now complete.** The remaining open items are:
- **1.2, 2.2**: Unit tests for pipeline helpers and detector internals (Phase 3)
- **4.2**: Debug writer tests (Phase 3)
- **8.1**: CLI tests (Phase 3)
- **11.1**: Documentation updates (Phase 5)

---

## Findings by Module

### 1. `__init__.py` — Pipeline orchestration (315 lines)

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 1.1 | **High** | Architecture | **Pipeline logic lives in `__init__.py`.**  Contains ~250 lines of orchestration logic (`identify_cards`, `identify_cards_from_array`, `_explore_crops`) plus image processing helpers (`_denoise_clahe`, `_expand_corners`, `_rewarp`, `_make_whole_image_card`). An `__init__.py` should export the public API and re-export symbols — not host the core pipeline. |
| 1.2 | **High** | Missing tests | **No unit tests for any `__init__.py` helper function.** `_make_whole_image_card`, `_denoise_clahe`, `_expand_corners`, `_rewarp`, and `_explore_crops` are all untested directly. Integration tests cover the happy path but do not exercise edge cases (e.g. landscape image in `_make_whole_image_card`, extreme padding in `_expand_corners`). |
| 1.3 | **Medium** | Coupling | **Imports private symbols from `detector.py`.** `_refine_corners_edge_intersect` and `CARD_DST_PORTRAIT` are accessed directly. Crop exploration re-warps cards by reaching into detector internals rather than calling a public warp function. |
| 1.4 | **Medium** | Complexity | **`identify_cards_from_array` is 60+ lines with deep nesting.** Handles whole-image injection, detection loop, two-orientation matching, confident-match short-circuit, and crop-exploration fallback in a single function. Each concern should be a separate helper. |
| 1.5 | **Low** | Duplication | **`_denoise_clahe` duplicates CLAHE creation.** The same `clipLimit=2.0, tileGridSize=(8,8)` CLAHE configuration also appears twice in `detector.py` (lines ~86 and ~200). A shared factory would avoid drift. |
| 1.6 | **Low** | Code smell | **`_explore_crops` uses `nonlocal` mutation.** The nested `_try_image` function mutates `best` via `nonlocal`, making the data flow hard to trace and impossible to test the inner function in isolation. |

#### Remediation

- **1.1 / 1.2**: Move all pipeline logic to a new `src/card_reco/pipeline.py`. Keep `__init__.py` as a thin re-export layer:
  ```python
  # __init__.py
  from card_reco.pipeline import identify_cards, identify_cards_from_array
  ```
  Then add unit tests for every pure helper (`_make_whole_image_card`, `_denoise_clahe`, `_expand_corners`, `_rewarp`) in a new `tests/test_pipeline.py`.
- **1.3**: Promote `_refine_corners_edge_intersect` to a public API in `detector.py` (rename to `refine_corners_edge_intersect`) or move it to a shared `geometry.py`. Export `CARD_DST_PORTRAIT`, `CARD_WIDTH`, `CARD_HEIGHT` as public constants.
- **1.4**: Extract `_match_card_orientations(card, matcher, ...)` and `_explore_crops(...)` into clearly separated steps in `pipeline.py`.
- **1.5**: Create a `_make_clahe()` factory in a shared location (e.g. `preprocessing.py` or constants module).
- **1.6**: Refactor `_explore_crops` to return the best match from a list of `(image, label)` tuples, eliminating the `nonlocal` pattern.

---

### 2. `detector.py` — Card detection (916 lines)

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 2.1 | **High** | Architecture | **Approaching the ~1 000-line threshold.** Contains five detection strategies, corner extraction and refinement, quality scoring, NMS, centroid dedup, Hough-line detection, perspective transform, and debug output dispatch — all in one file. |
| 2.2 | **High** | Missing tests | **Most internal functions lack unit tests.** `_contour_quality`, `_corner_edge_fraction`, `_has_card_aspect_ratio`, `_find_card_contours`, `_collect_hsv_contours`, `_hough_quad`, `_angle_dist`, `_pick_two_lines`, `_line_intersection`, `_centroid_dedup`, `_compute_overlap`, `_four_point_transform`, `_intersect_param_lines` have no direct tests. Only `_order_corners` and `_refine_corners_from_hull` are tested. |
| 2.3 | **Medium** | Duplication | **Aspect ratio geometry computed twice for the same corners.** `_has_card_aspect_ratio` and `_contour_quality` both compute `avg_width`, `avg_height`, `short/long` ratio from the same corner array. `detect_cards` calls both. |
| 2.4 | **Medium** | Duplication | **Area computation pattern repeated in NMS and centroid dedup.** `cv2.contourArea(corners.reshape(4,1,2).astype(np.int32))` appears 4× across `_non_max_suppression` and `_centroid_dedup`, including re-computation inside inner loops comparing the same `kept` detection each time. |
| 2.5 | **Medium** | Magic numbers | **Detection strategy thresholds are inline.** Canny pairs `(50, 150)` and `(80, 200)`, adaptive threshold block size `15`, HSV range tuples, morphological kernel sizes `(3,3)`, `(5,5)`, `(7,7)`, and Hough parameters are scattered throughout the code without named constants. |
| 2.6 | **Low** | Duplication | **`_order_corners` is duplicated in `tests/test_edge_detection.py`.** The test file has its own identical copy (`_order_corners`) instead of importing from `detector`. |
| 2.7 | **Low** | Performance | **Areas recomputed inside O(n²) NMS loops.** Each detection's area is recomputed every time it is compared, rather than being precomputed once per detection before the loops begin. |

#### Remediation

- **2.1**: Split into focused submodules:
  - `detector/strategies.py` — `_find_card_contours`, `_collect_hsv_contours`, `_hough_quad` and helpers
  - `detector/corners.py` — `_extract_corners`, `_refine_corners_from_hull`, `_refine_corners_edge_intersect`, `_order_corners`, `_has_card_aspect_ratio`
  - `detector/nms.py` — `_non_max_suppression`, `_centroid_dedup`, `_compute_overlap`
  - `detector/quality.py` — `_contour_quality`, `_corner_edge_fraction`
  - `detector/__init__.py` — `detect_cards` (top-level orchestrator), `_four_point_transform`, constants
- **2.2**: Add `tests/test_detector_internals.py` with tests for each pure function (see Phase 3).
- **2.3**: Extract a shared `_corner_geometry(corners) -> (avg_width, avg_height, ratio)` helper; use in both functions.
- **2.4 / 2.7**: Precompute areas into a `list[float]` before entering the NMS/dedup loops.
- **2.5**: Extract all strategy-specific thresholds into module-level named constants (e.g. `_CANNY_PAIRS`, `_ADAPTIVE_BLOCK_SIZE`, `_HSV_RANGES`).
- **2.6**: Remove the duplicate; import `_order_corners` from `card_reco.detector` in the test.

---

### 3. `database.py` — Hash database (131 lines)

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 3.1 | **Medium** | Duplication | **CardRecord construction from Row is duplicated.** `get_all_cards()` and `get_card_by_id()` both contain identical 12-field `CardRecord(...)` mapping blocks. |
| 3.2 | **Medium** | Duplication | **Column list is repeated in SQL.** The `SELECT id, name, set_id, ...` column list appears in two queries. If a column is added or renamed, both must be updated. |
| 3.3 | **Low** | Missing API | **No `get_all_ids()` method.** `build_hash_db.py` calls `get_all_cards()` just to extract IDs (`{c.id for c in db.get_all_cards()}`), loading all hash data into memory unnecessarily. |

#### Remediation

- **3.1 / 3.2**: Extract a `_CARD_COLUMNS` constant and a `_row_to_card(row) -> CardRecord` factory:
  ```python
  _CARD_COLUMNS = "id, name, set_id, set_name, number, rarity, image_path, ahash, phash, dhash, whash"

  def _row_to_card(row: sqlite3.Row) -> CardRecord:
      return CardRecord(**{k: row[k] for k in CardRecord.__dataclass_fields__})
  ```
- **3.3**: Add `get_all_ids() -> set[str]` that runs `SELECT id FROM cards` — much cheaper than loading all rows.

---

### 4. `debug.py` — Debug writer (269 lines)

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 4.1 | **Medium** | Type annotations | **Loose `list` types on public methods.** `save_candidates(contours: list)`, `save_nms_result(after_detections: list)`, and `save_match_summary(matches: list)` lack specific generic types. Should be `list[np.ndarray]`, `list[DetectedCard]`, and `list[MatchResult]` respectively. |
| 4.2 | **Low** | Missing tests | **Individual debug methods are untested.** Only `save_input` and directory cleaning are exercised via `test_detector.py::TestDebugWriter`. `save_match_summary`, `save_nms_result`, `save_corners`, `save_candidates` have no direct tests confirming they handle edge cases (e.g. empty match lists, zero-area contours). |

#### Remediation

- **4.1**: Add full type annotations. Import `DetectedCard` and `MatchResult` via `TYPE_CHECKING` (already done for `DebugWriter` in other modules):
  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from card_reco.models import DetectedCard, MatchResult
  ```
  Then annotate: `contours: list[np.ndarray]`, `after_detections: list[DetectedCard]`, `matches: list[MatchResult]`.
- **4.2**: Add a `tests/test_debug.py` with focused tests for each save method, verifying files are created and no exception on empty inputs.

---

### 5. `matcher.py` — Card matching (244 lines)

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 5.1 | **Low** | Readability | **`_name_group_fallback` is complex (50+ lines).** Builds a name-group dict, ranks by distance, applies two acceptance strategies, then returns filtered indices. Could be decomposed into `_build_name_groups`, `_accept_by_consensus`, `_accept_by_separation`. |

#### Remediation

- Extract substeps of `_name_group_fallback` into small pure functions. This improves readability and enables targeted unit tests for each acceptance strategy.

---

### 6. `hasher.py` — Perceptual hashing (43 lines)

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| — | — | — | No significant debt. Module is small, well-focused, fully typed, and well-tested. |

---

### 7. `models.py` — Data models (53 lines)

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| — | — | — | No significant debt. Clean dataclass definitions with appropriate defaults. |

---

### 8. `cli.py` — Command-line interface (98 lines)

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 8.1 | **Medium** | Missing tests | **No tests at all for the CLI module.** Argument parsing, error paths (`command is None`, missing image file), debug flag handling, and output formatting are untested. |

#### Remediation

- Add `tests/test_cli.py` using `main(argv=[...])` with `capsys` or `monkeypatch` to test:
  - `identify` with valid image (mock `identify_cards` return)
  - Missing image prints error and exits
  - `--debug` flag creates a `DebugWriter`
  - No command prints help and exits
  - `--top-n` and `--threshold` are passed through correctly

---

### 9. `scripts/` — Utility scripts

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 9.1 | **Low** | Code quality | **`find_cards.py` uses `open()` without context manager.** `json.load(open(...))` leaks file descriptors. |
| 9.2 | **Low** | Code quality | **`check_cards.py` has no error handling.** If the DB doesn't exist, the script crashes with an unhelpful traceback. |
| 9.3 | **Low** | Type safety | **Scripts lack type annotations.** `find_cards.py`, `find_sets.py`, and `check_cards.py` have no type annotations at all. |
| 9.4 | **Low** | Portability | **`build_hash_db.py` uses `sys.path.insert` hack.** Manipulates `sys.path` to import `card_reco`. Should rely on `uv run` or an editable install. |

#### Remediation

- **9.1**: Replace with `with open(...) as f: json.load(f)`.
- **9.2**: Wrap DB access in try/except or check path existence first.
- **9.3**: Add return types and parameter types to all functions.
- **9.4**: Remove the `sys.path.insert` line and update the usage comment to `uv run python scripts/build_hash_db.py`.

---

### 10. `tests/` — Test suite

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 10.1 | **Medium** | Coverage gap | **`test_edge_detection.py` duplicates `_order_corners` from detector.** 15-line function copy-pasted instead of imported. Changes to the canonical version will silently diverge from the test version. |
| 10.2 | **Low** | Maintenance | **Integration tests (565 lines) are the largest file.** `test_integration.py` contains 6 test classes with heavy parametrization. Consider splitting into `test_integration_single.py`, `test_integration_grid.py`, `test_integration_graded.py` for maintainability. |

#### Remediation

- **10.1**: Import `_order_corners` from `card_reco.detector` instead of re-implementing.
- **10.2**: Optional — split only if the file continues to grow.

---

### 11. `docs/` — Documentation

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| 11.1 | **Low** | Stale docs | **`ARCHITECTURE.md` references scripts that no longer exist.** Lists `save_crops.py`, `debug_detector.py`, `debug_detector2.py`, `debug_matching.py` in the Scripts table, but these files are not in the repo. |

#### Remediation

- Remove or update the stale script entries in the Architecture doc's Scripts table.

---

## Implementation Roadmap

### Dependency Graph

```
Phase 1 (Constants & Types) ──→ Phase 3 (Test Coverage) ─────────→ Phase 5 (Polish & Docs)
                                                          ╲
Phase 2 (Module Restructure) ─→ Phase 3 (Test Coverage) ──→ Phase 5 (Polish & Docs)
                                                          ╱
Phase 4 (Performance) ────────────────────────────────────
```

Phases 1, 2, and 4 can begin in parallel. Phase 3 depends on Phases 1 and 2.
Phase 5 is the final sweep after all structural changes land.

---

### Phase 1 — Constants & Type Safety ✅ COMPLETED

**Goal:** Eliminate magic numbers, fix type annotations, clean up duplication of small values.

**Scope:** Low-risk, no behavior change. Can be reviewed independently.

**Status:** All tasks completed. Verified by ruff, ty, pylint (10.00/10), and pytest (96 passed, 3 xfailed, 3 failed pre-existing).

| Task | Files | Items addressed | Status |
|------|-------|-----------------|--------|
| Extract shared CLAHE factory function | `detector/constants.py` | 1.5 | ✅ `make_clahe()` factory in `detector/constants.py`; used by `pipeline.py` and `strategies.py` |
| Extract detection-strategy constants | `detector/constants.py` | 2.5 | ✅ `CANNY_PAIRS`, `ADAPTIVE_BLOCK_SIZE`, `ADAPTIVE_C`, `HSV_COLOR_GROUPS`, `HSV_GROUP_NAMES`, `CLAHE_CLIP_LIMIT`, `CLAHE_TILE_GRID` |
| Extract `_CARD_COLUMNS` and `_row_to_card` helper in `database.py` | `database.py` | 3.1, 3.2 | ✅ `_CARD_COLUMNS` constant and `_row_to_card()` factory |
| Add `get_all_ids()` to `HashDatabase` | `database.py`, `scripts/build_hash_db.py` | 3.3 | ✅ `get_all_ids() -> set[str]`, used in `build_hash_db.py` |
| Fix loose `list` types in `debug.py` | `debug.py` | 4.1 | ✅ `TYPE_CHECKING` imports for `DetectedCard`, `MatchResult`; annotated all methods |
| Remove `_order_corners` duplicate from `test_edge_detection.py` | `tests/test_edge_detection.py` | 2.6, 10.1 | ✅ Removed 15-line duplicate; imports `order_corners` from `detector.corners` |
| Fix `open()` without context manager in `find_cards.py` | `scripts/find_cards.py` | 9.1 | ✅ Context managers throughout |
| Remove `sys.path.insert` from `build_hash_db.py` | `scripts/build_hash_db.py` | 9.4 | ✅ Removed; updated usage comment to `uv run` |
| Add type annotations to scripts | `scripts/*.py` | 9.3 | ✅ All scripts annotated |
| Add error handling to `check_cards.py` | `scripts/check_cards.py` | 9.2 | ✅ `main()` function, try/except, `if __name__` guard |

---

### Phase 2 — Module Restructuring ✅ COMPLETED

**Goal:** Split oversized modules and move pipeline logic out of `__init__.py`.

**Scope:** Structural refactor — no behavior change, but broad file moves.

**Status:** All tasks completed. `detector.py` (916 lines) → `detector/` subpackage (6 files, ~1 025 lines total). Pipeline logic moved from `__init__.py` (315 lines) → `pipeline.py` (270 lines) + thin re-export `__init__.py` (8 lines). Backward-compat aliases preserved for existing test imports.

| Task | Description | Items addressed | Status |
|------|-------------|-----------------|--------|
| **Create `src/card_reco/pipeline.py`** | Moved `identify_cards`, `identify_cards_from_array`, `_explore_crops`, `_make_whole_image_card`, `_denoise_clahe`, `_expand_corners`, `_rewarp` from `__init__.py`. `__init__.py` is now a thin re-export layer. | 1.1, 1.3, 1.4, 1.6 | ✅ |
| **Promote private detector APIs** | `refine_corners_edge_intersect` (public), `refine_corners_from_hull` (public), `order_corners` (public). `CARD_DST_PORTRAIT`, `CARD_WIDTH`, `CARD_HEIGHT` exported from `detector/constants.py` and re-exported from `detector/__init__.py`. | 1.3 | ✅ |
| **Split `detector.py` into subpackage** | Created `detector/__init__.py` (orchestrator), `detector/strategies.py`, `detector/corners.py`, `detector/nms.py`, `detector/quality.py`, `detector/constants.py`. Deleted `detector_old.py`. | 2.1 | ✅ |
| **Extract shared geometry** | `corner_geometry(corners) -> tuple[float, float, float]` in `corners.py`, used by both `has_card_aspect_ratio` and `contour_quality`. | 2.3 | ✅ |
| **Refactor `_explore_crops`** | Replaced `nonlocal` pattern with a loop that builds `candidate_images` list, then evaluates all candidates and returns the best match. | 1.6 | ✅ |
| **Decompose `_name_group_fallback`** | Split into `_build_name_groups()` and `_accept_by_consensus_or_separation()` as module-level pure functions. | 5.1 | ✅ |
| **Precompute areas in NMS/dedup** | Areas precomputed into a list before O(n²) loops in `non_max_suppression` and `centroid_dedup`. | 2.4, 2.7 | ✅ (done during Phase 2 restructuring) |

---

### Phase 3 — Test Coverage

**Goal:** Reach unit-test coverage for every public and non-trivial private function.

**Scope:** Depends on Phases 1 & 2 for stable module boundaries.

| Task | New test file | Functions to test | Items addressed |
|------|--------------|-------------------|-----------------|
| **Pipeline helpers** | `tests/test_pipeline.py` | `_make_whole_image_card` (card-shaped / non-card / landscape), `_denoise_clahe` (smoke), `_expand_corners` (contract/expand), `_rewarp` (clamp to bounds) | 1.2 |
| **Detector internals** | `tests/test_detector_internals.py` | `_contour_quality`, `_corner_edge_fraction`, `_has_card_aspect_ratio`, `_hough_quad` (no lines / valid quad), `_angle_dist`, `_pick_two_lines`, `_line_intersection`, `_intersect_param_lines`, `_four_point_transform`, `_centroid_dedup`, `_compute_overlap` | 2.2 |
| **CLI** | `tests/test_cli.py` | `main()` with valid args, missing image, no command, `--debug` flag, `--top-n`/`--threshold` pass-through | 8.1 |
| **Debug writer** | `tests/test_debug.py` | `save_candidates` (empty list), `save_match_summary` (empty matches), `save_nms_result`, `save_corners` (multiple detections) | 4.2 |
| **Database `get_all_ids`** | `tests/test_database.py` (append) | `get_all_ids()` | 3.3 |

---

### Phase 4 — Performance Optimizations

**Goal:** Eliminate redundant computation in hot paths.

**Scope:** Independent of restructuring; can run in parallel.

| Task | Description | Items addressed |
|------|-------------|-----------------|
| **Precompute areas in NMS/dedup** | Build a `list[float]` of areas before entering the comparison loops in `_non_max_suppression` and `_centroid_dedup`. Pass precomputed values into the inner loop. | 2.4, 2.7 | ✅ Done in Phase 2 |
| **Avoid re-hashing original crop in `_explore_crops`** | The 0°-orientation hash of `card.image` is already computed by the caller. Pass it in to avoid a redundant `compute_hashes` call. | — |
| **Use `get_all_ids()` in `build_hash_db.py`** | Replace `{c.id for c in db.get_all_cards()}` with `db.get_all_ids()`. Reduces startup memory when resuming builds against large databases. | 3.3 | ✅ Done in Phase 1 |

---

### Phase 5 — Polish & Documentation

**Goal:** Final cleanup after all structural changes have landed.

| Task | Description | Items addressed |
|------|-------------|-----------------|
| **Update `ARCHITECTURE.md`** | Remove stale script references (`save_crops.py`, `debug_detector.py`, etc.). Add new modules (`pipeline.py`, `detector/` subpackage) to the module map. | 11.1 |
| **Update `copilot-instructions.md`** | Reflect any new project layout (e.g. `detector/` subpackage). |
| **Run full lint/type/test suite** | `uv run ruff check`, `uv run ruff format`, `uv run ty check`, `uv run pylint src/card_reco/`, `uv run pytest` — ensure 10.00/10 pylint and all tests green. |
| **Optional: split `test_integration.py`** | If the file has grown past 600 lines, split into per-category files. | 10.2 |
