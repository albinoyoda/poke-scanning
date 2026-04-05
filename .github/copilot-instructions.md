# Project Guidelines

Pokemon card recognition pipeline: detects cards in photos via OpenCV contour
analysis and perspective transforms, then identifies them using either
perceptual hashes (ahash, phash, dhash via `imagehash`) against a SQLite
reference database, or CNN embeddings (MobileNetV3-Small via ONNX Runtime)
matched by cosine similarity via a FAISS index.

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the full architecture,
module map, data layout, performance profile, and improvement roadmap.

## Tech Stack

- **Language**: Python 3.10+
- **Build**: hatchling, managed with uv
- **Core deps**: OpenCV (`opencv-python`), NumPy, Pillow, `imagehash`, ONNX Runtime, FAISS-cpu
- **Reference DB**: SQLite 3 (stdlib `sqlite3`) for hashes; FAISS index for CNN embeddings
- **Quality**: ruff (lint + format), ty (type checking), pylint (10.00/10 required)
- **Tests**: pytest

## Project Layout

```
src/card_reco/        # Library package
  __init__.py         # Thin re-export layer (public API)
  pipeline.py         # Pipeline orchestration, crop exploration, preprocessing
  detector/           # Card detection subpackage
    __init__.py       #   Top-level orchestrator: detect_cards()
    strategies.py     #   Detection strategies (Canny, adaptive, HSV, Hough)
    corners.py        #   Corner extraction, refinement, ordering
    nms.py            #   Non-max suppression, centroid dedup
    quality.py        #   Contour quality scoring, edge verification
    constants.py      #   Shared constants and CLAHE factory
  hasher.py           # Perceptual hashing (ahash, phash, dhash)
  matcher.py          # Vectorised NumPy hash matching
  database.py         # SQLite reference database wrapper
  embedder.py         # CNN embedding extraction (ONNX Runtime, MobileNetV3-Small)
  faiss_index.py      # FAISS vector index for CNN embedding search
  models.py           # Dataclasses (DetectedCard, CardHashes, etc.)
  debug.py            # Debug image writer
  scanner.py          # Live scanner GUI with real-time CNN identification
  cli.py              # CLI entry point
scripts/              # Data download, DB build, debugging utilities
data/                 # Reference images, metadata, hash DB, test photos
tests/                # Integration tests (pytest)
docs/                 # Architecture docs, known issues
```

## Build and Test

This project uses **uv** for dependency management. Install with:

```sh
uv sync --dev
```

## After Every Code Change

After making any code change, always (for all files, not just the ones you edited):

1. **Add or update tests** for the changed code using `pytest`. Run them:
   ```sh
   uv run pytest
   ```
   Use multiple workers to speed up the test suite (pytest-xdist is installed):
   ```sh
   uv run pytest -n 8
   ```

2. **Lint with ruff**:
   ```sh
   uv run ruff check
   ```

3. **Format with ruff**:
   ```sh
   uv run ruff format
   ```

4. **Type check with ty**:
   ```sh
   uv run ty check
   ```

5. **Lint with pylint** — a score of **10.00/10** is required:
   ```sh
   uv run pylint src/ tests/ scripts/
   ```
   If the score is below 10.00, fix all issues before considering the change complete.

6. If the repo architechture or data layout has changed, update this file or relevant files in `docs/` accordingly.

## Debugging Detection Failures

The CLI has a `--debug` flag that writes intermediate pipeline images to a
directory.  This is invaluable for diagnosing why a card fails detection or
matching:

```sh
uv run card-reco identify data/tests/single_cards/tilted/moltres_151.png --debug debug/moltres
```

The output directory contains numbered PNGs for every pipeline stage (input,
preprocessing, edge maps, candidates, corners, NMS, warped cards, match
summaries).  See [docs/debug_output.md](../docs/debug_output.md) for a
detailed guide on what each image shows and how to interpret match quality.

**Agent workflow**: When investigating why a card is not detected or matched
correctly, run the identify command with `--debug`, then inspect the
resulting images using the image viewing tool.  Compare edge maps, candidate
overlays, and corner placements to understand which pipeline stage is
failing.  This is the primary way to diagnose algorithm issues and verify
that code changes improve results.

**Image viewing limit**: The VS Code Copilot extension has a bug where calling
`view_image` more than twice in the main agent session causes a 413 error and
crashes the session.  To inspect many debug images, use `runSubagent` —
subagents are exempt from this limit and can safely view large batches of
images.  Dispatch multiple subagents in parallel, each viewing ~10 images, and
collect their descriptions.
