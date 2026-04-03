# Project Guidelines

Pokemon card recognition pipeline: detects cards in photos via OpenCV contour
analysis and perspective transforms, then identifies them by comparing
perceptual hashes (ahash, phash, dhash, whash via `imagehash`) against a
pre-built SQLite reference database.

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the full architecture,
module map, data layout, performance profile, and improvement roadmap.

## Tech Stack

- **Language**: Python 3.10+
- **Build**: hatchling, managed with uv
- **Core deps**: OpenCV (`opencv-python`), NumPy, Pillow, `imagehash`
- **Reference DB**: SQLite 3 (stdlib `sqlite3`)
- **Quality**: ruff (lint + format), ty (type checking), pylint (10.00/10 required)
- **Tests**: pytest

## Project Layout

```
src/card_reco/        # Library: detector, hasher, matcher, database, models, cli
scripts/              # Data download, DB build, debugging utilities
data/                 # Reference images, metadata, hash DB, test photos
tests/                # Unit + integration tests (pytest)
docs/                 # Architecture docs, known issues
```

## Build and Test

This project uses **uv** for dependency management. Install with:

```sh
uv sync --dev
```

## After Every Code Change

After making any code change, always:

1. **Add or update tests** for the changed code using `pytest`. Run them:
   ```sh
   uv run pytest
   ```

2. **Lint with ruff**:
   ```sh
   uv run ruff check src/ tests/
   ```

3. **Format with ruff**:
   ```sh
   uv run ruff format src/ tests/
   ```

4. **Type check with ty**:
   ```sh
   uv run ty check src/ tests/
   ```

5. **Lint with pylint** — a score of **10.00/10** is required:
   ```sh
   uv run pylint src/card_reco/
   ```
   If the score is below 10.00, fix all issues before considering the change complete.

6. If the repo architechture or data layout has changed, update this file and the docs in `docs/ARCHITECTURE.md` accordingly.