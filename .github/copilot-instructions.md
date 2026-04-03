# Project Guidelines

Pokemon card recognition pipeline that detects cards in photos using OpenCV contour analysis and perspective transforms, then identifies them by comparing perceptual hashes (ahash, phash, dhash, whash via the `imagehash` library) against a pre-built SQLite reference database. The project is written in Python 3.10+, built with hatchling, and managed with uv for dependency resolution. Key dependencies include OpenCV, NumPy, Pillow, and imagehash, with ruff, ty, and pylint enforcing code quality.

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
