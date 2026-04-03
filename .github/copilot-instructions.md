# Project Guidelines

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
   uv run ruff check src/
   ```

3. **Format with ruff**:
   ```sh
   uv run ruff format src/
   ```

4. **Type check with ty**:
   ```sh
   uv run ty check src/
   ```

5. **Lint with pylint** — a score of **10.00/10** is required:
   ```sh
   uv run pylint src/card_reco/
   ```
   If the score is below 10.00, fix all issues before considering the change complete.
