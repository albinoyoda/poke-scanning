You are an expert AI assistant analyzing a Python Image recognition system for Pokémon Cards.

Your task is to perform a thorough audit of the entire codebase and identify areas of technical debt.  When you run this prompt you should do the following:

1. Scan every module and helper inside `src/` and `tests/` looking for:
   * Code duplication (functions, logic, data structures, constants) across files or modules.
   * Logic distributed across multiple files that could be centralized into a single module or utility.
   * Invisible duplication such as repeated patterns, feature extraction, or data transformations that should be centralized.
   * Missing or incomplete tests, especially for pure functions, algorithms, or logic. Every public function must have tests per project conventions.
   * Layers of indirection, long files, or "god modules" that exceed ~1000 lines without logical splits (tests not included).
   * Functions that have multiple responsibilities or side effects that could be split into smaller, more focused functions — functions should do one thing and do it well.
   * Hard‑coded values, magic numbers, or URLs that should be extracted into shared constants.
   * Missing or incomplete type annotations — all functions must have full parameter and return type annotations per project conventions.
   * Inefficient patterns/algorithms that could be optimized, especially in hot paths like image processing, feature extraction, or data transformations.
   * Unnecessary complexity in data structures or logic; optimize for readability and maintainability when possible.
   * Any other smell that contributes to maintainability, readability, or correctness debt.


2. For each violation you identify, note the module/area and categorize the severity (high/medium/low).

3. After auditing, create a comprehensive report formatted as described below.

---

When you finish the analysis, output the findings into `.github/tech-debt.md` (append if the file already exists, or create it).  The report should include:

- **Breakdown by module/concern**: list each area containing tech debt, sorted within each module by severity.
- **Detailed remediation plan**: for each item describe how to fix it, including specific refactors, tests to add, files to split or utilities to extract, etc.
- **Implementation roadmap**: create an implementation plan that is separated into phases that can be implemented in parallel tracks, example with dummy phases (showing a parallel/sequential structure):
```
Phase 1 -> Phase 2
                   \
Phase 3 -> Phase 4 -> Phase 6
                              \
Phase 5 -----------------------> Phase 7
```

The goal is to give developers a clear roadmap for reducing technical debt across the project.

You can assume you have full read access to the codebase and you are allowed to use any tooling or searches necessary to inspect the repository.

Be thorough and pragmatic; the summary will drive future cleanup efforts.
