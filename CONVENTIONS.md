# OPForch conventions

OPForch adopts the code-style conventions used by cpmux and phitrain.
These rules apply to human and Copilot changes alike. Code defines behavior;
this document defines the standards for changing it.

## Architecture and compatibility

- Keep the package organized around classifiers, graph state, mathematics, streaming, and shared utilities.
- `Subgraph` owns per-node tensor initialization. Reuse that initialization rather than duplicating its field layout.
- Keep the callable distance registry instead of adding a class hierarchy for individual distance functions.
- Preserve public imports, method signatures, return values, and checkpoint compatibility unless a change is explicit.
- Public `fit()` and `predict()` overrides are APIs, not undocumented framework callbacks.
- Document numerical domains and implemented variants. Do not change a formula or normalization as a style cleanup.
- Library logging stays in `opforch.utils.logging`. Do not import cpmux or adopt its application-specific architecture.

## Python and imports

- Target Python 3.12 and newer. Use `X | None`, builtin generics, and `collections.abc` for collection and callable ABCs.
- Import only types without suitable builtin or `collections.abc` equivalents from `typing`.
- Keep imports top-level, absolute, and grouped as standard library, third-party, then local, with blank lines between groups.
- Import internal behavior from its defining module. Retain the existing public package exports.
- Use double quotes and a 120-character line length.
- Begin every Python file with the project header:

```python
# Copyright (c) 2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.
```

## Documentation

- Public functions, classes, and explicit constructors have Google-style docstrings.
- Start with a single-sentence summary. Keep each `Args:`, `Returns:`, and `Raises:` entry on one line.
- A regular class has a short class summary. Document constructor arguments on `__init__`.
- Data classes without an explicit constructor document every field in an `Attributes:` section.
- Keep a blank line before the closing triple quotes and after the docstring before the first statement or field.
- Black collapses summary-only docstrings. Use `# fmt: skip` on their closing line to preserve this required padding.
- Limit that formatting directive to the docstring statement, never a surrounding block of executable code.
- Private helpers and framework-dispatched callbacks have no docstrings.
- Preserve useful explanation when moving it out of a private helper or a class summary.
- Explain public I/O, tensor shape and dtype, device ownership, mutation, persistence, and failure behavior where relevant.
- Do not use semicolons or append "defaults to" tails to entries.
- Keep readable prose within 120 characters. Prefer useful contract information to repeated signature descriptions.

## Errors and logging

- Import `get_logger` from `opforch.utils.logging` and initialize module loggers with `get_logger(__name__)`.
- Do not use `print()` in library modules. Standalone examples may print their results.
- Diagnostic warnings and errors identify a backticked offender and end with a period, such as
  `` f"`name={value}` could not be loaded." ``.
- Informational and debug messages use plain prose.
- Raised messages identify a backticked argument or state and end with a period, such as
  `` f"`name` must be positive, but got {value}." ``.
- Use `is None` and `is True` prose rather than shorthand comparisons.
- Validate with `if` and a specific exception, not `assert`. Bare `except:` is forbidden.
- Preserve expected failure-return contracts and let unexpected operational failures propagate.

## Readability and reuse

- Comments explain why, not what. Prefer no comment or one line, with a three-line maximum.
- Do not add banner comments, section separators, or trailing comment periods. The fixed legal header is an exception.
- Separate logical phases with a single blank line in function bodies of at least 12 lines.
- Keep a cohesive group together rather than inserting a blank line after every statement.
- Inline first. Extract a new helper, constant, or parameter only when there is a second call site.
- Do not remove an existing public API solely because the repository has no internal caller.

## Tests and tooling

- Keep tests aligned with their source modules. Test names identify the function or class and the behavior.
- Test functions and pytest fixtures have no docstrings or type hints.
- Use bare assertions without failure-message strings. Test names describe the expectation.
- Separate successful behavior from expected failures. Use `pytest.raises` rather than catching arbitrary exceptions.
- Cover observable contracts, not incidental implementation structure.
- Use the existing pytest, Black, isort, Flake8, Sphinx, and uv tooling.
- Black, isort, and Flake8 use 120 columns. Formatting does not replace review of documentation or logical grouping.

The version remains defined in `opforch/__init__.py`; packaging and documentation derive it from that source.
