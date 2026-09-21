"""Command-line interface package for `npu-rag`.

The public surface is the console script declared as
`npu-rag = "npu_rag.cli.main:main"` under `[project.scripts]`, resolved as the
`main` attribute of the `npu_rag.cli.main` **module**.

This `__init__` deliberately re-exports nothing. `from npu_rag.cli.main import
main` here would bind the name `main` on the package to the *function*,
shadowing the `main` *submodule* of the same name: `from npu_rag.cli import
main` would then hand a caller a function where it asked for a module, and
which of the two wins depends on import order. Keeping the namespace empty
makes `npu_rag.cli.main` mean exactly one thing.

Nothing imported here may be heavy either. design.md §1 principle 5 requires
the top-level entry, `--help` and `--version` to touch neither `onnxruntime`,
`torch`, `transformers` nor `lancedb`, and this module runs on every one of
those paths. The `cmd_*` handler modules are imported by name at dispatch time,
inside `npu_rag.cli.main.dispatch()`.
"""

from __future__ import annotations

__all__: list[str] = []
