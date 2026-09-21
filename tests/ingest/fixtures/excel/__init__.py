"""Package marker.

Present so that test modules sharing a basename across sibling directories
resolve to distinct dotted module paths. Five basenames collide today
(`test_package_baseline.py`, `test_types.py`, `test_errors.py`,
`test_service.py`, `generate.py`), and without these markers mypy stops at the
first collision with "Duplicate module named ..." and checks nothing at all.
pytest is unaffected either way: it uses `--import-mode=importlib`.

Do not delete. See `.kiro/specs/npu-embedding-runtime/tasks.md` for the
project's standing rule that a verification gate must fail loudly, not
silently stop checking.
"""