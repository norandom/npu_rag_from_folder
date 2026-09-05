"""Repository tooling.

Not part of the distributed package - the wheel builds only ``src/npu_rag``.
Code here mutates the developer's machine (environment provisioning), which
design.md places outside the ``npu_rag.embedding`` boundary: that package
*detects and reports* environment state and never changes it.
"""
