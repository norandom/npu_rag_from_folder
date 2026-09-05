"""NPU-accelerated text embedding runtime.

Public surface for this package re-exports the embedding service and its types
only. Nothing is exported yet; the contracts land with the components that
define them.

This package sits at the outward end of the project's dependency direction and
must not import from any sibling ``npu_rag`` sub-package.
"""

__all__: list[str] = []
