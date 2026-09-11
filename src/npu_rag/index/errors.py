"""Error hierarchy for the vector-index package."""

from __future__ import annotations


class VectorIndexError(Exception):
    """Base class for all vector index exceptions."""


class StoreError(VectorIndexError):
    """Raised when an error occurs during database or storage operations."""


class QueryError(VectorIndexError):
    """Raised when an invalid query or search parameter is provided."""


class SyncError(VectorIndexError):
    """Raised when synchronization between ingest and index fails."""


class SchemaError(VectorIndexError):
    """Raised when record conversion or schema validation fails."""


class ConfigurationError(VectorIndexError):
    """Raised when index configuration is invalid or missing."""
