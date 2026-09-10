"""The OpenRouter vision credential (task 1.5).

A secret that cannot render itself. The value is reachable only through
``reveal``, so the ordinary ways a value escapes — an f-string, a ``repr``
in a traceback, an exception message — yield the redaction marker instead.

Discovery reads ``OPENROUTER_API_KEY`` from the process environment or the
nearest ``.env``, using the runtime's dotenv helpers rather than a copy.
This module sits immediately to the right of config in the ingest
dependency direction; it imports nothing from this package.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from npu_rag.embedding.models.acquire import REDACTED, find_dotenv, parse_dotenv

__all__ = ["OpenRouterCredential", "discover_openrouter_credential"]

_ENV_KEY = "OPENROUTER_API_KEY"


class OpenRouterCredential:
    """An OpenRouter API key that does not render itself.

    The value is reachable only through ``reveal``, which exists so that
    every place the secret is genuinely needed is one grep away, and so
    that the ordinary ways a value escapes yield the redaction instead.

    ``source`` says where the credential came from, never what it is.
    """

    __slots__ = ("_value", "source")

    def __init__(self, value: str, *, source: str) -> None:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(
                "a credential cannot be blank: an empty token is an absent one, "
                "and absence is spelled None so the two cannot be confused"
            )
        self._value = cleaned
        self.source = source

    def reveal(self) -> str:
        """The key itself. The only way to obtain it, and deliberately so."""
        return self._value

    def redact(self, text: str) -> str:
        """``text`` with every occurrence of the key replaced."""
        return text.replace(self._value, REDACTED)

    def __repr__(self) -> str:
        return f"OpenRouterCredential(source={self.source!r}, value={REDACTED})"

    __str__ = __repr__


def discover_openrouter_credential(
    *,
    env: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> OpenRouterCredential | None:
    """Find an OpenRouter credential, or report honestly that there is none.

    The process environment wins over the file, so a one-off override does
    not require editing a gitignored file. ``None`` means no credential
    was found — image-to-text is then unavailable, and the run continues
    on the local paths alone.
    """
    environment = env if env is not None else os.environ
    value = environment.get(_ENV_KEY)
    if value is not None and value.strip():
        return OpenRouterCredential(
            value, source=f"the {_ENV_KEY} environment variable"
        )

    dotenv = find_dotenv(start)
    if dotenv is None:
        return None
    try:
        parsed = parse_dotenv(dotenv.read_text(encoding="utf-8"))
    except OSError:
        return None
    value = parsed.get(_ENV_KEY)
    if value is not None and value.strip():
        return OpenRouterCredential(value, source=f"{_ENV_KEY} in {dotenv}")
    return None
