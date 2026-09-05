"""Live acquisition against the real Hugging Face Hub (task 3.1).

These are the real-network half of the evidence. They skip wholesale when the
Hub is unreachable, so the suite stays green offline - the unit tests in
``test_acquire.py`` cover the same logic against an injected client.

**Nothing here downloads model weights.** ``bge-large-en-v1.5`` is roughly
1.3 GB and EmbeddingGemma roughly 1.2 GB; every test below either makes a
metadata-only call or restricts the download to ``config.json``, a few kilobytes
that the Hub cache then serves for free on subsequent runs.

**What can and cannot be proved on this machine.** The Gemma terms *are*
accepted for the credential stored in this project's ``.env``, so the gated
repository answers 200 here and the failure requirement 4.5 is about cannot be
reproduced with it. It *can* be reproduced anonymously: an unauthenticated
request for the same repository is refused by the Hub, and the assertion is that
this package reports it as a licensing requirement carrying the acceptance step
rather than as a missing repository, which is what the transport says.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from huggingface_hub import HfApi

from npu_rag.embedding.errors import LicenseAcceptanceRequired
from npu_rag.embedding.models.acquire import (
    acquire_model,
    discover_credential,
)
from npu_rag.embedding.profiles import profile_for

GATED = profile_for("embeddinggemma-300m")
OPEN = profile_for("bge-large-en-v1.5")

#: A few kilobytes. Never the weights.
METADATA_ONLY = ["config.json"]


def _hub_reachable() -> bool:
    try:
        HfApi().model_info(OPEN.model_id, token=False, timeout=10)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hub_reachable(), reason="the Hugging Face Hub is not reachable"
)


def test_the_gated_candidate_refuses_an_anonymous_request_with_the_licence_step(
    tmp_path: Path,
) -> None:
    """Task 3.1's Observable, against the real gate.

    The Hub answers an unauthenticated *file* request for a gated repository
    with 401, which ``huggingface_hub`` surfaces as ``RepositoryNotFoundError``
    - the generic retrieval error requirement 4.5 forbids reporting.

    The cache is deliberately a fresh directory. Measured on 2026-09-05: a gated
    repository's **metadata** is public - ``model_info`` returns 200 anonymously
    with ``gated="manual"`` - so the gate closes on the files, and a commit-
    pinned file already in the shared cache is served from disk without any
    request at all. Both behaviours are correct; together they would let this
    test pass on a machine where the licence had never been accepted and fail to
    detect a regression on one where it had. An empty cache removes the
    ambiguity.
    """
    with pytest.raises(LicenseAcceptanceRequired) as caught:
        acquire_model(
            GATED,
            credential=None,
            allow_patterns=METADATA_ONLY,
            cache_dir=tmp_path,
        )

    error = caught.value
    assert error.acceptance_url == GATED.license_acceptance_url
    assert error.model_id == GATED.model_id
    assert error.stage == "acquisition"
    assert "authenticat" in str(error).lower()


def test_an_open_comparator_needs_no_credential() -> None:
    """Requirement 4.5 must not turn every failure into a licence problem, and
    the two comparators are the control: they are ungated and resolve
    anonymously."""
    acquired = acquire_model(OPEN, credential=None, allow_patterns=METADATA_ONLY)

    assert re.fullmatch(r"[0-9a-f]{40}", acquired.revision)
    assert (acquired.local_path / "config.json").is_file()


def test_the_stored_credential_opens_the_gate() -> None:
    """Task 1.6's Observable, now exercised through the package rather than by
    hand: "fetching the primary candidate's configuration with the stored
    credential succeeds"."""
    credential = discover_credential()
    if credential is None:
        pytest.skip("no Hugging Face credential in the environment or .env")

    acquired = acquire_model(
        GATED, credential=credential, allow_patterns=METADATA_ONLY
    )

    assert acquired.model_id == GATED.model_id
    assert re.fullmatch(r"[0-9a-f]{40}", acquired.revision)
    assert (acquired.local_path / "config.json").is_file()


def test_a_second_acquisition_reuses_the_hub_cache(tmp_path: Path) -> None:
    """Acquisition is lazy and cache-aware: pinning the download to the resolved
    commit is what lets the Hub serve the second call from disk.

    An isolated cache directory makes the first call a genuine fetch, so the
    second one is genuinely a reuse rather than two reuses of a cache some
    earlier run populated.
    """
    first = acquire_model(
        OPEN, credential=None, allow_patterns=METADATA_ONLY, cache_dir=tmp_path
    )
    second = acquire_model(
        OPEN, credential=None, allow_patterns=METADATA_ONLY, cache_dir=tmp_path
    )

    assert second == first
    assert (second.local_path / "config.json").is_file()


def test_the_credential_is_loaded_from_the_project_dotenv() -> None:
    """Task 1.6's remaining implementation step: the credential comes from the
    gitignored file, not from a shell profile."""
    credential = discover_credential(env={})
    if credential is None:
        pytest.skip("no .env credential on this machine")

    assert ".env" in credential.source
    assert credential.reveal() not in repr(credential)
