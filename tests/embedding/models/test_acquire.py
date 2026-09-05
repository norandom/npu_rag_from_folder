"""Unit tests for model acquisition and gated-license detection (task 3.1).

Nothing here touches the network and nothing here downloads a model. Every test
drives ``acquire_model`` through an injected ``ModelRepositoryClient``, which is
the seam that exists precisely so the failure this task is about - a gated
repository whose terms have not been accepted - can be exercised on a machine
where the terms *have* been accepted and the gate therefore opens.

The download-size discipline is a hard constraint rather than a preference:
``bge-large-en-v1.5`` is roughly 1.3 GB and EmbeddingGemma roughly 1.2 GB, so a
test that "just downloads the model" would cost gigabytes per run. The live
counterpart in ``test_acquire_live.py`` makes real requests, but only for
metadata and a few-kilobyte ``config.json``.
"""

from __future__ import annotations

import ast
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

from npu_rag.embedding.errors import (
    EmbeddingRuntimeError,
    LicenseAcceptanceRequired,
    PreparationError,
)
from npu_rag.embedding.models import acquire
from npu_rag.embedding.models.acquire import (
    ACQUISITION_STAGE,
    DEFAULT_REVISION,
    REDACTED,
    AcquiredModel,
    HfCredential,
    HubRepositoryClient,
    acquire_model,
    discover_credential,
    parse_dotenv,
)
from npu_rag.embedding.profiles import PROFILES, ModelProfile, profile_for
from npu_rag.embedding.reporting import ProgressUpdate

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "npu_rag"
    / "embedding"
    / "models"
    / "acquire.py"
)

#: A plausible resolved commit hash: forty lowercase hex characters, which is
#: what the Hub returns as ``ModelInfo.sha``.
SHA = "0f4a5c2b9d1e6f7a8b3c4d5e6f708192a3b4c5d6"
OTHER_SHA = "1111111111111111111111111111111111111111"

GATED = profile_for("embeddinggemma-300m")
OPEN = profile_for("bge-large-en-v1.5")


# --------------------------------------------------------------------------
# Dependency direction (design.md, Architecture)
# --------------------------------------------------------------------------

MODULE_PACKAGE = "npu_rag.embedding.models"

#: design.md, Architecture: "types, errors -> reporting -> profiles ->
#: environment -> models -> providers -> service -> bench". This module sits in
#: ``models``, so everything to its right is off limits.
LAYERS_RIGHT_OF_MODELS = ("providers", "service", "bench")


def absolute_imports_of(source: str, package: str) -> list[str]:
    """Every name ``source`` imports, as an absolute dotted path.

    Relative imports are resolved rather than skipped, so ``from .. import
    providers`` is recognised as the same violation as
    ``import npu_rag.embedding.providers``.
    """
    parts = package.split(".")
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                root = node.module or ""
            else:
                base = ".".join(parts[: len(parts) - node.level + 1])
                root = f"{base}.{node.module}" if node.module else base
            names.append(root)
            names.extend(f"{root}.{alias.name}" for alias in node.names)
    return names


def test_module_imports_nothing_from_a_later_layer() -> None:
    imported = absolute_imports_of(
        MODULE_PATH.read_text(encoding="utf-8"), MODULE_PACKAGE
    )

    forbidden = sorted(
        {
            name
            for name in imported
            for layer in LAYERS_RIGHT_OF_MODELS
            if name == f"npu_rag.embedding.{layer}"
            or name.startswith(f"npu_rag.embedding.{layer}.")
        }
    )
    assert forbidden == []


@pytest.mark.parametrize(
    "statement",
    [
        "import npu_rag.embedding.providers",
        "from npu_rag.embedding.providers import base",
        "from npu_rag.embedding import providers",
        "from .. import providers",
        "from ..providers import base",
        "from ...embedding.service import EmbeddingService",
    ],
)
def test_the_layer_guard_recognises_every_import_spelling(statement: str) -> None:
    """The guard above is only worth having if a different import spelling
    cannot side-step it."""
    imported = absolute_imports_of(statement, MODULE_PACKAGE)

    assert any(
        name.startswith(f"npu_rag.embedding.{layer}")
        for name in imported
        for layer in LAYERS_RIGHT_OF_MODELS
    )


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


def http_error(
    kind: type[HfHubHTTPError], status: int, message: str = "boom"
) -> HfHubHTTPError:
    """A real Hub error carrying a real status code.

    Built from the genuine exception classes rather than a stand-in, because
    the mapping under test keys on those types and on the status code the
    response carries; a hand-rolled double would let a mapping that recognised
    neither still pass.
    """
    request = httpx.Request(
        "GET", "https://huggingface.co/api/models/google/embeddinggemma-300m"
    )
    return kind(message, response=httpx.Response(status, request=request))


class FakeClient:
    """A ``ModelRepositoryClient`` that records its calls and never uses a
    network."""

    def __init__(
        self,
        *,
        sha: str = SHA,
        local_path: Path | None = None,
        resolve_error: Exception | None = None,
        download_error: Exception | None = None,
    ) -> None:
        self.sha = sha
        self.local_path = local_path
        self.resolve_error = resolve_error
        self.download_error = download_error
        self.resolve_calls: list[dict[str, Any]] = []
        self.download_calls: list[dict[str, Any]] = []

    def resolve_revision(
        self, *, repo_id: str, revision: str, credential: HfCredential | None
    ) -> str:
        self.resolve_calls.append(
            {"repo_id": repo_id, "revision": revision, "credential": credential}
        )
        if self.resolve_error is not None:
            raise self.resolve_error
        return self.sha

    def download(
        self,
        *,
        repo_id: str,
        revision: str,
        credential: HfCredential | None,
        allow_patterns: Sequence[str] | None,
        cache_dir: Path | None,
    ) -> Path:
        self.download_calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "credential": credential,
                "allow_patterns": allow_patterns,
                "cache_dir": cache_dir,
            }
        )
        if self.download_error is not None:
            raise self.download_error
        assert self.local_path is not None
        return self.local_path


# --------------------------------------------------------------------------
# The .env parser and credential discovery (task 1.6's remaining half)
# --------------------------------------------------------------------------


def test_parse_dotenv_reads_plain_assignments() -> None:
    assert parse_dotenv("HF_TOKEN=abc\nOTHER=def\n") == {
        "HF_TOKEN": "abc",
        "OTHER": "def",
    }


def test_parse_dotenv_ignores_comments_and_blank_lines() -> None:
    text = "\n# a comment\n\n   \nHF_TOKEN=abc\n  # indented comment\n"

    assert parse_dotenv(text) == {"HF_TOKEN": "abc"}


def test_parse_dotenv_strips_export_prefix_and_surrounding_quotes() -> None:
    text = 'export HF_TOKEN="abc"\nB=\'def\'\nC = ghi \n'

    assert parse_dotenv(text) == {"HF_TOKEN": "abc", "B": "def", "C": "ghi"}


def test_parse_dotenv_keeps_equals_signs_inside_a_value() -> None:
    assert parse_dotenv("HF_TOKEN=a=b=c\n") == {"HF_TOKEN": "a=b=c"}


def test_parse_dotenv_skips_lines_that_are_not_assignments() -> None:
    assert parse_dotenv("not an assignment\nHF_TOKEN=abc\n=novalue\n") == {
        "HF_TOKEN": "abc"
    }


def test_discover_credential_prefers_the_process_environment(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("HF_TOKEN=from-dotenv\n", encoding="utf-8")

    credential = discover_credential(
        env={"HF_TOKEN": "from-environment"}, start=tmp_path
    )

    assert credential is not None
    assert credential.reveal() == "from-environment"
    assert ".env" not in credential.source


def test_discover_credential_reads_the_project_dotenv_walking_upward(
    tmp_path: Path,
) -> None:
    """Task 1.6: "load the credential from the environment file in tooling
    rather than requiring a shell-profile variable"."""
    (tmp_path / ".env").write_text("HF_TOKEN=from-dotenv\n", encoding="utf-8")
    nested = tmp_path / "src" / "npu_rag"
    nested.mkdir(parents=True)

    credential = discover_credential(env={}, start=nested)

    assert credential is not None
    assert credential.reveal() == "from-dotenv"


def test_discover_credential_returns_none_when_nothing_supplies_one(
    tmp_path: Path,
) -> None:
    assert discover_credential(env={}, start=tmp_path) is None


def test_discover_credential_ignores_a_blank_assignment(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("HF_TOKEN=   \n", encoding="utf-8")

    assert discover_credential(env={"HF_TOKEN": "  "}, start=tmp_path) is None


def test_credential_never_renders_its_value() -> None:
    credential = HfCredential("hf_secret_value", source=".env")

    assert "hf_secret_value" not in repr(credential)
    assert "hf_secret_value" not in str(credential)
    assert REDACTED in repr(credential)
    assert credential.reveal() == "hf_secret_value"


def test_credential_rejects_a_blank_value() -> None:
    with pytest.raises(ValueError):
        HfCredential("   ", source=".env")


# --------------------------------------------------------------------------
# Acquisition: the resolved revision (design.md, Security Considerations)
# --------------------------------------------------------------------------


def test_acquisition_returns_the_local_path_and_the_resolved_revision(
    tmp_path: Path,
) -> None:
    client = FakeClient(local_path=tmp_path)

    acquired = acquire_model(OPEN, client=client, credential=None)

    assert acquired == AcquiredModel(
        model_id=OPEN.model_id, revision=SHA, local_path=tmp_path
    )


def test_download_is_pinned_to_the_resolved_commit_not_the_branch(
    tmp_path: Path,
) -> None:
    """design.md, Security Considerations: "acquisition records model revision
    in the manifest so a silently changed upstream artifact is detectable".

    A branch name records nothing - ``main`` means something different tomorrow.
    Pinning the download to the resolved commit is also what makes the Hub cache
    a correct cache rather than a stale one.
    """
    client = FakeClient(local_path=tmp_path)

    acquire_model(OPEN, client=client, credential=None)

    assert client.resolve_calls[0]["revision"] == DEFAULT_REVISION
    assert client.download_calls[0]["revision"] == SHA


def test_a_branch_name_is_not_accepted_as_a_resolved_revision(
    tmp_path: Path,
) -> None:
    client = FakeClient(sha="main", local_path=tmp_path)

    with pytest.raises(PreparationError) as caught:
        acquire_model(OPEN, client=client, credential=None)

    assert caught.value.stage == ACQUISITION_STAGE
    assert caught.value.model_id == OPEN.model_id
    assert "main" in str(caught.value)
    # Nothing was downloaded against an unresolved reference.
    assert client.download_calls == []


def test_an_empty_revision_is_not_accepted(tmp_path: Path) -> None:
    client = FakeClient(sha="", local_path=tmp_path)

    with pytest.raises(PreparationError):
        acquire_model(OPEN, client=client, credential=None)


def test_acquired_model_refuses_an_unresolved_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AcquiredModel(model_id=OPEN.model_id, revision="main", local_path=tmp_path)


def test_acquired_model_refuses_a_path_that_is_not_a_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "nowhere"

    with pytest.raises(ValueError):
        AcquiredModel(model_id=OPEN.model_id, revision=SHA, local_path=missing)


def test_an_explicit_revision_is_resolved_rather_than_passed_through(
    tmp_path: Path,
) -> None:
    client = FakeClient(sha=OTHER_SHA, local_path=tmp_path)

    acquired = acquire_model(
        OPEN, client=client, credential=None, revision="refs/pr/3"
    )

    assert client.resolve_calls[0]["revision"] == "refs/pr/3"
    assert acquired.revision == OTHER_SHA


# --------------------------------------------------------------------------
# Acquisition: credential handling
# --------------------------------------------------------------------------


def test_an_explicit_none_credential_is_carried_to_both_calls(
    tmp_path: Path,
) -> None:
    """``None`` means anonymous, and must not quietly become "whatever token
    happens to be lying around in this process"."""
    client = FakeClient(local_path=tmp_path)

    acquire_model(OPEN, client=client, credential=None)

    assert client.resolve_calls[0]["credential"] is None
    assert client.download_calls[0]["credential"] is None


def test_a_supplied_credential_is_carried_to_both_calls(tmp_path: Path) -> None:
    client = FakeClient(local_path=tmp_path)
    credential = HfCredential("hf_secret_value", source=".env")

    acquire_model(GATED, client=client, credential=credential)

    assert client.resolve_calls[0]["credential"] is credential
    assert client.download_calls[0]["credential"] is credential


def test_the_cache_directory_and_patterns_reach_the_download(
    tmp_path: Path,
) -> None:
    client = FakeClient(local_path=tmp_path)
    cache = tmp_path / "cache"

    acquire_model(
        OPEN,
        client=client,
        credential=None,
        cache_dir=cache,
        allow_patterns=["config.json"],
    )

    assert client.download_calls[0]["cache_dir"] == cache
    assert client.download_calls[0]["allow_patterns"] == ["config.json"]


# --------------------------------------------------------------------------
# Acquisition: progress (design.md, Monitoring - a callback, never a print)
# --------------------------------------------------------------------------


def test_progress_is_reported_through_the_callback(tmp_path: Path) -> None:
    client = FakeClient(local_path=tmp_path)
    seen: list[ProgressUpdate] = []

    acquire_model(OPEN, client=client, credential=None, progress=seen.append)

    assert [update.completed for update in seen] == [0, 1, 2]
    assert all(update.total == 2 for update in seen)
    assert all(OPEN.name in update.operation for update in seen)
    assert seen[-1].remaining == 0


def test_acquisition_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = FakeClient(local_path=tmp_path)

    acquire_model(OPEN, client=client, credential=None)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --------------------------------------------------------------------------
# The Observable: a gated repository whose terms are not accepted
# --------------------------------------------------------------------------


def test_a_gated_repository_raises_the_licensing_error_with_the_acceptance_step(
    tmp_path: Path,
) -> None:
    """Task 3.1's Observable, and requirement 4.5."""
    client = FakeClient(
        local_path=tmp_path, resolve_error=http_error(GatedRepoError, 403)
    )

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        acquire_model(GATED, client=client, credential=None)

    error = caught.value
    assert error.acceptance_url == GATED.license_acceptance_url
    assert error.model_id == GATED.model_id
    assert error.stage == ACQUISITION_STAGE
    assert error.acceptance_url is not None
    assert error.acceptance_url in str(error)


def test_the_licensing_error_is_not_a_transport_error(tmp_path: Path) -> None:
    """Requirement 4.5: "rather than failing with a generic retrieval error"."""
    client = FakeClient(
        local_path=tmp_path, resolve_error=http_error(GatedRepoError, 403)
    )

    with pytest.raises(EmbeddingRuntimeError) as caught:
        acquire_model(GATED, client=client, credential=None)

    assert isinstance(caught.value, LicenseAcceptanceRequired)
    assert not isinstance(caught.value, HfHubHTTPError)


def test_a_forbidden_response_on_a_gated_repo_is_a_licensing_failure(
    tmp_path: Path,
) -> None:
    """403 means the credential is valid but the terms are not accepted."""
    client = FakeClient(
        local_path=tmp_path, resolve_error=http_error(HfHubHTTPError, 403)
    )

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        acquire_model(
            GATED,
            client=client,
            credential=HfCredential("hf_secret_value", source=".env"),
        )

    assert "accept" in str(caught.value).lower()


def test_an_unauthorized_response_on_a_gated_repo_is_a_licensing_failure(
    tmp_path: Path,
) -> None:
    """The Hub reports an anonymous request for a gated repo as 401, and
    ``huggingface_hub`` surfaces that as ``RepositoryNotFoundError`` - "401 is
    misleading ... we process them as RepoNotFound anyway". Left unmapped, the
    operator would be told the model does not exist."""
    client = FakeClient(
        local_path=tmp_path,
        resolve_error=http_error(RepositoryNotFoundError, 401),
    )

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        acquire_model(GATED, client=client, credential=None)

    assert caught.value.acceptance_url == GATED.license_acceptance_url


def test_the_two_gate_messages_name_different_next_steps(tmp_path: Path) -> None:
    """Both 401 and 403 mean "you cannot have this yet", but the operator's next
    move differs: authenticate, versus accept the terms."""
    unauthorized = FakeClient(
        local_path=tmp_path,
        resolve_error=http_error(RepositoryNotFoundError, 401),
    )
    forbidden = FakeClient(
        local_path=tmp_path, resolve_error=http_error(HfHubHTTPError, 403)
    )

    with pytest.raises(LicenseAcceptanceRequired) as first:
        acquire_model(GATED, client=unauthorized, credential=None)
    with pytest.raises(LicenseAcceptanceRequired) as second:
        acquire_model(GATED, client=forbidden, credential=None)

    assert "authenticat" in str(first.value).lower()
    assert "authenticat" not in second.value.message.lower()
    assert str(first.value) != str(second.value)


def test_the_unauthorized_message_says_no_credential_was_found(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        local_path=tmp_path,
        resolve_error=http_error(RepositoryNotFoundError, 401),
    )

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        acquire_model(GATED, client=client, credential=None)

    assert "no credential" in str(caught.value).lower()


def test_the_unauthorized_message_names_the_credential_source_when_there_is_one(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        local_path=tmp_path,
        resolve_error=http_error(RepositoryNotFoundError, 401),
    )

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        acquire_model(
            GATED,
            client=client,
            credential=HfCredential("hf_secret_value", source="the .env file"),
        )

    assert "the .env file" in str(caught.value)


def test_the_gate_is_detected_during_download_too(tmp_path: Path) -> None:
    """Resolution can succeed on cached metadata and the gate still close on the
    files themselves; both steps must map the same way."""
    client = FakeClient(
        local_path=tmp_path, download_error=http_error(GatedRepoError, 403)
    )

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        acquire_model(GATED, client=client, credential=None)

    assert caught.value.acceptance_url == GATED.license_acceptance_url


def test_a_gate_on_a_profile_declared_open_still_reports_the_licensing_step(
    tmp_path: Path,
) -> None:
    """A comparator that gains a gate upstream is still a licensing problem for
    the operator, so it is reported as one - with the model card, which is where
    the acceptance form lives, standing in for the profile's absent URL."""
    client = FakeClient(
        local_path=tmp_path, resolve_error=http_error(GatedRepoError, 403)
    )

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        acquire_model(OPEN, client=client, credential=None)

    assert caught.value.acceptance_url == f"https://huggingface.co/{OPEN.model_id}"


def test_an_authorization_failure_on_an_open_profile_is_not_a_licence_problem(
    tmp_path: Path,
) -> None:
    """MIT-licensed weights behind a 403 is an access failure, not a licence
    gate; calling it one would send the operator to a form that does not
    exist."""
    client = FakeClient(
        local_path=tmp_path, resolve_error=http_error(HfHubHTTPError, 403)
    )

    with pytest.raises(PreparationError) as caught:
        acquire_model(OPEN, client=client, credential=None)

    assert not isinstance(caught.value, LicenseAcceptanceRequired)
    assert caught.value.stage == ACQUISITION_STAGE


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_only_the_gated_candidate_declares_an_acceptance_url(name: str) -> None:
    profile: ModelProfile = PROFILES[name]

    assert (profile.license_acceptance_url is not None) is profile.license_gated


#: A profile whose acceptance URL is deliberately *not* its model card. The real
#: gated candidate's acceptance form happens to live on its model card, so an
#: implementation that ignored ``license_acceptance_url`` and derived the URL
#: from the model id would agree with it by coincidence and pass every
#: assertion above. This profile makes the two answers different.
ELSEWHERE_GATED = ModelProfile(
    model_id="acme/private-embedder",
    dimension=8,
    compiled_seq_len=16,
    architectural_context_limit=32,
    batch_size=1,
    pooling="mean",
    has_dense_stage=False,
    document_template="{content}",
    query_template="{content}",
    license_gated=True,
    license_acceptance_url="https://terms.example.invalid/accept-here",
)


def test_the_acceptance_url_comes_from_the_profile_not_from_the_model_id(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        local_path=tmp_path, resolve_error=http_error(GatedRepoError, 403)
    )

    with pytest.raises(LicenseAcceptanceRequired) as caught:
        acquire_model(ELSEWHERE_GATED, client=client, credential=None)

    assert caught.value.acceptance_url == ELSEWHERE_GATED.license_acceptance_url
    assert ELSEWHERE_GATED.model_id not in caught.value.acceptance_url


# --------------------------------------------------------------------------
# The Hub-backed client
#
# The seam above is where the gate is *simulated*; this is where the real client
# is checked for the two things a simulation cannot catch - that anonymity is
# requested explicitly, and that the credential is handed over exactly once, to
# the library, and nowhere else.
# --------------------------------------------------------------------------


class FakeApi:
    """Stands in for ``HfApi``, recording what ``model_info`` was asked."""

    def __init__(self, sha: str | None = SHA) -> None:
        self.sha = sha
        self.calls: list[dict[str, Any]] = []

    def model_info(
        self, repo_id: str, *, revision: str | None = None, token: str | bool | None = None
    ) -> Any:
        self.calls.append(
            {"repo_id": repo_id, "revision": revision, "token": token}
        )
        return SimpleNamespace(sha=self.sha)


def hub_client(api: FakeApi) -> HubRepositoryClient:
    return HubRepositoryClient(api=cast("Any", api))


def test_the_hub_client_asks_for_anonymity_explicitly() -> None:
    """``token=None`` is not anonymous to ``huggingface_hub``: it falls back to
    a cached login or an ambient variable. Only ``token=False`` is. An
    explicitly anonymous request that quietly authenticated would make the
    gated-repository behaviour depend on the machine it ran on."""
    api = FakeApi()

    hub_client(api).resolve_revision(
        repo_id="acme/model", revision="main", credential=None
    )

    assert api.calls[0]["token"] is False


def test_the_hub_client_passes_the_revealed_token_when_there_is_one() -> None:
    api = FakeApi()

    hub_client(api).resolve_revision(
        repo_id="acme/model",
        revision="main",
        credential=HfCredential(SECRET, source=".env"),
    )

    assert api.calls[0]["token"] == SECRET
    assert api.calls[0]["revision"] == "main"


def test_the_hub_client_returns_the_resolved_sha() -> None:
    api = FakeApi()

    resolved = hub_client(api).resolve_revision(
        repo_id="acme/model", revision="main", credential=None
    )

    assert resolved == SHA


def test_the_hub_client_reports_a_repository_that_resolves_to_nothing() -> None:
    """``ModelInfo.sha`` is optional; an absent one must not become the string
    ``"None"`` in a manifest."""
    api = FakeApi(sha=None)

    resolved = hub_client(api).resolve_revision(
        repo_id="acme/model", revision="main", credential=None
    )

    assert resolved == ""


def test_the_hub_client_download_pins_and_silences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: dict[str, Any] = {}

    def fake_snapshot_download(**kwargs: Any) -> str:
        recorded.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(acquire, "snapshot_download", fake_snapshot_download)

    local = hub_client(FakeApi()).download(
        repo_id="acme/model",
        revision=SHA,
        credential=None,
        allow_patterns=["config.json"],
        cache_dir=tmp_path / "cache",
    )

    assert local == tmp_path
    assert recorded["revision"] == SHA
    assert recorded["token"] is False
    assert recorded["allow_patterns"] == ["config.json"]
    assert recorded["cache_dir"] == tmp_path / "cache"
    # design.md, Monitoring: no stream output from this package.
    assert recorded["tqdm_class"] is not None
    bar = recorded["tqdm_class"](total=1)
    assert bar.disable is True


def test_the_hub_client_rejects_a_download_that_is_not_a_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        acquire, "snapshot_download", lambda **kwargs: [1, 2, 3]
    )

    with pytest.raises(PreparationError) as caught:
        hub_client(FakeApi()).download(
            repo_id="acme/model",
            revision=SHA,
            credential=None,
            allow_patterns=None,
            cache_dir=None,
        )

    assert caught.value.stage == ACQUISITION_STAGE


# --------------------------------------------------------------------------
# Honest failure for everything that is not a licence gate
# --------------------------------------------------------------------------


def test_a_missing_repository_is_reported_as_a_preparation_failure(
    tmp_path: Path,
) -> None:
    """404 on a gated profile means the repository is gone, not that terms are
    unaccepted; conflating them would send the operator to accept terms for a
    model that no longer exists."""
    client = FakeClient(
        local_path=tmp_path,
        resolve_error=http_error(RepositoryNotFoundError, 404),
    )

    with pytest.raises(PreparationError) as caught:
        acquire_model(GATED, client=client, credential=None)

    assert not isinstance(caught.value, LicenseAcceptanceRequired)
    assert caught.value.stage == ACQUISITION_STAGE
    assert caught.value.model_id == GATED.model_id


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(OSError("[Errno 28] No space left on device"), id="disk-full"),
        pytest.param(
            httpx.ConnectError("connection refused"), id="network-unreachable"
        ),
        pytest.param(
            http_error(RemoteEntryNotFoundError, 404), id="entry-missing"
        ),
        pytest.param(RuntimeError("something unforeseen"), id="unforeseen"),
    ],
)
def test_every_other_failure_arrives_as_a_preparation_error(
    tmp_path: Path, failure: Exception
) -> None:
    """Nothing raw escapes: a caller catching the preparation category catches
    every acquisition failure (requirement 8.2)."""
    client = FakeClient(local_path=tmp_path, resolve_error=failure)

    with pytest.raises(PreparationError) as caught:
        acquire_model(OPEN, client=client, credential=None)

    assert caught.value.stage == ACQUISITION_STAGE
    assert type(failure).__name__ in str(caught.value)


def test_an_error_from_this_package_passes_through_unchanged(
    tmp_path: Path,
) -> None:
    original = PreparationError("already diagnosed", stage="export")
    client = FakeClient(local_path=tmp_path, resolve_error=original)

    with pytest.raises(PreparationError) as caught:
        acquire_model(OPEN, client=client, credential=None)

    assert caught.value is original


def test_a_download_that_returns_no_directory_is_a_preparation_failure(
    tmp_path: Path,
) -> None:
    client = FakeClient(local_path=tmp_path / "does-not-exist")

    with pytest.raises(PreparationError) as caught:
        acquire_model(OPEN, client=client, credential=None)

    assert caught.value.stage == ACQUISITION_STAGE


# --------------------------------------------------------------------------
# The credential never leaks
# --------------------------------------------------------------------------

SECRET = "hf_ThisIsTheSecretTokenValue"


def _formatted(error: BaseException) -> str:
    return "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )


@pytest.mark.parametrize(
    "failure_factory",
    [
        pytest.param(
            lambda: http_error(
                GatedRepoError, 403, f"401 Client Error. Authorization: Bearer {SECRET}"
            ),
            id="gated",
        ),
        pytest.param(
            lambda: http_error(
                RepositoryNotFoundError,
                401,
                f"Invalid credentials: {SECRET}",
            ),
            id="unauthorized",
        ),
        pytest.param(
            lambda: RuntimeError(f"transport dumped headers: token={SECRET}"),
            id="generic",
        ),
    ],
)
def test_the_credential_never_reaches_a_message_or_a_traceback(
    tmp_path: Path, failure_factory: Callable[[], Exception]
) -> None:
    """A hostile-but-realistic transport echoes the Authorization header into
    its error text. Whatever this package raises must not carry it onward - not
    in the message, not in ``repr``, not in the rendered traceback, which means
    the causing exception must not be chained into it either."""
    credential = HfCredential(SECRET, source=".env")
    client = FakeClient(local_path=tmp_path, resolve_error=failure_factory())

    with pytest.raises(EmbeddingRuntimeError) as caught:
        acquire_model(GATED, client=client, credential=credential)

    error = caught.value
    assert SECRET not in str(error)
    assert SECRET not in repr(error)
    assert SECRET not in _formatted(error)
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    assert REDACTED in str(error)


def test_the_credential_never_reaches_a_progress_update(tmp_path: Path) -> None:
    client = FakeClient(local_path=tmp_path)
    seen: list[ProgressUpdate] = []

    acquire_model(
        GATED,
        client=client,
        credential=HfCredential(SECRET, source=".env"),
        progress=seen.append,
    )

    assert all(SECRET not in str(update.as_mapping()) for update in seen)


def test_the_credential_never_reaches_the_acquisition_result(
    tmp_path: Path,
) -> None:
    client = FakeClient(local_path=tmp_path)

    acquired = acquire_model(
        GATED, client=client, credential=HfCredential(SECRET, source=".env")
    )

    assert SECRET not in repr(acquired)
