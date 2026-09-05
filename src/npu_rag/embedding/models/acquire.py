"""Model acquisition and gated-license detection (task 3.1).

Requirement 4.5 is the whole reason this module is separate from the export and
artifact steps that follow it: *if a candidate model requires acceptance of
license terms before it can be obtained, then the Embedding Runtime shall report
the licensing requirement and the acceptance step, rather than failing with a
generic retrieval error.* EmbeddingGemma - the initial default candidate (4.2) -
is gated behind the Gemma Terms of Use, so this is a routine path, not an edge
case, and an operator who meets it once needs to be told to click a form rather
than to retry a download.

**The two facts this module produces.** Acquisition returns where the files
landed *and* the revision they came from, resolved to a concrete commit hash.
design.md's Security Considerations require the second: "acquisition records
model revision in the manifest so a silently changed upstream artifact is
detectable". A branch name records nothing - ``main`` means one thing today and
another tomorrow - so a revision that is not a forty-character commit hash is
refused here rather than written into a manifest that would then be unable to
tell staleness from identity (4.7). Pinning the download to the resolved commit
is also what makes the Hub's own cache a correct cache: a second call for the
same commit re-reads it instead of re-fetching.

**Why the Hub client is behind a protocol.** The failure this task exists to
handle cannot be reproduced on the developing machine: the Gemma terms *are*
accepted here, so the real repository answers 200. `ModelRepositoryClient` is the
seam that lets the gated-and-unaccepted path be exercised anyway, and the live
tests make real requests for metadata and a few-kilobyte ``config.json`` only -
the weights are over a gigabyte per candidate and no test downloads them.

**How the Hub reports a closed gate**, and why all three shapes are mapped:

- ``GatedRepoError`` - the repository is gated and this account is not on the
  authorised list.
- **403** with a valid credential - authenticated, terms not accepted.
- **401** anonymously, which ``huggingface_hub`` deliberately surfaces as
  ``RepositoryNotFoundError`` ("401 is misleading as it is returned for private
  and gated repos if the user is not authenticated ... we process them as
  RepoNotFound anyway"). Left unmapped, the operator would be told the model
  does not exist, which is the generic retrieval error 4.5 forbids.

401 and 403 both become `LicenseAcceptanceRequired` because both mean "you
cannot have this until you both authenticate and accept", but their *messages*
differ, because the operator's next move differs. A **404** is not folded in:
that is a repository that is genuinely gone, and sending someone to accept terms
for a model that no longer exists would be a worse diagnosis than the transport
error it replaced.

**The credential never appears anywhere.** It is loaded from the project's
gitignored ``.env`` (task 1.6's remaining half: "load the credential from the
environment file in tooling rather than requiring a shell-profile variable"),
carried in an `HfCredential` whose ``repr`` is redacted, and unwrapped at exactly
one place - `HfCredential.reveal` - which is greppable for that reason. Upstream
error text is redacted before it is quoted, and mapped errors are raised ``from
None``: a chained cause would render the transport's own unredacted message into
the traceback, undoing the redaction one line further down.

This module sits in ``models`` in design.md's dependency direction - ``types,
errors -> reporting -> profiles -> environment -> models -> providers -> service
-> bench`` - so it reads errors, reporting and profiles, and nothing to its
right.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError
from huggingface_hub.utils.tqdm import tqdm as hub_tqdm

from npu_rag.embedding.errors import (
    EmbeddingRuntimeError,
    LicenseAcceptanceRequired,
    PreparationError,
)
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.reporting import ProgressCallback, ProgressUpdate

__all__ = [
    "ACQUISITION_STAGE",
    "DEFAULT_REVISION",
    "DISCOVER",
    "DOTENV_FILENAME",
    "REDACTED",
    "TOKEN_KEYS",
    "AcquiredModel",
    "HfCredential",
    "HubRepositoryClient",
    "ModelRepositoryClient",
    "acquire_model",
    "discover_credential",
    "find_dotenv",
    "parse_dotenv",
]

#: The stage every failure here reports (8.1, 4.6). It matches
#: ``LicenseAcceptanceRequired.default_stage`` deliberately: the licensing
#: failure and the honest transport failure happen at the same point in the
#: flow, and only their *type* distinguishes them.
ACQUISITION_STAGE: Final = "acquisition"

#: What is asked for when no revision is named. It is a branch, which is exactly
#: why it is resolved to a commit before anything is downloaded or recorded.
DEFAULT_REVISION: Final = "main"

#: What stands in for a credential wherever text is rendered.
REDACTED: Final = "<redacted>"

#: Where a Hugging Face credential is looked for, in order. The second name is
#: ``huggingface_hub``'s own older variable, honoured so a machine already set up
#: for the Hub does not have to be set up twice.
TOKEN_KEYS: Final = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

#: The project's gitignored credential file (task 1.6).
DOTENV_FILENAME: Final = ".env"

#: Acquisition's two steps, for progress reporting: resolve, then download.
_STEPS: Final = 2

#: A resolved Hub revision: the forty-character commit hash ``ModelInfo.sha``
#: carries. Anything else - a branch, a tag, ``refs/pr/3``, an empty string - is
#: a moving target and is refused.
_RESOLVED_REVISION: Final = re.compile(r"\A[0-9a-f]{40}\Z")


# --------------------------------------------------------------------------
# The credential
# --------------------------------------------------------------------------


class HfCredential:
    """A Hugging Face access token that does not render itself.

    The value is reachable only through `reveal`, which exists so that every
    place the secret is genuinely needed is one grep away, and so that the
    ordinary ways a value escapes - an f-string, a ``repr`` in a traceback, a
    dataclass ``__repr__``, a debugger's variable pane - yield the redaction
    instead.

    ``source`` says *where the credential came from*, never what it is. It is
    safe to print, and the licensing message quotes it so an operator whose
    token is being rejected knows which file or variable to fix.
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
        """The token itself. The only way to obtain it, and deliberately so."""
        return self._value

    def redact(self, text: str) -> str:
        """``text`` with every occurrence of the token replaced."""
        return text.replace(self._value, REDACTED)

    def __repr__(self) -> str:
        return f"HfCredential(source={self.source!r}, value={REDACTED})"

    __str__ = __repr__


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse a ``.env`` file into a mapping.

    Deliberately a dozen lines of the standard library rather than a
    dependency: this project reads exactly one key from one file, and
    ``python-dotenv``'s interpolation, multi-line values and shell semantics are
    surface no one here asks for. The subset supported is the subset the file
    actually uses - ``KEY=value``, an optional ``export`` prefix, ``#``
    comments, blank lines, and one layer of matching quotes.

    Inline comments are **not** stripped, because an unquoted value containing
    ``#`` is far likelier than a trailing comment, and silently truncating a
    credential would produce an authentication failure with no visible cause.
    A line that is not an assignment is skipped rather than raising: a malformed
    line elsewhere in the file must not make the credential unreadable.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        values[key] = _unquoted(value.strip())
    return values


def _unquoted(value: str) -> str:
    """One layer of matching quotes removed, if present."""
    for quote in ('"', "'"):
        if len(value) >= 2 and value.startswith(quote) and value.endswith(quote):
            return value[1:-1]
    return value


def find_dotenv(start: Path | None = None) -> Path | None:
    """The nearest ``.env`` at or above ``start`` (default: the process's cwd).

    Walking upward rather than assuming a fixed location is what lets the file
    be found from a test's temporary directory, from a nested working directory,
    and from an installed package that has no idea where the project root is.
    """
    origin = (start if start is not None else Path.cwd()).resolve()
    for directory in (origin, *origin.parents):
        candidate = directory / DOTENV_FILENAME
        if candidate.is_file():
            return candidate
    return None


def discover_credential(
    *,
    env: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> HfCredential | None:
    """Find a Hub credential, or report honestly that there is none.

    The process environment wins over the file, so a one-off override does not
    require editing a gitignored file. ``None`` means no credential was found -
    an anonymous request follows, and if the repository is gated the resulting
    401 is turned into a licensing error that says so.
    """
    environment = env if env is not None else os.environ
    for key in TOKEN_KEYS:
        value = environment.get(key)
        if value is not None and value.strip():
            return HfCredential(value, source=f"the {key} environment variable")

    dotenv = find_dotenv(start)
    if dotenv is None:
        return None
    try:
        parsed = parse_dotenv(dotenv.read_text(encoding="utf-8"))
    except OSError:
        # An unreadable .env is an absent credential, not a crash: the caller's
        # next failure will be the licensing error, which names the file.
        return None
    for key in TOKEN_KEYS:
        value = parsed.get(key)
        if value is not None and value.strip():
            return HfCredential(value, source=f"{key} in {dotenv}")
    return None


class _Discover:
    """Sentinel: work out the credential rather than being handed one.

    A plain ``None`` default cannot express this, because ``None`` already means
    something specific and useful here - *make the request anonymously* - and
    the two must stay distinguishable, or a caller asking for an anonymous
    request would silently get whatever token the machine happens to hold.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "DISCOVER"


DISCOVER: Final[_Discover] = _Discover()


# --------------------------------------------------------------------------
# What acquisition produces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AcquiredModel:
    """Where a model's files are, and exactly which commit they came from.

    ``revision`` is a resolved commit hash, enforced here as an invariant rather
    than trusted: task 3.3's manifest records this value and compares it on the
    next run, and a branch name recorded there would compare equal to itself
    forever while the files underneath changed (design.md, Security
    Considerations; requirement 4.7).
    """

    model_id: str
    revision: str
    local_path: Path

    def __post_init__(self) -> None:
        if not _RESOLVED_REVISION.match(self.revision):
            raise ValueError(
                f"revision {self.revision!r} is not a resolved commit hash; a "
                "branch or tag cannot identify the artifact that was actually "
                "downloaded"
            )
        if not self.local_path.is_dir():
            raise ValueError(
                f"local_path {self.local_path} is not a directory: acquisition "
                "reports where the files are, so a path with nothing at it is a "
                "failed acquisition wearing a success"
            )


# --------------------------------------------------------------------------
# The repository seam
# --------------------------------------------------------------------------


class ModelRepositoryClient(Protocol):
    """The two things acquisition needs from a model repository.

    Kept to two methods on purpose. This is the boundary at which the gated
    path is simulated, and a wide protocol would make the simulation less like
    the thing it stands in for.
    """

    def resolve_revision(
        self, *, repo_id: str, revision: str, credential: HfCredential | None
    ) -> str:
        """The concrete commit hash ``revision`` currently points at."""
        ...

    def download(
        self,
        *,
        repo_id: str,
        revision: str,
        credential: HfCredential | None,
        allow_patterns: Sequence[str] | None,
        cache_dir: Path | None,
    ) -> Path:
        """Fetch the repository at ``revision`` and return the local directory."""
        ...


def _token_argument(credential: HfCredential | None) -> str | bool:
    """What ``huggingface_hub`` should be told about authentication.

    ``False`` - not ``None`` - is how the Hub client is told to be genuinely
    anonymous. Passing ``None`` lets it fall back to a cached login or an
    ambient environment variable, which would make an explicitly anonymous
    request quietly authenticated and would make the gated-repository test
    depend on the machine it runs on.
    """
    return credential.reveal() if credential is not None else False


class _SilentTqdm(hub_tqdm):
    """The Hub's own progress bar, constructed disabled.

    design.md's Monitoring decision: "Progress is a callback, not logging, so
    callers choose presentation". The Hub renders a download bar to stderr,
    which would make that choice on the caller's behalf and would break the
    benchmark harness's requirement not to assume a terminal.

    Disabling it *per call* rather than through
    ``huggingface_hub.disable_progress_bars()`` matters: the latter is a
    process-global switch, and a library that flipped it would silence progress
    bars in unrelated code the host application never asked us to touch.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["disable"] = True
        # ``tqdm`` ships no annotations, so strict mode sees an untyped call
        # here. The ignore is scoped to this one line rather than widened to a
        # per-module override, which would also hide real type errors in this
        # file.
        super().__init__(*args, **kwargs)  # type: ignore[no-untyped-call]


class HubRepositoryClient:
    """`ModelRepositoryClient` backed by ``huggingface_hub``."""

    def __init__(self, *, api: HfApi | None = None) -> None:
        self._api = api if api is not None else HfApi()

    def resolve_revision(
        self, *, repo_id: str, revision: str, credential: HfCredential | None
    ) -> str:
        info = self._api.model_info(
            repo_id, revision=revision, token=_token_argument(credential)
        )
        # A metadata call only: no weights move here, and this is where a closed
        # gate is met - before a byte of a gigabyte-sized repository is fetched.
        return info.sha or ""

    def download(
        self,
        *,
        repo_id: str,
        revision: str,
        credential: HfCredential | None,
        allow_patterns: Sequence[str] | None,
        cache_dir: Path | None,
    ) -> Path:
        local = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=_token_argument(credential),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            cache_dir=cache_dir,
            tqdm_class=_SilentTqdm,
        )
        if not isinstance(local, str):
            raise PreparationError(
                f"the model repository returned {type(local).__name__} instead "
                "of a local path",
                model_id=repo_id,
                stage=ACQUISITION_STAGE,
            )
        return Path(local)


# --------------------------------------------------------------------------
# Failure mapping
# --------------------------------------------------------------------------


def _status_code(error: Exception) -> int | None:
    """The HTTP status behind ``error``, when there is one.

    Read defensively through ``getattr`` because the attribute belongs to the
    Hub's exception hierarchy, not to ours, and an acquisition failure is a bad
    moment to raise a second, unrelated error while diagnosing the first.
    """
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _acceptance_url(profile: ModelProfile) -> str:
    """Where the terms are accepted.

    The profile's URL when it declares one. When it does not - a comparator that
    gained a gate upstream - the model card stands in, because that is where the
    acceptance form lives: research.md records that "the acceptance form is
    embedded at the top of the model card when logged in; there is no separate
    terms page".
    """
    return (
        profile.license_acceptance_url
        or f"https://huggingface.co/{profile.model_id}"
    )


def _credential_state(credential: HfCredential | None) -> str:
    if credential is None:
        return (
            "no credential was found (looked for "
            f"{' and '.join(TOKEN_KEYS)} in the environment and in the nearest "
            f"{DOTENV_FILENAME})"
        )
    return f"the credential from {credential.source} was not accepted"


def _acquisition_failure(
    error: Exception,
    *,
    profile: ModelProfile,
    credential: HfCredential | None,
    step: str,
) -> EmbeddingRuntimeError:
    """Turn whatever the repository raised into this feature's vocabulary.

    Every path returns one of ours, so nothing raw reaches a caller and a
    ``except PreparationError`` covers acquisition entirely (8.2). The detail
    from upstream is quoted, redacted, because it is often the only clue about
    a transport failure - but it is quoted as *text*, never chained, so the
    redaction cannot be undone by the traceback printing the original.
    """
    detail = f"{type(error).__name__}: {error}"
    if credential is not None:
        detail = credential.redact(detail)
    status = _status_code(error)

    gated = isinstance(error, GatedRepoError)
    if profile.license_gated and status in (401, 403):
        gated = True

    if gated:
        if status == 401:
            message = (
                f"{profile.model_id} is gated, and the request was not "
                f"authenticated (HTTP 401): {_credential_state(credential)}. "
                f"Authenticate with a Hugging Face token that has read access, "
                f"then accept the terms for this repository. "
                f"[{step}] {detail}"
            )
        else:
            message = (
                f"{profile.model_id} is gated and its terms have not been "
                f"accepted for the account in use"
                f"{'' if status is None else f' (HTTP {status})'}. This is a "
                f"one-time acceptance, not a transient failure, so retrying "
                f"the download will not resolve it. [{step}] {detail}"
            )
        return LicenseAcceptanceRequired(
            message,
            acceptance_url=_acceptance_url(profile),
            model_id=profile.model_id,
            stage=ACQUISITION_STAGE,
        )

    return PreparationError(
        f"could not acquire {profile.model_id} at the {step} step"
        f"{'' if status is None else f' (HTTP {status})'}: {detail}",
        model_id=profile.model_id,
        stage=ACQUISITION_STAGE,
    )


def _guarded[T](
    action: Callable[[], T],
    *,
    step: str,
    profile: ModelProfile,
    credential: HfCredential | None,
) -> T:
    """Run one repository step, translating any failure it raises.

    ``from None`` is load-bearing rather than tidy: the cause's own message is
    not redacted, and a chained traceback would print it verbatim under the
    redacted one. The cause's type and text survive inside the message instead.
    """
    try:
        return action()
    except EmbeddingRuntimeError:
        # Already diagnosed in this feature's vocabulary; re-diagnosing it would
        # bury the specific stage under a generic one.
        raise
    except Exception as error:
        raise _acquisition_failure(
            error, profile=profile, credential=credential, step=step
        ) from None


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------


def _emit(
    progress: ProgressCallback | None, operation: str, completed: int
) -> None:
    if progress is None:
        return
    progress(
        ProgressUpdate(operation=operation, completed=completed, total=_STEPS)
    )


def acquire_model(
    profile: ModelProfile,
    *,
    client: ModelRepositoryClient | None = None,
    credential: HfCredential | None | _Discover = DISCOVER,
    revision: str = DEFAULT_REVISION,
    cache_dir: Path | None = None,
    allow_patterns: Sequence[str] | None = None,
    progress: ProgressCallback | None = None,
) -> AcquiredModel:
    """Obtain ``profile``'s model files and the commit they came from.

    The revision is resolved first, on a metadata-only call, for three reasons:
    a closed gate is met before any weights move, the value design.md requires
    the manifest to record is a commit rather than a branch, and pinning the
    download to that commit lets the Hub cache serve a second call instead of
    re-fetching.

    ``credential`` is tri-state. Omitted, a credential is discovered from the
    environment or the project's ``.env``; ``None`` means an explicitly
    anonymous request; an `HfCredential` is used as given.

    Raises `LicenseAcceptanceRequired` when the repository is gated and its
    terms are not accepted (4.5), and `PreparationError` naming the acquisition
    stage for every other failure (4.6, 8.1).
    """
    repository = client if client is not None else HubRepositoryClient()
    resolved_credential = (
        discover_credential() if isinstance(credential, _Discover) else credential
    )
    operation = f"acquire:{profile.name}"

    _emit(progress, operation, 0)

    resolved_revision = _guarded(
        lambda: repository.resolve_revision(
            repo_id=profile.model_id,
            revision=revision,
            credential=resolved_credential,
        ),
        step="revision resolution",
        profile=profile,
        credential=resolved_credential,
    )
    if not _RESOLVED_REVISION.match(resolved_revision):
        raise PreparationError(
            f"the model repository resolved {revision!r} to "
            f"{resolved_revision!r}, which is not a commit hash; the artifact "
            "manifest records a revision so a silently changed upstream model "
            "is detectable, and a moving reference cannot do that",
            model_id=profile.model_id,
            stage=ACQUISITION_STAGE,
        )
    _emit(progress, operation, 1)

    local_path = _guarded(
        lambda: repository.download(
            repo_id=profile.model_id,
            revision=resolved_revision,
            credential=resolved_credential,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
        ),
        step="download",
        profile=profile,
        credential=resolved_credential,
    )

    try:
        acquired = AcquiredModel(
            model_id=profile.model_id,
            revision=resolved_revision,
            local_path=local_path,
        )
    except ValueError as error:
        raise PreparationError(
            f"acquisition of {profile.model_id} did not produce a usable "
            f"result: {error}",
            model_id=profile.model_id,
            stage=ACQUISITION_STAGE,
        ) from None

    _emit(progress, operation, _STEPS)
    return acquired
