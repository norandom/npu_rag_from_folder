"""Image resolution and the vision seam (task 5.1).

This module resolves each ImageRef emitted by extractors into either a
FigureSegment (with recognizer id and pipeline version provenance) or an
Omission describing why it was excluded or failed.

Resolution order per ImageRef:
1. Threshold: shorter side below IngestConfig.min_image_pixels -> BELOW_THRESHOLD
2. Chart ranges: native Excel chart ranges bypass OCR and emit figure text directly
3. Cache: image hash, recognizer id, and PIPELINE_VERSION match -> cached text
4. Models present: describer is None or raises VisionUnavailable -> VISION_UNAVAILABLE
5. Describe: describer.describe(ref) called for one image at a time
6. Store: retained in StateStore.vision_cache on success

This module sits at rank 8 in design.md's dependency direction. It never
imports httpx, never opens a socket, and imports nothing from chunk, report,
or pipeline.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, ClassVar, Protocol

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import VisionError, VisionUnavailable
from npu_rag.ingest.state import StateStore
from npu_rag.ingest.types import (
    ChartRanges,
    FigureSegment,
    ImageRef,
    Omission,
    OmissionCategory,
    Provenance,
    Resolved,
)

__all__ = [
    "CHECKER_ID",
    "DETECTOR_ID",
    "PIPELINE_VERSION",
    "RECOGNIZER_ID",
    "LocalOcrDescriber",
    "NumericDisagreedError",
    "VisionDescriber",
    "filter_numeric_tokens",
    "format_chart_ranges",
    "resolve",
    "resolve_all",
]

# Engine identities (requirement 5.7)
DETECTOR_ID = "infiniflow/deepdoc/det"
RECOGNIZER_ID = "infiniflow/deepdoc/rec"
CHECKER_ID = "tesseract-5-lstm"

#: Module constant derived from detector, recognizer and checker identities.
#: Changing an engine without bumping this version is a review failure.
PIPELINE_VERSION = f"ocr:det={DETECTOR_ID},rec={RECOGNIZER_ID},checker={CHECKER_ID};v1"


class NumericDisagreedError(VisionError):
    """Numeric tokens disagreed between OCR engines and no text remains."""

    default_stage: ClassVar[str] = "vision"


class VisionDescriber(Protocol):
    """Protocol for image-to-text conversion (design.md, Service Interface)."""

    def describe(self, image: ImageRef) -> str:
        """Transcribe visible text from image locally.

        Raises VisionError on failure; raises VisionUnavailable if model files
        or required external binaries are absent.
        """
        ...


_CURRENCY_AND_SYMBOLS = "$€£¥%,"


def _is_numeric_token(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def _normalize_token(token: str) -> str:
    cleaned = token
    for sym in _CURRENCY_AND_SYMBOLS:
        cleaned = cleaned.replace(sym, "")
    return cleaned.strip("()[]{}'\" \t\r\n").lower()


def filter_numeric_tokens(primary_text: str, checker_text: str) -> str:
    """Keep numeric tokens only when primary recognizer and checker agree.

    Drop numbers that only one engine produced. Non-numeric text is kept.
    If dropping disagreed numbers leaves no text, raises NumericDisagreedError.
    (Requirement 5.9).
    """
    primary_words = primary_text.split()
    checker_words = checker_text.split()

    checker_normalized_numbers = {
        _normalize_token(w) for w in checker_words if _is_numeric_token(w)
    }

    kept: list[str] = []
    for word in primary_words:
        if _is_numeric_token(word):
            norm = _normalize_token(word)
            if norm in checker_normalized_numbers:
                kept.append(word)
            # otherwise dropped!
        else:
            kept.append(word)

    result = " ".join(kept).strip()
    if not result:
        raise NumericDisagreedError(
            f"numeric token disagreement between primary ({primary_text!r}) and "
            f"checker ({checker_text!r}) left no text"
        )
    return result


class LocalOcrDescriber:
    """Local OCR using InfiniFlow det+rec with Tesseract 5 digit verification.

    Conforms to VisionDescriber protocol. Never imports httpx, never opens a
    network socket.
    """

    def __init__(
        self,
        *,
        det_model: Path | Any = None,
        rec_model: Path | Any = None,
        checker: Path | str | Any = None,
        recognizer_id: str = RECOGNIZER_ID,
    ) -> None:
        self.recognizer_id = recognizer_id

        # Detector: loads det.onnx on CPU
        if det_model is None:
            default_det = Path(".npu_rag/models/ocr/det.onnx")
            if not default_det.is_file():
                raise VisionUnavailable(
                    "detector model absent: det.onnx", missing_capability="det.onnx"
                )
            self._det: Any = self._init_ort_session(default_det)
        elif isinstance(det_model, Path):
            if not det_model.is_file():
                raise VisionUnavailable(
                    f"detector model absent: {det_model.name}",
                    missing_capability=det_model.name,
                )
            self._det = self._init_ort_session(det_model)
        else:
            self._det = det_model

        # Recognizer: loads rec.onnx on CPU
        if rec_model is None:
            default_rec = Path(".npu_rag/models/ocr/rec.onnx")
            if not default_rec.is_file():
                raise VisionUnavailable(
                    "recognition model absent: rec.onnx", missing_capability="rec.onnx"
                )
            self._rec: Any = self._init_ort_session(default_rec)
        elif isinstance(rec_model, Path):
            if not rec_model.is_file():
                raise VisionUnavailable(
                    f"recognition model absent: {rec_model.name}",
                    missing_capability=rec_model.name,
                )
            self._rec = self._init_ort_session(rec_model)
        else:
            self._rec = rec_model

        # Checker: Tesseract 5 LSTM
        if checker is None:
            tess = shutil.which("tesseract")
            if not tess:
                raise VisionUnavailable(
                    "tesseract 5 absent", missing_capability="tesseract_5"
                )
            self._checker: Any = tess
        elif isinstance(checker, Path):
            if not checker.is_file():
                raise VisionUnavailable(
                    f"tesseract binary absent: {checker.name}",
                    missing_capability="tesseract_5",
                )
            self._checker = str(checker)
        else:
            self._checker = checker

    @staticmethod
    def _init_ort_session(path: Path) -> Any:
        import onnxruntime as ort  # type: ignore[import-untyped]

        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def describe(self, image: ImageRef) -> str:
        """Transcribe text locally from image and verify digits."""
        from io import BytesIO
        from PIL import Image

        try:
            pil_image = Image.open(BytesIO(image.data)).convert("RGB")
        except Exception as exc:
            raise VisionError(f"could not decode image: {exc}", stage="vision") from exc

        # 1. Detect text crops
        crops: list[Any]
        if callable(self._det):
            crops = self._det(pil_image)
        else:
            crops = [pil_image]

        if not crops:
            crops = [pil_image]

        texts: list[str] = []
        for crop in crops:
            # 2. Primary recognition
            if callable(self._rec):
                primary_text = self._rec(crop)
            else:
                primary_text = ""

            # 3. Checker recognition
            if callable(self._checker):
                checker_text = self._checker(crop)
            else:
                checker_text = self._run_tesseract(crop)

            # 4. Filter numeric tokens
            filtered = filter_numeric_tokens(primary_text, checker_text)
            if filtered:
                texts.append(filtered)

        full = "\n".join(texts).strip()
        if not full:
            raise NumericDisagreedError("no text remained after OCR numeric agreement")
        return full

    def _run_tesseract(self, crop: Any) -> str:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                crop.save(tmp_path)
                cmd = [
                    str(self._checker),
                    str(tmp_path),
                    "stdout",
                    "--oem",
                    "1",
                    "-l",
                    "eng",
                    "--psm",
                    "6",
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return proc.stdout.strip()
            except Exception as exc:
                raise VisionError(f"tesseract execution failed: {exc}", stage="vision") from exc
            finally:
                tmp_path.unlink(missing_ok=True)


def format_chart_ranges(ranges: ChartRanges) -> str:
    """Format native Excel chart series ranges into figure text."""
    parts: list[str] = []
    if ranges.value_ranges:
        parts.append(f"values: {', '.join(ranges.value_ranges)}")
    if ranges.category_ranges:
        parts.append(f"categories: {', '.join(ranges.category_ranges)}")
    return "; ".join(parts)


def _recognizer_id(describer: VisionDescriber | None, config: IngestConfig) -> str:
    if describer is not None:
        rec_id = getattr(describer, "recognizer_id", None) or getattr(describer, "model_id", None)
        if rec_id:
            return str(rec_id)
    return config.vision_model or RECOGNIZER_ID


def resolve(
    ref: ImageRef,
    *,
    store: StateStore,
    describer: VisionDescriber | None,
    config: IngestConfig,
    source_path: Path | None = None,
) -> Resolved:
    """Resolve a single ImageRef in strict gate order.

    Order: threshold -> chart_ranges -> cache -> models present -> describe -> store.
    """
    path = source_path if source_path is not None else Path("")
    rec_id = _recognizer_id(describer, config)

    # Gate 1: Threshold (requirement 5.3)
    shorter_side = min(ref.width, ref.height)
    if shorter_side < config.min_image_pixels:
        return Resolved(
            figure=None,
            omission=Omission(
                category=OmissionCategory.BELOW_THRESHOLD,
                path=path,
                reason=(
                    f"image dimensions ({ref.width}x{ref.height}) shorter side "
                    f"{shorter_side}px is below minimum {config.min_image_pixels}px"
                ),
            ),
        )

    # Gate 2: Native chart ranges short-circuit (requirement 5.10)
    if ref.chart_ranges is not None:
        text = format_chart_ranges(ref.chart_ranges)
        figure = FigureSegment(
            text=text,
            locator=ref.locator,
            provenance=Provenance(
                vision_model=rec_id,
                prompt_version=PIPELINE_VERSION,
            ),
        )
        return Resolved(figure=figure, omission=None)

    # Gate 3: Cache lookup (requirements 5.4, 5.5)
    image_hash = hashlib.sha256(ref.data).hexdigest()
    cached = store.cached_description(
        image_sha256=image_hash,
        model_id=rec_id,
        prompt_version=PIPELINE_VERSION,
    )
    if cached is not None:
        figure = FigureSegment(
            text=cached,
            locator=ref.locator,
            provenance=Provenance(
                vision_model=rec_id,
                prompt_version=PIPELINE_VERSION,
            ),
        )
        return Resolved(figure=figure, omission=None)

    # Gate 4: Models present / describer configured (requirements 5.2, 9.4)
    if describer is None:
        return Resolved(
            figure=None,
            omission=Omission(
                category=OmissionCategory.VISION_UNAVAILABLE,
                path=path,
                reason="no vision describer configured or OCR models not provided",
                missing_capability="ocr_models",
            ),
        )

    # Gate 5: Describe (one image per call; requirements 5.1, 5.6, 5.9)
    try:
        text = describer.describe(ref)
    except VisionUnavailable as exc:
        missing_cap = (
            getattr(exc, "missing_capability", None)
            or exc.message
            or "ocr_models"
        )
        return Resolved(
            figure=None,
            omission=Omission(
                category=OmissionCategory.VISION_UNAVAILABLE,
                path=path,
                reason=exc.message,
                missing_capability=missing_cap,
            ),
        )
    except NumericDisagreedError as exc:
        return Resolved(
            figure=None,
            omission=Omission(
                category=OmissionCategory.NUMERIC_DISAGREED,
                path=path,
                reason=exc.message,
            ),
        )
    except VisionError as exc:
        return Resolved(
            figure=None,
            omission=Omission(
                category=OmissionCategory.VISION_FAILED,
                path=path,
                reason=exc.message,
            ),
        )
    except Exception as exc:
        return Resolved(
            figure=None,
            omission=Omission(
                category=OmissionCategory.VISION_FAILED,
                path=path,
                reason=str(exc),
            ),
        )

    # Gate 6: Cache store (requirement 5.4)
    store.store_description(
        image_sha256=image_hash,
        model_id=rec_id,
        prompt_version=PIPELINE_VERSION,
        text=text,
    )
    figure = FigureSegment(
        text=text,
        locator=ref.locator,
        provenance=Provenance(
            vision_model=rec_id,
            prompt_version=PIPELINE_VERSION,
        ),
    )
    return Resolved(figure=figure, omission=None)


def resolve_all(
    refs: Sequence[ImageRef],
    *,
    store: StateStore,
    describer: VisionDescriber | None,
    config: IngestConfig,
    source_path: Path | None = None,
) -> tuple[Resolved, ...]:
    """Resolve a sequence of ImageRefs preserving input order.

    Cache misses that require description are executed with bounded concurrency
    via a thread pool whose maximum worker count is config.vision_concurrency.
    """
    if not refs:
        return ()

    path = source_path if source_path is not None else Path("")
    rec_id = _recognizer_id(describer, config)
    results: list[Resolved | None] = [None] * len(refs)

    # Track indices that need calling describer.describe()
    miss_indices: list[int] = []

    for idx, ref in enumerate(refs):
        # Gate 1: Threshold
        shorter_side = min(ref.width, ref.height)
        if shorter_side < config.min_image_pixels:
            results[idx] = Resolved(
                figure=None,
                omission=Omission(
                    category=OmissionCategory.BELOW_THRESHOLD,
                    path=path,
                    reason=(
                        f"image dimensions ({ref.width}x{ref.height}) shorter side "
                        f"{shorter_side}px is below minimum {config.min_image_pixels}px"
                    ),
                ),
            )
            continue

        # Gate 2: Chart ranges
        if ref.chart_ranges is not None:
            text = format_chart_ranges(ref.chart_ranges)
            results[idx] = Resolved(
                figure=FigureSegment(
                    text=text,
                    locator=ref.locator,
                    provenance=Provenance(
                        vision_model=rec_id,
                        prompt_version=PIPELINE_VERSION,
                    ),
                ),
                omission=None,
            )
            continue

        # Gate 3: Cache lookup
        image_hash = hashlib.sha256(ref.data).hexdigest()
        cached = store.cached_description(
            image_sha256=image_hash,
            model_id=rec_id,
            prompt_version=PIPELINE_VERSION,
        )
        if cached is not None:
            results[idx] = Resolved(
                figure=FigureSegment(
                    text=cached,
                    locator=ref.locator,
                    provenance=Provenance(
                        vision_model=rec_id,
                        prompt_version=PIPELINE_VERSION,
                    ),
                ),
                omission=None,
            )
            continue

        # Gate 4: Models present / describer configured
        if describer is None:
            results[idx] = Resolved(
                figure=None,
                omission=Omission(
                    category=OmissionCategory.VISION_UNAVAILABLE,
                    path=path,
                    reason="no vision describer configured or OCR models not provided",
                    missing_capability="ocr_models",
                ),
            )
            continue

        miss_indices.append(idx)

    # If no misses require execution, return immediately
    if not miss_indices:
        return tuple(res for res in results if res is not None)

    # Process misses with bounded concurrency
    def _execute_miss(idx: int) -> tuple[int, Resolved]:
        ref = refs[idx]
        resolved = resolve(
            ref,
            store=store,
            describer=describer,
            config=config,
            source_path=source_path,
        )
        return idx, resolved

    max_workers = max(1, config.vision_concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_execute_miss, idx) for idx in miss_indices]
        for future in concurrent.futures.as_completed(futures):
            idx, res = future.result()
            results[idx] = res

    return tuple(res for res in results if res is not None)
