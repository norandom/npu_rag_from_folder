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
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Protocol

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
    "NumericDisagreedError",
    "VisionDescriber",
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
