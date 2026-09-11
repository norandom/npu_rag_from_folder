"""Unit tests for ImageRef resolution (task 5.1).

Order of resolution per ImageRef:
1. threshold (min_image_pixels on shorter side -> BELOW_THRESHOLD)
2. chart_ranges present -> FigureSegment from ranges, no OCR call
3. cache hit -> FigureSegment from store, no OCR call
4. models present / describer not None -> if absent, VISION_UNAVAILABLE
5. describe -> calls describer.describe(ref)
6. cache store -> retained in vision_cache on success

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.8, 5.10, 9.4
Boundary: npu_rag.ingest.vision
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Sequence

import pytest

from npu_rag.ingest.config import IngestConfig
from npu_rag.ingest.errors import VisionError, VisionUnavailable
from npu_rag.ingest.state import StateStore
from npu_rag.ingest.types import (
    ChartRanges,
    FigureSegment,
    ImageFileLocator,
    ImageRef,
    OmissionCategory,
    PageLocator,
    Resolved,
)
from npu_rag.ingest.vision import (
    PIPELINE_VERSION,
    LocalOcrDescriber,
    NumericDisagreedError,
    VisionDescriber,
    filter_numeric_tokens,
    resolve,
    resolve_all,
)

from io import BytesIO
from PIL import Image

ROOT = Path("test_root")
BUDGET = 256

_buf = BytesIO()
Image.new("RGB", (16, 16), color="white").save(_buf, format="PNG")
VALID_PNG = _buf.getvalue()


def make_config(**overrides: object) -> IngestConfig:
    values: dict[str, object] = {
        "roots": (ROOT,),
        "token_budget": BUDGET,
        "min_image_pixels": 200,
        "vision_concurrency": 2,
    }
    values.update(overrides)
    return IngestConfig(**values)  # type: ignore[arg-type]


def make_ref(
    *,
    data: bytes = VALID_PNG,
    width: int = 300,
    height: int = 300,
    locator: object = None,
    chart_ranges: ChartRanges | None = None,
) -> ImageRef:
    if locator is None:
        locator = PageLocator(page=1)
    return ImageRef(
        data=data,
        mime="image/png",
        width=width,
        height=height,
        locator=locator,  # type: ignore[arg-type]
        chart_ranges=chart_ranges,
    )


class FakeDescriber:
    def __init__(
        self,
        *,
        text: str = "transcribed text",
        delay_s: float = 0.0,
        recognizer_id: str = "fake-rec-v1",
        raise_exc: Exception | None = None,
    ) -> None:
        self.text = text
        self.delay_s = delay_s
        self.recognizer_id = recognizer_id
        self.raise_exc = raise_exc
        self.call_count = 0
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        self.received_refs: list[ImageRef] = []

    def describe(self, image: ImageRef) -> str:
        with self._lock:
            self.call_count += 1
            self._in_flight += 1
            if self._in_flight > self.max_in_flight:
                self.max_in_flight = self._in_flight
            self.received_refs.append(image)

        try:
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            if self.raise_exc is not None:
                raise self.raise_exc
            return self.text
        finally:
            with self._lock:
                self._in_flight -= 1


def test_pipeline_version_is_defined_and_non_empty() -> None:
    assert isinstance(PIPELINE_VERSION, str)
    assert len(PIPELINE_VERSION) > 0


def test_below_threshold_image_never_reaches_describer(tmp_path: Path) -> None:
    """Requirement 5.3: Shorter side below min_image_pixels produces BELOW_THRESHOLD."""
    config = make_config(min_image_pixels=200)
    store = StateStore(tmp_path / "state.sqlite")
    fake = FakeDescriber()

    ref = make_ref(width=150, height=300)  # min side 150 < 200
    res = resolve(ref, store=store, describer=fake, config=config, source_path=Path("doc.pdf"))

    assert res.figure is None
    assert res.omission is not None
    assert res.omission.category is OmissionCategory.BELOW_THRESHOLD
    assert res.omission.path == Path("doc.pdf")
    assert fake.call_count == 0


def test_chart_ranges_short_circuit_never_reaches_describer(tmp_path: Path) -> None:
    """Requirement 5.10: chart_ranges emits FigureSegment directly without OCR."""
    config = make_config(min_image_pixels=100)
    store = StateStore(tmp_path / "state.sqlite")
    fake = FakeDescriber()

    ranges = ChartRanges(
        value_ranges=("Sheet1!$B$2:$B$10",),
        category_ranges=("Sheet1!$A$2:$A$10",),
    )
    ref = make_ref(width=300, height=300, chart_ranges=ranges)
    res = resolve(ref, store=store, describer=fake, config=config, source_path=Path("sheet.xlsx"))

    assert res.omission is None
    assert res.figure is not None
    assert isinstance(res.figure, FigureSegment)
    assert "Sheet1!$B$2:$B$10" in res.figure.text
    assert "Sheet1!$A$2:$A$10" in res.figure.text
    assert res.figure.provenance.prompt_version == PIPELINE_VERSION
    assert fake.call_count == 0


def test_cache_hit_returns_stored_description_and_never_calls_describer(tmp_path: Path) -> None:
    """Requirements 5.4, 5.5: cache hit reuses stored description without OCR."""
    config = make_config()
    store = StateStore(tmp_path / "state.sqlite")
    ref = make_ref(data=b"unique_image_bytes", width=400, height=400)
    image_hash = hashlib.sha256(b"unique_image_bytes").hexdigest()

    fake = FakeDescriber(text="First run OCR", recognizer_id="rec-1")
    store.store_description(
        image_sha256=image_hash,
        model_id="rec-1",
        prompt_version=PIPELINE_VERSION,
        text="Cached OCR result",
    )

    res = resolve(ref, store=store, describer=fake, config=config)
    assert res.omission is None
    assert res.figure is not None
    assert res.figure.text == "Cached OCR result"
    assert res.figure.provenance.vision_model == "rec-1"
    assert res.figure.provenance.prompt_version == PIPELINE_VERSION
    assert fake.call_count == 0


def test_null_describer_never_raises_and_yields_vision_unavailable(tmp_path: Path) -> None:
    """Requirements 5.2, 9.4: describer is None returns VISION_UNAVAILABLE naming capability."""
    config = make_config()
    store = StateStore(tmp_path / "state.sqlite")
    ref = make_ref(width=400, height=400)

    res = resolve(ref, store=store, describer=None, config=config, source_path=Path("scans/page.png"))
    assert res.figure is None
    assert res.omission is not None
    assert res.omission.category is OmissionCategory.VISION_UNAVAILABLE
    assert res.omission.missing_capability is not None
    assert len(res.omission.missing_capability.strip()) > 0


def test_absent_models_describer_yields_vision_unavailable(tmp_path: Path) -> None:
    """Requirement 5.2: describer raising VisionUnavailable names the missing capability."""
    config = make_config()
    store = StateStore(tmp_path / "state.sqlite")
    fake = FakeDescriber(
        raise_exc=VisionUnavailable("det.onnx missing", missing_capability="det.onnx")
    )
    ref = make_ref(width=400, height=400)

    res = resolve(ref, store=store, describer=fake, config=config, source_path=Path("page.pdf"))
    assert res.figure is None
    assert res.omission is not None
    assert res.omission.category is OmissionCategory.VISION_UNAVAILABLE
    assert res.omission.missing_capability == "det.onnx"


def test_describer_success_stores_in_cache_and_produces_figure(tmp_path: Path) -> None:
    """Requirements 5.1, 5.4: Successful describe returns FigureSegment and stores in cache."""
    config = make_config()
    store = StateStore(tmp_path / "state.sqlite")
    fake = FakeDescriber(text="Detected Chart Content 42", recognizer_id="infiniflow-rec")
    ref = make_ref(data=b"chart_image_data", width=500, height=500)
    image_hash = hashlib.sha256(b"chart_image_data").hexdigest()

    res = resolve(ref, store=store, describer=fake, config=config)
    assert res.omission is None
    assert res.figure is not None
    assert res.figure.text == "Detected Chart Content 42"
    assert res.figure.provenance.vision_model == "infiniflow-rec"
    assert res.figure.provenance.prompt_version == PIPELINE_VERSION
    assert fake.call_count == 1

    # Verify cached in store
    cached = store.cached_description(image_hash, "infiniflow-rec", PIPELINE_VERSION)
    assert cached == "Detected Chart Content 42"

    # Second call hits cache
    res2 = resolve(ref, store=store, describer=fake, config=config)
    assert res2.figure is not None
    assert res2.figure.text == "Detected Chart Content 42"
    assert fake.call_count == 1


def test_describer_numeric_disagreement_yields_numeric_disagreed(tmp_path: Path) -> None:
    """Requirement 5.9: when digit agreement drops all text, omission is NUMERIC_DISAGREED."""
    config = make_config()
    store = StateStore(tmp_path / "state.sqlite")
    fake = FakeDescriber(raise_exc=NumericDisagreedError("digits disagreed"))
    ref = make_ref(width=300, height=300)

    res = resolve(ref, store=store, describer=fake, config=config)
    assert res.figure is None
    assert res.omission is not None
    assert res.omission.category is OmissionCategory.NUMERIC_DISAGREED
    assert res.omission.missing_capability is None


def test_describer_general_error_yields_vision_failed(tmp_path: Path) -> None:
    """Requirement 5.6: VisionError becomes VISION_FAILED and does not crash."""
    config = make_config()
    store = StateStore(tmp_path / "state.sqlite")
    fake = FakeDescriber(raise_exc=VisionError("inference error", stage="vision"))
    ref = make_ref(width=300, height=300)

    res = resolve(ref, store=store, describer=fake, config=config)
    assert res.figure is None
    assert res.omission is not None
    assert res.omission.category is OmissionCategory.VISION_FAILED


def test_resolve_all_bounded_concurrency_and_order_preservation(tmp_path: Path) -> None:
    """Task 5.1: Concurrency is bounded and input order is preserved."""
    concurrency = 2
    config = make_config(vision_concurrency=concurrency, min_image_pixels=100)
    store = StateStore(tmp_path / "state.sqlite")
    fake = FakeDescriber(delay_s=0.05, text="delayed text")

    # 1. Below threshold
    ref_small = make_ref(data=b"img0", width=50, height=50)
    # 2. Chart ranges
    ref_chart = make_ref(
        data=b"img1",
        width=200,
        height=200,
        chart_ranges=ChartRanges(value_ranges=("Sheet!$A$1:$A$5",)),
    )
    # 3. Cache hit
    ref_cached = make_ref(data=b"img2_cached", width=200, height=200)
    store.store_description(
        hashlib.sha256(b"img2_cached").hexdigest(),
        "fake-rec-v1",
        PIPELINE_VERSION,
        "pre-cached text",
    )
    # 4-7. Cache misses that should run concurrently
    ref_miss1 = make_ref(data=b"miss1", width=200, height=200)
    ref_miss2 = make_ref(data=b"miss2", width=200, height=200)
    ref_miss3 = make_ref(data=b"miss3", width=200, height=200)
    ref_miss4 = make_ref(data=b"miss4", width=200, height=200)

    refs = [ref_small, ref_chart, ref_cached, ref_miss1, ref_miss2, ref_miss3, ref_miss4]
    results = resolve_all(refs, store=store, describer=fake, config=config)

    assert len(results) == len(refs)
    # Check 1: below threshold
    assert results[0].omission is not None
    assert results[0].omission.category is OmissionCategory.BELOW_THRESHOLD
    # Check 2: chart ranges
    fig1 = results[1].figure
    assert fig1 is not None
    assert "Sheet!$A$1:$A$5" in fig1.text
    # Check 3: cache hit
    fig2 = results[2].figure
    assert fig2 is not None
    assert fig2.text == "pre-cached text"
    # Check 4-7: misses described
    for i in range(3, 7):
        fig = results[i].figure
        assert fig is not None
        assert fig.text == "delayed text"

    # Only the 4 misses should have reached the describer
    assert fake.call_count == 4
    # Max in-flight count should not exceed concurrency limit
    assert fake.max_in_flight <= concurrency


# --------------------------------------------------------------------------
# Task 5.2: Local OCR describer and second vote on digits
# --------------------------------------------------------------------------


def test_filter_numeric_tokens_keeps_agreed_number() -> None:
    """Requirement 5.9: Primary reads 12.4 and checker reads 12.4 -> keeps 12.4."""
    result = filter_numeric_tokens("12.4", "12.4")
    assert result == "12.4"


def test_filter_numeric_tokens_drops_disagreed_number_and_raises() -> None:
    """Requirement 5.9: Primary reads 12.4 and checker reads 12.9 -> drops 12.4;
    emptying leaves no text, so it raises NumericDisagreedError."""
    with pytest.raises(NumericDisagreedError):
        filter_numeric_tokens("12.4", "12.9")


def test_filter_numeric_tokens_keeps_prose_when_number_disagreed() -> None:
    """Requirement 5.9: Disagreed number is dropped, but prose words remain."""
    result = filter_numeric_tokens("Q1 Revenue 12.4", "Q1 Revenue 12.9")
    assert result == "Q1 Revenue"


def test_absent_det_model_names_missing_capability() -> None:
    """Requirements 5.2, 10.5: Absent det.onnx raises VisionUnavailable naming det.onnx."""
    with pytest.raises(VisionUnavailable) as exc_info:
        LocalOcrDescriber(det_model=Path("nonexistent/det.onnx"))
    assert "det.onnx" in (exc_info.value.missing_capability or "")


def _dummy_engine(img: object) -> list[object]:
    return []


def test_absent_rec_model_names_missing_capability() -> None:
    """Requirements 5.2, 10.5: Absent rec.onnx raises VisionUnavailable naming rec.onnx."""
    with pytest.raises(VisionUnavailable) as exc_info:
        LocalOcrDescriber(det_model=_dummy_engine, rec_model=Path("nonexistent/rec.onnx"))
    assert "rec.onnx" in (exc_info.value.missing_capability or "")


def test_absent_checker_names_missing_capability() -> None:
    """Requirements 5.2, 10.5: Absent tesseract raises VisionUnavailable naming tesseract_5."""
    with pytest.raises(VisionUnavailable) as exc_info:
        LocalOcrDescriber(
            det_model=_dummy_engine,
            rec_model=_dummy_engine,
            checker=Path("nonexistent/tesseract.exe"),
        )
    assert "tesseract" in (exc_info.value.missing_capability or "").lower()


def test_local_ocr_describer_agreed_digits_emits_text() -> None:
    """Requirement 5.9: Both engines agree on 12.4 -> emits 12.4."""
    describer = LocalOcrDescriber(
        det_model=lambda img: [img],  # 1 crop
        rec_model=lambda crop: "12.4",
        checker=lambda crop: "12.4",
    )
    ref = make_ref(width=256, height=256)
    text = describer.describe(ref)
    assert text == "12.4"


def test_local_ocr_describer_disagreed_digits_raises_numeric_disagreed() -> None:
    """Requirement 5.9: Primary reads 12.4 and checker reads 12.9 -> raises NumericDisagreedError."""
    describer = LocalOcrDescriber(
        det_model=lambda img: [img],  # 1 crop
        rec_model=lambda crop: "12.4",
        checker=lambda crop: "12.9",
    )
    ref = make_ref(width=256, height=256)
    with pytest.raises(NumericDisagreedError):
        describer.describe(ref)


def test_local_ocr_describer_never_issues_http_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirements 10.2, 10.3: Never opens a socket or makes network requests."""
    import socket

    def _fail_socket(*args: object, **kwargs: object) -> None:
        pytest.fail("Network socket opened during local OCR!")

    monkeypatch.setattr(socket, "socket", _fail_socket)

    describer = LocalOcrDescriber(
        det_model=lambda img: [img],
        rec_model=lambda crop: "local text 100",
        checker=lambda crop: "local text 100",
    )
    ref = make_ref(width=256, height=256)
    assert describer.describe(ref) == "local text 100"

