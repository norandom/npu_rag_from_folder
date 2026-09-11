"""Run sustained inference load on the AMD Ryzen AI NPU for 60 seconds (or custom duration).

Demonstrates and verifies active NPU utilization in Windows Task Manager by:
1. Ensuring an NPU-compiled model artifact exists (using all-MiniLM-L6-v2 on VitisAIExecutionProvider).
2. Initializing a Vitis AI NPU session on AMD XDNA2 silicon.
3. Executing a tight inference loop for 60 seconds.
4. Concurrently polling xrt-smi telemetry (power in Watts, AIE column occupancy).
5. Streaming live progress and throughput metrics to stdout.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

from npu_rag.embedding.environment.capability import check_capability
from npu_rag.embedding.environment.xrt import XrtSmiWrapper
from npu_rag.embedding.models.artifacts import OrtContextCompiler, ensure_prepared
from npu_rag.embedding.profiles import ModelProfile
from npu_rag.embedding.providers.vitisai import (
    MINIMUM_PARTITION_SHARE,
    VITISAI_PROVIDER,
    VitisAIBackend,
)
from npu_rag.embedding.reporting import ProgressUpdate
from npu_rag.embedding.types import ProviderChoice

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTROL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONTROL_SEQ_LEN = 128

CONTROL_PROFILE = ModelProfile(
    model_id=CONTROL_MODEL,
    dimension=384,
    compiled_seq_len=CONTROL_SEQ_LEN,
    architectural_context_limit=256,
    batch_size=1,
    pooling="mean",
    has_dense_stage=False,
    document_template="{content}",
    query_template="{content}",
    license_gated=False,
)


def _progress_listener(update: ProgressUpdate) -> None:
    pct = (update.completed / update.total * 100.0) if update.total > 0 else 0.0
    print(
        f"[{update.completed}/{update.total}] {update.operation} ({pct:.0f}%)"
    )


def run_load(duration_seconds: float = 60.0, artifacts_dir: Path | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    artifacts_root = artifacts_dir or (REPO_ROOT / "artifacts")
    artifacts_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AMD RYZEN AI NPU SUSTAINED LOAD GENERATOR")
    print("=" * 70)

    # 1. Capability check
    print("\n[Step 1/3] Checking NPU capability...")
    cap = check_capability()
    print(f"  Device: {cap.device_name or 'Unknown'}")
    print(f"  Driver: {cap.driver_version or 'Unknown'}")
    print(f"  Runtime: {cap.runtime_version or 'Unknown'}")
    print(f"  Execution mode: {cap.execution_mode.value}")

    if cap.execution_mode.value not in ("in_process", "supported"):
        print(
            f"\nWARNING: Capability check reports {cap.execution_mode.value}. "
            "Attempting to proceed with VitisAIExecutionProvider...",
            file=sys.stderr,
        )

    # 2. Artifact preparation / loading
    print(f"\n[Step 2/3] Preparing / loading NPU model ({CONTROL_MODEL} @ seq {CONTROL_SEQ_LEN})...")
    compiler = OrtContextCompiler()
    prepared = ensure_prepared(
        CONTROL_PROFILE,
        ProviderChoice.NPU,
        artifacts_root,
        compiler=compiler,
        progress=_progress_listener,
    )
    print(f"  Artifact ready at: {prepared.directory}")
    print(f"  Compiled context: {prepared.context_path.name if prepared.context_path else 'None'}")

    # 3. Create NPU backend session
    print("\n[Step 3/3] Initializing VitisAIExecutionProvider session on NPU silicon...")
    backend = VitisAIBackend(prepared, CONTROL_PROFILE, ProviderChoice.NPU)
    print(f"  Provider confirmed: {backend.provider.value}")
    if backend.npu_partition_share is not None:
        print(f"  NPU partition share: {backend.npu_partition_share * 100:.1f}%")

    # Prepare static dummy input
    row_ids = np.zeros(CONTROL_SEQ_LEN, dtype=np.int64)
    row_ids[:8] = np.array([101, 2023, 2003, 1037, 3231, 6251, 1012, 102], np.int64)
    input_ids = row_ids.reshape(1, CONTROL_SEQ_LEN)

    row_mask = np.zeros(CONTROL_SEQ_LEN, dtype=np.int64)
    row_mask[:8] = 1
    attention_mask = row_mask.reshape(1, CONTROL_SEQ_LEN)

    # Warm-up inference
    print("\nExecuting warm-up forward pass on NPU...")
    out = backend.run(input_ids, attention_mask)
    print(f"  Warm-up completed. Output tensor shape: {out.shape}, dtype: {out.dtype}")

    # Telemetry monitor setup
    wrapper = XrtSmiWrapper()
    latest_telemetry: dict[str, str] = {
        "power": "sampling...",
        "partitions": "checking...",
    }
    stop_event = threading.Event()

    def telemetry_poller() -> None:
        while not stop_event.is_set():
            if wrapper.available:
                try:
                    p = wrapper.read_platform()
                    if p.power.watts is not None:
                        latest_telemetry["power"] = f"{p.power.watts:.3f} W"
                    elif p.power.reason:
                        latest_telemetry["power"] = p.power.reason
                except Exception:
                    pass

                try:
                    occ = wrapper.read_partitions()
                    if occ.context_live:
                        cols = ", ".join(f"P{p.index}[{p.columns}]" for p in occ.partitions)
                        latest_telemetry["partitions"] = f"LIVE ({cols})"
                    elif occ.context_live is False:
                        latest_telemetry["partitions"] = "idle"
                    elif occ.unavailable_reason:
                        latest_telemetry["partitions"] = occ.unavailable_reason
                except Exception:
                    pass
            stop_event.wait(1.0)

    poller_thread = threading.Thread(target=telemetry_poller, daemon=True)
    poller_thread.start()

    print("\n" + "=" * 70)
    print(f"STARTING {duration_seconds:.0f}-SECOND SUSTAINED LOAD ON NPU")
    print(">>> Open Windows Task Manager -> Performance -> NPU to observe load! <<<")
    print("=" * 70 + "\n")

    start_time = time.perf_counter()
    deadline = start_time + duration_seconds
    next_report = start_time + 5.0

    inferences = 0
    recent_inferences = 0
    recent_time = start_time

    try:
        while time.perf_counter() < deadline:
            backend.run(input_ids, attention_mask)
            inferences += 1
            recent_inferences += 1

            now = time.perf_counter()
            if now >= next_report:
                elapsed = now - start_time
                recent_interval = now - recent_time
                current_rate = recent_inferences / recent_interval if recent_interval > 0 else 0.0
                recent_inferences = 0
                recent_time = now
                next_report = now + 5.0

                power_str = latest_telemetry["power"]
                part_str = latest_telemetry["partitions"]
                print(
                    f"[{elapsed:4.1f}s / {duration_seconds:4.1f}s] "
                    f"Inferences: {inferences:5d} ({current_rate:5.1f} inf/s) | "
                    f"NPU Power: {power_str:9s} | "
                    f"AIE Context: {part_str}"
                )
    finally:
        stop_event.set()
        poller_thread.join(timeout=2.0)

    total_time = time.perf_counter() - start_time
    avg_rate = inferences / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 70)
    print("NPU SUSTAINED LOAD RUN COMPLETE")
    print("=" * 70)
    print(f"  Duration:          {total_time:.2f} seconds")
    print(f"  Total Inferences:  {inferences:,}")
    print(f"  Average Rate:      {avg_rate:.1f} inferences/second")
    print(f"  Hardware Provider: {VITISAI_PROVIDER}")
    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sustained inference load on the AMD Ryzen AI NPU."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration of sustained load in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Directory to store compiled NPU model artifacts (default: %(default)s)",
    )
    args = parser.parse_args()
    run_load(duration_seconds=args.duration, artifacts_dir=args.artifacts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
