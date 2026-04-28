#!/usr/bin/env python3
# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: GPL-3.0-only
#
"""B3b multi-dispatch benchmark: linear_projection per-group vs fused-perts.

Simulates the encoder's realistic load pattern — many chunks, one
shared weight matrix, one silicon-lock acquire across the entire
batch. Each chunk produces a ``(L=334, N=1, HIDDEN=96)`` FP32 tensor
that the linear_projection op turns into ``(L=334, N=1, OUT=256)``.

Usage::

    python3 benchmarks/basecalling/bench_linear_projection_block.py \\
        --chunks 8 --variants per_group,fused_perts \\
        --output-dir results/basecalling

The default ``--chunks 64`` is a representative mid-pod5 load. The
B3b validation gate uses ``--chunks 8`` (cheap on silicon, large
enough to exercise multi-dispatch).

Notes
-----

* The benchmark wraps the entire batch in ONE
  :func:`bionpu.dispatch.npu_silicon_lock.npu_silicon_lock` context.
  NPU resources serialise device-side; multiple lock acquires per
  benchmark would waste both wall time and dmesg-tail noise.

* Per-chunk wall time is captured both as Python perf-counter wall
  (subprocess fork + XRT setup + kernel + readback) AND as the
  kernel's self-reported ``avg_us`` from ``op.last_run`` (NPU-internal
  timing only). The aggregate ``total_wall_seconds`` is what
  matters for throughput-vs-baseline comparisons.

* HALT / no-retry on NaN/Inf or all-zero output per CLAUDE.md
  silicon-wedge etiquette.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# These constants must match the kernel package geometry.
T_LSTM = 334
HIDDEN = 96
OUT_DIM = 256
CLAMP_LO = -5.0
CLAMP_HI = 5.0

DEFAULT_CHUNKS = 64
DEFAULT_SEED = 20260428
DEFAULT_VARIANTS: tuple[str, ...] = ("per_group", "fused_perts")
BF16_TOLERANCE_GATE = 5e-2  # matches DESIGN-fusion.md / B2 validation


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _cpu_fp32_clamped(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """FP32 reference (matches per-group artifact's contract)."""
    x_flat = x.reshape(T_LSTM, HIDDEN)
    y = x_flat @ weight.T
    y = np.clip(y, CLAMP_LO, CLAMP_HI)
    return y.reshape(T_LSTM, 1, OUT_DIM).astype(np.float32)


def _cpu_bf16ref(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """bf16-narrowed reference (matches fused-perts contract)."""
    def _to_bf16(a: np.ndarray) -> np.ndarray:
        u32 = a.astype(np.float32).view(np.uint32)
        rounded = (u32 + 0x8000) & 0xFFFF0000
        return rounded.view(np.float32).astype(np.float32)

    x_bf = _to_bf16(x)
    w_bf = _to_bf16(weight)
    y = x_bf.reshape(T_LSTM, HIDDEN) @ w_bf.T
    y = np.clip(y, CLAMP_LO, CLAMP_HI)
    return y.reshape(T_LSTM, 1, OUT_DIM).astype(np.float32)


def _generate_chunks(n_chunks: int, seed: int):
    """Deterministic chunk batch + a single shared weight matrix.

    Mirrors the encoder usage pattern: the CRF head's weight is fixed
    per model, but each chunk's input changes. Re-seeding with the
    same seed across variants guarantees both variants compute the
    same arithmetic.
    """
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal(size=(OUT_DIM, HIDDEN)) * 0.1).astype(
        np.float32
    )
    chunks = [
        rng.standard_normal(size=(T_LSTM, 1, HIDDEN)).astype(np.float32)
        for _ in range(n_chunks)
    ]
    return weight, chunks


def _validate_chunk_output(
    y: np.ndarray, *, variant: str, chunk_idx: int
) -> None:
    """HALT-on-failure validation per CLAUDE.md silicon-wedge etiquette."""
    if y.shape != (T_LSTM, 1, OUT_DIM):
        raise RuntimeError(
            f"[bench:{variant}:chunk={chunk_idx}] output shape "
            f"{y.shape} != ({T_LSTM}, 1, {OUT_DIM})"
        )
    n_nan = int(np.isnan(y).sum())
    n_inf = int(np.isinf(y).sum())
    if n_nan or n_inf:
        raise RuntimeError(
            f"[bench:{variant}:chunk={chunk_idx}] HALT — output has "
            f"{n_nan} NaN / {n_inf} Inf values. Per CLAUDE.md no retry."
        )
    out_max_abs = float(np.abs(y).max())
    if out_max_abs == 0.0:
        raise RuntimeError(
            f"[bench:{variant}:chunk={chunk_idx}] HALT — output all "
            f"zero. Likely silicon wedge / firmware drop. Per CLAUDE.md "
            f"no retry."
        )


def _run_variant(
    *,
    variant: str,
    op,
    weight: np.ndarray,
    chunks: list,
    fp32_refs: list,
    bf16_refs: list,
) -> dict:
    """Dispatch one variant across all chunks; collect per-chunk + aggregate.

    Caller is responsible for holding ``npu_silicon_lock`` across the
    whole call. ``op.last_run.avg_us`` is read after each per-chunk
    invocation (n_iters=1, warmup=0 — we want per-chunk wall, not a
    microbenchmark loop).
    """
    print(
        f"[bench:{variant}] dispatching {len(chunks)} chunks "
        f"(per-chunk n_iters=1 warmup=0)…"
    )
    per_chunk: list[dict] = []
    outputs: list[np.ndarray] = []
    n_chunks = len(chunks)

    t_total0 = time.perf_counter()
    for idx, chunk in enumerate(chunks):
        t0 = time.perf_counter()
        y = op(x=chunk, weight=weight, n_iters=1, warmup=0)
        wall_sec = time.perf_counter() - t0
        _validate_chunk_output(y, variant=variant, chunk_idx=idx)
        r = op.last_run
        per_chunk.append(
            {
                "chunk_idx": idx,
                "wall_seconds": wall_sec,
                "kernel_avg_us": r.avg_us,
                "kernel_min_us": r.min_us,
                "kernel_max_us": r.max_us,
                "kernel_n_iters": r.n_iters,
                "output_max_abs": float(np.abs(y).max()),
                "max_abs_vs_fp32_ref": float(
                    np.abs(y - fp32_refs[idx]).max()
                ),
                "max_abs_vs_bf16_ref": float(
                    np.abs(y - bf16_refs[idx]).max()
                ),
            }
        )
        outputs.append(y)
        if idx == 0 or (idx + 1) % 16 == 0 or idx + 1 == n_chunks:
            print(
                f"[bench:{variant}]   chunk {idx + 1}/{n_chunks}: "
                f"wall {wall_sec * 1e3:.2f} ms, kernel avg {r.avg_us:.1f} us"
            )
    total_wall_sec = time.perf_counter() - t_total0

    walls = np.array([c["wall_seconds"] for c in per_chunk], dtype=np.float64)
    kavgs = np.array([c["kernel_avg_us"] for c in per_chunk], dtype=np.float64)
    fp32_drifts = np.array(
        [c["max_abs_vs_fp32_ref"] for c in per_chunk], dtype=np.float64
    )
    bf16_drifts = np.array(
        [c["max_abs_vs_bf16_ref"] for c in per_chunk], dtype=np.float64
    )

    aggregate = {
        "n_chunks": n_chunks,
        "total_wall_seconds": total_wall_sec,
        "throughput_chunks_per_sec": (
            n_chunks / total_wall_sec if total_wall_sec > 0 else float("nan")
        ),
        "wall_seconds_mean": float(walls.mean()),
        "wall_seconds_min": float(walls.min()),
        "wall_seconds_max": float(walls.max()),
        "wall_seconds_p50": float(np.percentile(walls, 50)),
        "wall_seconds_p95": float(np.percentile(walls, 95)),
        "kernel_avg_us_mean": float(kavgs.mean()),
        "kernel_avg_us_min": float(kavgs.min()),
        "kernel_avg_us_max": float(kavgs.max()),
        "max_abs_vs_fp32_ref": float(fp32_drifts.max()),
        "max_abs_vs_bf16_ref": float(bf16_drifts.max()),
    }
    return {
        "verdict": "PASS",
        "per_chunk": per_chunk,
        "aggregate": aggregate,
        "outputs": outputs,
    }


def _ops_for_variants(variants: list[str]) -> dict:
    from bionpu.kernels.basecalling.linear_projection import (
        DoradoFastLinearProjection,
        DoradoFastLinearProjectionFusedPerts,
    )

    factory = {
        "per_group": DoradoFastLinearProjection,
        "fused_perts": DoradoFastLinearProjectionFusedPerts,
    }
    out = {}
    for v in variants:
        if v not in factory:
            raise ValueError(
                f"unknown variant {v!r}; expected one of {list(factory)}"
            )
        cls = factory[v]
        if not cls.artifacts_present():
            raise FileNotFoundError(
                f"NPU artifacts missing for variant {v!r} (class "
                f"{cls.__name__}). Build them per the kernel's MANIFEST.md "
                f"or set --variants to skip."
            )
        out[v] = cls()
    return out


def _output_dir(arg_dir: str | None, iso: str) -> Path:
    if arg_dir:
        base = Path(arg_dir)
    else:
        # Default location matches the existing b-m6/b-m6-fused-perts pattern.
        base = Path("results/basecalling")
    out = base / f"b-m7-linear-projection-block-{iso}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "B3b multi-dispatch benchmark: linear_projection per-group "
            "vs fused-perts under encoder-realistic load."
        )
    )
    p.add_argument(
        "--chunks", type=int, default=DEFAULT_CHUNKS,
        help=f"chunks to dispatch per variant (default {DEFAULT_CHUNKS})",
    )
    p.add_argument(
        "--variants", type=str,
        default=",".join(DEFAULT_VARIANTS),
        help=(
            "comma-separated subset of {per_group,fused_perts}; "
            f"default {','.join(DEFAULT_VARIANTS)}"
        ),
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed for chunk + weight generation (default {DEFAULT_SEED})",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "base directory for results JSON (a "
            "b-m7-linear-projection-block-<iso>/ subdir is created)"
        ),
    )
    p.add_argument(
        "--no-silicon", action="store_true",
        help=(
            "skip silicon dispatch (CPU-only smoke; useful when the "
            "device is not available — produces no measurements JSON)"
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    iso = _now_iso()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    n_chunks = int(args.chunks)
    if n_chunks <= 0:
        print(f"[bench] --chunks must be > 0; got {n_chunks}", file=sys.stderr)
        return 2

    print(f"[bench] iso={iso}")
    print(f"[bench] chunks={n_chunks} variants={variants} seed={args.seed}")
    weight, chunks = _generate_chunks(n_chunks=n_chunks, seed=args.seed)

    # Pre-compute CPU references once per chunk — same for both variants.
    print(f"[bench] precomputing CPU fp32 + bf16 references…")
    fp32_refs = [_cpu_fp32_clamped(c, weight) for c in chunks]
    bf16_refs = [_cpu_bf16ref(c, weight) for c in chunks]

    if args.no_silicon:
        print(f"[bench] --no-silicon: skipped silicon dispatch")
        return 0

    out_dir = _output_dir(args.output_dir, iso)
    measurements_path = out_dir / "measurements.json"

    results: dict = {
        "schema_version": "1.0.0",
        "task_id": "B3b",
        "kernel_name": "dorado_fast_linear_projection_block",
        "generated_at_iso8601": iso,
        "host": os.environ.get("HOSTNAME", "unknown"),
        "config": {
            "n_chunks": n_chunks,
            "seed": args.seed,
            "variants": variants,
            "shape": {"L": T_LSTM, "N": 1, "HIDDEN": HIDDEN, "OUT": OUT_DIM},
            "clamp_lo": CLAMP_LO,
            "clamp_hi": CLAMP_HI,
            "bf16_tolerance_gate": BF16_TOLERANCE_GATE,
        },
        "verdict": "pending",
    }

    # Build ops up-front so an artifact-missing failure is surfaced
    # before we acquire the lock (avoids holding /tmp/bionpu-npu-silicon.lock
    # while raising for a missing xclbin).
    try:
        ops = _ops_for_variants(variants)
    except Exception as e:
        results["verdict"] = "FAIL_ARTIFACTS"
        results["error"] = f"{type(e).__name__}: {e}"
        results["traceback"] = traceback.format_exc()
        measurements_path.write_text(json.dumps(results, indent=2))
        print(f"[bench] FAIL_ARTIFACTS: {e}", file=sys.stderr)
        return 1

    from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

    label = f"bench_lin_proj_block:variants={','.join(variants)}:n={n_chunks}"
    print(f"[bench] acquiring npu_silicon_lock(label={label!r})")

    variant_outputs: dict[str, list] = {}
    try:
        with npu_silicon_lock(label=label):
            for variant in variants:
                op = ops[variant]
                try:
                    rv = _run_variant(
                        variant=variant,
                        op=op,
                        weight=weight,
                        chunks=chunks,
                        fp32_refs=fp32_refs,
                        bf16_refs=bf16_refs,
                    )
                    # Strip outputs for JSON serialisation; keep
                    # for the cross-comparison block below.
                    variant_outputs[variant] = rv.pop("outputs")
                    results[variant] = rv
                except Exception as e:
                    results[variant] = {
                        "verdict": "FAIL",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }
                    print(
                        f"[bench:{variant}] FAIL: {e}",
                        file=sys.stderr,
                    )
    except Exception as e:
        # Lock acquisition or wedge-detection failure.
        results["verdict"] = "FAIL_LOCK"
        results["error"] = f"{type(e).__name__}: {e}"
        results["traceback"] = traceback.format_exc()
        measurements_path.write_text(json.dumps(results, indent=2))
        print(f"[bench] FAIL_LOCK: {e}", file=sys.stderr)
        return 1

    # Cross-comparison only when both variants ran.
    if (
        "per_group" in variant_outputs
        and "fused_perts" in variant_outputs
        and len(variant_outputs["per_group"]) == n_chunks
        and len(variant_outputs["fused_perts"]) == n_chunks
    ):
        max_abs_diffs = []
        mean_abs_diffs = []
        for yp, yf in zip(
            variant_outputs["per_group"], variant_outputs["fused_perts"]
        ):
            d = np.abs(yp - yf)
            max_abs_diffs.append(float(d.max()))
            mean_abs_diffs.append(float(d.mean()))
        per_group_total = results["per_group"]["aggregate"]["total_wall_seconds"]
        fused_total = results["fused_perts"]["aggregate"]["total_wall_seconds"]
        per_group_kus_mean = results["per_group"]["aggregate"]["kernel_avg_us_mean"]
        fused_kus_mean = results["fused_perts"]["aggregate"]["kernel_avg_us_mean"]
        results["cross_comparison"] = {
            "max_abs_diff_overall": float(max(max_abs_diffs)),
            "mean_abs_diff_overall": float(np.mean(mean_abs_diffs)),
            "throughput_speedup_total_wall": (
                per_group_total / fused_total if fused_total > 0 else None
            ),
            "kernel_speedup_avg_us_mean": (
                per_group_kus_mean / fused_kus_mean
                if fused_kus_mean > 0 else None
            ),
        }

    # Verdict: PASS if every requested variant ran AND all bf16 drifts
    # are under the 5e-2 gate. Per-group is allowed to fail the bf16
    # gate (it computes in FP32) — only fused-perts is gated on bf16.
    fused_ok = True
    if "fused_perts" in variants:
        fr = results.get("fused_perts", {})
        if fr.get("verdict") != "PASS":
            fused_ok = False
        else:
            fused_ok = (
                fr["aggregate"]["max_abs_vs_bf16_ref"] < BF16_TOLERANCE_GATE
            )
    per_group_ok = True
    if "per_group" in variants:
        pr = results.get("per_group", {})
        if pr.get("verdict") != "PASS":
            per_group_ok = False

    results["verdict"] = "B3b_PASS" if (fused_ok and per_group_ok) else "B3b_FAIL"

    measurements_path.write_text(json.dumps(results, indent=2))
    print(f"[bench] verdict: {results['verdict']}")
    print(f"[bench] wrote {measurements_path}")

    # Console summary lines for quick caller readback.
    if "per_group" in results and results["per_group"].get("verdict") == "PASS":
        a = results["per_group"]["aggregate"]
        print(
            f"[bench] per_group     total {a['total_wall_seconds']:.3f} s "
            f"({a['throughput_chunks_per_sec']:.2f} chunks/s) "
            f"kernel avg {a['kernel_avg_us_mean']:.1f} us"
        )
    if "fused_perts" in results and results["fused_perts"].get("verdict") == "PASS":
        a = results["fused_perts"]["aggregate"]
        print(
            f"[bench] fused_perts   total {a['total_wall_seconds']:.3f} s "
            f"({a['throughput_chunks_per_sec']:.2f} chunks/s) "
            f"kernel avg {a['kernel_avg_us_mean']:.1f} us"
        )
    if "cross_comparison" in results:
        cx = results["cross_comparison"]
        print(
            f"[bench] speedup (total wall): "
            f"{cx['throughput_speedup_total_wall']:.2f}x  "
            f"speedup (kernel avg us): "
            f"{cx['kernel_speedup_avg_us_mean']:.2f}x  "
            f"max-abs cross-diff: {cx['max_abs_diff_overall']:.3e}"
        )

    return 0 if results["verdict"] == "B3b_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
