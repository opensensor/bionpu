# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Short tandem repeat (STR) NPU op (v0).

For each base position in the input, scan for runs of >= MIN_COPIES
consecutive identical motifs of period in [MIN_PERIOD, MAX_PERIOD].
Emit ``(start, end, period, motif)`` records.

Wire format mirrors cpg_island/primer_scan with a 16-byte record:
``uint32 start | uint32 end | uint32 period | uint32 motif_canonical``.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.data.tandem_repeat_oracle import (
    TR_MAX_PERIOD,
    TR_MIN_COPIES,
    TR_MIN_PERIOD,
    motif_to_canonical_u32,
)
from bionpu.dispatch._pyxrt_helpers import resolve_dispatch_impl
from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    _xrt_env,  # noqa: PLC2701
    register_npu_op,
)

__all__ = [
    "BionpuTandemRepeat",
    "MAX_EMIT_IDX",
    "PARTIAL_OUT_BYTES_PADDED",
    "RECORD_BYTES",
    "SEQ_IN_CHUNK_BYTES_BASE",
    "SUPPORTED_N_CHUNKS_PER_LAUNCH",
    "SUPPORTED_N_TILES",
    "TANDEM_REPEAT_DISPATCH_ENV",
    "canonical_to_motif",
]

SUPPORTED_N_TILES: tuple[int, ...] = (1, 2, 4, 8)
SUPPORTED_N_CHUNKS_PER_LAUNCH: tuple[int, ...] = (1, 2, 4, 8)
SEQ_IN_CHUNK_BYTES_BASE: int = 4096
PARTIAL_OUT_BYTES_PADDED: int = 32768
RECORD_BYTES: int = 16
MAX_EMIT_IDX: int = 2046
TANDEM_REPEAT_DISPATCH_ENV: str = "BIONPU_TANDEM_REPEAT_DISPATCH"
_KERNEL_NAME: str = "MLIR_AIE"

_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)

_2BIT_TO_BASE = "ACGT"


def canonical_to_motif(motif_canon: int, period: int) -> str:
    """Inverse of :func:`motif_to_canonical_u32`. MSB-first decode."""
    if not 1 <= period <= TR_MAX_PERIOD:
        raise ValueError(f"period must be in 1..{TR_MAX_PERIOD}; got {period}")
    chars = []
    for i in range(period):
        shift = 2 * (period - 1 - i)
        chars.append(_2BIT_TO_BASE[(motif_canon >> shift) & 0x3])
    return "".join(chars)


def _parse_us_label(stdout: str, label: str) -> float | None:
    m = re.search(
        rf"^{label} NPU time:\s*([0-9.eE+\-]+)us\.?\s*$",
        stdout,
        re.MULTILINE,
    )
    return float(m.group(1)) if m else None


@dataclass(frozen=True)
class _RunInfo:
    avg_us: float
    min_us: float
    max_us: float
    n_iters: int
    used_npu: bool
    n_records: int = 0


def _parse_records_blob(path: Path) -> list[tuple[int, int, int, str]]:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 8:
        raise NpuRunFailed(
            0, "",
            f"[bionpu] binary output blob {path} too small "
            f"({raw.size} bytes; expected >=8)",
        )
    n_records = int(np.frombuffer(raw[:8], dtype=np.uint64)[0])
    expected_bytes = 8 + n_records * RECORD_BYTES
    if raw.size < expected_bytes:
        raise NpuRunFailed(
            0, "",
            f"[bionpu] binary output blob {path} truncated: "
            f"n_records={n_records}, bytes={raw.size}, need={expected_bytes}",
        )
    if n_records == 0:
        return []
    rec_dt = np.dtype([
        ("start", "<u4"),
        ("end", "<u4"),
        ("period", "<u4"),
        ("motif_canon", "<u4"),
    ])
    arr = np.frombuffer(raw[8 : 8 + n_records * RECORD_BYTES], dtype=rec_dt)
    out: list[tuple[int, int, int, str]] = []
    for s, e, p, mc in arr:
        out.append((int(s), int(e), int(p),
                    canonical_to_motif(int(mc), int(p))))
    return out


class BionpuTandemRepeat(NpuOp):
    """Short tandem repeat (STR) scan on AIE2P (v0).

    Returns ``[(start, end, period, motif), ...]`` records, sorted by
    ``(start asc)`` and de-duplicated so each base position appears in
    at most one record.

    Mirrors :func:`bionpu.data.tandem_repeat_oracle.find_tandem_repeats`
    byte-for-byte.
    """

    INPUT_DTYPE = np.uint8
    SUPPORTED_N_TILES = SUPPORTED_N_TILES
    SUPPORTED_N_CHUNKS_PER_LAUNCH = SUPPORTED_N_CHUNKS_PER_LAUNCH

    def __init__(
        self,
        *,
        n_tiles: int = 4,
        n_chunks_per_launch: int = 1,
    ) -> None:
        if int(n_tiles) not in SUPPORTED_N_TILES:
            valid = ", ".join(str(x) for x in SUPPORTED_N_TILES)
            raise ValueError(
                f"BionpuTandemRepeat: n_tiles={n_tiles!r} not in {{{valid}}}."
            )
        if int(n_chunks_per_launch) not in SUPPORTED_N_CHUNKS_PER_LAUNCH:
            valid = ", ".join(str(x) for x in SUPPORTED_N_CHUNKS_PER_LAUNCH)
            raise ValueError(
                "BionpuTandemRepeat: n_chunks_per_launch="
                f"{n_chunks_per_launch!r} not in {{{valid}}}."
            )
        self.n_tiles = int(n_tiles)
        self.n_chunks_per_launch = int(n_chunks_per_launch)
        self.name = "bionpu_tandem_repeat"
        self.last_run: _RunInfo | None = None

    @property
    def artifact_dir(self) -> Path:
        base = f"bionpu_tandem_repeat_n{self.n_tiles}"
        if self.n_chunks_per_launch == 1:
            return _ART_ROOT / base
        return _ART_ROOT / f"{base}_b{self.n_chunks_per_launch}"

    @property
    def xclbin(self) -> Path:
        return self.artifact_dir / "final.xclbin"

    @property
    def insts(self) -> Path:
        return self.artifact_dir / "insts.bin"

    @property
    def host_runner(self) -> Path:
        return self.artifact_dir / "host_runner"

    def artifacts_present(self) -> bool:
        return all(p.exists() for p in (self.xclbin, self.insts, self.host_runner))

    def _resolve_dispatch_impl(self, impl: str | None) -> str:
        return resolve_dispatch_impl(impl, env_var=TANDEM_REPEAT_DISPATCH_ENV)

    def _validate_inputs(self, packed_seq: np.ndarray) -> None:
        if not isinstance(packed_seq, np.ndarray):
            raise TypeError("packed_seq must be a numpy.ndarray of dtype uint8")
        if packed_seq.dtype != np.uint8:
            raise ValueError(
                f"packed_seq dtype must be uint8; got {packed_seq.dtype}"
            )
        if packed_seq.ndim != 1:
            raise ValueError(
                f"packed_seq must be 1-D; got shape {packed_seq.shape}"
            )
        # Need at least MIN_PERIOD * MIN_COPIES bases (=5) to emit anything.
        min_bytes = (TR_MIN_PERIOD * TR_MIN_COPIES + 3) // 4
        if packed_seq.size < min_bytes:
            raise ValueError(
                f"packed_seq too short for MIN_PERIOD*MIN_COPIES="
                f"{TR_MIN_PERIOD * TR_MIN_COPIES}: need at least "
                f"{min_bytes} bytes, got {packed_seq.size}"
            )

    def _run_subprocess(
        self,
        *,
        packed_seq: np.ndarray,
        top_n: int,
        n_iters: int,
        warmup: int,
        timeout_s: float,
    ) -> tuple[list[tuple[int, int, int, str]], _RunInfo]:
        from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as tdir:
            t = Path(tdir)
            input_path = t / "packed_seq.2bit.bin"
            output_path = t / "tandem_repeats.bin"
            np.ascontiguousarray(packed_seq).tofile(input_path)
            cmd = [
                str(self.host_runner),
                "-x", str(self.xclbin),
                "-i", str(self.insts),
                "-k", _KERNEL_NAME,
                "--input", str(input_path),
                "--output", str(output_path),
                "--output-format", "binary",
                "--launch-chunks", str(self.n_tiles),
                "--n-chunks-per-launch", str(self.n_chunks_per_launch),
                "--top", str(int(top_n)),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
            ]
            with npu_silicon_lock():
                proc = subprocess.run(  # noqa: S603 — argv strictly controlled
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_s,
                    env=_xrt_env(),
                )
            if proc.returncode != 0:
                raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker missing",
                )
            avg = _parse_us_label(proc.stdout, "Avg")
            mn = _parse_us_label(proc.stdout, "Min")
            mx = _parse_us_label(proc.stdout, "Max")
            if avg is None or mn is None or mx is None:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] could not parse NPU timing lines",
                )
            if not output_path.exists():
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr
                    + f"\n[bionpu] runner did not produce {output_path}",
                )
            records = _parse_records_blob(output_path)

        info = _RunInfo(
            avg_us=float(avg),
            min_us=float(mn),
            max_us=float(mx),
            n_iters=int(n_iters),
            used_npu=True,
            n_records=len(records),
        )
        return records, info

    def __call__(
        self,
        *,
        packed_seq: np.ndarray,
        top_n: int = 0,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 600.0,
        _impl: str | None = None,
        **_unused: Any,
    ) -> list[tuple[int, int, int, str]]:
        self._validate_inputs(packed_seq)

        impl = self._resolve_dispatch_impl(_impl)
        if impl == "pyxrt":
            raise NotImplementedError(
                "BionpuTandemRepeat pyxrt path not implemented; "
                "use BIONPU_TANDEM_REPEAT_DISPATCH=subprocess (default)."
            )
        if not self.host_runner.exists():
            raise NpuArtifactsMissingError(
                f"NPU artifact missing for {self.name}: {self.host_runner}. "
                "Build via `make NPU2=1 experiment=wide4 all` in the "
                "tandem_repeat kernel directory."
            )
        for need in (self.xclbin, self.insts):
            if not need.exists():
                raise NpuArtifactsMissingError(f"NPU artifact missing: {need}")

        records, info = self._run_subprocess(
            packed_seq=packed_seq,
            top_n=top_n,
            n_iters=n_iters,
            warmup=warmup,
            timeout_s=timeout_s,
        )
        self.last_run = info
        return records


# Force-use of motif_to_canonical_u32 import (silence unused-import lint).
_ = motif_to_canonical_u32

register_npu_op("bionpu_tandem_repeat", BionpuTandemRepeat(n_tiles=4))
