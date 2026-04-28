# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""CpG island candidate-position NPU op (v0).

The tile emits candidate window-start positions whose length-200 window
passes the fixed-point Gardiner-Garden thresholds. The host merges those
positions into half-open CpG-island intervals using
``bionpu.data.cpg_oracle.merge_streak_positions_to_islands``.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.data.cpg_oracle import (
    CPG_DEFAULT_W,
    merge_candidate_positions_to_islands,
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
    "BionpuCpgIsland",
    "CPG_ISLAND_DISPATCH_ENV",
    "MAX_EMIT_IDX",
    "PARTIAL_OUT_BYTES_PADDED",
    "RECORD_BYTES",
    "SEQ_IN_CHUNK_BYTES_BASE",
    "SUPPORTED_N_CHUNKS_PER_LAUNCH",
    "SUPPORTED_N_TILES",
]

SUPPORTED_N_TILES: tuple[int, ...] = (1, 2, 4, 8)
SUPPORTED_N_CHUNKS_PER_LAUNCH: tuple[int, ...] = (1, 2, 4, 8)
SEQ_IN_CHUNK_BYTES_BASE: int = 4096
PARTIAL_OUT_BYTES_PADDED: int = 32768
RECORD_BYTES: int = 4
MAX_EMIT_IDX: int = 8190
CPG_ISLAND_DISPATCH_ENV: str = "BIONPU_CPG_ISLAND_DISPATCH"
_KERNEL_NAME: str = "MLIR_AIE"

_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)


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
    n_candidates: int = 0
    n_islands: int = 0


def _parse_position_blob(path: Path) -> list[int]:
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
    arr = np.frombuffer(raw[8:expected_bytes], dtype="<u4")
    return [int(x) for x in arr]


class BionpuCpgIsland(NpuOp):
    """CpG island candidate scan on AIE2P.

    Returns ``[(start, end), ...]`` island intervals by default. Pass
    ``return_candidates=True`` to get the raw candidate window-start
    positions emitted by silicon before host-side run merging.
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
                f"BionpuCpgIsland: n_tiles={n_tiles!r} not in {{{valid}}}."
            )
        if int(n_chunks_per_launch) not in SUPPORTED_N_CHUNKS_PER_LAUNCH:
            valid = ", ".join(str(x) for x in SUPPORTED_N_CHUNKS_PER_LAUNCH)
            raise ValueError(
                "BionpuCpgIsland: n_chunks_per_launch="
                f"{n_chunks_per_launch!r} not in {{{valid}}}."
            )
        self.n_tiles = int(n_tiles)
        self.n_chunks_per_launch = int(n_chunks_per_launch)
        self.name = "bionpu_cpg_island"
        self.last_run: _RunInfo | None = None

    @property
    def artifact_dir(self) -> Path:
        base = f"bionpu_cpg_island_n{self.n_tiles}"
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
        return resolve_dispatch_impl(impl, env_var=CPG_ISLAND_DISPATCH_ENV)

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
        min_bytes = (CPG_DEFAULT_W + 3) // 4
        if packed_seq.size < min_bytes:
            raise ValueError(
                f"packed_seq too short for W={CPG_DEFAULT_W}: need at least "
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
    ) -> tuple[list[int], _RunInfo]:
        from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as tdir:
            t = Path(tdir)
            input_path = t / "packed_seq.2bit.bin"
            output_path = t / "cpg_candidates.bin"
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
                proc = subprocess.run(
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
            candidates = _parse_position_blob(output_path)

        info = _RunInfo(
            avg_us=float(avg),
            min_us=float(mn),
            max_us=float(mx),
            n_iters=int(n_iters),
            used_npu=True,
            n_candidates=len(candidates),
            n_islands=len(merge_candidate_positions_to_islands(candidates)),
        )
        return candidates, info

    def __call__(
        self,
        *,
        packed_seq: np.ndarray,
        top_n: int = 0,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 600.0,
        return_candidates: bool = False,
        _impl: str | None = None,
        **_unused: Any,
    ) -> list[tuple[int, int]] | list[int]:
        self._validate_inputs(packed_seq)

        impl = self._resolve_dispatch_impl(_impl)
        if impl == "pyxrt":
            raise NotImplementedError(
                "BionpuCpgIsland pyxrt path not implemented; "
                "use BIONPU_CPG_ISLAND_DISPATCH=subprocess (default)."
            )
        if not self.host_runner.exists():
            raise NpuArtifactsMissingError(
                f"NPU artifact missing for {self.name}: {self.host_runner}. "
                "Build via `make NPU2=1 experiment=wide4 all` in the "
                "cpg_island kernel directory."
            )
        for need in (self.xclbin, self.insts):
            if not need.exists():
                raise NpuArtifactsMissingError(f"NPU artifact missing: {need}")

        candidates, info = self._run_subprocess(
            packed_seq=packed_seq,
            top_n=top_n,
            n_iters=n_iters,
            warmup=warmup,
            timeout_s=timeout_s,
        )
        islands = merge_candidate_positions_to_islands(candidates)
        self.last_run = _RunInfo(
            avg_us=info.avg_us,
            min_us=info.min_us,
            max_us=info.max_us,
            n_iters=info.n_iters,
            used_npu=True,
            n_candidates=len(candidates),
            n_islands=len(islands),
        )
        return candidates if return_candidates else islands


register_npu_op("bionpu_cpg_island", BionpuCpgIsland(n_tiles=4))
