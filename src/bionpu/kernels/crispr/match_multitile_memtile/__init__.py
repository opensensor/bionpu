# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""CRISPR memtile-aggregated 4-into-1 multi-tile match op.

**AM020 cross-walk follow-up to .** Recovers the original PRD §4.2
sketch (4 match tiles × 32 guides) that 's ship had to reduce to
2 match × 64 guides because compute tiles only have 2 input DMA channels
. The fix per AM020 Ch. 5 p. 74 is to aggregate via the
memtile, which has 6 MM2S + 6 S2MM channels (3× the compute-tile fan-in
budget) and east/west neighbour memtile addressing on channels 0..3.

Registers ``crispr_match_multitile_memtile`` in
:data:`bionpu.dispatch.npu.NPU_OPS` at import time. **Same byte-equality
contract as ** (and ): same encoding (2-bit packed: A=00, C=01,
G=10, T=11), same per-(guide, window) mismatch-count arithmetic. The
architectural change is **fan-in width only** (4 vs 2 match tiles); the
output is identical bit-for-bit.

Topology (canonical AIE-ML memtile aggregation per AM020 Figure 22):

* shim → guides + windows broadcast → 4 match tiles
* match_<i> → memtile slot i (i ∈ {0, 1, 2, 3})
* memtile reorgs the 4 partials into a single (n_windows × N_GUIDES)
  buffer via 5D address generation, then memtile MM2S → shim DMA out.

No joiner compute tile — the join is fabric-side via memtile DMA.

Throughput hypothesis: 4 match tiles × 32 guides ≈ 2× the 2-tile 
ceiling, less the memtile aggregation overhead. baseline: ~240 ms
wall-clock per launch on chr22 × 128 guides; ~395 K windows/sec
kernel-only. Predicted: 130-180 ms / 600-700 K windows/sec. Actual is
the load-bearing measurement.

See ``DESIGN.md`` for the full AM020 cross-walk rationale and
``MANIFEST.md`` for the vendored xclbin/insts/host-runner provenance.

Out of scope (deferred):

    * On-tile PAM filtering.
    * On-tile sparse-emit + ring buffer to host.
    * Slide-by-1 windowing on the tile (host enumerates windows in v1).
    * Genome-scale walk.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    _xrt_env,  # noqa: PLC2701 — internal helper
    register_npu_op,
)

# Re-export the codec from (= ) so callers don't need three
# import paths to build inputs. The encoding is identical (byte-equality
# requires it).
from bionpu.kernels.crispr.match_singletile import (
    SPACER_BYTES,
    SPACER_LEN,
    _cpu_mismatch_count_matrix,  # noqa: PLC2701 — used by tests for parity
    decode_2bit,
    encode_2bit,
)
from bionpu.kernels.crispr.match_multitile import (
    MatchRecord,
    extract_hits,
)

__all__ = [
    "CrisprMatchMultiTileMemtile",
    "GUIDES_PER_TILE",
    "MEMTILE_MEMORY_BYTES",
    "MatchRecord",
    "N_GUIDES",
    "N_MATCH_TILES",
    "N_WINDOWS",
    "SPACER_BYTES",
    "SPACER_LEN",
    "TILE_MEMORY_BREAKDOWN",
    "TILE_MEMORY_BYTES",
    "_cpu_mismatch_count_matrix",
    "decode_2bit",
    "encode_2bit",
    "extract_hits",
]

# --- pinned shape (must match multitile_memtile.py / runner.cpp / kernel) ---
N_GUIDES = 128
N_WINDOWS = 4096

# Fan-out: **4 match tiles × 32 guides each** — recovers the original PRD
# §4.2 sketch via memtile-mediated aggregation.
N_MATCH_TILES = 4
GUIDES_PER_TILE = N_GUIDES // N_MATCH_TILES  # 32

# Per-chunk geometry (mirrors / so the host-side window stream
# is byte-equivalent — required for byte-equality with both).
WINDOWS_PER_CHUNK = 64
N_CHUNKS = N_WINDOWS // WINDOWS_PER_CHUNK  # 64

# Artifact paths (populated by `make NPU2=1` + the MANIFEST.md vendor step).
_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
    / "crispr_match_multitile_memtile"
)
_XCLBIN = _ART_ROOT / "final.xclbin"
_INSTS = _ART_ROOT / "insts.bin"
_BINARY = _ART_ROOT / "host_runner"

_KERNEL_NAME = "MLIR_AIE"

# --------------------------------------------------------------------------- #
# Tile-memory accounting (per multitile_memtile.py's IRON lowering)
# --------------------------------------------------------------------------- #

# Per match tile (Tiles match_0..match_3 — all identical):
_MATCH_GUIDES_RESIDENT = N_GUIDES * SPACER_BYTES                  # 640
_MATCH_WINDOWS_DBL = 2 * WINDOWS_PER_CHUNK * SPACER_BYTES         # 640
_MATCH_PARTIAL_OUT_DBL = 2 * WINDOWS_PER_CHUNK * GUIDES_PER_TILE  # 4096
_MATCH_TILE_TOTAL = (
    _MATCH_GUIDES_RESIDENT + _MATCH_WINDOWS_DBL + _MATCH_PARTIAL_OUT_DBL
) # 5376 B per match tile (vs 's 9472 B — narrower guide slice)

# Memtile (the new "joiner" — fabric-side aggregation, not a compute tile):
# 4 partials × 2 dbl-buf × WINDOWS_PER_CHUNK × GUIDES_PER_TILE
_MEMTILE_PARTIALS_IN = (
    N_MATCH_TILES * 2 * WINDOWS_PER_CHUNK * GUIDES_PER_TILE
)  # 16384 B (4 × 2 × 64 × 32)
# 1 joined output × 2 dbl-buf × WINDOWS_PER_CHUNK × N_GUIDES
_MEMTILE_OUT_DBL = 2 * WINDOWS_PER_CHUNK * N_GUIDES  # 16384 B
_MEMTILE_TOTAL = _MEMTILE_PARTIALS_IN + _MEMTILE_OUT_DBL  # 32768 B

#: per-tile breakdown so MANIFEST and measurements.json can cite it.
TILE_MEMORY_BREAKDOWN: dict[str, int] = {
    "match_tile": _MATCH_TILE_TOTAL,
    "memtile": _MEMTILE_TOTAL,
}

#: peak per-compute-tile memory in bytes. Memtile is excluded from the
#: 64 KiB compute-tile DM cap (memtile's 512 KiB cap is per AM020 Ch. 5
#: p. 74, not the compute-tile budget).
TILE_MEMORY_BYTES: int = _MATCH_TILE_TOTAL

#: peak memtile memory in bytes (separate budget, ~512 KiB).
MEMTILE_MEMORY_BYTES: int = _MEMTILE_TOTAL

# Sanity: every compute tile fits the 64 KiB DM cap.
_AIE2P_DM_BUDGET_BYTES = 64 * 1024
assert TILE_MEMORY_BYTES < _AIE2P_DM_BUDGET_BYTES, (
    f"per-match-tile memory {TILE_MEMORY_BYTES} >= AIE2P DM cap "
    f"{_AIE2P_DM_BUDGET_BYTES}; rebalance the topology."
)

# Sanity: memtile fits AIE-ML's 512 KiB cap (AM020 Ch. 5 p. 74).
# AIE2P's memtile size is unverified at this layer — see gaps.yaml.
_AIEML_MEMTILE_BUDGET_BYTES = 512 * 1024
assert MEMTILE_MEMORY_BYTES < _AIEML_MEMTILE_BUDGET_BYTES, (
    f"memtile memory {MEMTILE_MEMORY_BYTES} >= AIE-ML memtile cap "
    f"{_AIEML_MEMTILE_BUDGET_BYTES}; verify AIE2P memtile size."
)

# --------------------------------------------------------------------------- #
# NpuOp registration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _RunInfo:
    """Timing for the last NPU run, parsed from the host binary."""

    avg_us: float
    min_us: float
    max_us: float
    n_iters: int

def _parse_us(stdout: str, label: str) -> float | None:
    import re

    # Allow scientific notation (e.g. ``6.14402e+06us``) — long kernel
    # runs print stream-style scientific values via std::cout's default
    # float format. 's parser missed these; fixed here.
    m = re.search(
        rf"^{label} NPU time:\s*([0-9.]+(?:[eE][+-]?\d+)?)us\.?\s*$",
        stdout,
        re.MULTILINE,
    )
    return float(m.group(1)) if m else None

class CrisprMatchMultiTileMemtile(NpuOp):
    """``crispr_match_multitile_memtile`` — registered NPU op.

    Same input/output contract as 's ``CrisprMatchMultiTile``; the only
    architectural difference is the fan-in width (4 vs 2 match tiles) and
    that the join happens fabric-side via memtile DMA aggregation rather
    than on a joiner compute tile. The output dense matrix is byte-equal
    to (and , and the NumPy oracle) by construction.

    Inputs (keyword-only):
        guides_2bit (np.ndarray): shape ``(N_GUIDES, SPACER_BYTES)``, uint8.
        windows_2bit (np.ndarray): shape ``(N_WINDOWS, SPACER_BYTES)``, uint8.
        max_mismatches (int): threshold for the sparse-hit extraction
            (default 4 — Cas-OFFinder's standard ceiling).
        n_iters (int): NPU iterations to time. Default 1.
        warmup (int): warmup iterations not counted in the timing. Default 0.
        return_dense (bool): if True (default), returns the dense
            ``(N_GUIDES, N_WINDOWS)`` matrix — byte-equal to / .
            If False, returns ``list[MatchRecord]`` (sparse hit list at
            ``max_mismatches``).
    """

    name = "crispr_match_multitile_memtile"

    def __init__(self) -> None:
        self.last_run: _RunInfo | None = None

    @staticmethod
    def _validate_inputs(
        guides_2bit: np.ndarray, windows_2bit: np.ndarray
    ) -> None:
        if not isinstance(guides_2bit, np.ndarray):
            raise TypeError("guides_2bit must be a numpy ndarray")
        if not isinstance(windows_2bit, np.ndarray):
            raise TypeError("windows_2bit must be a numpy ndarray")
        if guides_2bit.shape != (N_GUIDES, SPACER_BYTES):
            raise ValueError(
                f"guides_2bit must be shape ({N_GUIDES}, {SPACER_BYTES}); "
                f"got {guides_2bit.shape}"
            )
        if windows_2bit.shape != (N_WINDOWS, SPACER_BYTES):
            raise ValueError(
                f"windows_2bit must be shape ({N_WINDOWS}, {SPACER_BYTES}); "
                f"got {windows_2bit.shape}"
            )
        if guides_2bit.dtype != np.uint8:
            raise ValueError(
                f"guides_2bit dtype must be uint8; got {guides_2bit.dtype}"
            )
        if windows_2bit.dtype != np.uint8:
            raise ValueError(
                f"windows_2bit dtype must be uint8; got {windows_2bit.dtype}"
            )

    @staticmethod
    def artifacts_present() -> bool:
        """True iff all three NPU artifacts are on disk."""
        return all(p.exists() for p in (_XCLBIN, _INSTS, _BINARY))

    def _run_npu(
        self,
        guides_2bit: np.ndarray,
        windows_2bit: np.ndarray,
        n_iters: int,
        warmup: int,
    ) -> tuple[np.ndarray, _RunInfo]:
        """Invoke the precompiled host binary and return (dense_matrix, timing)."""
        with tempfile.TemporaryDirectory(prefix="crispr_memtile_") as tdir:
            t = Path(tdir)
            guides_bin = t / "guides.bin"
            windows_bin = t / "windows.bin"
            out_bin = t / "out.bin"
            guides_bin.write_bytes(np.ascontiguousarray(guides_2bit).tobytes())
            windows_bin.write_bytes(np.ascontiguousarray(windows_2bit).tobytes())

            cmd = [
                str(_BINARY),
                "-x", str(_XCLBIN),
                "-i", str(_INSTS),
                "-k", _KERNEL_NAME,
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
                "--guides", str(guides_bin),
                "--windows", str(windows_bin),
                "--out", str(out_bin),
            ]
            proc = subprocess.run(  # noqa: S603 — argv strictly controlled
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=120.0,
                env=_xrt_env(),
            )
            if proc.returncode != 0:
                raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker not found",
                )
            avg = _parse_us(proc.stdout, "Avg")
            mn = _parse_us(proc.stdout, "Min")
            mx = _parse_us(proc.stdout, "Max")
            if avg is None or mn is None or mx is None:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] could not parse NPU timing",
                )
            raw = np.frombuffer(out_bin.read_bytes(), dtype=np.uint8)
            if raw.size != N_WINDOWS * N_GUIDES:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr
                    + f"\n[bionpu] output size {raw.size} != "
                    + f"{N_WINDOWS * N_GUIDES} (per-chunk guide-major)",
                )
            # The kernel writes per-chunk (N_GUIDES, WINDOWS_PER_CHUNK)
            # guide-major partial buffers; the memtile flat-concatenates
            # within a chunk so each chunk's 8192-byte block is
            # (N_GUIDES, WINDOWS_PER_CHUNK) guide-major. The host then
            # walks N_CHUNKS such blocks; flat layout in the output file
            # is therefore (N_CHUNKS, N_GUIDES, WINDOWS_PER_CHUNK), which
            # we reshape and transpose to (N_GUIDES, N_WINDOWS) by
            # walking the window axis through chunks.
            chunked = raw.reshape(N_CHUNKS, N_GUIDES, WINDOWS_PER_CHUNK)
            # Move chunk axis to be inner-window-axis: (N_GUIDES, N_CHUNKS, WINDOWS_PER_CHUNK)
            #                                       → (N_GUIDES, N_WINDOWS)
            out_gw = np.ascontiguousarray(
                chunked.transpose(1, 0, 2).reshape(N_GUIDES, N_WINDOWS)
            )
            info = _RunInfo(
                avg_us=avg, min_us=mn, max_us=mx, n_iters=n_iters
            )
            return out_gw, info

    def __call__(
        self,
        *,
        guides_2bit: np.ndarray,
        windows_2bit: np.ndarray,
        max_mismatches: int = 4,
        n_iters: int = 1,
        warmup: int = 0,
        return_dense: bool = True,
        **_unused: Any,
    ) -> np.ndarray | list[MatchRecord]:
        self._validate_inputs(guides_2bit, windows_2bit)

        if not self.artifacts_present():
            missing = [p for p in (_XCLBIN, _INSTS, _BINARY) if not p.exists()]
            raise NpuArtifactsMissingError(
                "crispr_match_multitile_memtile NPU artifacts missing: "
                + ", ".join(str(p) for p in missing)
                + ". See "
                + str(_ART_ROOT / "MANIFEST.md")
                + " for build instructions."
            )

        dense, info = self._run_npu(guides_2bit, windows_2bit, n_iters, warmup)
        self.last_run = info

        if return_dense:
            return dense
        return extract_hits(dense, max_mismatches=max_mismatches)

# Register at import time.
register_npu_op(
    "crispr_match_multitile_memtile", CrisprMatchMultiTileMemtile()
)
