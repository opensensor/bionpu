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

"""CRISPR multi-tile streaming match op — registered NPU op.

Registers ``crispr_match_multitile`` in :data:`bionpu.dispatch.npu.NPU_OPS`
at import time. Same byte-equality contract as 's
``crispr_match_singletile`` — same encoding (2-bit packed: A=00, C=01, G=10,
T=11), same per-(guide, window) mismatch-count arithmetic — but split across
multiple AIE2P tiles via IRON's multi-worker ObjectFifo dataflow. See
``multitile.py`` and ``MANIFEST.md`` for the topology.

Per CRISPR PRD §4.2 + plan ("the hard part"): the multi-tile streaming
kernel is the C-M4 architecture milestone. This op preserves 's dense
``(N_GUIDES, N_WINDOWS)`` output contract — the *sparse* hit-list emission
(threshold + emit ring buffer to host) is layered on as a thin host-side
wrapper here, and moves to Tile Z in along with the on-tile PAM
filter.

Encoding contract (host side):

    Each spacer is 20 nt. Pack 2 bits per base, 4 bases per byte:
    A=00, C=01, G=10, T=11. Bases 0..3 of a spacer go in byte 0
    (base 0 in bits 1:0, base 1 in bits 3:2, base 2 in bits 5:4,
    base 3 in bits 7:6); bases 4..7 in byte 1; etc. 20 nt = 5 bytes.

This is the same encoding uses; we re-export ``encode_2bit`` /
``decode_2bit`` from the singletile module so callers can build inputs
identically.

Two public entry points:

* :class:`CrisprMatchMultiTile` — full ``NpuOp`` with ``__call__(*,
  guides_2bit, windows_2bit, ...) -> dense (N_GUIDES, N_WINDOWS) uint8``,
  byte-equal to 's output on the same fixture.
* :class:`MatchRecord` + :func:`extract_hits` — the sparse hit list
  derived from the dense matrix on the host side, filtered by
  ``max_mismatches``. will move this filter inside Tile Z.

Out of scope for (deferred):

    * On-tile PAM filtering.
    * On-tile sparse-emit + ring buffer to host.
    * Slide-by-1 windowing on the tile (host enumerates windows in v1; ).
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

# Re-export the codec from so callers don't need two import paths to
# build inputs. The encoding is identical.
from bionpu.kernels.crispr.match_singletile import (
    SPACER_BYTES,
    SPACER_LEN,
    _cpu_mismatch_count_matrix,  # noqa: PLC2701 — used by tests for parity
    decode_2bit,
    encode_2bit,
)

__all__ = [
    "CrisprMatchMultiTile",
    "GUIDES_PER_TILE",
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

# --- pinned shape (must match multitile.py / runner.cpp / kernel) ---
N_GUIDES = 128
N_WINDOWS = 4096

# Fan-out: 2 match tiles × 64 guides each. (Original PRD §4.2 sketch was 4
# tiles × 32 guides; reduced after hitting the AIE2P 2-input-DMA-channels
# limit on the joiner — see  .)
N_MATCH_TILES = 2
GUIDES_PER_TILE = N_GUIDES // N_MATCH_TILES  # 64

# Per-chunk geometry (mirrors so the host-side window stream is reusable).
WINDOWS_PER_CHUNK = 64
N_CHUNKS = N_WINDOWS // WINDOWS_PER_CHUNK  # 64

# Artifact paths (populated by `make NPU2=1` + the MANIFEST.md vendor step).
_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
    / "crispr_match_multitile"
)
_XCLBIN = _ART_ROOT / "final.xclbin"
_INSTS = _ART_ROOT / "insts.bin"
_BINARY = _ART_ROOT / "host_runner"

_KERNEL_NAME = "MLIR_AIE"

# --------------------------------------------------------------------------- #
# Tile-memory accounting (per multitile.py's IRON lowering)
# --------------------------------------------------------------------------- #

# Per-match-tile (Tiles C, D — both identical):
_MATCH_GUIDES_RESIDENT = N_GUIDES * SPACER_BYTES                  # 640
_MATCH_WINDOWS_DBL = 2 * WINDOWS_PER_CHUNK * SPACER_BYTES         # 640
_MATCH_PARTIAL_OUT_DBL = 2 * WINDOWS_PER_CHUNK * GUIDES_PER_TILE  # 8192
_MATCH_TILE_TOTAL = (
    _MATCH_GUIDES_RESIDENT + _MATCH_WINDOWS_DBL + _MATCH_PARTIAL_OUT_DBL
)  # 9472 B per match tile

# Joiner tile (Tile Z):
_JOIN_PARTIAL_IN_DBL = (
    N_MATCH_TILES * 2 * WINDOWS_PER_CHUNK * GUIDES_PER_TILE
)  # 16384 B (2 partials × dbl-buffered)
_JOIN_OUT_DBL = 2 * WINDOWS_PER_CHUNK * N_GUIDES  # 16384 B
_JOIN_TILE_TOTAL = _JOIN_PARTIAL_IN_DBL + _JOIN_OUT_DBL  # 32768 B

#: per-tile breakdown so MANIFEST and measurements.json can cite it.
TILE_MEMORY_BREAKDOWN: dict[str, int] = {
    "match_tile": _MATCH_TILE_TOTAL,
    "joiner_tile": _JOIN_TILE_TOTAL,
}

#: peak per-tile memory in bytes (joiner is the worst case).
TILE_MEMORY_BYTES: int = max(TILE_MEMORY_BREAKDOWN.values())

# Sanity: every tile fits the 64 KiB DM cap.
_AIE2P_DM_BUDGET_BYTES = 64 * 1024
assert TILE_MEMORY_BYTES < _AIE2P_DM_BUDGET_BYTES, (
    f"per-tile memory {TILE_MEMORY_BYTES} >= AIE2P DM cap {_AIE2P_DM_BUDGET_BYTES}; "
    "rebalance the topology (more match tiles, smaller WINDOWS_PER_CHUNK, etc.)"
)

# --------------------------------------------------------------------------- #
# Sparse hit-list — the v1 host-side threshold + emit pipeline
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class MatchRecord:
    """One surviving (guide, window) pair after the threshold filter.

    Attributes:
        guide_idx: row index into the input ``guides_2bit`` (0..N_GUIDES-1).
        window_idx: column index into the input ``windows_2bit`` (0..N_WINDOWS-1).
        mismatches: spacer-only mismatch count, 0..20.

     will move the threshold to Tile Z and emit these records via a
    ring buffer to the host; for we extract them on the host from
    the dense kernel output. The ``MatchRecord`` schema is forward-
    compatible with that move ( only adds ``strand`` and ``chrom``
    metadata, which here live in the calling :mod:`tracks.crispr.npu.match_multitile`
    runner that owns the chr22 fixture).
    """

    guide_idx: int
    window_idx: int
    mismatches: int

def extract_hits(
    dense_matrix: np.ndarray, *, max_mismatches: int = 4
) -> list[MatchRecord]:
    """Threshold + emit on the host.

    Args:
        dense_matrix: ``(N_GUIDES, N_WINDOWS)`` uint8 from the NPU op.
        max_mismatches: only (guide, window) pairs with
            ``mismatches <= max_mismatches`` survive.

    Returns:
        List of :class:`MatchRecord`, sorted by
        ``(window_idx, guide_idx)`` — the same order 's on-tile
        ring-buffer emission will produce.

    Sort key: window-major (mirrors how the kernel writes the dense
    matrix internally) so byte-equality through the canonical TSV
    pipeline is deterministic. The canonical-sites normalization
 does its own resort by ``(chrom, start, mismatches,
    guide_id, strand)`` — the order *here* doesn't reach the final TSV,
    but pinning it makes the intermediate easier to diff.
    """
    if dense_matrix.shape != (N_GUIDES, N_WINDOWS):
        raise ValueError(
            f"dense_matrix must have shape ({N_GUIDES}, {N_WINDOWS}); "
            f"got {dense_matrix.shape}"
        )
    if dense_matrix.dtype != np.uint8:
        raise ValueError(
            f"dense_matrix must be uint8; got {dense_matrix.dtype}"
        )
    # Vectorized extraction: find indices where mismatch count <= threshold.
    g_idx, w_idx = np.where(dense_matrix <= max_mismatches)
    mm = dense_matrix[g_idx, w_idx]
    # Sort window-major.
    order = np.lexsort((g_idx, w_idx))  # primary: w_idx, secondary: g_idx
    return [
        MatchRecord(
            guide_idx=int(g_idx[i]),
            window_idx=int(w_idx[i]),
            mismatches=int(mm[i]),
        )
        for i in order
    ]

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

    m = re.search(rf"^{label} NPU time:\s*([0-9.]+)us\.?\s*$", stdout, re.MULTILINE)
    return float(m.group(1)) if m else None

class CrisprMatchMultiTile(NpuOp):
    """``crispr_match_multitile`` — registered NPU op.

    Inputs (keyword-only):
        guides_2bit (np.ndarray): shape ``(N_GUIDES, SPACER_BYTES)``, uint8.
        windows_2bit (np.ndarray): shape ``(N_WINDOWS, SPACER_BYTES)``, uint8.
        max_mismatches (int): threshold for the sparse-hit extraction
            (default 4 — Cas-OFFinder's standard ceiling).
        n_iters (int): NPU iterations to time. Default 1.
        warmup (int): warmup iterations not counted in the timing. Default 0.
        return_dense (bool): if True (default), returns the dense
            ``(N_GUIDES, N_WINDOWS)`` matrix — byte-equal to . If False,
            returns ``list[MatchRecord]`` (sparse hit list at
            ``max_mismatches``).

    The op:
        1. Validates shapes/dtypes (hard fails on mismatch — v1 kernel is
           shape-pinned to 's 128×4096 fixture, same as ).
        2. If the NPU artifacts are present, runs the precompiled host
           binary via XRT. Output is read back from a tmp file as
           window-major and transposed to guide-major.
        3. If artifacts are missing, raises :class:`NpuArtifactsMissingError`.

    Notes on host I/O:
        Same file-backed contract as (``--guides PATH``,
        ``--windows PATH``, ``--out PATH``). The xclbin internally fans
        the work across 2 match tiles + 1 joiner tile but the host
        interface is identical so the dispatch layer is unchanged.
    """

    name = "crispr_match_multitile"

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
        with tempfile.TemporaryDirectory(prefix="crispr_multitile_") as tdir:
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
                    + f"{N_WINDOWS * N_GUIDES} (window-major (n_windows, n_guides))",
                )
            window_major = raw.reshape(N_WINDOWS, N_GUIDES)
            out_gw = np.ascontiguousarray(window_major.T)
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
                "crispr_match_multitile NPU artifacts missing: "
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
register_npu_op("crispr_match_multitile", CrisprMatchMultiTile())
