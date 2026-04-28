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

"""Sliding-window (w, k) minimizer NPU op (v0).

Mirrors the kmer_count v0.5 NpuOp shape (subprocess host_runner +
binary --output blob parsed via numpy) but specialised for sliding-
window minimizers per the canonical algorithm
(minimap2 ``mm_sketch``, *Bioinformatics* 2016, lines 77+ of
``sketch.c``). v0 uses a SIMPLER pure-canonical scheme — no secondary
``hash64`` ordering — so the silicon byte-equal contract is tight and
implementable in scalar AIE2P code with a tiny ring buffer. See
``DESIGN.md`` §2 for the divergence from minimap2 and ``minimizer_oracle.py``
for the reference algorithm.

Wire format (per ``minimizer_constants.h`` + ``data/minimizer_oracle.py``):

* Input ``packed_seq``: 1-D ``np.uint8`` packed-2-bit DNA (A=00, C=01,
  G=10, T=11; first base = bits[7:6] of byte 0, MSB-first).
* The host runner streams the input in 4096-byte chunks with a
  per-(k, w) overlap (8 bytes for both pinned configs) plus an 8-byte
  in-band header (``actual_payload_bytes`` + ``owned_start_offset_bases``).
* Per chunk × per tile, the kernel emits 16-byte records
  ``{u64 canonical_LE, u32 position_LE, u32 _pad}`` into a 32 KiB
  pass-slot prefixed by ``[u32 emit_count_LE]``. Slot 0 is the
  authoritative output; slots 1..N_TILES-1 are duplicates (v0
  broadcast topology) ignored host-side.
* The host runner translates per-chunk positions to global
  (``+ src_offset * 4``), de-duplicates, sorts by ``(position asc,
  canonical asc)``, applies ``--top``, and emits a binary blob
  ``[u64 n_records][n_records × {u64 canonical, u32 position, u32 _pad}]``
  via ``--output-format binary``.

Two NPU_OPS registry entries:

* ``bionpu_minimizer_k15_w10`` — short-read default (matches minimap2's
  ``-k 15 -w 10`` short-read preset).
* ``bionpu_minimizer_k21_w11`` — long-read default (matches minimap2's
  ``-k 21 -w 11`` long-read preset).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.dispatch._pyxrt_helpers import resolve_dispatch_impl
from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    _xrt_env,  # noqa: PLC2701 — internal helper; mirrors kmer_count
    register_npu_op,
)

__all__ = [
    "BionpuMinimizer",
    "MAX_EMIT_IDX",
    "MINIMIZER_DISPATCH_ENV",
    "PARTIAL_OUT_BYTES_PADDED",
    "RECORD_BYTES",
    "SEQ_IN_CHUNK_BYTES_BASE",
    "SUPPORTED_KW",
    "SUPPORTED_N_CHUNKS_PER_LAUNCH",
    "SUPPORTED_N_TILES",
    "decode_canonical_to_ascii",
    "kmer_mask_for",
    "overlap_bytes_for_kw",
]

# --------------------------------------------------------------------------- #
# Pinned constants — mirror minimizer_constants.h byte-equal.
# --------------------------------------------------------------------------- #

#: Pinned (k, w) configurations matching the silicon artifacts.
SUPPORTED_KW: tuple[tuple[int, int], ...] = ((15, 10), (21, 11))

#: Supported tile fan-out (n_tiles constructor arg).
SUPPORTED_N_TILES: tuple[int, ...] = (1, 2, 4, 8)

#: Supported chunks-per-launch (constructor arg).
SUPPORTED_N_CHUNKS_PER_LAUNCH: tuple[int, ...] = (1, 2, 4, 8)

#: Per-k canonical bit masks (bit-equal to the C header).
_KMER_MASKS: dict[int, int] = {
    15: (1 << 30) - 1,
    21: (1 << 42) - 1,
}

#: 4096-byte primary chunk before per-(k, w) overlap padding.
SEQ_IN_CHUNK_BYTES_BASE: int = 4096

#: 32 KiB per pass-slot in the joined output buffer.
PARTIAL_OUT_BYTES_PADDED: int = 32768

#: Wire record size (uint64 canonical + uint32 position + uint32 pad).
RECORD_BYTES: int = 16

#: Max records per pass-slot (kernel caps at this).
MAX_EMIT_IDX: int = 2046

#: Env var selecting subprocess vs in-process pyxrt dispatch.
MINIMIZER_DISPATCH_ENV: str = "BIONPU_MINIMIZER_DISPATCH"

#: Kernel name baked into the xclbin (mirrors kmer_count convention).
_KERNEL_NAME: str = "MLIR_AIE"

# --------------------------------------------------------------------------- #
# Artifact root — same _npu_artifacts/ directory as every other kernel.
# --------------------------------------------------------------------------- #

_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)


def overlap_bytes_for_kw(k: int, w: int) -> int:
    """Per-(k, w) overlap (bytes) — matches IRON Python's ``_OVERLAP_BY_KW``.

    Both pinned configs land on 8 bytes (4-byte-aligned for aiecc).
    """
    if (int(k), int(w)) not in {(15, 10), (21, 11)}:
        raise ValueError(
            f"unsupported (k={k}, w={w}); pinned: (15,10), (21,11)"
        )
    return 8


def kmer_mask_for(k: int) -> int:
    """Return the per-k canonical mask. Raises ``ValueError`` for unsupported k."""
    if int(k) not in _KMER_MASKS:
        valid = ", ".join(str(x) for x in _KMER_MASKS)
        raise ValueError(
            f"k={k!r} is not supported; expected one of {{{valid}}}."
        )
    return _KMER_MASKS[int(k)]


def decode_canonical_to_ascii(canonical: int, k: int) -> str:
    """Decode a 2-bit-packed canonical k-mer back to ACGT ASCII."""
    if int(k) not in _KMER_MASKS:
        raise ValueError(f"k={k!r} unsupported")
    table = "ACGT"
    mask = _KMER_MASKS[int(k)]
    canonical &= mask
    chars: list[str] = []
    for i in range(int(k) - 1, -1, -1):
        chars.append(table[(canonical >> (2 * i)) & 0x3])
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Timing parser (mirrors kmer_count).
# --------------------------------------------------------------------------- #


def _parse_us_label(stdout: str, label: str) -> float | None:
    m = re.search(
        rf"^{label} NPU time:\s*([0-9.eE+\-]+)us\.?\s*$",
        stdout,
        re.MULTILINE,
    )
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Run info.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _RunInfo:
    """Timing + provenance for the last NPU run."""

    avg_us: float
    min_us: float
    max_us: float
    n_iters: int
    used_npu: bool
    n_records: int = 0


# --------------------------------------------------------------------------- #
# Binary blob parser — mirrors kmer_count's _parse_binary_blob shape.
# --------------------------------------------------------------------------- #


def _parse_binary_blob(path: Path) -> list[tuple[int, int]]:
    """Parse the runner's ``--output-format binary`` blob.

    Layout (little-endian):
      ``[uint64 n_records][n_records × {uint64 canonical, uint32 position, uint32 _pad}]``

    Returns ``[(canonical_u64, position_u32), ...]`` in the runner's
    sort order (position asc, canonical asc — applied host-side).
    """
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 8:
        raise NpuRunFailed(
            0, "",
            f"[bionpu] binary output blob {path} too small "
            f"({raw.size} bytes; expected >=8 for n_records header)",
        )
    n_records = int(np.frombuffer(raw[:8], dtype=np.uint64)[0])
    expected_bytes = 8 + n_records * RECORD_BYTES
    if raw.size < expected_bytes:
        raise NpuRunFailed(
            0, "",
            f"[bionpu] binary output blob {path} truncated: "
            f"n_records={n_records} but file has only "
            f"{raw.size} bytes (need {expected_bytes})",
        )
    if n_records == 0:
        return []
    records_buf = raw[8 : 8 + n_records * RECORD_BYTES]
    # View as (canonical_u64, position_u32, pad_u32) via structured dtype.
    rec_dt = np.dtype([
        ("canonical", "<u8"),
        ("position", "<u4"),
        ("_pad", "<u4"),
    ])
    arr = np.frombuffer(records_buf, dtype=rec_dt)
    return [(int(c), int(p)) for c, p in zip(arr["canonical"], arr["position"])]


# --------------------------------------------------------------------------- #
# Op class.
# --------------------------------------------------------------------------- #


class BionpuMinimizer(NpuOp):
    """Sliding-window (w, k) minimizers on AIE2P.

    Pinned (k, w) ∈ {(15, 10), (21, 11)}; n_tiles ∈ {1, 2, 4, 8};
    n_chunks_per_launch ∈ {1, 2, 4, 8}.

    Two NPU_OPS registry entries (one per supported (k, w)). The
    artifact directory is per-(k, w, n_tiles) at
    ``_npu_artifacts/bionpu_minimizer_k{k}_w{w}_n{n_tiles}/`` and
    contains a single ``final.xclbin`` + ``insts.bin`` (single-pass —
    no hash-slice partition; emit volume is low) plus one
    ``host_runner``.
    """

    INPUT_DTYPE = np.uint8
    SUPPORTED_KW = SUPPORTED_KW
    SUPPORTED_N_TILES = SUPPORTED_N_TILES
    SUPPORTED_N_CHUNKS_PER_LAUNCH = SUPPORTED_N_CHUNKS_PER_LAUNCH

    def __init__(
        self,
        k: int = 15,
        w: int = 10,
        n_tiles: int = 4,
        n_chunks_per_launch: int = 1,
    ) -> None:
        if (int(k), int(w)) not in SUPPORTED_KW:
            valid = ", ".join(f"({a},{b})" for a, b in SUPPORTED_KW)
            raise ValueError(
                f"BionpuMinimizer: (k, w)=({k!r}, {w!r}) not in "
                f"{{{valid}}}."
            )
        if int(n_tiles) not in SUPPORTED_N_TILES:
            valid = ", ".join(str(x) for x in SUPPORTED_N_TILES)
            raise ValueError(
                f"BionpuMinimizer: n_tiles={n_tiles!r} not in {{{valid}}}."
            )
        if int(n_chunks_per_launch) not in SUPPORTED_N_CHUNKS_PER_LAUNCH:
            valid = ", ".join(str(x) for x in SUPPORTED_N_CHUNKS_PER_LAUNCH)
            raise ValueError(
                f"BionpuMinimizer: n_chunks_per_launch="
                f"{n_chunks_per_launch!r} not in {{{valid}}}."
            )
        self.k: int = int(k)
        self.w: int = int(w)
        self.n_tiles: int = int(n_tiles)
        self.n_chunks_per_launch: int = int(n_chunks_per_launch)
        self.name: str = f"bionpu_minimizer_k{self.k}_w{self.w}"
        self.last_run: _RunInfo | None = None

    # ----- artifact paths ------------------------------------------------- #

    @property
    def artifact_dir(self) -> Path:
        """Per-(k, w, n_tiles) artifact directory."""
        return (
            _ART_ROOT
            / f"bionpu_minimizer_k{self.k}_w{self.w}_n{self.n_tiles}"
        )

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
        return (
            self.host_runner.exists()
            and self.xclbin.exists()
            and self.insts.exists()
        )

    # ----- dispatch routing ----------------------------------------------- #

    def _resolve_dispatch_impl(self, impl: str | None) -> str:
        return resolve_dispatch_impl(impl, env_var=MINIMIZER_DISPATCH_ENV)

    # ----- input validation ---------------------------------------------- #

    def _validate_inputs(self, packed_seq: np.ndarray) -> None:
        if not isinstance(packed_seq, np.ndarray):
            raise TypeError(
                "packed_seq must be a numpy.ndarray of dtype uint8"
            )
        if packed_seq.dtype != np.uint8:
            raise ValueError(
                f"packed_seq dtype must be uint8; got {packed_seq.dtype}"
            )
        if packed_seq.ndim != 1:
            raise ValueError(
                f"packed_seq must be 1-D; got shape {packed_seq.shape}"
            )
        # Need at least k + w - 1 bases to emit anything; require enough
        # bytes to potentially fit the first window.
        min_bases = self.k + self.w - 1
        min_bytes = (min_bases + 3) // 4
        if packed_seq.size < min_bytes:
            raise ValueError(
                f"packed_seq too short for (k={self.k}, w={self.w}): need "
                f"at least {min_bytes} bytes, got {packed_seq.size}"
            )

    # ----- subprocess path ------------------------------------------------ #

    def _run_subprocess(
        self,
        *,
        packed_seq: np.ndarray,
        top_n: int,
        n_iters: int,
        warmup: int,
        timeout_s: float,
    ) -> tuple[list[tuple[int, int]], _RunInfo]:
        """Spawn the host_runner and parse the binary --output blob.

        Per CLAUDE.md (2026-04-25 swarm): subprocess-based silicon
        submissions MUST be wrapped in
        :func:`bionpu.dispatch.npu_silicon_lock.npu_silicon_lock`. The
        op-class subprocess path enters host_runner which submits to
        ``/dev/accel/accel0``, so we wrap.
        """
        from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as tdir:
            t = Path(tdir)
            input_path = t / "packed_seq.2bit.bin"
            output_path = t / "minimizers.bin"
            np.ascontiguousarray(packed_seq).tofile(input_path)

            cmd = [
                str(self.host_runner),
                "-x", str(self.xclbin),
                "-i", str(self.insts),
                "-k", _KERNEL_NAME,
                "--input", str(input_path),
                "--output", str(output_path),
                "--output-format", "binary",
                "--k", str(self.k),
                "--w", str(self.w),
                "--top", str(int(top_n)),
                "--launch-chunks", str(self.n_tiles),
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
                    + f"\n[bionpu] runner did not produce --output {output_path}",
                )
            records = _parse_binary_blob(output_path)

        info = _RunInfo(
            avg_us=float(avg),
            min_us=float(mn),
            max_us=float(mx),
            n_iters=int(n_iters),
            used_npu=True,
            n_records=len(records),
        )
        return records, info

    # ----- entry point ---------------------------------------------------- #

    def __call__(
        self,
        *,
        packed_seq: np.ndarray,
        top_n: int = 0,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 120.0,
        _impl: str | None = None,
        **_unused: Any,
    ) -> list[tuple[int, int]]:
        """Run the kernel and return ``[(canonical_u64, position_u32), ...]``.

        Output is sorted by ``position`` ascending (matches the oracle
        convention in :mod:`bionpu.data.minimizer_oracle`). Per-position
        uniqueness is enforced by the host runner's de-duplication pass.
        ``top_n=0`` (default) returns all records; ``top_n>0`` truncates
        to the first ``top_n`` entries (still position-sorted).
        """
        self._validate_inputs(packed_seq)

        impl = self._resolve_dispatch_impl(_impl)
        if impl == "pyxrt":
            # The v0 pyxrt path is non-trivial (single xclbin + chunked
            # streaming + host-side dedup); for smoke validation the
            # subprocess path is the gate. pyxrt support tracked as a
            # follow-up.
            raise NotImplementedError(
                "BionpuMinimizer v0 pyxrt path not implemented; "
                "use BIONPU_MINIMIZER_DISPATCH=subprocess (default)."
            )

        # Subprocess path: enforce all required artifacts exist.
        if not self.host_runner.exists():
            raise NpuArtifactsMissingError(
                f"NPU artifact missing for {self.name} "
                f"(k={self.k}, w={self.w}, n_tiles={self.n_tiles}): "
                f"{self.host_runner}. Build via "
                f"`make NPU2=1 K={self.k} W={self.w} "
                f"experiment={'production' if self.n_tiles == 1 else f'wide{self.n_tiles}'} all` "
                f"in this kernel directory."
            )
        for need in (self.xclbin, self.insts):
            if not need.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name}: {need}"
                )

        records, info = self._run_subprocess(
            packed_seq=packed_seq,
            top_n=top_n,
            n_iters=n_iters,
            warmup=warmup,
            timeout_s=timeout_s,
        )

        self.last_run = info
        return records


# --------------------------------------------------------------------------- #
# Registry (2 entries, one per supported (k, w); n_tiles=4 default).
# --------------------------------------------------------------------------- #

for _k, _w in SUPPORTED_KW:
    register_npu_op(
        f"bionpu_minimizer_k{_k}_w{_w}",
        BionpuMinimizer(k=_k, w=_w, n_tiles=4),
    )
del _k, _w
