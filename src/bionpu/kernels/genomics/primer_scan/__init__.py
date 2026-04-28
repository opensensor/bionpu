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

"""Primer / adapter scan NPU op (v0).

Path B (runtime primer canonical): the primer's forward + RC canonical
uint64 pair lives in the chunk header, so a single xclbin per primer
length P handles any primer of length P at runtime.

Wire format (per ``primer_scan_constants.h`` + ``data/primer_oracle.py``):

* Input ``packed_seq``: 1-D ``np.uint8`` packed-2-bit DNA, MSB-first
  (matches kmer_count + minimizer).
* Per-chunk in-band header (24 bytes):
    bytes [0..3]:    uint32 actual_payload_bytes LE
    bytes [4..7]:    int32  owned_start_offset_bases LE
    bytes [8..15]:   uint64 primer_fwd_canonical LE
    bytes [16..23]:  uint64 primer_rc_canonical  LE
  Payload starts at byte 24.
* Per-tile output: N_TILES × 32 KiB pass-slots. Slot 0 is authoritative
  (broadcast topology; slots 1..N_TILES-1 are duplicates).
* Per-slot record (16 bytes):
    bytes [0..3]:   uint32 query_pos LE
    bytes [4]:      uint8 strand (0 = fwd, 1 = rc)
    bytes [5]:      uint8 primer_idx (0 for v0)
    bytes [6..7]:   uint16 _pad
    bytes [8..15]:  uint64 _pad2
* Output binary blob:
    [uint64 n_records LE][n × 16-byte records as above]

Three NPU_OPS registry entries (one per supported P):
    ``bionpu_primer_scan_p13``
    ``bionpu_primer_scan_p20``
    ``bionpu_primer_scan_p25``
"""

from __future__ import annotations

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
    _xrt_env,  # noqa: PLC2701 — internal helper
    register_npu_op,
)

__all__ = [
    "BionpuPrimerScan",
    "MAX_EMIT_IDX",
    "PARTIAL_OUT_BYTES_PADDED",
    "PRIMER_SCAN_DISPATCH_ENV",
    "RECORD_BYTES",
    "SEQ_IN_CHUNK_BYTES_BASE",
    "SUPPORTED_P",
    "SUPPORTED_N_CHUNKS_PER_LAUNCH",
    "SUPPORTED_N_TILES",
    "TRUSEQ_P5_ADAPTER",
]

# --------------------------------------------------------------------------- #
# Pinned constants — mirror primer_scan_constants.h byte-equal.
# --------------------------------------------------------------------------- #

#: Pinned primer lengths matching silicon artifacts.
SUPPORTED_P: tuple[int, ...] = (13, 20, 25)

#: Supported tile fan-out (n_tiles constructor arg).
SUPPORTED_N_TILES: tuple[int, ...] = (1, 2, 4, 8)

#: Supported chunks-per-launch (constructor arg).
SUPPORTED_N_CHUNKS_PER_LAUNCH: tuple[int, ...] = (1, 2, 4, 8)

#: 4096-byte primary chunk.
SEQ_IN_CHUNK_BYTES_BASE: int = 4096

#: 32 KiB per pass-slot in the joined output buffer.
PARTIAL_OUT_BYTES_PADDED: int = 32768

#: Wire record size (16 B).
RECORD_BYTES: int = 16

#: Max records per pass-slot (kernel caps at this).
MAX_EMIT_IDX: int = 2046

#: Env var selecting subprocess vs in-process pyxrt dispatch.
PRIMER_SCAN_DISPATCH_ENV: str = "BIONPU_PRIMER_SCAN_DISPATCH"

#: Kernel name baked into the xclbin (mirrors kmer_count / minimizer).
_KERNEL_NAME: str = "MLIR_AIE"

#: Re-export the canonical default primer.
TRUSEQ_P5_ADAPTER: str = "AGATCGGAAGAGC"

# --------------------------------------------------------------------------- #
# Artifact root — same _npu_artifacts/ directory as every other kernel.
# --------------------------------------------------------------------------- #

_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)


# --------------------------------------------------------------------------- #
# Timing parser (mirrors kmer_count / minimizer).
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
# Binary blob parser — matches the runner's --output-format binary layout.
# --------------------------------------------------------------------------- #


def _parse_binary_blob(path: Path) -> list[tuple[int, int]]:
    """Parse the runner's ``--output-format binary`` blob.

    Layout (little-endian):
      ``[uint64 n_records][n_records × 16-byte record]``
    where each record is:
      ``uint32 position | uint8 strand | uint8 primer_idx | uint16 _pad |
        uint64 _pad2``

    Returns ``[(query_pos, strand), ...]`` in the runner's sort order
    (position asc, strand asc — applied host-side).
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
    rec_dt = np.dtype([
        ("position", "<u4"),
        ("strand", "u1"),
        ("primer_idx", "u1"),
        ("_pad", "<u2"),
        ("_pad2", "<u8"),
    ])
    arr = np.frombuffer(records_buf, dtype=rec_dt)
    return [(int(p), int(s)) for p, s in zip(arr["position"], arr["strand"])]


# --------------------------------------------------------------------------- #
# Op class.
# --------------------------------------------------------------------------- #


class BionpuPrimerScan(NpuOp):
    """Primer / adapter exact-match scan on AIE2P (v0).

    Pinned P ∈ {13, 20, 25}; n_tiles ∈ {1, 2, 4, 8};
    n_chunks_per_launch ∈ {1, 2, 4, 8}.

    Three NPU_OPS registry entries (one per supported P).

    Artifact directory layout:
      - n_chunks_per_launch=1: ``_npu_artifacts/bionpu_primer_scan_p{P}_n{n_tiles}/``
      - n_chunks_per_launch>1: ``_npu_artifacts/bionpu_primer_scan_p{P}_n{n_tiles}_b{N}/``

    Each artifact dir contains ``final.xclbin``, ``insts.bin``, and
    ``host_runner``. (No multi-pass fan-out for v0; emit volume is far
    below MAX_EMIT_IDX so a single xclbin per (P, n_tiles) suffices.)
    """

    INPUT_DTYPE = np.uint8
    SUPPORTED_P = SUPPORTED_P
    SUPPORTED_N_TILES = SUPPORTED_N_TILES
    SUPPORTED_N_CHUNKS_PER_LAUNCH = SUPPORTED_N_CHUNKS_PER_LAUNCH

    def __init__(
        self,
        primer: str = TRUSEQ_P5_ADAPTER,
        n_tiles: int = 4,
        n_chunks_per_launch: int = 1,
    ) -> None:
        if not isinstance(primer, str) or not primer:
            raise ValueError(
                f"BionpuPrimerScan: primer must be a non-empty ACGT string; "
                f"got {primer!r}"
            )
        primer_upper = primer.upper()
        for ch in primer_upper:
            if ch not in "ACGT":
                raise ValueError(
                    f"BionpuPrimerScan: non-ACGT base {ch!r} in primer "
                    f"{primer!r}; only ACGT is supported in v0."
                )
        p_len = len(primer_upper)
        if p_len not in SUPPORTED_P:
            valid = ", ".join(str(x) for x in SUPPORTED_P)
            raise ValueError(
                f"BionpuPrimerScan: primer length P={p_len} not in "
                f"{{{valid}}}."
            )
        if int(n_tiles) not in SUPPORTED_N_TILES:
            valid = ", ".join(str(x) for x in SUPPORTED_N_TILES)
            raise ValueError(
                f"BionpuPrimerScan: n_tiles={n_tiles!r} not in {{{valid}}}."
            )
        if int(n_chunks_per_launch) not in SUPPORTED_N_CHUNKS_PER_LAUNCH:
            valid = ", ".join(str(x) for x in SUPPORTED_N_CHUNKS_PER_LAUNCH)
            raise ValueError(
                f"BionpuPrimerScan: n_chunks_per_launch="
                f"{n_chunks_per_launch!r} not in {{{valid}}}."
            )
        self.primer: str = primer_upper
        self.p: int = p_len
        self.n_tiles: int = int(n_tiles)
        self.n_chunks_per_launch: int = int(n_chunks_per_launch)
        self.name: str = f"bionpu_primer_scan_p{self.p}"
        self.last_run: _RunInfo | None = None

    # ----- artifact paths ------------------------------------------------- #

    @property
    def artifact_dir(self) -> Path:
        """Per-(P, n_tiles[, n_chunks_per_launch]) artifact directory."""
        base = f"bionpu_primer_scan_p{self.p}_n{self.n_tiles}"
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

    # ----- dispatch routing ----------------------------------------------- #

    def _resolve_dispatch_impl(self, impl: str | None) -> str:
        return resolve_dispatch_impl(impl, env_var=PRIMER_SCAN_DISPATCH_ENV)

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
        # Need at least P bases to emit anything.
        min_bases = self.p
        min_bytes = (min_bases + 3) // 4
        if packed_seq.size < min_bytes:
            raise ValueError(
                f"packed_seq too short for P={self.p}: need at least "
                f"{min_bytes} bytes, got {packed_seq.size}"
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

        Per CLAUDE.md: subprocess-based silicon submissions MUST be
        wrapped in :func:`bionpu.dispatch.npu_silicon_lock.npu_silicon_lock`.
        """
        from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as tdir:
            t = Path(tdir)
            input_path = t / "packed_seq.2bit.bin"
            output_path = t / "primer_hits.bin"
            np.ascontiguousarray(packed_seq).tofile(input_path)

            cmd = [
                str(self.host_runner),
                "-x", str(self.xclbin),
                "-i", str(self.insts),
                "-k", _KERNEL_NAME,
                "--input", str(input_path),
                "--output", str(output_path),
                "--output-format", "binary",
                "--p", str(self.p),
                "--primer-ascii", self.primer,
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
        timeout_s: float = 600.0,
        _impl: str | None = None,
        **_unused: Any,
    ) -> list[tuple[int, int]]:
        """Run the kernel and return ``[(query_pos, strand), ...]``.

        Output is sorted by ``(position asc, strand asc)`` (matches the
        oracle's convention in
        :mod:`bionpu.data.primer_oracle`). ``top_n=0`` (default) returns
        all records; ``top_n>0`` truncates to the first ``top_n``.
        """
        self._validate_inputs(packed_seq)

        impl = self._resolve_dispatch_impl(_impl)
        if impl == "pyxrt":
            raise NotImplementedError(
                "BionpuPrimerScan pyxrt path not implemented; "
                "use BIONPU_PRIMER_SCAN_DISPATCH=subprocess (default)."
            )

        if not self.host_runner.exists():
            raise NpuArtifactsMissingError(
                f"NPU artifact missing for {self.name} "
                f"(P={self.p}, n_tiles={self.n_tiles}): "
                f"{self.host_runner}. Build via "
                f"`make NPU2=1 P={self.p} "
                f"experiment={'production' if self.n_tiles == 1 else f'wide{self.n_tiles}'} "
                f"all` in this kernel directory."
            )
        for need in (self.xclbin, self.insts):
            if not need.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing: {need}"
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
# Registry (3 entries, one per supported P; n_tiles=4 default).
# --------------------------------------------------------------------------- #

for _p in SUPPORTED_P:
    # Pick a default ACGT primer of the right length so the registry-
    # resident instance can be constructed. Callers re-instantiate via
    # ``get_primer_scan_op`` with their actual primer.
    _default_primer = "A" * _p
    register_npu_op(
        f"bionpu_primer_scan_p{_p}",
        BionpuPrimerScan(primer=_default_primer, n_tiles=4),
    )
del _p, _default_primer
