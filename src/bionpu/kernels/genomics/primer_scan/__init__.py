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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.dispatch._pyxrt_helpers import resolve_dispatch_impl
from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    _get_pyxrt,  # noqa: PLC2701 — used by _run_pyxrt
    _xrt_env,  # noqa: PLC2701 — internal helper
    default_backend,
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
# Per-dispatch profiling (v2 pyxrt path).
# --------------------------------------------------------------------------- #


@dataclass
class _PyxrtPhaseProfile:
    """Cumulative per-phase timing for the pyxrt path (debug/profiling)."""

    stage_us: float = 0.0          # build header + memcpy payload + bo.write
    sync_to_us: float = 0.0        # bo.sync(TO_DEVICE) on seq_in
    kernel_run_us: float = 0.0     # kernel(...) launch (non-blocking)
    wait_us: float = 0.0           # run.wait()
    sync_from_us: float = 0.0      # bo.sync(FROM_DEVICE) on sparse_out
    parse_us: float = 0.0          # bo.read + tile-0 parse
    n_dispatches: int = 0


# --------------------------------------------------------------------------- #
# Primer canonical encoding (mirrors runner.cpp::encode_primer_ascii).
# --------------------------------------------------------------------------- #


_BASE_TO_2BIT: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode_primer_canon(primer: str) -> tuple[int, int]:
    """Return ``(fwd_canonical_u64, rc_canonical_u64)`` for an ACGT primer.

    Byte-equal to ``runner.cpp::encode_primer_ascii``: each base in
    ``primer`` contributes 2 bits with the first base in the highest
    2-bit lane.
    """
    p = len(primer)
    if not 1 <= p <= 32:
        raise ValueError(f"primer length must be 1..32; got {p}")
    fwd = 0
    for ch in primer:
        v = _BASE_TO_2BIT.get(ch.upper())
        if v is None:
            raise ValueError(f"non-ACGT base in primer: {ch!r}")
        fwd = (fwd << 2) | v
    mask = (~0 & ((1 << 64) - 1)) if p == 32 else ((1 << (2 * p)) - 1)
    fwd &= mask
    comp = fwd ^ mask
    rc = 0
    for _ in range(p):
        rc = (rc << 2) | (comp & 0x3)
        comp >>= 2
    return fwd, rc


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

    # Per-instance pyxrt cache slot.
    #
    # Populated lazily on the first ``_run_pyxrt`` call; subsequent calls
    # reuse the cached BO ring + kernel handle. Stored as an instance
    # attribute (not a class attribute) so multiple op-class instances
    # bound to the same artifact don't share BOs (avoids accidental
    # cross-instance race in the rare multi-op case).
    _pyxrt_state: dict[str, Any] | None = None

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
        # Populated on first _run_pyxrt invocation; see _ensure_pyxrt_state.
        self._pyxrt_state = None
        # Cumulative per-phase profile (cleared via reset_pyxrt_profile).
        self._pyxrt_profile: _PyxrtPhaseProfile = _PyxrtPhaseProfile()

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

    # ----- in-process pyxrt path (v2) ------------------------------------- #

    # Kernel arg slots for primer_scan runner.cpp:
    #   arg0=opcode, arg1=bo_instr, arg2=instr_size_uint,
    #   arg3=bo_seq_in, arg4=bo_sparse_out, arg5=n_input_bytes_int.
    _ARG_SEQ_IN = 3
    _ARG_SPARSE_OUT = 4

    def _ensure_pyxrt_state(self) -> dict[str, Any]:
        """Lazily build + cache the pyxrt device + kernel + BO state.

        First call: opens the device (via the process-level
        :func:`default_backend`), loads + caches the xclbin, allocates the
        bo_instr + ring of (bo_seq_in, bo_sparse_out). Subsequent calls
        return the cached dict in O(1).

        The XRT handle lifetime is tied to the op-class instance; the
        bo_instr is uploaded exactly once. This is what amortises away the
        ~100 ms subprocess fork + xclbin-register cost that bottlenecked
        v1 — successive ``_run_pyxrt`` calls reuse all of the above and
        only pay BO write + sync + run.wait per dispatch (3-5 ms each).
        """
        if self._pyxrt_state is not None:
            return self._pyxrt_state

        for need in (self.xclbin, self.insts):
            if not need.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing: {need}"
                )

        pyxrt = _get_pyxrt()
        backend = default_backend()
        loaded = backend.load_xclbin(
            self.xclbin, self.insts, kernel_name=_KERNEL_NAME
        )
        device = backend.device()
        kernel = loaded.kernel
        instr_u32 = loaded.instr_u32

        # Per-batch buffer geometry (mirrors runner.cpp).
        seq_in_slot_bytes = SEQ_IN_CHUNK_BYTES_BASE + 8  # PS_OVERLAP_BYTES=8
        n_batch = self.n_chunks_per_launch
        per_batch_seq_in_bytes = n_batch * seq_in_slot_bytes
        per_dispatch_out_bytes = (
            n_batch * self.n_tiles * PARTIAL_OUT_BYTES_PADDED
        )

        # Upload instruction stream once.
        bo_instr = pyxrt.bo(
            device,
            int(instr_u32.nbytes),
            pyxrt.bo.cacheable,
            kernel.group_id(1),
        )
        bo_instr.write(instr_u32.tobytes(), 0)
        bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Single-slot ring (synchronous dispatch). The runner uses RING=4
        # for pipelined dispatch; for v2 we keep the BO model simple and
        # synchronous so byte-equality vs subprocess is straightforward.
        # Per-dispatch silicon time is ~110-130 us measured in v1; the
        # 100 ms subprocess overhead disappears entirely with one BO ring.
        bo_seq_in = pyxrt.bo(
            device,
            per_batch_seq_in_bytes,
            pyxrt.bo.host_only,
            kernel.group_id(self._ARG_SEQ_IN),
        )
        bo_sparse_out = pyxrt.bo(
            device,
            per_dispatch_out_bytes,
            pyxrt.bo.host_only,
            kernel.group_id(self._ARG_SPARSE_OUT),
        )

        # Pre-encode the primer canonical pair once per op-class lifetime
        # (the primer is fixed in the constructor).
        fwd_canon, rc_canon = _encode_primer_canon(self.primer)

        state: dict[str, Any] = {
            "pyxrt": pyxrt,
            "device": device,
            "kernel": kernel,
            "bo_instr": bo_instr,
            "bo_seq_in": bo_seq_in,
            "bo_sparse_out": bo_sparse_out,
            "instr_size": int(instr_u32.size),
            "seq_in_slot_bytes": seq_in_slot_bytes,
            "per_batch_seq_in_bytes": per_batch_seq_in_bytes,
            "per_dispatch_out_bytes": per_dispatch_out_bytes,
            "n_batch": n_batch,
            "fwd_canon": fwd_canon,
            "rc_canon": rc_canon,
        }
        self._pyxrt_state = state
        return state

    def reset_pyxrt_profile(self) -> None:
        """Zero the cumulative per-phase profile (used by benchmarks)."""
        self._pyxrt_profile = _PyxrtPhaseProfile()

    @property
    def pyxrt_profile(self) -> _PyxrtPhaseProfile:
        """Cumulative per-phase profile from the last sequence of pyxrt
        dispatches. Reset via :meth:`reset_pyxrt_profile`.
        """
        return self._pyxrt_profile

    def _plan_chunks(
        self, n_input_bytes: int
    ) -> list[tuple[int, int]]:
        """Mirror of runner.cpp::plan_chunks.

        Returns ``[(src_offset, payload_bytes), ...]``.
        """
        if n_input_bytes == 0:
            return []
        overlap_bytes = 8  # PS_OVERLAP_BYTES
        total_chunk = SEQ_IN_CHUNK_BYTES_BASE + overlap_bytes
        # The kernel's seq_in slot is total_chunk (4104) bytes; payload
        # capacity inside the slot is total_chunk - 24 (header).
        payload_cap = total_chunk - 24
        advance = payload_cap - overlap_bytes
        plan: list[tuple[int, int]] = []
        off = 0
        while off < n_input_bytes:
            end = min(off + payload_cap, n_input_bytes)
            plan.append((off, end - off))
            if end >= n_input_bytes:
                break
            off += advance
        return plan

    @staticmethod
    def _parse_tile0_blob(
        blob: np.ndarray, chunk_global_base_bases: int
    ) -> list[tuple[int, int]]:
        """Parse one tile-0 32 KiB output slot into ``(global_pos, strand)``.

        Mirrors runner.cpp::parse_chunk_tile0. ``blob`` is a uint8 view
        of length ``PARTIAL_OUT_BYTES_PADDED``.
        """
        emit_count = int(np.frombuffer(blob[:4], dtype=np.uint32)[0])
        if emit_count > MAX_EMIT_IDX:
            emit_count = MAX_EMIT_IDX
        if emit_count == 0:
            return []
        # Records start at byte 4; each record is 16 B.
        rec_dt = np.dtype([
            ("position", "<u4"),
            ("strand", "u1"),
            ("primer_idx", "u1"),
            ("_pad", "<u2"),
            ("_pad2", "<u8"),
        ])
        recs = np.frombuffer(
            blob[4 : 4 + emit_count * RECORD_BYTES], dtype=rec_dt
        )
        return [
            (int(p) + chunk_global_base_bases, int(s))
            for p, s in zip(recs["position"], recs["strand"])
        ]

    def _run_pyxrt(
        self,
        *,
        packed_seq: np.ndarray,
        top_n: int = 0,
        timeout_s: float = 60.0,
    ) -> tuple[list[tuple[int, int]], _RunInfo]:
        """In-process pyxrt dispatch path (v2).

        Loads xclbin + insts.bin once per op-class lifetime; reuses the
        device + kernel + bo_instr + bo_seq_in + bo_sparse_out across
        all subsequent ``__call__`` invocations. Per-dispatch wall is
        bounded by BO write + sync + kernel-run + sync, single-digit ms.

        Per CLAUDE.md: in-process pyxrt path takes only the in-process
        ``_dispatch_lock`` (a no-op nullcontext at this point in the
        codebase) — does NOT wrap ``npu_silicon_lock``. Silicon-level
        multi-use protection handles same-process concurrency.

        Args:
            packed_seq: 1-D uint8 packed-2-bit input.
            top_n: as in :meth:`__call__` (0 = all).
            timeout_s: per-dispatch wait timeout (currently advisory;
                pyxrt's ``run.wait()`` is bounded by the driver's TDR).

        Returns:
            ``([(query_pos, strand), ...], _RunInfo)``.
        """
        del timeout_s  # advisory only; driver TDR enforces an upper bound.
        state = self._ensure_pyxrt_state()
        pyxrt = state["pyxrt"]
        kernel = state["kernel"]
        bo_instr = state["bo_instr"]
        bo_seq_in = state["bo_seq_in"]
        bo_sparse_out = state["bo_sparse_out"]
        instr_size = state["instr_size"]
        seq_in_slot_bytes = state["seq_in_slot_bytes"]
        per_batch_seq_in_bytes = state["per_batch_seq_in_bytes"]
        per_dispatch_out_bytes = state["per_dispatch_out_bytes"]
        n_batch = state["n_batch"]
        fwd_canon = state["fwd_canon"]
        rc_canon = state["rc_canon"]
        n_tiles = self.n_tiles
        p = self.p

        # Zero output (tile-0 emit_count prefix is the gate; non-zeroed
        # bytes are still safe because the prefix bounds reads). We zero
        # anyway for parity with the runner.
        zero_out = bytes(per_dispatch_out_bytes)

        # Chunk plan over the entire input.
        input_buf = np.ascontiguousarray(packed_seq, dtype=np.uint8)
        chunks = self._plan_chunks(int(input_buf.size))

        prof = self._pyxrt_profile
        all_records: list[tuple[int, int]] = []
        timings_us: list[float] = []
        avg_opcode = 3
        n_chunks = len(chunks)
        if n_chunks == 0:
            info = _RunInfo(0.0, 0.0, 0.0, 0, used_npu=True, n_records=0)
            return [], info

        n_batches = (n_chunks + n_batch - 1) // n_batch

        # Pre-build header template bytes for primer canonical (constant
        # across all dispatches for this op-class instance).
        primer_canon_bytes = (
            int(fwd_canon).to_bytes(8, "little", signed=False)
            + int(rc_canon).to_bytes(8, "little", signed=False)
        )

        for bi in range(n_batches):
            batch_first = bi * n_batch
            # ---------------- stage seq_in ----------------
            t_stage0 = time.perf_counter()
            seq_in_buf = bytearray(per_batch_seq_in_bytes)  # zeroed
            for slot in range(n_batch):
                ci = batch_first + slot
                if ci >= n_chunks:
                    break  # tail-pad: leave zeros
                src_off, payload = chunks[ci]
                base = slot * seq_in_slot_bytes
                # bytes [0..3]: actual_payload_bytes LE (uint32)
                seq_in_buf[base + 0 : base + 4] = int(payload).to_bytes(
                    4, "little", signed=False
                )
                # bytes [4..7]: owned_start_offset_bases LE (int32)
                if ci == 0:
                    owned = 0
                else:
                    owned = (8 * 4) - (p - 1)  # overlap_bytes*4 - (P-1)
                seq_in_buf[base + 4 : base + 8] = int(owned).to_bytes(
                    4, "little", signed=True
                )
                # bytes [8..23]: primer_fwd_canon | primer_rc_canon
                seq_in_buf[base + 8 : base + 24] = primer_canon_bytes
                # bytes [24..24+payload]: payload
                seq_in_buf[base + 24 : base + 24 + payload] = (
                    input_buf[src_off : src_off + payload].tobytes()
                )
            bo_seq_in.write(bytes(seq_in_buf), 0)
            t_stage1 = time.perf_counter()
            prof.stage_us += 1e6 * (t_stage1 - t_stage0)

            # ---------------- sync to device ----------------
            bo_seq_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            # We also zero the output BO host-side and push so stale data
            # from the prior dispatch can't leak through.
            bo_sparse_out.write(zero_out, 0)
            bo_sparse_out.sync(
                pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            )
            t_sync_to1 = time.perf_counter()
            prof.sync_to_us += 1e6 * (t_sync_to1 - t_stage1)

            # ---------------- launch ----------------
            t_launch0 = time.perf_counter()
            run = kernel(
                avg_opcode,
                bo_instr,
                int(instr_size),
                bo_seq_in,
                bo_sparse_out,
                int(per_batch_seq_in_bytes),
            )
            t_launch1 = time.perf_counter()
            prof.kernel_run_us += 1e6 * (t_launch1 - t_launch0)

            # ---------------- wait ----------------
            state_v = run.wait()
            t_wait1 = time.perf_counter()
            prof.wait_us += 1e6 * (t_wait1 - t_launch1)
            # Only fail loud if we have a clearly-not-COMPLETED state.
            try:
                ok = (
                    str(state_v).endswith("COMPLETED")
                    or getattr(state_v, "name", "")
                    == "ERT_CMD_STATE_COMPLETED"
                )
            except Exception:  # noqa: BLE001
                ok = True
            if not ok:
                raise NpuRunFailed(
                    int(getattr(state_v, "value", -1)),
                    "",
                    f"kernel state={state_v!r} (expected COMPLETED) "
                    f"in pyxrt path on batch {bi}/{n_batches}",
                )

            # ---------------- sync from + parse ----------------
            bo_sparse_out.sync(
                pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            )
            t_sync_from1 = time.perf_counter()
            prof.sync_from_us += 1e6 * (t_sync_from1 - t_wait1)

            # Read raw bytes once per dispatch; subset to tile-0 of each
            # slot. We treat the read as one numpy array view to avoid
            # the int8-to-uint8 round-trip overhead.
            raw = bo_sparse_out.read(int(per_dispatch_out_bytes), 0)
            raw_np = np.asarray(raw, dtype=np.int8).view(np.uint8)

            per_chunk_block = n_tiles * PARTIAL_OUT_BYTES_PADDED
            for slot in range(n_batch):
                ci = batch_first + slot
                if ci >= n_chunks:
                    break
                src_off, _ = chunks[ci]
                base = slot * per_chunk_block
                # Tile 0 only (broadcast topology; slots 1..n_tiles-1 are
                # duplicates, exactly as runner.cpp ignores them).
                tile0 = raw_np[base : base + PARTIAL_OUT_BYTES_PADDED]
                global_base_bases = src_off * 4
                all_records.extend(
                    self._parse_tile0_blob(tile0, global_base_bases)
                )
            t_parse1 = time.perf_counter()
            prof.parse_us += 1e6 * (t_parse1 - t_sync_from1)
            prof.n_dispatches += 1
            timings_us.append(1e6 * (t_parse1 - t_stage0))

        # Sort + dedup (mirrors runner.cpp's std::sort + std::unique by
        # (position, strand)). Without this our output order would differ
        # from the subprocess path on multi-batch inputs (where chunks are
        # processed in order but multiple chunks may emit at the same
        # global position via the overlap region).
        all_records.sort()
        # Dedup adjacent equal records.
        deduped: list[tuple[int, int]] = []
        prev: tuple[int, int] | None = None
        for r in all_records:
            if r != prev:
                deduped.append(r)
                prev = r

        if top_n > 0 and len(deduped) > top_n:
            deduped = deduped[:top_n]

        avg_us = float(sum(timings_us) / max(len(timings_us), 1))
        min_us = float(min(timings_us)) if timings_us else 0.0
        max_us = float(max(timings_us)) if timings_us else 0.0
        info = _RunInfo(
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=len(timings_us),
            used_npu=True,
            n_records=len(deduped),
        )
        return deduped, info

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
            # v2 in-process path. xclbin + insts are required; the
            # subprocess host_runner is NOT (it's a build-time artifact
            # consumed by the subprocess path only).
            for need in (self.xclbin, self.insts):
                if not need.exists():
                    raise NpuArtifactsMissingError(
                        f"NPU artifact missing for {self.name} "
                        f"(P={self.p}, n_tiles={self.n_tiles}): {need}. "
                        f"Build via `make NPU2=1 P={self.p} "
                        f"experiment={'production' if self.n_tiles == 1 else f'wide{self.n_tiles}'} "
                        f"all` in this kernel directory."
                    )
            try:
                records, info = self._run_pyxrt(
                    packed_seq=packed_seq,
                    top_n=top_n,
                    timeout_s=timeout_s,
                )
            except ImportError:
                # pyxrt not available — fall back to subprocess so the
                # caller still gets a result rather than a hard failure.
                # This is the documented escape hatch for hosts that
                # can't run in-process pyxrt for whatever reason.
                impl = "subprocess"
            else:
                self.last_run = info
                return records

        # Subprocess path — same as v0/v1.
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
