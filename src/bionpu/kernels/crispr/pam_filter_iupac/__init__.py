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

"""IUPAC PAM filter NPU op (Track A v0 — base editor design).

Single xclbin handles every Cas9 PAM variant via runtime header args
(pam_mask + pam_length live in the per-chunk 24-byte header). This is
the silicon counterpart to :mod:`bionpu.data.pam_iupac_oracle`.

Wire format mirrors primer_scan v0:

* Input ``packed_seq``: 1-D ``np.uint8`` packed-2-bit DNA, MSB-first.
* Per-chunk in-band header (24 bytes):
    bytes [0..3]:    uint32 pam_mask LE   (packed 4-bit IUPAC nibbles)
    bytes [4]:       uint8  pam_length
    bytes [5..7]:    padding
    bytes [8..15]:   uint64 _reserved (zero)
    bytes [16..19]:  uint32 actual_payload_bytes LE
    bytes [20..23]:  int32  owned_start_offset_bases LE
  Payload starts at byte 24.
* Per-tile output: N_TILES × 32 KiB pass-slots. Slot 0 authoritative.
* Per-slot record (16 bytes):
    bytes [0..3]:   uint32 query_pos LE
    bytes [4]:      uint8 strand (0 = fwd; v0 emits only fwd)
    bytes [5]:      uint8 _pad
    bytes [6..7]:   uint16 _pad2
    bytes [8..15]:  uint64 _pad3

NPU_OPS registry: ``bionpu_pam_filter_iupac``.
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
    "BionpuPamFilterIupac",
    "MAX_EMIT_IDX",
    "PAM_FILTER_IUPAC_DISPATCH_ENV",
    "PARTIAL_OUT_BYTES_PADDED",
    "RECORD_BYTES",
    "SEQ_IN_CHUNK_BYTES_BASE",
    "SUPPORTED_N_CHUNKS_PER_LAUNCH",
    "SUPPORTED_N_TILES",
]

# --------------------------------------------------------------------------- #
# Pinned constants (mirror pam_filter_iupac_constants.h byte-equal).
# --------------------------------------------------------------------------- #

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
PAM_FILTER_IUPAC_DISPATCH_ENV: str = "BIONPU_PAM_FILTER_IUPAC_DISPATCH"

#: Kernel name baked into the xclbin.
_KERNEL_NAME: str = "MLIR_AIE"

#: Maximum supported PAM length (must match constants.h PFI_PAM_LEN_MAX).
PAM_LEN_MAX: int = 8

#: Per-chunk overlap, must match runner.cpp's PFI_OVERLAP_BYTES.
PFI_OVERLAP_BYTES: int = 8

#: Header size (bytes), must match runner.cpp's HEADER_BYTES.
PFI_HEADER_BYTES: int = 24

# --------------------------------------------------------------------------- #
# Artifact root.
# --------------------------------------------------------------------------- #

_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)


# --------------------------------------------------------------------------- #
# Timing parser.
# --------------------------------------------------------------------------- #


def _parse_us_label(stdout: str, label: str) -> float | None:
    m = re.search(
        rf"^{label} NPU time:\s*([0-9.eE+\-]+)us\.?\s*$",
        stdout,
        re.MULTILINE,
    )
    return float(m.group(1)) if m else None


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
# Per-dispatch profiling (v1 pyxrt path) — mirrors primer_scan v2.
# --------------------------------------------------------------------------- #


@dataclass
class _PyxrtPhaseProfile:
    """Cumulative per-phase timing for the pyxrt path (debug/profiling)."""

    stage_us: float = 0.0       # build header + memcpy payload + bo.write
    sync_to_us: float = 0.0     # bo.sync(TO_DEVICE) on seq_in + zero output
    kernel_run_us: float = 0.0  # kernel(...) launch (non-blocking)
    wait_us: float = 0.0        # run.wait()
    sync_from_us: float = 0.0   # bo.sync(FROM_DEVICE) on sparse_out
    parse_us: float = 0.0       # bo.read + tile-0 parse
    n_dispatches: int = 0


# --------------------------------------------------------------------------- #
# IUPAC PAM encoder (mirrors runner.cpp::encode_pam_iupac).
# --------------------------------------------------------------------------- #


_IUPAC_NIBBLE: dict[str, int] = {
    "A": 0x1, "C": 0x2, "G": 0x4, "T": 0x8,
    "R": 0x5, "Y": 0xA, "S": 0x6, "W": 0x9,
    "K": 0xC, "M": 0x3, "B": 0xE, "D": 0xD,
    "H": 0xB, "V": 0x7, "N": 0xF,
}


def _encode_pam_iupac_runtime(pam: str) -> tuple[int, int]:
    """Return ``(pam_mask_u32, pam_length)`` for an IUPAC PAM string.

    Byte-equal to ``runner.cpp::encode_pam_iupac``.
    """
    p = len(pam)
    if not 1 <= p <= PAM_LEN_MAX:
        raise ValueError(
            f"PAM length must be 1..{PAM_LEN_MAX}; got {p}"
        )
    mask = 0
    for i, ch in enumerate(pam):
        nib = _IUPAC_NIBBLE.get(ch.upper())
        if nib is None:
            raise ValueError(f"non-IUPAC PAM base: {ch!r}")
        mask |= (nib & 0xF) << (4 * i)
    return mask, p


# --------------------------------------------------------------------------- #
# Binary blob parser.
# --------------------------------------------------------------------------- #


def _parse_binary_blob(path: Path) -> list[tuple[int, int]]:
    """Parse the runner's ``--output-format binary`` blob.

    Layout (little-endian):
      ``[uint64 n_records][n_records × 16-byte record]``
    where each record is:
      ``uint32 position | uint8 strand | uint8 _pad | uint16 _pad2 |
        uint64 _pad3``.
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
        ("_pad", "u1"),
        ("_pad2", "<u2"),
        ("_pad3", "<u8"),
    ])
    arr = np.frombuffer(records_buf, dtype=rec_dt)
    return [(int(p), int(s)) for p, s in zip(arr["position"], arr["strand"])]


# --------------------------------------------------------------------------- #
# Op class.
# --------------------------------------------------------------------------- #


class BionpuPamFilterIupac(NpuOp):
    """Multi-PAM IUPAC filter on AIE2P (Track A v0).

    A single xclbin per (n_tiles, n_chunks_per_launch) cell handles
    every Cas9 variant via runtime PAM-mask header. Construct with the
    PAM template (e.g. ``"NGG"``, ``"NRN"``, ``"NNNRRT"``) and dispatch
    with ``__call__(packed_seq=...)``.

    Subprocess dispatch path only in v0 (mirrors primer_scan v0). The
    in-process pyxrt path is the same shape as primer_scan v0; v1 will
    add it.
    """

    INPUT_DTYPE = np.uint8
    SUPPORTED_N_TILES = SUPPORTED_N_TILES
    SUPPORTED_N_CHUNKS_PER_LAUNCH = SUPPORTED_N_CHUNKS_PER_LAUNCH

    # Per-instance pyxrt cache slot — populated lazily on first
    # ``_run_pyxrt`` call. See :meth:`_ensure_pyxrt_state` for the
    # cache contract; mirrors primer_scan v2's pattern verbatim.
    _pyxrt_state: dict[str, Any] | None = None

    def __init__(
        self,
        pam: str = "NGG",
        n_tiles: int = 4,
        n_chunks_per_launch: int = 1,
    ) -> None:
        if not isinstance(pam, str) or not pam:
            raise ValueError(
                f"BionpuPamFilterIupac: pam must be a non-empty IUPAC "
                f"string; got {pam!r}"
            )
        pam_upper = pam.upper()
        for ch in pam_upper:
            if ch not in "ACGTRYSWKMBDHVN":
                raise ValueError(
                    f"BionpuPamFilterIupac: non-IUPAC base {ch!r} in PAM "
                    f"{pam!r}"
                )
        if not 1 <= len(pam_upper) <= PAM_LEN_MAX:
            raise ValueError(
                f"BionpuPamFilterIupac: PAM length must be 1..{PAM_LEN_MAX}; "
                f"got {len(pam_upper)}"
            )
        if int(n_tiles) not in SUPPORTED_N_TILES:
            valid = ", ".join(str(x) for x in SUPPORTED_N_TILES)
            raise ValueError(
                f"BionpuPamFilterIupac: n_tiles={n_tiles!r} not in "
                f"{{{valid}}}."
            )
        if int(n_chunks_per_launch) not in SUPPORTED_N_CHUNKS_PER_LAUNCH:
            valid = ", ".join(str(x) for x in SUPPORTED_N_CHUNKS_PER_LAUNCH)
            raise ValueError(
                f"BionpuPamFilterIupac: n_chunks_per_launch="
                f"{n_chunks_per_launch!r} not in {{{valid}}}."
            )
        self.pam: str = pam_upper
        self.n_tiles: int = int(n_tiles)
        self.n_chunks_per_launch: int = int(n_chunks_per_launch)
        self.name: str = "bionpu_pam_filter_iupac"
        self.last_run: _RunInfo | None = None
        # Populated on first _run_pyxrt invocation; see _ensure_pyxrt_state.
        self._pyxrt_state = None
        # Cumulative per-phase profile (cleared via reset_pyxrt_profile).
        self._pyxrt_profile: _PyxrtPhaseProfile = _PyxrtPhaseProfile()

    # ----- artifact paths ------------------------------------------------- #

    @property
    def artifact_dir(self) -> Path:
        base = f"bionpu_pam_filter_iupac_n{self.n_tiles}"
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
            output_path = t / "pam_hits.bin"
            np.ascontiguousarray(packed_seq).tofile(input_path)

            cmd = [
                str(self.host_runner),
                "-x", str(self.xclbin),
                "-i", str(self.insts),
                "-k", _KERNEL_NAME,
                "--input", str(input_path),
                "--output", str(output_path),
                "--output-format", "binary",
                "--pam", self.pam,
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

    # ----- in-process pyxrt path (v1) ------------------------------------- #

    # Kernel arg slots for pam_filter_iupac runner.cpp:
    #   arg0=opcode, arg1=bo_instr, arg2=instr_size_uint,
    #   arg3=bo_seq_in, arg4=bo_sparse_out, arg5=n_input_bytes_int.
    _ARG_SEQ_IN = 3
    _ARG_SPARSE_OUT = 4

    def _resolve_dispatch_impl(self, impl: str | None) -> str:
        return resolve_dispatch_impl(impl, env_var=PAM_FILTER_IUPAC_DISPATCH_ENV)

    def _ensure_pyxrt_state(self) -> dict[str, Any]:
        """Lazily build + cache the pyxrt device + kernel + BO state.

        Mirrors :class:`bionpu.kernels.genomics.primer_scan.BionpuPrimerScan`
        v2: opens the device once via :func:`default_backend`, loads +
        caches the xclbin, allocates bo_instr + a single (bo_seq_in,
        bo_sparse_out) ring slot. Subsequent calls return the cached
        dict in O(1).
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
        seq_in_slot_bytes = SEQ_IN_CHUNK_BYTES_BASE + PFI_OVERLAP_BYTES
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

        # Single-slot ring (synchronous dispatch). Per-dispatch silicon
        # time is sub-millisecond; the ~100 ms subprocess overhead
        # disappears entirely with one BO ring.
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

        # Pre-encode the PAM mask + length once per op-class lifetime
        # (the PAM is fixed in the constructor).
        pam_mask, pam_length = _encode_pam_iupac_runtime(self.pam)

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
            "pam_mask": pam_mask,
            "pam_length": pam_length,
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

    def _plan_chunks(self, n_input_bytes: int) -> list[tuple[int, int]]:
        """Mirror of runner.cpp::plan_chunks.

        Returns ``[(src_offset, payload_bytes), ...]``.
        """
        if n_input_bytes == 0:
            return []
        total_chunk = SEQ_IN_CHUNK_BYTES_BASE + PFI_OVERLAP_BYTES
        payload_cap = total_chunk - PFI_HEADER_BYTES
        advance = payload_cap - PFI_OVERLAP_BYTES
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
        rec_dt = np.dtype([
            ("position", "<u4"),
            ("strand", "u1"),
            ("_pad", "u1"),
            ("_pad2", "<u2"),
            ("_pad3", "<u8"),
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
        """In-process pyxrt dispatch path (v1).

        Loads xclbin + insts.bin once per op-class lifetime; reuses the
        device + kernel + bo_instr + bo_seq_in + bo_sparse_out across
        all subsequent calls. Per-dispatch wall is bounded by BO write
        + sync + kernel-run + sync, sub-millisecond.

        Per CLAUDE.md: in-process pyxrt path takes only the in-process
        ``_dispatch_lock`` (a no-op nullcontext); does NOT wrap
        ``npu_silicon_lock``. Silicon-level multi-use protection
        handles same-process concurrency.

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
        pam_mask = state["pam_mask"]
        pam_length = state["pam_length"]
        n_tiles = self.n_tiles

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

        # Pre-build header static bytes (pam_mask + pam_length are
        # constant across all dispatches for this op-class instance).
        pam_mask_bytes = int(pam_mask).to_bytes(4, "little", signed=False)

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
                # Header layout (24 bytes; mirrors runner.cpp::stage_batch):
                #   [0..3]   pam_mask  (uint32 LE)
                #   [4]      pam_length (uint8)
                #   [5..15]  zero (already memset)
                #   [16..19] actual_payload_bytes (uint32 LE)
                #   [20..23] owned_start_offset_bases (int32 LE)
                seq_in_buf[base + 0 : base + 4] = pam_mask_bytes
                seq_in_buf[base + 4] = int(pam_length) & 0xFF
                seq_in_buf[base + 16 : base + 20] = int(payload).to_bytes(
                    4, "little", signed=False
                )
                if ci == 0:
                    owned = 0
                else:
                    owned = (PFI_OVERLAP_BYTES * 4) - (int(pam_length) - 1)
                seq_in_buf[base + 20 : base + 24] = int(owned).to_bytes(
                    4, "little", signed=True
                )
                # Payload at byte 24.
                seq_in_buf[
                    base + PFI_HEADER_BYTES :
                    base + PFI_HEADER_BYTES + payload
                ] = input_buf[src_off : src_off + payload].tobytes()
            bo_seq_in.write(bytes(seq_in_buf), 0)
            t_stage1 = time.perf_counter()
            prof.stage_us += 1e6 * (t_stage1 - t_stage0)

            # ---------------- sync to device ----------------
            bo_seq_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            # Zero output BO host-side and push so stale data from prior
            # dispatch can't leak through.
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
        # (position, strand)).
        all_records.sort()
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
        target_seq: str | None = None,  # for parity with composition layer
        top_n: int = 0,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 600.0,
        force_host: bool = False,
        _impl: str | None = None,
        **_unused: Any,
    ) -> list[tuple[int, int]]:
        """Run the kernel and return ``[(query_pos, strand), ...]``.

        v0 emits only forward-strand records; the host runs RC as a
        second dispatch on a pre-flipped buffer (mirrors locked
        ``crispr/pam_filter`` design — RC is host bookkeeping).

        Args:
            packed_seq: 1-D uint8 packed-2-bit input.
            target_seq: unused at the silicon layer; accepted for API
                parity with the composition layer's CPU oracle path.
            top_n: 0 = all; >0 truncates after ranking.
            n_iters: timed iterations.
            warmup: untimed warmup iterations.
            timeout_s: subprocess timeout (advisory).
            force_host: if True, raise NpuArtifactsMissingError instead
                of attempting the subprocess. Used by tests that want a
                deterministic CPU-only path.
        """
        del target_seq  # parity-only
        if force_host:
            raise NpuArtifactsMissingError(
                "force_host=True; refusing silicon dispatch"
            )
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

        impl = self._resolve_dispatch_impl(_impl)
        if impl == "pyxrt":
            # In-process pyxrt path (v1). xclbin + insts are required;
            # the host_runner subprocess binary is NOT (it's a build-time
            # artifact consumed by the subprocess path only).
            for need in (self.xclbin, self.insts):
                if not need.exists():
                    raise NpuArtifactsMissingError(
                        f"NPU artifact missing for {self.name} "
                        f"(n_tiles={self.n_tiles}): {need}. Build via "
                        f"`make NPU2=1 experiment="
                        f"{'production' if self.n_tiles == 1 else f'wide{self.n_tiles}'} "
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
                impl = "subprocess"
            else:
                self.last_run = info
                return records

        # Subprocess path — same as v0.
        if not self.host_runner.exists():
            raise NpuArtifactsMissingError(
                f"NPU artifact missing for {self.name} "
                f"(n_tiles={self.n_tiles}): "
                f"{self.host_runner}. Build via "
                f"`make NPU2=1 experiment={'production' if self.n_tiles == 1 else f'wide{self.n_tiles}'} "
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
# Registry (single entry; pam is passed at construction time).
# --------------------------------------------------------------------------- #

register_npu_op(
    "bionpu_pam_filter_iupac",
    BionpuPamFilterIupac(pam="NGG", n_tiles=4),
)
