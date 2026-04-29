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
