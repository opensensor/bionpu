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

"""CRISPR single-tile match op — registered NPU op.

Registers the ``"crispr_match_singletile"`` entry in
:data:`bionpu.dispatch.npu.NPU_OPS` at import time. The op takes packed
2-bit guides + windows on the host side, runs the precompiled AIE2P
xclbin via the vendored host C++ binary, and returns a dense
``(n_guides, n_windows)`` mismatch-count matrix at ``uint8``.

The artifact root is colocated with 's vector_scalar_mul artifacts:
``bionpu/dispatch/_npu_artifacts/crispr_match_singletile/{final.xclbin,
insts.bin, host_runner, MANIFEST.md}``. Rebuild instructions live in
that directory's ``MANIFEST.md``.

Encoding contract (host side):

    Each spacer is 20 nt. Pack 2 bits per base, 4 bases per byte:
    A=00, C=01, G=10, T=11. Bases 0..3 of a spacer go in byte 0
    (base 0 in bits 1:0, base 1 in bits 3:2, base 2 in bits 5:4,
    base 3 in bits 7:6); bases 4..7 in byte 1; etc. 20 nt = 5 bytes.

The ``encode_2bit`` helper exposes this packing so callers can build
inputs from ASCII strings or numpy uint8 arrays. ``decode_2bit`` is the
inverse, used in tests for round-trip checks.

This module deliberately does **not** modify ``bionpu/dispatch/{__init__,
npu,devices}.py`` — that's 's territory. The registration happens at
import time via ``register_npu_op``.

Out of scope for (deferred):

    * PAM filtering
    * Tile-local thresholding / sparse output
    * DMA streaming + multi-tile fan-out
    * Genome-scale walk
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    register_npu_op,
)
from bionpu.dispatch.npu import _xrt_env  # noqa: PLC2701 — internal helper

__all__ = [
    "CrisprMatchSingleTile",
    "N_GUIDES",
    "N_WINDOWS",
    "SPACER_BYTES",
    "SPACER_LEN",
    "decode_2bit",
    "encode_2bit",
]

# --- pinned shape (must match match_singletile.py / test.cpp / kernel) ---
N_GUIDES = 128
N_WINDOWS = 4096
SPACER_LEN = 20
SPACER_BYTES = 5  # 20 nt × 2 bits / 8

# Artifact paths (populated by `make NPU2=1` + the MANIFEST.md vendor step).
_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
    / "crispr_match_singletile"
)
_XCLBIN = _ART_ROOT / "final.xclbin"
_INSTS = _ART_ROOT / "insts.bin"
_BINARY = _ART_ROOT / "host_runner"

_KERNEL_NAME = "MLIR_AIE"

# --------------------------------------------------------------------------- #
# 2-bit ASCII <-> packed-byte codec
# --------------------------------------------------------------------------- #

# Lookup tables (initialized lazily so module import is cheap).
def _base_to_2bit(b: int) -> int:
    if b == ord("A"):
        return 0
    if b == ord("C"):
        return 1
    if b == ord("G"):
        return 2
    if b == ord("T"):
        return 3
    raise ValueError(f"non-ACGT byte: {b!r}")

def encode_2bit(spacer_ascii: bytes | str | np.ndarray) -> np.ndarray:
    """Pack a 20-nt ASCII spacer into 5 packed bytes.

    Args:
        spacer_ascii: 20-base sequence as bytes, str, or uint8 ndarray. Must
            contain only ``ACGT`` (uppercase).

    Returns:
        np.ndarray of shape ``(SPACER_BYTES,)`` and dtype uint8. Bits 1:0 hold
        the first base, bits 3:2 the second, ..., bits 7:6 the fourth — i.e.
        little-endian-within-byte 2-bit packing as documented in
        ``match_kernel.cc``.

    Raises:
        ValueError: spacer length is not :data:`SPACER_LEN` or contains a
            non-ACGT base.
    """
    if isinstance(spacer_ascii, str):
        data = spacer_ascii.encode("ascii")
    elif isinstance(spacer_ascii, np.ndarray):
        if spacer_ascii.dtype != np.uint8:
            raise ValueError("ndarray spacer must be uint8")
        data = bytes(spacer_ascii.tolist())
    else:
        data = bytes(spacer_ascii)
    if len(data) != SPACER_LEN:
        raise ValueError(f"spacer must be {SPACER_LEN} nt; got {len(data)}")
    out = np.zeros(SPACER_BYTES, dtype=np.uint8)
    for i, b in enumerate(data):
        byte_idx = i // 4
        bit_off = (i % 4) * 2
        out[byte_idx] |= (_base_to_2bit(b) & 0x3) << bit_off
    return out

def decode_2bit(packed: np.ndarray) -> str:
    """Inverse of :func:`encode_2bit`. Returns a 20-character ACGT string."""
    if packed.shape != (SPACER_BYTES,) or packed.dtype != np.uint8:
        raise ValueError(
            f"packed shape/dtype mismatch: got {packed.shape}/{packed.dtype}; "
            f"expected ({SPACER_BYTES},)/uint8"
        )
    bases = "ACGT"
    chars = []
    for i in range(SPACER_LEN):
        byte_idx = i // 4
        bit_off = (i % 4) * 2
        v = (packed[byte_idx] >> bit_off) & 0x3
        chars.append(bases[v])
    return "".join(chars)

# --------------------------------------------------------------------------- #
# CPU fallback — bit-equal to the AIE-tile kernel
# --------------------------------------------------------------------------- #

def _cpu_mismatch_count_matrix(
    guides_packed: np.ndarray, windows_packed: np.ndarray
) -> np.ndarray:
    """Compute the mismatch-count matrix on CPU using the same bit-twiddling
    arithmetic the AIE C++ kernel uses (see ``match_kernel.cc``).

    Returns:
        np.ndarray of shape ``(N_GUIDES, N_WINDOWS)`` and dtype uint8.

    Used both as the "no NPU available" fallback and as a
    cross-check for the unit tests. A separate path exists in the
    oracle that computes mismatch from ASCII strings — those two
    must agree byte-for-byte by construction (the encoding is bijective
    and the algorithm is base-equality).
    """
    # guides_packed: (n_guides, SPACER_BYTES) uint8
    # windows_packed: (n_windows, SPACER_BYTES) uint8
    g = guides_packed.astype(np.uint8)
    w = windows_packed.astype(np.uint8)
    # Broadcast XOR to (n_guides, n_windows, SPACER_BYTES).
    xor = g[:, None, :] ^ w[None, :, :]
    # Collapse each 2-bit pair to one bit: pair == 00 → 0, else → 1.
    m = ((xor | (xor >> 1)) & 0x55).astype(np.uint8)
    # Popcount via shifts/masks (bit-equivalent to the C kernel).
    c = (m & 0x55) + ((m >> 1) & 0x55)
    c = (c & 0x33) + ((c >> 2) & 0x33)
    c = (c & 0x0F) + ((c >> 4) & 0x0F)
    return c.sum(axis=2).astype(np.uint8)

# --------------------------------------------------------------------------- #
# NpuOp registration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _RunInfo:
    """Timing + provenance for the last NPU run, parsed from the host binary."""

    avg_us: float
    min_us: float
    max_us: float
    n_iters: int
    used_npu: bool  # False if we fell back to the CPU bit-equivalent path

def _parse_us(stdout: str, label: str) -> float | None:
    import re

    m = re.search(rf"^{label} NPU time:\s*([0-9.]+)us\.?\s*$", stdout, re.MULTILINE)
    return float(m.group(1)) if m else None

class CrisprMatchSingleTile(NpuOp):
    """``crispr_match_singletile`` — registered NPU op.

    Inputs (keyword-only):
        guides_2bit (np.ndarray): shape ``(N_GUIDES, SPACER_BYTES)``, uint8.
        windows_2bit (np.ndarray): shape ``(N_WINDOWS, SPACER_BYTES)``, uint8.
        n_iters (int): NPU iterations to time. Default 1.
        warmup (int): warmup iterations not counted in the timing. Default 0.

    Returns:
        np.ndarray: ``(N_GUIDES, N_WINDOWS)`` uint8. Cell (g, w) is the
        spacer-only mismatch count between guide ``g`` and window ``w``.

    The op:
        1. Validates shapes/dtypes (hard fails on mismatch — the v1 kernel
           is shape-pinned to 's 128×4096 fixture).
        2. If the NPU artifacts are present, runs the precompiled host
           binary via XRT. The binary writes its output to a tmp file we
           then read back as a numpy array (window-major; we transpose).
        3. If artifacts are missing, raises :class:`NpuArtifactsMissingError`
           — same surface as (the build-and-vendor flow is
           documented in this op's MANIFEST.md). Tests skip cleanly when
           this is raised.

    Notes on host I/O:
        The host binary takes ``--guides PATH``, ``--windows PATH``, ``--out
        PATH`` so the Python side can pump arbitrary inputs through. This
        differs from 's ``vector_scalar_mul`` (which hardcodes the
        fixture inside the binary) because byte-equality on caller-supplied
        inputs is mandatory for .
    """

    name = "crispr_match_singletile"

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
        """True iff all four NPU artifacts are on disk."""
        return all(p.exists() for p in (_XCLBIN, _INSTS, _BINARY))

    def _run_npu(
        self,
        guides_2bit: np.ndarray,
        windows_2bit: np.ndarray,
        n_iters: int,
        warmup: int,
    ) -> tuple[np.ndarray, _RunInfo]:
        """Invoke the precompiled host binary and return (output, timing)."""
        with tempfile.TemporaryDirectory(prefix="crispr_singletile_") as tdir:
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
            # Kernel emits window-major (w, g); transpose to (g, w).
            window_major = raw.reshape(N_WINDOWS, N_GUIDES)
            out_gw = np.ascontiguousarray(window_major.T)
            info = _RunInfo(
                avg_us=avg, min_us=mn, max_us=mx, n_iters=n_iters, used_npu=True
            )
            return out_gw, info

    def __call__(
        self,
        *,
        guides_2bit: np.ndarray,
        windows_2bit: np.ndarray,
        n_iters: int = 1,
        warmup: int = 0,
        **_unused: Any,
    ) -> np.ndarray:
        self._validate_inputs(guides_2bit, windows_2bit)

        if not self.artifacts_present():
            missing = [p for p in (_XCLBIN, _INSTS, _BINARY) if not p.exists()]
            raise NpuArtifactsMissingError(
                "crispr_match_singletile NPU artifacts missing: "
                + ", ".join(str(p) for p in missing)
                + ". See "
                + str(_ART_ROOT / "MANIFEST.md")
                + " for build instructions."
            )

        out, info = self._run_npu(guides_2bit, windows_2bit, n_iters, warmup)
        self.last_run = info
        return out

# Register at import time.
register_npu_op("crispr_match_singletile", CrisprMatchSingleTile())
