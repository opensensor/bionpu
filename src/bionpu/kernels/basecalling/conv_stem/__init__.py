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

"""Dorado fast stem Conv1d layer 1 — NPU op.

Registers ``dorado_fast_conv_stem_layer1`` with the
:data:`bionpu.dispatch.NPU_OPS` registry and ships the call wrapper that
turns a caller-supplied ``(1, 1, T)`` FP32 signal + ``(16, 1, 5)``
weights + ``(16,)`` bias into an NPU run that writes a
``(1, 16, T)`` FP32 output.

The xclbin is fixed at ``T = 2000``; passing a different time dimension
raises :class:`ValueError` ( will rebuild for ``T = 10000``, the
production Dorado chunk length).

Why a custom-runner path and not a 1-shot template fork? The
mlir-aie ``xrt_test_wrapper.h`` template assumes compiled-in fixtures.
This op needs caller-driven signals AND caller-driven weights, so the
host runner reads both via files; the Python wrapper marshals the
chunked layout the IRON kernel expects (see MANIFEST.md).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    register_npu_op,
)

# Pinned op geometry. Mirrors `architecture.md §1` (Dorado fast layer 0
# = Conv1d(1, 16, kernel=5, stride=1, padding=2)) and the IRON
# kernel constants in `conv1d_layer1.py`. T_CHUNK is the per-iteration
# tile-resident chunk; T is the public input/output time dimension the
# xclbin was built for.
T = 2000
T_CHUNK = 200
N_CHUNKS = T // T_CHUNK
PAD = 2
K = 5
IN_CH = 1
OUT_CH = 16
WB_LEN = OUT_CH * K + OUT_CH  # 96 floats

# Vendored artifact paths. Same layout as 's vector_scalar_mul/.
# This file lives at bionpu/kernels/basecalling/conv_stem/__init__.py
# (5 path components from repo root); the artifacts live at
# bionpu/dispatch/_npu_artifacts/dorado_fast_conv_stem_layer1/. So we
# walk up 4 parents (.../conv_stem/.../basecalling/.../kernels/.../bionpu)
# then descend into dispatch/_npu_artifacts/.
_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
    / "dorado_fast_conv_stem_layer1"
)
_XCLBIN = _ARTIFACT_DIR / "final.xclbin"
_INSTS = _ARTIFACT_DIR / "insts.bin"
_HOST_RUNNER = _ARTIFACT_DIR / "host_runner"

# stdout regexes, same anchored shape as 's NpuBackend.
_RE_AVG = re.compile(r"^Avg NPU time:\s*([0-9.]+)us\.?\s*$", re.MULTILINE)
_RE_MIN = re.compile(r"^Min NPU time:\s*([0-9.]+)us\.?\s*$", re.MULTILINE)
_RE_MAX = re.compile(r"^Max NPU time:\s*([0-9.]+)us\.?\s*$", re.MULTILINE)

@dataclass(frozen=True)
class ConvStemRunResult:
    """Per-call timing + output data for the Conv stem op."""

    output: np.ndarray  # shape (1, 16, T) FP32
    avg_us: float
    min_us: float
    max_us: float
    n_iters: int

def _xrt_env() -> dict[str, str]:
    """Same XRT env helper as `bionpu.dispatch.npu._xrt_env`."""
    env = os.environ.copy()
    xrt_root = env.get("XILINX_XRT", "/opt/xilinx/xrt")
    env["XILINX_XRT"] = xrt_root
    extra_lib = f"{xrt_root}/lib"
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if extra_lib not in existing_ld.split(":"):
        env["LD_LIBRARY_PATH"] = (
            f"{extra_lib}:{existing_ld}" if existing_ld else extra_lib
        )
    return env

def _pack_signal(signal: np.ndarray) -> np.ndarray:
    """Convert ``(1, 1, T)`` FP32 input into the chunked format the
    host runner expects.

    Layout: zero-pad ``T`` to ``T + 2*PAD`` then slice into ``N_CHUNKS``
    overlapping chunks of length ``T_CHUNK + 2*PAD``. Chunk ``i`` covers
    samples ``[i*T_CHUNK - PAD : (i+1)*T_CHUNK + PAD]`` from the padded
    buffer. Adjacent chunks share ``2*PAD = 4`` samples of context (the
    receptive-field overlap needed to keep stride=1, kernel=5 boundary
    samples correct).

    Returns:
        ndarray shape ``(N_CHUNKS, T_CHUNK + 2*PAD)``, FP32, contiguous.
    """
    if signal.shape != (1, 1, T):
        raise ValueError(
            f"signal must have shape (1, 1, {T}); got {signal.shape}. "
            f"T is hardcoded by the xclbin; will rebuild for "
            f"T=10000."
        )
    if signal.dtype != np.float32:
        raise ValueError(f"signal must be FP32; got dtype={signal.dtype}")
    s = signal.reshape(T)
    padded = np.zeros(T + 2 * PAD, dtype=np.float32)
    padded[PAD : PAD + T] = s
    chunks = np.empty((N_CHUNKS, T_CHUNK + 2 * PAD), dtype=np.float32)
    for i in range(N_CHUNKS):
        # padded indices: each chunk's center is [i*T_CHUNK + PAD :
        # (i+1)*T_CHUNK + PAD] with PAD samples of context on each side.
        start = i * T_CHUNK
        chunks[i] = padded[start : start + T_CHUNK + 2 * PAD]
    return chunks

def _pack_wb(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Pack `(16, 1, 5)` weights + `(16,)` bias into the kernel's wb buffer.

    The kernel reads weights as a contiguous ``(16, 5)`` block (`oc`
    major) followed immediately by ``(16,)`` bias. Returns FP32
    contiguous of length :data:`WB_LEN`.
    """
    if weight.shape != (OUT_CH, IN_CH, K):
        raise ValueError(
            f"weight must have shape ({OUT_CH}, {IN_CH}, {K}); "
            f"got {weight.shape}"
        )
    if bias.shape != (OUT_CH,):
        raise ValueError(f"bias must have shape ({OUT_CH},); got {bias.shape}")
    W = weight.reshape(OUT_CH, K).astype(np.float32, copy=False)
    B = bias.astype(np.float32, copy=False)
    return np.concatenate([W.flatten(), B.flatten()]).astype(np.float32)

def _unpack_output(flat: np.ndarray) -> np.ndarray:
    """Convert chunked kernel output ``(N_CHUNKS * 16 * T_CHUNK,)`` into
    the public ``(1, 16, T)`` FP32 layout.

    Per-chunk layout from the kernel: ``(16, T_CHUNK)`` row-major
    (out_ch major). We reshape to ``(N_CHUNKS, 16, T_CHUNK)`` then
    transpose+concat along the time axis.
    """
    if flat.size != N_CHUNKS * OUT_CH * T_CHUNK:
        raise ValueError(
            f"output length mismatch: got {flat.size}, "
            f"expected {N_CHUNKS * OUT_CH * T_CHUNK}"
        )
    chunked = flat.reshape(N_CHUNKS, OUT_CH, T_CHUNK)
    return chunked.transpose(1, 0, 2).reshape(1, OUT_CH, T)

class DoradoFastConvStemLayer1(NpuOp):
    """First Dorado fast stem Conv1d layer on AIE2P.

    Op: ``Conv1d(in=1, out=16, kernel=5, stride=1, padding=2, bias=True)``
    on a fixed ``(1, 1, T=2000)`` FP32 signal. Returns ``(1, 16, T)``.

    Two calling conventions, both supported:

    1. Signal-only (BN-folded weights from PASSPORT.json / model file):
       ``op(signal=...)``. The op loads the pinned weights once and
       reuses them.
    2. Caller-supplied weights and bias:
       ``op(signal=..., weight=..., bias=...)``. Useful for the
       equivalence harness which wants to assert NPU vs ONNX-CPU
       parity using the same weights both runs see.

    For the v1 path, weights+bias travel to the device with every call
    (no on-device weight caching); will optimise this.
    """

    name = "dorado_fast_conv_stem_layer1"

    # Public so callers can introspect the pinned shape.
    INPUT_SHAPE = (1, 1, T)
    OUTPUT_SHAPE = (1, OUT_CH, T)
    KERNEL_SIZE = K
    PADDING = PAD
    STRIDE = 1
    IN_CHANNELS = IN_CH
    OUT_CHANNELS = OUT_CH

    def __init__(self) -> None:
        self.last_run: ConvStemRunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(p.exists() for p in (_XCLBIN, _INSTS, _HOST_RUNNER))

    def __call__(
        self,
        *,
        signal: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 60.0,
    ) -> np.ndarray:
        """Run the Conv stem on the NPU and return the output.

        Args:
            signal: ``(1, 1, 2000)`` FP32.
            weight: ``(16, 1, 5)`` FP32.
            bias:   ``(16,)`` FP32.
            n_iters: number of NPU iterations (timing is averaged).
            warmup: warmup iterations excluded from timing.
            timeout_s: subprocess timeout for the host runner.

        Returns:
            ``(1, 16, 2000)`` FP32 numpy array.

        Raises:
            NpuArtifactsMissingError: xclbin/insts/runner missing on disk.
            NpuRunFailed: host runner exited non-zero.
            ValueError: shape/dtype mismatch on inputs.
        """
        for p in (_XCLBIN, _INSTS, _HOST_RUNNER):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for "
                    f"dorado_fast_conv_stem_layer1: {p}. See "
                    f"{_ARTIFACT_DIR}/MANIFEST.md to rebuild."
                )

        # Pack inputs into the on-disk binary format the runner expects.
        sig_packed = _pack_signal(signal)
        wb_packed = _pack_wb(weight, bias)

        with tempfile.TemporaryDirectory(prefix="t42_conv_stem_") as td:
            tdp = Path(td)
            sig_path = tdp / "signal.f32.bin"
            wb_path = tdp / "wb.f32.bin"
            out_path = tdp / "output.f32.bin"

            sig_packed.tofile(sig_path)
            wb_packed.tofile(wb_path)

            cmd = [
                str(_HOST_RUNNER),
                "-x", str(_XCLBIN),
                "-i", str(_INSTS),
                "-k", "MLIR_AIE",
                "--signal", str(sig_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--time", str(T),
                "--chunk", str(T_CHUNK),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
            ]
            proc = subprocess.run(  # noqa: S603 — argv is fully controlled
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
                env=_xrt_env(),
            )
            if proc.returncode != 0:
                raise NpuRunFailed(
                    proc.returncode, proc.stdout, proc.stderr
                )
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker missing",
                )
            m_avg = _RE_AVG.search(proc.stdout)
            m_min = _RE_MIN.search(proc.stdout)
            m_max = _RE_MAX.search(proc.stdout)
            if not (m_avg and m_min and m_max):
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] could not parse NPU timing lines",
                )
            avg_us = float(m_avg.group(1))
            min_us = float(m_min.group(1))
            max_us = float(m_max.group(1))

            flat = np.fromfile(out_path, dtype=np.float32)
        out = _unpack_output(flat)
        self.last_run = ConvStemRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out

# Tile-local memory used by the AIE compute tile, parsed from the
# linker script `main_core_0_2.ld.script` produced by aiecc. See
# MANIFEST.md for the per-buffer breakdown. Stored here so the
# equivalence runner / measurement writer can record it without
# re-running the build.
TILE_MEMORY_USED_BYTES = 29024  # 28.3 KiB on a 64 KiB tile

# Register at import time so `dispatch(op="dorado_fast_conv_stem_layer1",
# device="npu", ...)` works as soon as the package is imported.
register_npu_op(
    "dorado_fast_conv_stem_layer1",
    DoradoFastConvStemLayer1(),
)

__all__ = [
    "DoradoFastConvStemLayer1",
    "ConvStemRunResult",
    "TILE_MEMORY_USED_BYTES",
    "T",
    "T_CHUNK",
    "N_CHUNKS",
    "PAD",
    "K",
    "IN_CH",
    "OUT_CH",
    "WB_LEN",
]
