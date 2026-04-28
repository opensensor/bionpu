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

from bionpu.dispatch._pyxrt_helpers import (
    resolve_dispatch_impl,
    run_pyxrt_with_buffers,
)
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

# Production-shape (long) input length, paired with the L=1667 LSTM
# artifact. The encoder's production_long path runs the full 10000-
# sample chunk through stem L1 -> L2 -> L3 (stride=6) -> 1667 LSTM
# timesteps without internal slab resets. The op's ``seq_len`` arg
# names the *LSTM timestep count* (334 short / 1667 long), matching
# DoradoFastLstmCellBf16Acc; the conv-stem T_in is derived from it.
L1667_T = 10000
L1667_T_CHUNK = 200
L1667_N_CHUNKS = L1667_T // L1667_T_CHUNK  # 50

# Map LSTM seq_len -> conv-stem layer-1 input time length. Layer 1 keeps
# the time axis (stride=1, padding=2), so the conv input is the same
# raw-sample T_in that feeds stem L3 (which is what reduces to L_lstm
# via stride=6). Derived: T_in = (L_lstm - 1) * 6 + 19 - 18 = L_lstm * 6
# - 6 + 1 - 0 ... in practice: L=334 -> T_in=2000; L=1667 -> T_in=10000.
_SEQ_LEN_TO_T_IN: dict[int, int] = {
    334: T,           # short-shape: T_in=2000
    1667: L1667_T,    # long-shape:  T_in=10000
}

# Vendored artifact paths. Same layout as 's vector_scalar_mul/.
# This file lives at bionpu/kernels/basecalling/conv_stem/__init__.py
# (5 path components from repo root); the artifacts live at
# bionpu/dispatch/_npu_artifacts/dorado_fast_conv_stem_layer1/. So we
# walk up 4 parents (.../conv_stem/.../basecalling/.../kernels/.../bionpu)
# then descend into dispatch/_npu_artifacts/.
_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_ARTIFACT_NAME = "dorado_fast_conv_stem_layer1"
_ARTIFACT_DIR = _PACKAGE_ROOT / _ARTIFACT_NAME
_XCLBIN = _ARTIFACT_DIR / "final.xclbin"
_INSTS = _ARTIFACT_DIR / "insts.bin"
_HOST_RUNNER = _ARTIFACT_DIR / "host_runner"

# stdout regexes, same anchored shape as 's NpuBackend.
_RE_AVG = re.compile(
    r"^Avg NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MIN = re.compile(
    r"^Min NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MAX = re.compile(
    r"^Max NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)

# Kernel-arg slot layout for the conv_stem layer1 runner (see
# runner.cpp: kernel(opcode, bo_instr, instr_size, bo_signal, bo_wb,
# bo_output, bo_ctrlpkts, bo_trace)). Slots 6/7 are 8-byte/1-byte
# placeholders the pyxrt path fills with zero-payload BOs to keep
# group_id valid on those args.
_ARG_SIGNAL = 3
_ARG_WB = 4
_ARG_OUTPUT = 5
_ARG_CTRLPKTS = 6
_ARG_TRACE = 7

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

def _pack_signal(signal: np.ndarray, total_t: int = T, chunk_t: int = T_CHUNK) -> np.ndarray:
    """Convert ``(1, 1, total_t)`` FP32 input into the chunked format
    the host runner expects.

    Layout: zero-pad ``total_t`` to ``total_t + 2*PAD`` then slice into
    ``total_t // chunk_t`` overlapping chunks of length ``chunk_t +
    2*PAD``. Chunk ``i`` covers samples ``[i*chunk_t - PAD :
    (i+1)*chunk_t + PAD]`` from the padded buffer. Adjacent chunks
    share ``2*PAD = 4`` samples of context (the receptive-field overlap
    needed to keep stride=1, kernel=5 boundary samples correct).

    The ``total_t`` / ``chunk_t`` knobs let the L1667 op
    (``total_t=10000``, ``chunk_t=200``, 50 chunks) reuse the same
    packing primitive as the legacy L334 op (T=2000, T_CHUNK=200, 10
    chunks).

    Returns:
        ndarray shape ``(total_t // chunk_t, chunk_t + 2*PAD)``, FP32, contiguous.
    """
    if total_t % chunk_t != 0:
        raise ValueError(
            f"total_t ({total_t}) must be a multiple of chunk_t ({chunk_t})"
        )
    if signal.shape != (1, 1, total_t):
        raise ValueError(
            f"signal must have shape (1, 1, {total_t}); got {signal.shape}. "
            f"total_t is hardcoded by the xclbin; rebuild for a different "
            f"length."
        )
    if signal.dtype != np.float32:
        raise ValueError(f"signal must be FP32; got dtype={signal.dtype}")
    n_chunks = total_t // chunk_t
    s = signal.reshape(total_t)
    padded = np.zeros(total_t + 2 * PAD, dtype=np.float32)
    padded[PAD : PAD + total_t] = s
    chunks = np.empty((n_chunks, chunk_t + 2 * PAD), dtype=np.float32)
    for i in range(n_chunks):
        # padded indices: each chunk's center is [i*chunk_t + PAD :
        # (i+1)*chunk_t + PAD] with PAD samples of context on each side.
        start = i * chunk_t
        chunks[i] = padded[start : start + chunk_t + 2 * PAD]
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

def _unpack_output(flat: np.ndarray, total_t: int = T, chunk_t: int = T_CHUNK) -> np.ndarray:
    """Convert chunked kernel output ``(n_chunks * 16 * chunk_t,)`` into
    the public ``(1, 16, total_t)`` FP32 layout.

    Per-chunk layout from the kernel: ``(16, chunk_t)`` row-major
    (out_ch major). We reshape to ``(n_chunks, 16, chunk_t)`` then
    transpose+concat along the time axis.
    """
    n_chunks = total_t // chunk_t
    if flat.size != n_chunks * OUT_CH * chunk_t:
        raise ValueError(
            f"output length mismatch: got {flat.size}, "
            f"expected {n_chunks * OUT_CH * chunk_t}"
        )
    chunked = flat.reshape(n_chunks, OUT_CH, chunk_t)
    return chunked.transpose(1, 0, 2).reshape(1, OUT_CH, total_t)

class DoradoFastConvStemLayer1(NpuOp):
    """First Dorado fast stem Conv1d layer on AIE2P.

    Op: ``Conv1d(in=1, out=16, kernel=5, stride=1, padding=2, bias=True)``
    on a fixed ``(1, 1, T)`` FP32 signal. Returns ``(1, 16, T)``.

    Two on-disk artifacts:

    - ``seq_len=T`` (=2000, default; 10 chunks): legacy short-shape
      artifact paired with the LSTM L=334 path. Loaded from
      ``_npu_artifacts/dorado_fast_conv_stem_layer1/``.
    - ``seq_len=L1667_T`` (=10000; 50 chunks): production-shape (long)
      artifact paired with the LSTM L=1667 path. Loaded from
      ``_npu_artifacts/dorado_fast_conv_stem_layer1_L1667/``. Mirrors
      :class:`bionpu.kernels.basecalling.lstm_cell_bf16_acc.DoradoFastLstmCellBf16Acc`.

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

    # Public so callers can introspect the pinned shape (legacy
    # short-shape; the long-shape op exposes per-instance properties).
    INPUT_SHAPE = (1, 1, T)
    OUTPUT_SHAPE = (1, OUT_CH, T)
    KERNEL_SIZE = K
    PADDING = PAD
    STRIDE = 1
    IN_CHANNELS = IN_CH
    OUT_CHANNELS = OUT_CH

    def __init__(self, seq_len: int = 334) -> None:
        # ``seq_len`` names the LSTM timestep count (334 short, 1667
        # long), matching DoradoFastLstmCellBf16Acc. The conv-stem
        # layer-1 T_in is derived via the _SEQ_LEN_TO_T_IN map.
        if seq_len not in _SEQ_LEN_TO_T_IN:
            raise ValueError(
                f"seq_len={seq_len} not supported; expected one of "
                f"{tuple(_SEQ_LEN_TO_T_IN.keys())}. Add a Makefile "
                f"invocation + an "
                f"_npu_artifacts/dorado_fast_conv_stem_layer1_L<seq_len>/ "
                f"directory to extend."
            )
        self.seq_len = int(seq_len)
        self._t_in = _SEQ_LEN_TO_T_IN[self.seq_len]
        # Per-instance T_chunk: same 200 for both supported shapes.
        self._chunk = T_CHUNK if self._t_in == T else L1667_T_CHUNK
        self.last_run: ConvStemRunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(p.exists() for p in (_XCLBIN, _INSTS, _HOST_RUNNER))

    @property
    def artifact_dir(self) -> Path:
        if self.seq_len == 334:
            return _ARTIFACT_DIR
        return _PACKAGE_ROOT / f"{_ARTIFACT_NAME}_L{self.seq_len}"

    @property
    def xclbin(self) -> Path:
        return self.artifact_dir / "final.xclbin"

    @property
    def insts(self) -> Path:
        return self.artifact_dir / "insts.bin"

    @property
    def host_runner(self) -> Path:
        return self.artifact_dir / "host_runner"

    def artifacts_available(self) -> bool:
        return all(
            p.exists() for p in (self.xclbin, self.insts, self.host_runner)
        )

    def __call__(
        self,
        *,
        signal: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 60.0,
        _impl: str | None = None,
    ) -> np.ndarray:
        """Run the Conv stem on the NPU and return the output.

        Args:
            signal: ``(1, 1, 2000)`` FP32.
            weight: ``(16, 1, 5)`` FP32.
            bias:   ``(16,)`` FP32.
            n_iters: number of NPU iterations (timing is averaged).
            warmup: warmup iterations excluded from timing.
            timeout_s: subprocess timeout for the host runner.
            _impl: explicit dispatch impl override
                (``"subprocess"`` (default) or ``"pyxrt"``); takes
                precedence over ``BIONPU_CONV_STEM_LAYER1_DISPATCH``.

        Returns:
            ``(1, 16, 2000)`` FP32 numpy array.

        Raises:
            NpuArtifactsMissingError: xclbin/insts/runner missing on disk.
            NpuRunFailed: host runner exited non-zero.
            ValueError: shape/dtype mismatch on inputs.
        """
        impl = resolve_dispatch_impl(
            _impl, env_var="BIONPU_CONV_STEM_LAYER1_DISPATCH"
        )
        xclbin = self.xclbin
        insts = self.insts
        host_runner = self.host_runner
        artifact_dir = self.artifact_dir
        total_t = self._t_in
        chunk_t = self._chunk
        n_chunks = total_t // chunk_t
        required = (
            (xclbin, insts) if impl == "pyxrt"
            else (xclbin, insts, host_runner)
        )
        for p in required:
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for "
                    f"dorado_fast_conv_stem_layer1 (seq_len={self.seq_len}, "
                    f"T_in={total_t}): {p}. See "
                    f"{artifact_dir}/MANIFEST.md to rebuild."
                )

        # Pack inputs into the on-disk binary format the runner expects.
        sig_packed = _pack_signal(signal, total_t=total_t, chunk_t=chunk_t)
        wb_packed = _pack_wb(weight, bias)

        if impl == "pyxrt":
            out_size = n_chunks * OUT_CH * chunk_t * 4  # FP32
            raw_out, avg_us, min_us, max_us = run_pyxrt_with_buffers(
                xclbin_path=xclbin,
                insts_path=insts,
                in_buffers=[
                    (sig_packed.tobytes(), _ARG_SIGNAL),
                    (wb_packed.tobytes(), _ARG_WB),
                    # Match runner.cpp: ctrlpkts is 8 bytes, trace is 1 byte.
                    (bytes(8), _ARG_CTRLPKTS),
                    (bytes(1), _ARG_TRACE),
                ],
                out_size=out_size,
                out_arg_index=_ARG_OUTPUT,
                n_iters=n_iters,
                warmup=warmup,
            )
            flat = np.frombuffer(raw_out, dtype=np.float32).copy()
            out = _unpack_output(flat, total_t=total_t, chunk_t=chunk_t)
            self.last_run = ConvStemRunResult(
                output=out,
                avg_us=avg_us,
                min_us=min_us,
                max_us=max_us,
                n_iters=int(n_iters),
            )
            return out

        with tempfile.TemporaryDirectory(prefix="t42_conv_stem_") as td:
            tdp = Path(td)
            sig_path = tdp / "signal.f32.bin"
            wb_path = tdp / "wb.f32.bin"
            out_path = tdp / "output.f32.bin"

            sig_packed.tofile(sig_path)
            wb_packed.tofile(wb_path)

            cmd = [
                str(host_runner),
                "-x", str(xclbin),
                "-i", str(insts),
                "-k", "MLIR_AIE",
                "--signal", str(sig_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--time", str(total_t),
                "--chunk", str(chunk_t),
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
        out = _unpack_output(flat, total_t=total_t, chunk_t=chunk_t)
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
    "L1667_T",
    "L1667_T_CHUNK",
    "L1667_N_CHUNKS",
    "PAD",
    "K",
    "IN_CH",
    "OUT_CH",
    "WB_LEN",
]
