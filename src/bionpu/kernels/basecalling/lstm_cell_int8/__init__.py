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

"""Dorado fast LSTM cell — INT8 NPU op.

Registers ``dorado_fast_lstm_cell_int8`` with
:data:`bionpu.dispatch.NPU_OPS`. **INT8 sibling** of 's bf16
``dorado_fast_lstm_cell_bf16``: same wire-format topology (input +
output + per-chunk weight stream with prefix), but per-element type
drops to INT8 throughout. Closes — the AIE2P INT8 LSTM
kernel that 's INT8 sweep needed and that 's §3.2 ratification
gates on. Per the cross-walk doc (`docs/.md`),
AM020 Table 14 quotes 256 INT8 ops/tile/cycle on AIE-ML; AIE2P is
expected at >= that density (`docs/.md` §
"Caveats specific to our chip").

Op contract (FP32 in, FP32 out — keeps the encoder's FP32 interface):
    inputs:  x (L, 1, 96) fp32
             weight_ih, weight_hh, bias_ih, bias_hh (4*HIDDEN, ...) fp32
             calibration : per-channel symmetric quant scales from 
                           (consumed by the host packer; PASSPORT.json
                           records the calibration_data_id)
    outputs: y (L, 1, 96) fp32 (post-dequant from INT8 wire output)
    internal: int8 multiplier inputs, int32 MAC accumulator, FP32
              recurrent state (h, c) per 's lesson
. bf16 hardware tanh / sigmoid on the
              dequantized FP32 path (no INT8 transcendental on AIE2P).

Per-channel calibration: applied **host-side**. Each weight row's
per-channel scale is folded into the bias slab (the on-tile dequant
becomes a single per-gate scalar multiply); the per-channel ratio
is absorbed into the INT8 weight values themselves by the calibrator
. PASSPORT
records the per-cell + end-to-end max-abs achieved against the
calibration corpus.

Tile-memory used (parsed from
``build/aie_L334.mlir.prj/main_core_0_2.ld.script``):
    stack 0x400 + 2 * 0x1E40 (weight depth-2 buffer; smaller than
    bf16's 0x2A00 because INT8 elements are 1 byte vs 2)
    + 4 * 0x60 (input/output depth-2 buffers; 96 INT8 = 96 B vs 192 B)
    + .bss (h_state/c_state FP32, gate_acc INT32, bias_cache FP32,
    per-gate scales) ≈ 23.5 KiB on a 64 KiB AIE2P tile (close to
    's 22.3 KiB; the FP32 bias/state slab balances the smaller
    INT8 wire footprint).
"""

from __future__ import annotations

import json
import os
import re
import struct
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

# Pinned LSTM cell geometry — same as / (interface-compatible).
T_LSTM = 334
HIDDEN = 96
INPUT_DIM = 96
N_GATES = 4

# Per-gate weight slab sizes (counts in element-of-type).
WIH_GATE = HIDDEN * INPUT_DIM  # 9216 per gate
WHH_GATE = HIDDEN * HIDDEN     # 9216 per gate
BIH_GATE = HIDDEN              # 96 per gate
BHH_GATE = HIDDEN              # 96 per gate

# Compact WB layout matches runner.cpp parse_compact:
#   Wih_q (int8) + Whh_q (int8) + bih (fp32) + bhh (fp32)
#   + s_w_ih (fp32) + s_w_hh (fp32) + s_x + s_h + s_y (fp32)
COMPACT_INT8_BYTES = N_GATES * (WIH_GATE + WHH_GATE)
COMPACT_FP32_BYTES = (
    N_GATES * (BIH_GATE + BHH_GATE) * 4   # biases
    + N_GATES * 2 * HIDDEN * 4            # per-channel weight scales
    + 3 * 4                                # s_x, s_h, s_y
)
COMPACT_BLOB_BYTES = COMPACT_INT8_BYTES + COMPACT_FP32_BYTES

# Tile-local memory (parsed from linker script post-build).
LSTM_CELL_INT8_TILE_MEMORY_USED_BYTES = 0x400 + 2 * 0x1E40 + 4 * 0x60  # ~16480

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_ARTIFACT_DIR = _PACKAGE_ROOT / "dorado_fast_lstm_cell_int8"
_XCLBIN = _ARTIFACT_DIR / "final.xclbin"
_INSTS = _ARTIFACT_DIR / "insts.bin"
_HOST_RUNNER = _ARTIFACT_DIR / "host_runner"

_RE_AVG = re.compile(
    r"^Avg NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MIN = re.compile(
    r"^Min NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MAX = re.compile(
    r"^Max NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)

@dataclass(frozen=True)
class LstmCellInt8RunResult:
    output: np.ndarray  # (L, N=1, 96) FP32 (re-cast from INT8 via s_y)
    avg_us: float
    min_us: float
    max_us: float
    n_iters: int

@dataclass(frozen=True)
class Int8Calibration:
    """Per-channel symmetric quantization params.

    All scales are positive FP32. ``s_w_ih`` / ``s_w_hh`` are
    per-output-channel (gate-major); ``s_x`` / ``s_h`` / ``s_y`` are
    per-tensor.
    """

    s_w_ih: np.ndarray   # (N_GATES, HIDDEN) fp32, > 0
    s_w_hh: np.ndarray   # (N_GATES, HIDDEN) fp32, > 0
    s_x: float           # > 0
    s_h: float           # > 0
    s_y: float           # > 0

def _xrt_env() -> dict[str, str]:
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

# ---------------------------------------------------------------------------
# INT8 conversion helpers
# ---------------------------------------------------------------------------

def quantize_symmetric_per_channel(
    W: np.ndarray, *, axis: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel INT8 quantization.

    Args:
        W: FP32 weight tensor.
        axis: axis along which to sum the absolute values for the per-row
            scale. For ``W[oc, j]`` the natural choice is ``axis=1``.

    Returns:
        ``(W_q, scale)`` where ``W_q`` is int8 (saturated [-128, 127])
        and ``scale[oc] = max(|W[oc, :]|) / 127``. Rows with all zeros
        get ``scale=1.0`` (no quantization noise; emits int8 zeros).
    """
    Wf = np.asarray(W, dtype=np.float32)
    max_abs = np.max(np.abs(Wf), axis=axis, keepdims=True)
    # Avoid division-by-zero on all-zero rows.
    scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    Wq_float = Wf / scale
    Wq = np.clip(np.round(Wq_float), -128, 127).astype(np.int8)
    return Wq, scale.astype(np.float32).reshape(-1)

def quantize_symmetric_per_tensor(
    x: np.ndarray, *, scale: float | None = None
) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT8 quantization."""
    xf = np.asarray(x, dtype=np.float32)
    if scale is None:
        max_abs = float(np.max(np.abs(xf)))
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
    xq_float = xf / scale
    xq = np.clip(np.round(xq_float), -128, 127).astype(np.int8)
    return xq, float(scale)

def calibrate_lstm_weights(
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: np.ndarray,
    bias_hh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Int8Calibration]:
    """Compute per-channel INT8 calibration on Dorado LSTM weights.

    This is the host-side per-channel symmetric quantization that
    's PASSPORT records. Sister to ``bionpu/quant/calibrate.py``'s
    ``per_channel`` strategy (which targets ONNX QDQ; this targets the
    runner's compact WB packing).

    Args:
        weight_ih: ``(4*HIDDEN, INPUT_DIM)`` fp32, gate-major.
        weight_hh: ``(4*HIDDEN, HIDDEN)``    fp32, gate-major.
        bias_ih:   ``(4*HIDDEN,)``           fp32, gate-major.
        bias_hh:   ``(4*HIDDEN,)``           fp32, gate-major.

    Returns:
        ``(Wih_q, Whh_q, bih_f32, bhh_f32, calibration)`` — the int8
        quantized weights (gate-major, contiguous) and the calibration
        object the runner consumes.
    """
    if weight_ih.shape != (4 * HIDDEN, INPUT_DIM):
        raise ValueError(
            f"weight_ih must have shape ({4*HIDDEN}, {INPUT_DIM}); "
            f"got {weight_ih.shape}"
        )
    if weight_hh.shape != (4 * HIDDEN, HIDDEN):
        raise ValueError(
            f"weight_hh must have shape ({4*HIDDEN}, {HIDDEN}); "
            f"got {weight_hh.shape}"
        )

    # Per-output-channel scales. PyTorch nn.LSTM stacks i/f/g/o
    # gates on axis 0 — preserve that order.
    Wih_q_full = np.zeros((4 * HIDDEN, INPUT_DIM), dtype=np.int8)
    Whh_q_full = np.zeros((4 * HIDDEN, HIDDEN), dtype=np.int8)
    s_w_ih = np.zeros((N_GATES, HIDDEN), dtype=np.float32)
    s_w_hh = np.zeros((N_GATES, HIDDEN), dtype=np.float32)

    for g in range(N_GATES):
        sl = slice(g * HIDDEN, (g + 1) * HIDDEN)
        Wih_q, s_ih = quantize_symmetric_per_channel(weight_ih[sl], axis=1)
        Whh_q, s_hh = quantize_symmetric_per_channel(weight_hh[sl], axis=1)
        Wih_q_full[sl] = Wih_q
        Whh_q_full[sl] = Whh_q
        s_w_ih[g] = s_ih
        s_w_hh[g] = s_hh

    # Per-tensor activation scales for x (input) and h (recurrent).
    # We don't have ground-truth activations here — the calibrator
    # estimates them from the weight magnitudes as a pragmatic proxy.
    # In production, 's calibrate.py runs an activation reader
    # that observes real activation distributions; this fallback is
    # used by tests that don't have a calibration corpus.
    # For the test path: scale = 1.0 / 127 emulates a [-1, 1] dynamic
    # range typical of LSTM hidden states (post-tanh).
    s_x = 1.0 / 127.0
    s_h = 1.0 / 127.0
    # Output (h_t) range is bounded by tanh -> [-1, 1].
    s_y = 1.0 / 127.0

    cal = Int8Calibration(
        s_w_ih=s_w_ih,
        s_w_hh=s_w_hh,
        s_x=s_x,
        s_h=s_h,
        s_y=s_y,
    )
    return (
        Wih_q_full,
        Whh_q_full,
        bias_ih.astype(np.float32, copy=False),
        bias_hh.astype(np.float32, copy=False),
        cal,
    )

def _pack_input_int8(x: np.ndarray, s_x: float) -> tuple[np.ndarray, float]:
    """``(L, N=1, 96)`` fp32 -> ``(L * 96,)`` int8 contiguous."""
    if x.shape != (T_LSTM, 1, INPUT_DIM):
        raise ValueError(
            f"lstm_cell_int8 input must have shape ({T_LSTM}, 1, {INPUT_DIM}); "
            f"got {x.shape}"
        )
    if x.dtype != np.float32:
        raise ValueError(
            f"lstm_cell_int8 input must be FP32; got dtype={x.dtype}"
        )
    flat_fp32 = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    xq, s_x_actual = quantize_symmetric_per_tensor(flat_fp32, scale=s_x)
    return xq, s_x_actual

def _pack_compact_wb(
    Wih_q: np.ndarray,
    Whh_q: np.ndarray,
    bih: np.ndarray,
    bhh: np.ndarray,
    cal: Int8Calibration,
) -> bytes:
    """Pack the compact WB blob the runner.cpp parse_compact expects."""
    # Reorder per gate: PyTorch stacks i,f,g,o on axis 0; the runner
    # parser reads N_GATES * HIDDEN * INPUT_DIM contiguous blocks per
    # gate, so the gate-major axis-0 stacking matches directly.
    parts: list[bytes] = []
    parts.append(np.ascontiguousarray(Wih_q.reshape(-1)).tobytes())
    parts.append(np.ascontiguousarray(Whh_q.reshape(-1)).tobytes())
    parts.append(np.ascontiguousarray(bih.reshape(-1), dtype=np.float32).tobytes())
    parts.append(np.ascontiguousarray(bhh.reshape(-1), dtype=np.float32).tobytes())
    parts.append(np.ascontiguousarray(cal.s_w_ih.reshape(-1), dtype=np.float32).tobytes())
    parts.append(np.ascontiguousarray(cal.s_w_hh.reshape(-1), dtype=np.float32).tobytes())
    parts.append(struct.pack("<f", float(cal.s_x)))
    parts.append(struct.pack("<f", float(cal.s_h)))
    parts.append(struct.pack("<f", float(cal.s_y)))
    blob = b"".join(parts)
    if len(blob) != COMPACT_BLOB_BYTES:
        raise RuntimeError(
            f"compact WB blob size mismatch: built {len(blob)}, "
            f"expected {COMPACT_BLOB_BYTES}"
        )
    return blob

def _unpack_output_int8(buf_i8: np.ndarray, s_y: float) -> np.ndarray:
    """``(L * 96,)`` int8 -> ``(L, 1, 96)`` fp32 (de-quantized via s_y)."""
    expected = T_LSTM * HIDDEN
    if buf_i8.size != expected:
        raise ValueError(
            f"lstm output length mismatch: got {buf_i8.size}, expected {expected}"
        )
    yf = buf_i8.astype(np.float32) * float(s_y)
    return yf.reshape(T_LSTM, 1, HIDDEN)

class DoradoFastLstmCellInt8(NpuOp):
    """Single INT8 LSTM cell forward over a fixed time sequence."""

    name = "dorado_fast_lstm_cell_int8"

    INPUT_SHAPE = (T_LSTM, 1, INPUT_DIM)
    OUTPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    HIDDEN_SIZE = HIDDEN
    SEQ_LEN = T_LSTM
    PRECISION = "int8"

    def __init__(self) -> None:
        self.last_run: LstmCellInt8RunResult | None = None
        self.last_calibration: Int8Calibration | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(p.exists() for p in (_XCLBIN, _INSTS, _HOST_RUNNER))

    def __call__(
        self,
        *,
        x: np.ndarray,
        weight_ih: np.ndarray,
        weight_hh: np.ndarray,
        bias_ih: np.ndarray,
        bias_hh: np.ndarray,
        calibration: Int8Calibration | None = None,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 120.0,
    ) -> np.ndarray:
        for p in (_XCLBIN, _INSTS, _HOST_RUNNER):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name}: {p}. See "
                    f"{_ARTIFACT_DIR}/MANIFEST.md to rebuild."
                )

        # Compute per-channel calibration on-the-fly if the caller
        # didn't supply one. Production callers supply
        # the calibration from 's passport-tracked corpus.
        if calibration is None:
            (Wih_q, Whh_q, bih_f32, bhh_f32, calibration) = calibrate_lstm_weights(
                weight_ih, weight_hh, bias_ih, bias_hh,
            )
        else:
            (Wih_q, Whh_q, bih_f32, bhh_f32, _cal) = calibrate_lstm_weights(
                weight_ih, weight_hh, bias_ih, bias_hh,
            )
            # Caller-supplied calibration overrides the activation
            # scales (s_x, s_h, s_y) but keeps the freshly computed
            # per-channel weight scales (which are deterministic from
            # the weights).
            calibration = Int8Calibration(
                s_w_ih=_cal.s_w_ih,
                s_w_hh=_cal.s_w_hh,
                s_x=calibration.s_x,
                s_h=calibration.s_h,
                s_y=calibration.s_y,
            )

        x_packed, _ = _pack_input_int8(x, calibration.s_x)
        wb_blob = _pack_compact_wb(Wih_q, Whh_q, bih_f32, bhh_f32, calibration)

        with tempfile.TemporaryDirectory(prefix="t16_lstm_int8_") as td:
            tdp = Path(td)
            x_path = tdp / "input.int8.bin"
            wb_path = tdp / "wb.compact.bin"
            out_path = tdp / "output.int8.bin"

            x_packed.tofile(x_path)
            wb_path.write_bytes(wb_blob)

            cmd = [
                str(_HOST_RUNNER),
                "-x", str(_XCLBIN),
                "-i", str(_INSTS),
                "-k", "MLIR_AIE",
                "--input", str(x_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--seq", str(T_LSTM),
                "--hidden", str(HIDDEN),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
            ]
            proc = subprocess.run(  # noqa: S603 — argv fully controlled
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

            buf_i8 = np.fromfile(out_path, dtype=np.int8)

        out = _unpack_output_int8(buf_i8, calibration.s_y)
        self.last_run = LstmCellInt8RunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        self.last_calibration = calibration
        return out

register_npu_op(
    "dorado_fast_lstm_cell_int8",
    DoradoFastLstmCellInt8(),
)

__all__ = [
    "BIH_GATE",
    "BHH_GATE",
    "COMPACT_BLOB_BYTES",
    "DoradoFastLstmCellInt8",
    "HIDDEN",
    "INPUT_DIM",
    "Int8Calibration",
    "LSTM_CELL_INT8_TILE_MEMORY_USED_BYTES",
    "LstmCellInt8RunResult",
    "N_GATES",
    "T_LSTM",
    "WHH_GATE",
    "WIH_GATE",
    "_pack_compact_wb",
    "_pack_input_int8",
    "_unpack_output_int8",
    "calibrate_lstm_weights",
    "quantize_symmetric_per_channel",
    "quantize_symmetric_per_tensor",
]
