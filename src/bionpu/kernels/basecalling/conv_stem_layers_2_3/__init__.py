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

"""Dorado fast stem Conv1d layers 2 + 3 — NPU ops.

Registers two ops with :data:`bionpu.dispatch.NPU_OPS`:

- ``dorado_fast_conv_stem_layer2`` — Conv1d(16, 16, k=5, s=1, p=2)
  + SiLU + Clamp[-0.5, 3.5] on a fixed (1, 16, T=2000) FP32 input.
- ``dorado_fast_conv_stem_layer3`` — Conv1d(16, 96, k=19, s=6, p=9)
  + SiLU + Clamp[-0.5, 3.5] on a fixed (1, 16, T=2000) FP32 input,
  producing (1, 96, T_out=334).

Both layers reuse 's chunked-dataflow + custom-host-runner pattern
(``conv_stem/__init__.py``). Layer 2 is mechanically the same as
 (kernel=5, stride=1, padding=2) but multi-channel; layer 3 has
the harder geometry (kernel=19, stride=6, padding=9) so it's expressed
as a single non-chunked kernel call (see DESIGN comment below).

The activation+clamp are folded into the conv kernel — same architecture
as 's ConvBlock and as Dorado's inference graph (BN absorbed in
weights at training-serialize time).

Why the two layers ship together (one file): they share the pre/post
pack helpers, both use the same Makefile pattern (parameterised on
op-name), and they're called back-to-back in the encoder pipeline.
Splitting them across two packages added zero modularity but a lot of
duplicated plumbing.
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

# Pinned op geometry. T_IN is the public input time dimension across the
# encoder body; T_OUT_L3 is the conv-stem output length after stride 6.
T_IN = 2000

# --- Layer 2: Conv1d(16, 16, k=5, s=1, p=2) -------------------------------- #
L2_T_CHUNK = 200
L2_N_CHUNKS = T_IN // L2_T_CHUNK
L2_PAD = 2
L2_K = 5
L2_IN_CH = 16
L2_OUT_CH = 16
# Weights packed (oc-major, ic-row-major, kernel-minor) followed by bias
# (length OUT_CH). Length = 16*16*5 + 16 = 1296 floats.
L2_WB_LEN = L2_OUT_CH * L2_IN_CH * L2_K + L2_OUT_CH

# --- Layer 3: Conv1d(16, 96, k=19, s=6, p=9) ------------------------------- #
# Output length per PyTorch Conv1d: floor((T_IN + 2p - k) / s) + 1
#                = floor((2000 + 18 - 19) / 6) + 1 = 333 + 1 = 334.
# Stride 6 + kernel 19 means receptive-field overlap is 13 samples per
# adjacent chunk. We avoid the chunking complexity by building layer 3
# as a single non-chunked kernel call: input length 2000 + 18 = 2018
# samples; output length 334. With FP32, the weight buffer is 16*96*19
# + 96 = 29280 floats = 117 KiB; the input buffer is 16*2018 = 32288
# floats = 126 KiB. Both live in DRAM and stream through the tile.
# Tile-local working memory: per-output-step (96 floats) + a kernel
# window (16*19 = 304 floats) — well under 64 KiB.
L3_T_OUT = 334
L3_PAD = 9
L3_K = 19
L3_STRIDE = 6
L3_IN_CH = 16
L3_OUT_CH = 96
# Output channels are processed in groups of 16 (the IRON kernel
# acquires one wb slice covering a single OC group per call). 6 groups
# of 16 cover the full 96 output channels.
L3_OC_GROUP_SIZE = 16
# Weight packing: (oc, ic, k) row-major + (oc,) bias.
L3_WB_LEN = L3_OUT_CH * L3_IN_CH * L3_K + L3_OUT_CH
L3_INPUT_PADDED_LEN = T_IN + 2 * L3_PAD  # 2018 samples per channel
# T_OUT_L3 chunking: a single chunk of 334 output samples; layer 3 runs
# in one kernel invocation. may revisit chunking; for v1 the
# memory budget is fine.

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_L2_DIR = _PACKAGE_ROOT / "dorado_fast_conv_stem_layer2"
_L2_XCLBIN = _L2_DIR / "final.xclbin"
_L2_INSTS = _L2_DIR / "insts.bin"
_L2_RUNNER = _L2_DIR / "host_runner"

_L3_DIR = _PACKAGE_ROOT / "dorado_fast_conv_stem_layer3"
_L3_XCLBIN = _L3_DIR / "final.xclbin"
_L3_INSTS = _L3_DIR / "insts.bin"
_L3_RUNNER = _L3_DIR / "host_runner"

# Match plain decimal AND scientific notation (e.g. "1.76837e+06us"); the
# host runner's iostream emits scientific when the float is >= ~1e6 us.
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
class ConvLayerRunResult:
    output: np.ndarray  # FP32, shape per layer
    avg_us: float
    min_us: float
    max_us: float
    n_iters: int

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

# --------------------------------------------------------------------------- #
# Layer 2 helpers
# --------------------------------------------------------------------------- #

def _l2_pack_signal(signal: np.ndarray) -> np.ndarray:
    """(1, 16, T=2000) -> (N_CHUNKS, 16, T_CHUNK + 4) FP32 contiguous.

    Mirrors 's chunking but with 16 input channels: we zero-pad each
    channel along time, then slice into N overlapping chunks of length
    T_CHUNK + 4 samples. Per-channel layout within a chunk is row-major
    (chunk, channel, time).
    """
    if signal.shape != (1, L2_IN_CH, T_IN):
        raise ValueError(
            f"layer2 signal must have shape (1, {L2_IN_CH}, {T_IN}); "
            f"got {signal.shape}"
        )
    if signal.dtype != np.float32:
        raise ValueError(f"layer2 signal must be FP32; got dtype={signal.dtype}")
    s = signal.reshape(L2_IN_CH, T_IN)
    padded = np.zeros((L2_IN_CH, T_IN + 2 * L2_PAD), dtype=np.float32)
    padded[:, L2_PAD : L2_PAD + T_IN] = s
    chunks = np.empty(
        (L2_N_CHUNKS, L2_IN_CH, L2_T_CHUNK + 2 * L2_PAD), dtype=np.float32
    )
    for i in range(L2_N_CHUNKS):
        start = i * L2_T_CHUNK
        chunks[i] = padded[:, start : start + L2_T_CHUNK + 2 * L2_PAD]
    return chunks

def _l2_pack_wb(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """(16, 16, 5) + (16,) -> length-1296 FP32 contiguous buffer."""
    if weight.shape != (L2_OUT_CH, L2_IN_CH, L2_K):
        raise ValueError(
            f"layer2 weight must have shape "
            f"({L2_OUT_CH}, {L2_IN_CH}, {L2_K}); got {weight.shape}"
        )
    if bias.shape != (L2_OUT_CH,):
        raise ValueError(
            f"layer2 bias must have shape ({L2_OUT_CH},); got {bias.shape}"
        )
    W = weight.astype(np.float32, copy=False).reshape(-1)
    B = bias.astype(np.float32, copy=False)
    return np.concatenate([W, B]).astype(np.float32)

def _l2_unpack_output(flat: np.ndarray) -> np.ndarray:
    """(N_CHUNKS * 16 * T_CHUNK,) -> (1, 16, T) FP32."""
    expected = L2_N_CHUNKS * L2_OUT_CH * L2_T_CHUNK
    if flat.size != expected:
        raise ValueError(
            f"layer2 output length mismatch: got {flat.size}, expected {expected}"
        )
    chunked = flat.reshape(L2_N_CHUNKS, L2_OUT_CH, L2_T_CHUNK)
    return chunked.transpose(1, 0, 2).reshape(1, L2_OUT_CH, T_IN)

# --------------------------------------------------------------------------- #
# Layer 3 helpers
# --------------------------------------------------------------------------- #

def _l3_pack_signal(signal: np.ndarray) -> np.ndarray:
    """Pack ``(1, 16, T=2000)`` into the layer-3 per-step input stream.

    The IRON kernel acquires `signal_in` once per output time step
    (``T_OUT = 334`` total). Each acquire consumes one slice of
    ``(in_ch=16, kernel=19)`` floats covering the input window for
    that output: ``window[t] = padded_input[:, t*STRIDE : t*STRIDE + K]``
    where ``padded_input`` is the host's zero-padded raw signal of
    shape ``(16, T_IN + 2*PAD = 2018)``.

    Returns:
        ndarray shape ``(T_OUT * 16 * 19,)`` = ``(101_536,)`` floats,
        slice-major (slice, channel, kernel) row-major.
    """
    if signal.shape != (1, L3_IN_CH, T_IN):
        raise ValueError(
            f"layer3 signal must have shape (1, {L3_IN_CH}, {T_IN}); "
            f"got {signal.shape}"
        )
    if signal.dtype != np.float32:
        raise ValueError(f"layer3 signal must be FP32; got dtype={signal.dtype}")
    s = signal.reshape(L3_IN_CH, T_IN)
    padded = np.zeros((L3_IN_CH, L3_INPUT_PADDED_LEN), dtype=np.float32)
    padded[:, L3_PAD : L3_PAD + T_IN] = s
    # Build per-step windows: out shape (T_OUT, 16, 19).
    out = np.empty((L3_T_OUT, L3_IN_CH, L3_K), dtype=np.float32)
    for t in range(L3_T_OUT):
        start = t * L3_STRIDE
        out[t] = padded[:, start : start + L3_K]
    return out.reshape(-1)

def _l3_pack_wb(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Pack (96, 16, 19) + (96,) into the layer-3 wb stream.

    The IRON kernel acquires `wb_in` ``N_OC_GROUPS * T_OUT = 6 * 334 =
    2004`` times across one run; each acquire consumes one OC-group
    slice (4880 floats: 16*16*19 weights + 16 biases). The host
    therefore materialises ``T_OUT`` repetitions of the 6-group cycle
    into a single ``T_OUT * 6 * 4880 = 9_778_560`` float buffer (~37 MB
    FP32). Each cycle within a repetition lays the OC groups in order
    (OC group g covers output channels [g*16:(g+1)*16]). The slice for
    group g is the contiguous ``(weight[g*16:(g+1)*16], bias[g*16:(g+1)*16])``
    pair, in (oc_in_group, ic, k) row-major + bias-tail layout.

    A future revision can use BD-level repeat patterns to drop the
    repetition bloat; the v1 path prioritises correctness and stays
    inside host RAM.
    """
    if weight.shape != (L3_OUT_CH, L3_IN_CH, L3_K):
        raise ValueError(
            f"layer3 weight must have shape "
            f"({L3_OUT_CH}, {L3_IN_CH}, {L3_K}); got {weight.shape}"
        )
    if bias.shape != (L3_OUT_CH,):
        raise ValueError(
            f"layer3 bias must have shape ({L3_OUT_CH},); got {bias.shape}"
        )
    W = weight.astype(np.float32, copy=False)
    B = bias.astype(np.float32, copy=False)

    n_groups = L3_OUT_CH // L3_OC_GROUP_SIZE  # 6
    cycle = np.empty(
        (n_groups, L3_OC_GROUP_SIZE * L3_IN_CH * L3_K + L3_OC_GROUP_SIZE),
        dtype=np.float32,
    )
    for g in range(n_groups):
        oc_lo = g * L3_OC_GROUP_SIZE
        oc_hi = oc_lo + L3_OC_GROUP_SIZE
        gw = W[oc_lo:oc_hi].reshape(-1)  # (16*16*19,)
        gb = B[oc_lo:oc_hi]  # (16,)
        cycle[g, : gw.size] = gw
        cycle[g, gw.size :] = gb
    # Repeat the cycle T_OUT times: total buffer = T_OUT * n_groups *
    # slice. Use np.broadcast_to + np.ascontiguousarray to avoid
    # materialising the full buffer twice.
    repeated = np.broadcast_to(
        cycle, (L3_T_OUT, n_groups, cycle.shape[1])
    ).reshape(-1)
    return np.ascontiguousarray(repeated, dtype=np.float32)

def _l3_unpack_output(flat: np.ndarray) -> np.ndarray:
    """``(T_OUT * 6 * 16,)`` -> ``(1, 96, T_OUT)`` FP32 (NCL contiguous).

    The kernel produces output in (time, oc_group, oc_in_group)
    row-major order (one OC-group slice of 16 floats per kernel call,
    cycling through 6 groups per time step). We reshape to
    ``(T_OUT, 6, 16)``, flatten the last two dims to ``(T_OUT, 96)``
    (out_channel-major within each time step in (oc_group, oc_in_group)
    interleaving), then transpose+reshape to NCL.
    """
    n_groups = L3_OUT_CH // L3_OC_GROUP_SIZE
    expected = L3_T_OUT * n_groups * L3_OC_GROUP_SIZE  # = T_OUT * 96
    if flat.size != expected:
        raise ValueError(
            f"layer3 output length mismatch: got {flat.size}, expected {expected}"
        )
    # (T_OUT, 6, 16) -> (T_OUT, 96): groups stored back-to-back gives
    # the natural (g*16 + i_in_group)-th oc index = g*16 + i_in_group
    # (i.e. group 0 is oc 0..15, group 1 is oc 16..31, etc.). The
    # flat output is exactly that order.
    nlc = flat.reshape(L3_T_OUT, L3_OUT_CH)
    # NLC -> NCL with N=1.
    return nlc.transpose(1, 0).reshape(1, L3_OUT_CH, L3_T_OUT)

# --------------------------------------------------------------------------- #
# NpuOp wrappers
# --------------------------------------------------------------------------- #

def _run_host_runner(
    *,
    runner: Path,
    xclbin: Path,
    insts: Path,
    sig_path: Path,
    wb_path: Path,
    out_path: Path,
    extra_args: list[str],
    n_iters: int,
    warmup: int,
    timeout_s: float,
) -> tuple[float, float, float]:
    cmd = [
        str(runner),
        "-x", str(xclbin),
        "-i", str(insts),
        "-k", "MLIR_AIE",
        "--signal", str(sig_path),
        "--wb", str(wb_path),
        "--output", str(out_path),
        *extra_args,
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
    return float(m_avg.group(1)), float(m_min.group(1)), float(m_max.group(1))

class DoradoFastConvStemLayer2(NpuOp):
    """Second Dorado fast stem Conv1d layer."""

    name = "dorado_fast_conv_stem_layer2"

    INPUT_SHAPE = (1, L2_IN_CH, T_IN)
    OUTPUT_SHAPE = (1, L2_OUT_CH, T_IN)
    KERNEL_SIZE = L2_K
    PADDING = L2_PAD
    STRIDE = 1
    IN_CHANNELS = L2_IN_CH
    OUT_CHANNELS = L2_OUT_CH

    def __init__(self) -> None:
        self.last_run: ConvLayerRunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(p.exists() for p in (_L2_XCLBIN, _L2_INSTS, _L2_RUNNER))

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
        for p in (_L2_XCLBIN, _L2_INSTS, _L2_RUNNER):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name}: {p}. See "
                    f"{_L2_DIR}/MANIFEST.md to rebuild."
                )

        sig_packed = _l2_pack_signal(signal)
        wb_packed = _l2_pack_wb(weight, bias)

        with tempfile.TemporaryDirectory(prefix="t51_l2_") as td:
            tdp = Path(td)
            sig_path = tdp / "signal.f32.bin"
            wb_path = tdp / "wb.f32.bin"
            out_path = tdp / "output.f32.bin"

            sig_packed.tofile(sig_path)
            wb_packed.tofile(wb_path)

            avg_us, min_us, max_us = _run_host_runner(
                runner=_L2_RUNNER,
                xclbin=_L2_XCLBIN,
                insts=_L2_INSTS,
                sig_path=sig_path,
                wb_path=wb_path,
                out_path=out_path,
                extra_args=["--time", str(T_IN), "--chunk", str(L2_T_CHUNK)],
                n_iters=n_iters,
                warmup=warmup,
                timeout_s=timeout_s,
            )
            flat = np.fromfile(out_path, dtype=np.float32)
        out = _l2_unpack_output(flat)
        self.last_run = ConvLayerRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out

class DoradoFastConvStemLayer3(NpuOp):
    """Third Dorado fast stem Conv1d layer."""

    name = "dorado_fast_conv_stem_layer3"

    INPUT_SHAPE = (1, L3_IN_CH, T_IN)
    OUTPUT_SHAPE = (1, L3_OUT_CH, L3_T_OUT)
    KERNEL_SIZE = L3_K
    PADDING = L3_PAD
    STRIDE = L3_STRIDE
    IN_CHANNELS = L3_IN_CH
    OUT_CHANNELS = L3_OUT_CH

    def __init__(self) -> None:
        self.last_run: ConvLayerRunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(p.exists() for p in (_L3_XCLBIN, _L3_INSTS, _L3_RUNNER))

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
        for p in (_L3_XCLBIN, _L3_INSTS, _L3_RUNNER):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name}: {p}. See "
                    f"{_L3_DIR}/MANIFEST.md to rebuild."
                )

        sig_packed = _l3_pack_signal(signal)
        wb_packed = _l3_pack_wb(weight, bias)

        with tempfile.TemporaryDirectory(prefix="t51_l3_") as td:
            tdp = Path(td)
            sig_path = tdp / "signal.f32.bin"
            wb_path = tdp / "wb.f32.bin"
            out_path = tdp / "output.f32.bin"

            sig_packed.tofile(sig_path)
            wb_packed.tofile(wb_path)

            avg_us, min_us, max_us = _run_host_runner(
                runner=_L3_RUNNER,
                xclbin=_L3_XCLBIN,
                insts=_L3_INSTS,
                sig_path=sig_path,
                wb_path=wb_path,
                out_path=out_path,
                extra_args=[
                    "--in-time", str(L3_INPUT_PADDED_LEN),
                    "--out-time", str(L3_T_OUT),
                ],
                n_iters=n_iters,
                warmup=warmup,
                timeout_s=timeout_s,
            )
            flat = np.fromfile(out_path, dtype=np.float32)
        out = _l3_unpack_output(flat)
        self.last_run = ConvLayerRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out

# Tile-local memory used per kernel (parsed from main_core_*.ld.script
# at build time and recorded in the corresponding MANIFEST.md). The
# numbers below are placeholders that the build script re-stamps once
# the xclbins land. See MANIFEST.md for per-buffer breakdowns.
LAYER2_TILE_MEMORY_USED_BYTES: int | None = None
LAYER3_TILE_MEMORY_USED_BYTES: int | None = None

register_npu_op(
    "dorado_fast_conv_stem_layer2",
    DoradoFastConvStemLayer2(),
)
register_npu_op(
    "dorado_fast_conv_stem_layer3",
    DoradoFastConvStemLayer3(),
)

__all__ = [
    "ConvLayerRunResult",
    "DoradoFastConvStemLayer2",
    "DoradoFastConvStemLayer3",
    "L2_IN_CH",
    "L2_K",
    "L2_N_CHUNKS",
    "L2_OUT_CH",
    "L2_PAD",
    "L2_T_CHUNK",
    "L2_WB_LEN",
    "L3_INPUT_PADDED_LEN",
    "L3_IN_CH",
    "L3_K",
    "L3_OC_GROUP_SIZE",
    "L3_OUT_CH",
    "L3_PAD",
    "L3_STRIDE",
    "L3_T_OUT",
    "L3_WB_LEN",
    "T_IN",
    "_l2_pack_signal",
    "_l2_pack_wb",
    "_l2_unpack_output",
    "_l3_pack_signal",
    "_l3_pack_wb",
    "_l3_unpack_output",
]
