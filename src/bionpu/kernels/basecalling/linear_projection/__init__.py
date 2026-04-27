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

"""Dorado fast Linear projection (CRF head) — NPU op.

Registers ``dorado_fast_linear_projection`` with the NPU dispatch
registry. Implements ``Linear(in=96, out=256, bias=False) + Clamp(-5,5)``
on a fixed time-major input ``(L=334, N=1, 96)`` FP32, returning
``(L=334, N=1, 256)`` FP32.

The CRF head per is bias-free; the post-clamp range [-5, 5] is the
final FP32 logit feed for the host-side beam decoder. The op runs on a
single AIE2P tile with weights resident on tile (W_proj = 256*96 =
24,576 FP32 = 96 KiB — too big for one tile, so we shard along the
output dim into 4 OC groups of 64 outputs each, 24 KiB per group).
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

T_LSTM = 334
HIDDEN = 96
OUT_DIM = 256
OC_GROUP_SIZE = 64
N_OC_GROUPS = OUT_DIM // OC_GROUP_SIZE  # 4
WB_LEN = OUT_DIM * HIDDEN  # 24576 (no bias)
CLAMP_LO = -5.0
CLAMP_HI = 5.0

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dispatch"
    / "_npu_artifacts"
)
_ARTIFACT_DIR = _PACKAGE_ROOT / "dorado_fast_linear_projection"
_XCLBIN = _ARTIFACT_DIR / "final.xclbin"
_INSTS = _ARTIFACT_DIR / "insts.bin"
_HOST_RUNNER = _ARTIFACT_DIR / "host_runner"

# Match plain decimal AND scientific notation (e.g. "1.76837e+06us").
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
class LinearProjRunResult:
    output: np.ndarray  # (L, N=1, 256) FP32
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

def _pack_input(x: np.ndarray) -> np.ndarray:
    """``(L, N=1, 96)`` -> ``(L * 96,)`` FP32."""
    if x.shape != (T_LSTM, 1, HIDDEN):
        raise ValueError(
            f"linear_projection input must have shape ({T_LSTM}, 1, {HIDDEN}); "
            f"got {x.shape}"
        )
    if x.dtype != np.float32:
        raise ValueError(f"linear_projection input must be FP32; got {x.dtype}")
    return np.ascontiguousarray(x.reshape(-1), dtype=np.float32)

def _pack_wb(weight: np.ndarray) -> np.ndarray:
    """Pack ``(256, 96)`` weight into the host buffer.

    The IRON kernel runs ``T_LSTM * N_OC_GROUPS = 334 * 4 = 1336`` slabs
    per call; each slab is one OC group (64 rows × 96 cols = 6144 FP32).
    The host materialises ``T_LSTM`` repetitions of the 4-group cycle:
    ``T_LSTM * N_OC_GROUPS * OC_GROUP_SIZE * HIDDEN = 1336 * 6144 =
    8,208,384`` FP32 (~31 MB). A future revision can use BD-level
    repeat patterns to drop the bloat.
    """
    if weight.shape != (OUT_DIM, HIDDEN):
        raise ValueError(
            f"linear_projection weight must have shape "
            f"({OUT_DIM}, {HIDDEN}); got {weight.shape}"
        )
    W = weight.astype(np.float32, copy=False)
    # One cycle: (N_OC_GROUPS, OC_GROUP_SIZE, HIDDEN)
    cycle = W.reshape(N_OC_GROUPS, OC_GROUP_SIZE, HIDDEN)
    repeated = np.broadcast_to(
        cycle, (T_LSTM, N_OC_GROUPS, OC_GROUP_SIZE, HIDDEN)
    ).reshape(-1)
    return np.ascontiguousarray(repeated, dtype=np.float32)

def _unpack_output(flat: np.ndarray) -> np.ndarray:
    """``(L * 256,)`` -> ``(L, 1, 256)`` FP32."""
    expected = T_LSTM * OUT_DIM
    if flat.size != expected:
        raise ValueError(
            f"linear_projection output length mismatch: got {flat.size}, "
            f"expected {expected}"
        )
    return flat.reshape(T_LSTM, 1, OUT_DIM)

class DoradoFastLinearProjection(NpuOp):
    """CRF linear head + clamp on AIE2P."""

    name = "dorado_fast_linear_projection"

    INPUT_SHAPE = (T_LSTM, 1, HIDDEN)
    OUTPUT_SHAPE = (T_LSTM, 1, OUT_DIM)
    HIDDEN_SIZE = HIDDEN
    OUTPUT_SIZE = OUT_DIM
    SEQ_LEN = T_LSTM

    def __init__(self) -> None:
        self.last_run: LinearProjRunResult | None = None

    @classmethod
    def artifacts_present(cls) -> bool:
        return all(p.exists() for p in (_XCLBIN, _INSTS, _HOST_RUNNER))

    def __call__(
        self,
        *,
        x: np.ndarray,
        weight: np.ndarray,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 60.0,
    ) -> np.ndarray:
        for p in (_XCLBIN, _INSTS, _HOST_RUNNER):
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name}: {p}. See "
                    f"{_ARTIFACT_DIR}/MANIFEST.md to rebuild."
                )

        x_packed = _pack_input(x)
        wb_packed = _pack_wb(weight)

        with tempfile.TemporaryDirectory(prefix="t51_linear_") as td:
            tdp = Path(td)
            x_path = tdp / "input.f32.bin"
            wb_path = tdp / "wb.f32.bin"
            out_path = tdp / "output.f32.bin"

            x_packed.tofile(x_path)
            wb_packed.tofile(wb_path)

            cmd = [
                str(_HOST_RUNNER),
                "-x", str(_XCLBIN),
                "-i", str(_INSTS),
                "-k", "MLIR_AIE",
                "--input", str(x_path),
                "--wb", str(wb_path),
                "--output", str(out_path),
                "--seq", str(T_LSTM),
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

            flat = np.fromfile(out_path, dtype=np.float32)
        out = _unpack_output(flat)
        self.last_run = LinearProjRunResult(
            output=out,
            avg_us=avg_us,
            min_us=min_us,
            max_us=max_us,
            n_iters=int(n_iters),
        )
        return out

LINEAR_PROJECTION_TILE_MEMORY_USED_BYTES: int | None = None

register_npu_op(
    "dorado_fast_linear_projection",
    DoradoFastLinearProjection(),
)

__all__ = [
    "CLAMP_HI",
    "CLAMP_LO",
    "DoradoFastLinearProjection",
    "HIDDEN",
    "LINEAR_PROJECTION_TILE_MEMORY_USED_BYTES",
    "LinearProjRunResult",
    "N_OC_GROUPS",
    "OC_GROUP_SIZE",
    "OUT_DIM",
    "T_LSTM",
    "WB_LEN",
    "_pack_input",
    "_pack_wb",
    "_unpack_output",
]
