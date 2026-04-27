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

"""Peano export — quantized ONNX → MLIR-AIE → xclbin.

This module owns the lowering path that takes a quantized ONNX model
(produced by :func:`bionpu.quant.calibrate.quantize`) and emits a
ready-to-load AIE2P ``.xclbin`` plus matching userspace instruction
stream (``.bin``). Once lands, the artifacts will land under
``bionpu/dispatch/_npu_artifacts/<op>/`` next to the parity bundle
 ships.

 only ships the parity-test path against the canonical
``vector_scalar_mul`` example; the export pipeline itself is 's
deliverable. The function signature is fixed here so:

* the API contract is reviewable at review time without spinning
  up the full toolchain
* 's agent has a single, named entry point to fill in
* downstream callers can write their
  call sites against the final signature today

Implementation notes for whoever picks up 
----------------------------------------------

The pipeline (per umbrella PRD §4.3 + the bring-up agent's
PEANO_INSTALL_DIR convention) is:

1. Parse the ONNX graph; map each op node to an IRON
   ``aie.iron``-level construct. only needs ``Conv2d`` for the
   Dorado fast stem (basecalling B-M3 — see
   ``tracks/basecalling/reference/architecture.md``). / 
   extend the dispatch table.
2. Emit MLIR-AIE via the IRON API (``aie.iron.program``,
   placement helpers, etc).
3. Drive ``aiecc`` (which calls Peano via
   ``${PEANO_INSTALL_DIR}/bin/clang++`` + xchesscc fallback) to
   produce the xclbin and instruction stream. The bring-up environment
   exports ``PEANO_INSTALL_DIR`` from ironenv.
4. Verify the xclbin loads on the live NPU (probe via the same
   ``NpuBackend.probe`` :mod:`bionpu.dispatch.npu` uses).
5. Stamp a passport-like record next to the xclbin with the source
   ONNX sha256, target architecture, NPU2 flag, build host, and
   bring-up commit pins.

Until lands the function raises :class:`NotImplementedError`
with a message that names the task ID so anyone tripping over it from
a -later call site sees the right next step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# Public type alias — pinned here so 's call sites can import it.
PeanoTarget = Literal["aie2", "aie2p"]

def peano_export(
    quantized_onnx: Path | str,
    output_xclbin: Path | str,
    *,
    target: PeanoTarget = "aie2p",
    output_insts: Path | str | None = None,
    op_name: str | None = None,
) -> Path:
    """Lower a quantized ONNX model to an AIE2(P) xclbin via Peano.

    Args:
        quantized_onnx: path to the input ONNX-INT8 / ONNX-INT16 model
            (typically the output of :func:`bionpu.quant.quantize`).
        output_xclbin: where to write the compiled ``.xclbin``.
        target: ``"aie2"`` (Phoenix / NPU1) or ``"aie2p"`` (Strix /
            NPU2). The bring-up agent verified the host as
            ``aie2p`` (6×8 = 48 tiles); 's first run pins this.
        output_insts: where to write the userspace instruction stream
            (``.bin``). Defaults to ``output_xclbin.with_suffix('.bin')``.
        op_name: optional name to record in the artifact's MANIFEST so
            :data:`bionpu.dispatch.NPU_OPS` registration is symbolic
            rather than path-based.

    Returns:
        The resolved ``output_xclbin`` :class:`pathlib.Path` (for
        chaining).

    Raises:
        NotImplementedError: is the parity test against
            vector_scalar_mul; the ONNX → MLIR-AIE → xclbin pipeline
            is 's deliverable. The message includes "" so call
            sites can grep for the contract.
    """
    raise NotImplementedError(
        "peano_export is wired by ; only ships the parity "
        "test against the canonical vector_scalar_mul example "
        "(precompiled xclbin under "
        "bionpu/dispatch/_npu_artifacts/vector_scalar_mul/). "
        f"Called with quantized_onnx={quantized_onnx!r}, "
        f"output_xclbin={output_xclbin!r}, target={target!r}, "
        f"op_name={op_name!r}. See plan + "
        "bio-on-xdna-plan.md §."
    )

__all__ = ["PeanoTarget", "peano_export"]
