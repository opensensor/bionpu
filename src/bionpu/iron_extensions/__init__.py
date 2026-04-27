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

"""IRON extension layer for AIE2P primitives that the upstream
``mlir-aie`` wheel exposes at the MLIR-dialect level but not at the IRON
Python (``aie.iron``) layer.

This module sits **above** the installed wheel under
``<ironenv>/lib/python3.11/site-packages/mlir_aie``. It does
not fork the wheel; it re-uses ``aie.iron.{Worker,ObjectFifo,Program,
Runtime,Kernel}`` and the lower-level ``aie.dialects.aie`` op-builders,
adding small helpers that compose those primitives into the dataflow
patterns that AM020 documents but IRON doesn't expose directly.

Surface ():

- :func:`cascade_stream_chain` — declarative IRON topology helper for
  AM020 Ch. 4 p. 67 cascade-stream chaining of N CoreTile workers
  inside a single column. Vertical adjacency + matched
  ``put_mcd``/``get_scd`` C++ kernel intrinsics produce the
  accumulator-to-accumulator stream the upstream
  ``programming_examples/basic/matrix_multiplication/cascade/`` example
  uses for its 4-row matmul.

For a deeper dive into the IRON-vs-MLIR-dialect boundary and
which AM020 primitives are reachable as Python-level extensions vs
require a source rebuild, see :doc:`INVENTORY.md`.
"""

from __future__ import annotations

from .cascade_stream import (
    CASCADE_BITS,
    CASCADE_LANES_FP32,
    CASCADE_LANES_INT32,
    CascadeRole,
    CascadeStreamChain,
    cascade_stream_chain,
)

__all__ = [
    "CASCADE_BITS",
    "CASCADE_LANES_FP32",
    "CASCADE_LANES_INT32",
    "CascadeRole",
    "CascadeStreamChain",
    "cascade_stream_chain",
]
