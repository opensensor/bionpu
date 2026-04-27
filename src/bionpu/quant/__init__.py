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

"""Quantization toolkit + Peano export placeholder.

Public surface:

* :class:`CalibrationConfig` — strategy + precision + calibration corpus ID.
* :class:`CalibrationDataReader` — abstract base mirroring ORT's reader.
* :class:`NumpyCalibrationDataReader` — streams calibration batches from
  an ``.npz`` archive.
* :class:`AWQCalibrator` — placeholder for transformer activation-aware
  quant; raises :class:`NotImplementedError` with a message naming the
  LSTM-CRF case (Dorado fast does NOT need AWQ).
* :func:`quantize` — drive ORT static PTQ from FP32 ONNX → quantized ONNX.
* :class:`Passport` — per-model passport dataclass; round-trips through
  :mod:`bionpu.quant.passport` against ``passport.schema.json``.
* :func:`peano_export` — Peano lowering placeholder; raises
  :class:`NotImplementedError("......")`. ships only the
  parity test against the precompiled vector_scalar_mul example;
  the real ONNX → MLIR-AIE → xclbin pipeline is 's job.
* :data:`PeanoTarget` — type alias for the target architecture
  (``"aie2"`` / ``"aie2p"``); pinned at so 's call sites
  can import it.
"""

from .calibrate import (
    AWQCalibrator,
    CalibrationConfig,
    CalibrationDataReader,
    NumpyCalibrationDataReader,
    Precision,
    Strategy,
    quantize,
)
from .passport import Passport
from .peano_export import PeanoTarget, peano_export

__all__ = [
    "AWQCalibrator",
    "CalibrationConfig",
    "CalibrationDataReader",
    "NumpyCalibrationDataReader",
    "Passport",
    "PeanoTarget",
    "Precision",
    "Strategy",
    "peano_export",
    "quantize",
]
