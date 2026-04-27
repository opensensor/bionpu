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

"""Calibration + quantization driver.

Thin wrapper around :mod:`onnxruntime.quantization` that produces an
ONNX-INT8 / ONNX-INT16 model from a FP32 ONNX input, plus the matching
:class:`bionpu.quant.passport.Passport` (caller-driven; this module only
runs the quantizer and exposes the strategy enum).

Design notes
------------

* Per umbrella PRD §4.3: post-training calibration only at v1. AWQ-style
  hooks for transformers are stubbed (:class:`AWQCalibrator`) and raise
  :class:`NotImplementedError` with a message naming the LSTM-CRF case
  so a future agent doesn't accidentally route the LSTM Dorado model
  through them.
* Per-tensor and per-channel strategies are ORT's canonical PTQ paths;
  ``per_channel=True`` toggles per-channel weight quantization on
  Conv/MatMul nodes.
* INT16 (``QInt16``) was added to onnxruntime around 1.17. We probe the
  symbol at call time and fail loud if absent rather than silently
  degrading to INT8.
* No NPU-specific lowering. 's ``peano_export.py`` will consume the
  artifact emitted here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# onnxruntime is the heavy-lifter; we import lazily so the rest of the
# package is usable on a host without ORT (e.g. the writeup pipeline).

Strategy = Literal["per_tensor", "per_channel", "awq"]
Precision = Literal["int8", "int16"]

@dataclass
class CalibrationConfig:
    """User-facing PTQ configuration.

    Attributes:
        strategy: one of ``per_tensor``, ``per_channel``, ``awq``.
        precision: ``int8`` or ``int16``.
        calibration_data_id: stable identifier for the calibration corpus
            (e.g. ``pod5_hg002_subsample_seed42_n128``); recorded in the
            passport.
        n_calibration_samples: number of samples the reader is expected
            to expose; recorded in the passport. The reader MAY exhaust
            earlier (e.g. on small fixtures) — ``Passport.n_samples``
            captures the actual count.
    """

    strategy: Strategy
    precision: Precision
    calibration_data_id: str
    n_calibration_samples: int

class CalibrationDataReader(ABC):
    """Abstract base mirroring ``onnxruntime.quantization.CalibrationDataReader``.

    Implementers yield ``{input_name: np.ndarray}`` dicts via
    :meth:`get_next` until the data is exhausted, then return ``None``.
    The concrete implementation here is :class:`NumpyCalibrationDataReader`
    which streams from an .npz file.
    """

    @abstractmethod
    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next feed dict, or ``None`` when exhausted."""

    def rewind(self) -> None:  # noqa: B027  — intentionally empty default
        """Reset internal state so calibration can be re-driven.

        Default no-op so simple readers don't have to implement it.
        """

class NumpyCalibrationDataReader(CalibrationDataReader):
    """Stream calibration batches from an ``.npz`` archive.

    The archive must contain a single array (or one keyed by
    ``input_name``) with leading dimension == sample count. Each call to
    :meth:`get_next` yields one sample wrapped to its model batch axis.
    """

    def __init__(self, npz_file: Path, input_name: str) -> None:
        self._path = Path(npz_file)
        self._input_name = input_name
        self._iter = self._build_iter()

    def _build_iter(self) -> Iterator[dict[str, np.ndarray]]:
        with np.load(self._path) as data:
            if self._input_name in data.files:
                arr = np.asarray(data[self._input_name])
            elif len(data.files) == 1:
                arr = np.asarray(data[data.files[0]])
            else:
                raise KeyError(
                    f"NumpyCalibrationDataReader: input '{self._input_name}' "
                    f"not found in {self._path} (keys: {data.files})"
                )
        # Eager copy so the file handle can close above and we don't
        # depend on lazy NpzFile lifetime during iteration.
        for i in range(arr.shape[0]):
            yield {self._input_name: arr[i : i + 1]}

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = self._build_iter()

class AWQCalibrator(CalibrationDataReader):
    """Activation-aware weight-quantization placeholder.

    Dorado fast is LSTM-CRF (see ``tracks/basecalling/architecture.md``)
    and does NOT need AWQ. CRISPR scoring might if it picks a
    transformer (CRISPR-Net / DeepCRISPR with attention). Until that
    decision lands, this stub fails loud.
    """

    def get_next(self) -> dict[str, np.ndarray] | None:
        raise NotImplementedError(
            "AWQ calibration is not wired in . AWQ targets "
            "transformer attention layers; Dorado fast is LSTM-CRF and "
            "uses per_tensor / per_channel PTQ instead. Wire AWQ if and "
            "when a transformer model lands ( CRISPR scoring may "
            "qualify)."
        )

def _ort_quant_types(precision: Precision):
    """Resolve the ORT QuantType pair for a precision label.

    Raises:
        RuntimeError: if INT16 is requested on an ORT version without
            ``QInt16`` exposed.
    """
    from onnxruntime.quantization import QuantType

    if precision == "int8":
        return QuantType.QInt8, QuantType.QUInt8
    if precision == "int16":
        if not hasattr(QuantType, "QInt16"):
            raise RuntimeError(
                "INT16 quantization requested but the installed "
                "onnxruntime does not expose QuantType.QInt16. Upgrade "
                "onnxruntime>=1.17 or fall back to int8."
            )
        return QuantType.QInt16, QuantType.QInt16
    raise ValueError(f"unsupported precision: {precision!r}")

def quantize(
    input_onnx: Path,
    output_onnx: Path,
    config: CalibrationConfig,
    reader: CalibrationDataReader,
) -> Path:
    """Run static PTQ from FP32 ONNX → quantized ONNX.

    Args:
        input_onnx: path to the FP32 ONNX file (typically produced by
             export).
        output_onnx: where to write the quantized ONNX.
        config: calibration configuration (strategy, precision, IDs).
        reader: data source. Must be a :class:`CalibrationDataReader`
            subclass; the AWQ stub will raise here if used.

    Returns:
        ``output_onnx`` (for fluent chaining).
    """
    if config.strategy == "awq":
        # Force the contextual error rather than letting ORT discover an
        # empty calibration set.
        reader.get_next()
        # If get_next() somehow returned (it shouldn't), make sure we
        # still raise:
        raise NotImplementedError(
            "AWQ strategy is not yet implemented; see AWQCalibrator."
        )

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        quantize_static,
    )

    weight_type, activation_type = _ort_quant_types(config.precision)

    input_path = Path(input_onnx)
    output_path = Path(output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_channel = config.strategy == "per_channel"

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        weight_type=weight_type,
        activation_type=activation_type,
        calibrate_method=CalibrationMethod.MinMax,
    )
    return output_path
