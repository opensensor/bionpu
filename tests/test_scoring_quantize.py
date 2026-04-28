"""bionpu.scoring.quantize — quantization passport tests.

GPL-3.0. (c) 2026 OpenSensor.

Smoke tests for the AIE2P-targeted INT8 calibration. The full
end-to-end calibrator drives a 89M-parameter BERT body which is too
heavy for the test suite — we test the load/save cycle, the dataclass
shapes, and the quantization math on synthetic tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from bionpu.scoring.quantize import (
    ActivationScale,
    QuantizationPassport,
    QuantizedTensor,
    load_passport,
    save_passport,
)


def test_passport_round_trip_via_json(tmp_path: Path) -> None:
    p = QuantizationPassport(
        model_id="dnabert-epi-noEpi",
        base_model="zhihan1996/DNA_bert_3",
        source_checkpoint_sha256="a" * 64,
        calibration_corpus_id="test-corpus",
        n_calibration_samples=128,
        created_at_utc=1714000000.0,
        weights=[
            QuantizedTensor(
                name="bert.encoder.layer.0.attention.self.query.weight",
                int8_path="w0000",
                scale_per_channel=(0.001, 0.002, 0.003),
                fp_shape=(3, 768),
            ),
        ],
        activations=[
            ActivationScale(
                layer_name="bert.encoder.layer.0.attention.self.query.weight",
                scale=0.05,
                n_calibration_samples=128,
            ),
        ],
    )
    out = tmp_path / "passport.json"
    save_passport(p, out)
    p2 = load_passport(out)

    assert p2.model_id == p.model_id
    assert p2.source_checkpoint_sha256 == p.source_checkpoint_sha256
    assert p2.calibration_corpus_id == "test-corpus"
    assert p2.n_calibration_samples == 128
    assert len(p2.weights) == 1
    assert p2.weights[0].name == p.weights[0].name
    assert p2.weights[0].fp_shape == (3, 768)
    assert p2.weights[0].scale_per_channel == (0.001, 0.002, 0.003)
    assert len(p2.activations) == 1
    assert p2.activations[0].scale == 0.05


def test_passport_json_is_human_readable(tmp_path: Path) -> None:
    """Passport JSON must be indented + machine-parseable."""
    p = QuantizationPassport(
        model_id="dnabert-epi-noEpi",
        n_calibration_samples=128,
    )
    out = tmp_path / "p.json"
    save_passport(p, out)
    text = out.read_text()
    assert text.endswith("\n")
    assert '"model_id": "dnabert-epi-noEpi"' in text
    # Round-trips as JSON.
    parsed = json.loads(text)
    assert parsed["schema_version"] == "0.4-alpha"


def test_quantization_round_trip_bounded_error() -> None:
    """Per-output-channel symmetric INT8 must round-trip within
    ~1% of max-abs per channel — that's the design tolerance."""
    g = torch.Generator().manual_seed(0)
    W_fp = torch.randn(64, 32, generator=g, dtype=torch.float32).numpy()

    # Same math as in calibrate_dnabert_epi: per-out-channel max-abs.
    scale = np.maximum(np.abs(W_fp).max(axis=1), 1e-12) / 127.0
    W_int8 = np.clip(np.round(W_fp / scale[:, None]), -128, 127).astype(np.int8)
    W_dq = W_int8.astype(np.float32) * scale[:, None]

    # Per-row error <= scale (one quantization step) which is ~max-abs/127.
    err = np.abs(W_fp - W_dq)
    per_row_max = np.abs(W_fp).max(axis=1)
    assert (err.max(axis=1) <= per_row_max / 127.0 + 1e-6).all()


def test_quantized_tensor_shape_preserves() -> None:
    """The fp_shape field must capture the original FP weight shape."""
    qt = QuantizedTensor(
        name="bert.classifier.weight",
        int8_path="w0000",
        scale_per_channel=tuple(0.01 for _ in range(2)),
        fp_shape=(2, 768),
    )
    assert qt.fp_shape == (2, 768)
    assert len(qt.scale_per_channel) == qt.fp_shape[0]


def test_activation_scale_records_corpus_size() -> None:
    a = ActivationScale(
        layer_name="bert.encoder.layer.5.intermediate.dense.weight",
        scale=0.013,
        n_calibration_samples=128,
    )
    assert a.layer_name.endswith(".weight")
    assert a.n_calibration_samples == 128
    assert a.scale > 0
