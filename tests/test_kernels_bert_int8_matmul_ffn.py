"""Feed-forward INT8 matmul specializations for DNABERT-Epi scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bionpu.dispatch.npu import lookup_npu_op
from bionpu.kernels.scoring.bert_int8_matmul import (
    bert_int8_matmul_ffn1,
    bert_int8_matmul_ffn2,
)


_M = 5
_REPO = Path(__file__).resolve().parents[1]


def _missing_artifacts(tmp_path):
    return tmp_path / "missing-artifacts"


def test_ffn1_emulation_shape_and_correctness(tmp_path) -> None:
    rng = np.random.default_rng(7)
    x = rng.integers(-8, 8, size=(_M, 768), dtype=np.int8)
    w = rng.integers(-8, 8, size=(3072, 768), dtype=np.int8)
    s = np.full((3073,), 1e-3, dtype=np.float32)
    s[3072] = 0.0

    y = bert_int8_matmul_ffn1(x, w, s, artifacts_dir=_missing_artifacts(tmp_path))

    expected = np.clip(
        np.round(
            np.einsum("mk,nk->mn", x.astype(np.int32), w.astype(np.int32))
            .astype(np.float32)
            * s[:3072]
        ),
        -128,
        127,
    ).astype(np.int8)
    assert y.shape == (_M, 3072)
    assert (y == expected).all()


def test_ffn2_emulation_shape_and_correctness(tmp_path) -> None:
    rng = np.random.default_rng(8)
    x = rng.integers(-8, 8, size=(_M, 3072), dtype=np.int8)
    w = rng.integers(-8, 8, size=(768, 3072), dtype=np.int8)
    s = np.full((769,), 1e-3, dtype=np.float32)
    s[768] = 0.0

    y = bert_int8_matmul_ffn2(x, w, s, artifacts_dir=_missing_artifacts(tmp_path))

    expected = np.clip(
        np.round(
            np.einsum("mk,nk->mn", x.astype(np.int32), w.astype(np.int32))
            .astype(np.float32)
            * s[:768]
        ),
        -128,
        127,
    ).astype(np.int8)
    assert y.shape == (_M, 768)
    assert (y == expected).all()


def test_ffn_ops_registered() -> None:
    assert lookup_npu_op("bert_int8_matmul_ffn1").name == "bert_int8_matmul_ffn1"
    assert lookup_npu_op("bert_int8_matmul_ffn2").name == "bert_int8_matmul_ffn2"


def test_ffn1_silicon_dispatch_uses_ddr_grouped_payloads(monkeypatch, tmp_path) -> None:
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    (art_dir / "final.xclbin").write_bytes(b"x")
    (art_dir / "insts.bin").write_bytes(b"i")

    x = np.zeros((_M, 768), dtype=np.int8)
    w = np.zeros((3072, 768), dtype=np.int8)
    s = np.ones((3073,), dtype=np.float32)
    raw = bytes(48 * 3072)
    calls = []

    class _Backend:
        def run_xclbin(self, **kwargs):
            calls.append(kwargs)
            return raw, 1.0, 1.0, 1.0

    monkeypatch.setattr(
        "bionpu.kernels.scoring.bert_int8_matmul.default_backend",
        lambda: _Backend(),
    )

    y = bert_int8_matmul_ffn1(x, w, s, artifacts_dir=art_dir)

    assert y.shape == (_M, 3072)
    assert calls[0]["out_size"] == 48 * 3072
    xs_payload = calls[0]["in_buffers"][0][0]
    w_payload = calls[0]["in_buffers"][1][0]
    assert len(xs_payload) == 4 * 96 * (47 * 8 + 769 * 4)
    assert len(w_payload) == 4 * 96 * 768 * 8


def test_bert_int8_matmul_kernel_uses_aie_api_vector_dot() -> None:
    cc = (
        _REPO
        / "src"
        / "bionpu"
        / "kernels"
        / "scoring"
        / "bert_int8_matmul"
        / "bert_int8_matmul.cc"
    ).read_text()

    assert "BIONPU_HAS_AIE_API" in cc
    assert "aie::accum<acc32, VEC_I8>" in cc
    assert "aie::load_v<VEC_I8>" in cc
    assert "aie::mac(acc, w_v, x_v)" in cc
    assert "return dot_i8_scalar(x, w, k_len);" in cc
