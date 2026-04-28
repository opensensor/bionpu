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

"""DNABERT-Epi → AIE2P quantization passport.

The AIE2P scorer port consumes raw INT8 weight tensors plus per-channel
weight scales plus per-tensor activation scales — *not* an
``onnxruntime.quantization`` end-to-end model. This module produces
that passport from a fine-tuned DNABERT-Epi state-dict + a calibration
corpus.

Compare to :mod:`bionpu.quant.calibrate` (the ORT-based PTQ that backs
the basecalling track): the ORT path emits a runnable INT8 ONNX model,
which is the right shape for CPU/GPU deployment. AIE2P kernels don't
run an ONNX runtime — they consume the raw INT8 weights + scales as
arrays. This module is the AIE2P-targeted alternative.

What the passport pins
----------------------

For a given trained checkpoint:

- Per-Linear weight tensors as INT8 ndarrays (per-output-channel
  symmetric quantization: ``W_int8 = round(W_fp / scale_per_channel)``,
  scale_per_channel = max(abs(W_fp[oc, :])) / 127).
- Per-Linear input-activation scale (per-tensor symmetric, set at
  calibration time over a corpus of N representative inputs).
- LayerNorm weights stay FP16 — LN is not quantized; numeric stability
  matters more than memory there.
- Embedding tables stay FP16 — fits memtile, and per-token gather
  doesn't need INT8.

Scope (v0.4 alpha)
------------------

This module covers the *no-epi* DNABERT-Epi variant only — BERT-base
body + 2-class classifier head. The with-epi gating MLP is a
follow-up (separate calibration corpus required for the BigWig
features).

Output format is JSON+npz so that the downstream IRON-Python kernels
can mmap the int8 weights without parsing torch state-dicts.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "QuantizationPassport",
    "QuantizedTensor",
    "calibrate_dnabert_epi",
    "load_passport",
    "save_passport",
]


@dataclass(frozen=True)
class QuantizedTensor:
    """One quantized weight tensor + its scale metadata.

    Attributes
    ----------
    name:
        State-dict key of the source FP tensor (e.g.
        ``"bert.encoder.layer.0.intermediate.dense.weight"``).
    int8_path:
        Relative path to the INT8 ndarray inside the passport's npz
        archive (``f"{idx}.npy"`` convention).
    scale_per_channel:
        Tuple of FP32 scales, length == number of output channels.
        ``W_fp ≈ W_int8 * scale_per_channel[oc]``.
    fp_shape:
        Original FP weight shape (e.g. ``(3072, 768)``). For sanity
        checks at load time.
    """

    name: str
    int8_path: str
    scale_per_channel: tuple[float, ...]
    fp_shape: tuple[int, ...]


@dataclass(frozen=True)
class ActivationScale:
    """Per-tensor activation scale captured at calibration time."""

    layer_name: str
    scale: float
    n_calibration_samples: int


@dataclass
class QuantizationPassport:
    """Top-level passport written alongside the int8 weight archive."""

    schema_version: str = "0.4-alpha"
    model_id: str = "dnabert-epi-noEpi"
    base_model: str = "zhihan1996/DNA_bert_3"
    source_checkpoint_sha256: str = ""
    calibration_corpus_id: str = ""
    n_calibration_samples: int = 0
    created_at_utc: float = 0.0
    weights: list[QuantizedTensor] = field(default_factory=list)
    activations: list[ActivationScale] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "base_model": self.base_model,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "calibration_corpus_id": self.calibration_corpus_id,
            "n_calibration_samples": self.n_calibration_samples,
            "created_at_utc": self.created_at_utc,
            "weights": [asdict(w) for w in self.weights],
            "activations": [asdict(a) for a in self.activations],
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibrate_dnabert_epi(
    *,
    weights_path: Path,
    out_dir: Path,
    calibration_corpus_id: str,
    base_model: str = "zhihan1996/DNA_bert_3",
    calibration_samples: Iterable[tuple[str, str]] | None = None,
    n_samples_target: int = 128,
) -> QuantizationPassport:
    """Calibrate a fine-tuned DNABERT-Epi checkpoint to INT8 + write passport.

    Parameters
    ----------
    weights_path:
        State-dict ``.pth`` produced by the upstream training pipeline
        (full BERT body + classifier head).
    out_dir:
        Directory to write the passport into. Created if missing.
        Writes ``passport.json`` + ``weights.npz`` + ``activations.json``.
    calibration_corpus_id:
        Stable label for the calibration corpus, recorded in the
        passport (e.g. ``"crisproff_random_seed42_n128"``). Distinct
        IDs surface in the byte-equality verify output so a calibrated
        passport's identity is auditable.
    base_model:
        HuggingFace model id for the BERT body.
    calibration_samples:
        Iterable yielding ``(crrna, dna)`` 24-nt strings. If ``None``,
        a deterministic synthetic corpus is generated from the same
        seed as the smoke scorer (sufficient for validating the
        pipeline; not for production-quality calibration).
    n_samples_target:
        Cap on the number of calibration samples to consume. The
        actual count is recorded in the passport (may be less if the
        iterable is exhausted).

    Returns
    -------
    QuantizationPassport
        The metadata that was written to disk.

    Notes
    -----
    Lazily imports torch / transformers — the bionpu wheel doesn't
    pull these in for users that only run smoke or verify flows.
    """
    try:
        import numpy as np
        import torch
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "calibrate_dnabert_epi requires torch + transformers + numpy. "
            "Install them or stub via the smoke pipeline."
        ) from exc

    from ._head import build_no_epi_head
    from ._tokenize import tokenize_pair

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load FP model (BERT body + clean-room head).
    bert = AutoModel.from_pretrained(base_model).eval()
    head = build_no_epi_head(hidden_size=bert.config.hidden_size).eval()
    head_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    head.load_state_dict(head_state, strict=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 2. Quantize weights. Per-output-channel symmetric INT8 for every
    #    nn.Linear in the BERT body + the head. LayerNorm and embedding
    #    tables stay FP (handled separately by the kernel side).
    weight_records: list[QuantizedTensor] = []
    int8_arrays: dict[str, np.ndarray] = {}
    int8_idx = 0

    for module_root, module_name in (
        (bert, "bert"),
        (head, "head"),
    ):
        for name, mod in module_root.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            full_name = f"{module_name}.{name}.weight"
            W_fp = mod.weight.detach().to(torch.float32).cpu().numpy()
            # max-abs along the input axis → per-output-channel scale
            scale = np.maximum(np.abs(W_fp).max(axis=1), 1e-12) / 127.0
            W_int8 = np.clip(
                np.round(W_fp / scale[:, None]), -128, 127
            ).astype(np.int8)
            arr_key = f"w{int8_idx:04d}"
            int8_arrays[arr_key] = W_int8
            weight_records.append(QuantizedTensor(
                name=full_name,
                int8_path=arr_key,
                scale_per_channel=tuple(scale.astype(float).tolist()),
                fp_shape=tuple(W_fp.shape),
            ))
            int8_idx += 1

    # 3. Calibrate activations. Hook every nn.Linear's *input* via
    #    forward_pre_hook; track running max-abs across the calibration
    #    corpus. Per-tensor symmetric scale = max-abs / 127.
    act_max_abs: dict[str, float] = {}
    hook_handles = []

    def _make_hook(layer_name: str):
        def hook(_mod, inputs):
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                m = float(x.detach().abs().max().item())
                act_max_abs[layer_name] = max(act_max_abs.get(layer_name, 0.0), m)
        return hook

    for module_root, module_name in ((bert, "bert"), (head, "head")):
        for name, mod in module_root.named_modules():
            if isinstance(mod, torch.nn.Linear):
                full_name = f"{module_name}.{name}.weight"
                hook_handles.append(
                    mod.register_forward_pre_hook(_make_hook(full_name))
                )

    # Run calibration samples. Default corpus: deterministic synthetic.
    if calibration_samples is None:
        calibration_samples = _synthetic_calibration_corpus(n_samples_target)

    n_consumed = 0
    with torch.inference_mode():
        for crrna, dna in calibration_samples:
            if n_consumed >= n_samples_target:
                break
            ids, mask, types = tokenize_pair(tokenizer, crrna, dna)
            out = bert(input_ids=ids, attention_mask=mask, token_type_ids=types)
            cls = out.last_hidden_state[:, 0, :]
            _ = head(cls)
            n_consumed += 1

    for h in hook_handles:
        h.remove()

    activation_records = [
        ActivationScale(
            layer_name=name,
            scale=max_abs / 127.0,
            n_calibration_samples=n_consumed,
        )
        for name, max_abs in sorted(act_max_abs.items())
    ]

    # 4. Save artifacts.
    np.savez_compressed(out_dir / "weights.npz", **int8_arrays)
    passport = QuantizationPassport(
        model_id="dnabert-epi-noEpi",
        base_model=base_model,
        source_checkpoint_sha256=_sha256_file(weights_path),
        calibration_corpus_id=calibration_corpus_id,
        n_calibration_samples=n_consumed,
        created_at_utc=time.time(),
        weights=weight_records,
        activations=activation_records,
    )
    save_passport(passport, out_dir / "passport.json")
    return passport


def _synthetic_calibration_corpus(n: int):
    """Deterministic ``(crrna, dna)`` 24-nt pairs for smoke calibration.

    Same hash-keyed pseudo-random generator as the smoke scorer, so a
    user calling :func:`calibrate_dnabert_epi` without a real corpus
    gets a stable, reproducible (but not biologically meaningful)
    calibration. Production calibration should pass a real corpus.
    """
    import struct
    bases = "ACGT"
    for i in range(n):
        crrna = "".join(
            bases[(struct.unpack(
                "<I",
                hashlib.sha256(f"crrna-{i}-{j}".encode()).digest()[:4],
            )[0]) & 0b11]
            for j in range(24)
        )
        dna = "".join(
            bases[(struct.unpack(
                "<I",
                hashlib.sha256(f"dna-{i}-{j}".encode()).digest()[:4],
            )[0]) & 0b11]
            for j in range(24)
        )
        yield crrna, dna


def save_passport(passport: QuantizationPassport, path: Path) -> None:
    """Write the passport JSON sidecar."""
    Path(path).write_text(json.dumps(passport.to_json(), indent=2) + "\n")


def load_passport(path: Path) -> QuantizationPassport:
    """Read a previously-saved passport.

    JSON erases the tuple/list distinction, so coerce
    ``scale_per_channel`` and ``fp_shape`` back to tuples to match the
    in-memory dataclass contract (immutable + hashable).
    """
    blob = json.loads(Path(path).read_text())
    return QuantizationPassport(
        schema_version=blob["schema_version"],
        model_id=blob["model_id"],
        base_model=blob["base_model"],
        source_checkpoint_sha256=blob["source_checkpoint_sha256"],
        calibration_corpus_id=blob["calibration_corpus_id"],
        n_calibration_samples=blob["n_calibration_samples"],
        created_at_utc=blob["created_at_utc"],
        weights=[
            QuantizedTensor(
                name=w["name"],
                int8_path=w["int8_path"],
                scale_per_channel=tuple(w["scale_per_channel"]),
                fp_shape=tuple(w["fp_shape"]),
            )
            for w in blob["weights"]
        ],
        activations=[ActivationScale(**a) for a in blob["activations"]],
    )
