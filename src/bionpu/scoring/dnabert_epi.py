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

"""DNABERT-Epi off-target probability scorer (no-epi variant).

This is the GPU-first port of the model from Kimata 2025
(*Improved CRISPR/Cas9 Off-target Prediction with DNABERT and
Epigenetic Features*, PLOS One, DOI 10.1371/journal.pone.0335863).
Upstream code lives at https://github.com/opensensor/CRISPR_DNABERT
(forked from kimatakai/CRISPR_DNABERT).

Scope of this module
--------------------

**No-epi variant.** Inputs are guide RNA + candidate DNA only — no
ATAC / H3K4me3 / H3K27ac BigWig stage. The with-epi variant adds a
small (~3M params) gating MLP on top of the BERT body and is a
follow-up; it requires cell-type-specific epigenetic tracks at
inference time, which fundamentally changes the deployment story
(see ``docs/model-selection-audit.md`` § 3 for the inference-time
BigWig dependency analysis).

**CPU + GPU backends only (v0.3 alpha).** AIE2P is a follow-up. The
load-bearing claim is that swapping ``device`` does not change the
user-visible output beyond declared numerical tolerance — the
:mod:`bionpu.verify.score` policies underwrite this.

Smoke mode
----------

Without weights, the scorer can run in ``smoke=True`` mode: scores are
a deterministic function (SHA-256-bucketed pseudo-random in ``[0,1]``)
of the row identity tuple, with no torch dependency. This makes the
end-to-end CLI demonstrable on any host, exercises the verify-score
policies, and lets CI run without GPUs / weights / ~700 MB of HF
checkpoints. Real inference requires ``smoke=False`` and a path to a
fine-tuned classifier checkpoint plus a DNABERT-3 base.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from bionpu.data.canonical_sites import CasOFFinderRow

from .types import ScoreRow

__all__ = [
    "DNABERTEpiNpuNotImplementedError",
    "DNABERTEpiScorer",
    "DNABERTEpiUnavailableError",
]


Device = Literal["cpu", "gpu", "npu"]


class DNABERTEpiUnavailableError(RuntimeError):
    """Raised when a real (non-smoke) scorer is requested but the
    runtime dependencies (``torch``, ``transformers``, weights) are
    not available."""


class DNABERTEpiNpuNotImplementedError(NotImplementedError):
    """Raised when ``device='npu'`` is requested but the AIE2P scorer
    port hasn't landed yet. Tracked in
    ``docs/aie2p-scorer-port-design.md``; v0.4 milestone."""


class DNABERTEpiScorer:
    """DNABERT-Epi (no-epi variant) off-target scorer.

    Parameters
    ----------
    device:
        ``"cpu"``, ``"gpu"``, or ``"npu"``. ``"gpu"`` requires
        ``torch.cuda`` to be importable and at least one CUDA device
        visible. ``"npu"`` is reserved for the AIE2P scorer port
        (PRD-1 v0.4 follow-up; see
        ``docs/aie2p-scorer-port-design.md``); it currently raises
        :class:`DNABERTEpiNpuNotImplementedError` rather than
        silently falling through.
    weights_path:
        Path to a fine-tuned classifier checkpoint (``state_dict``
        pickled by ``torch.save``). Required when ``smoke=False``.
    base_model:
        HuggingFace model id for the DNABERT-3 base. Default
        ``"zhihan1996/DNA_bert_3"``. Loaded via
        :func:`transformers.AutoModel.from_pretrained`.
    smoke:
        If ``True``, produce deterministic pseudo-random scores
        without loading torch / transformers / any model. Intended
        for demonstrating the pipeline end-to-end on hosts without
        GPUs or trained weights. Treats ``device`` as a label only —
        the smoke output is byte-identical between ``cpu`` and
        ``gpu`` smoke runs (so verify-score with BITWISE_EXACT
        passes).
    seed:
        Salt for the smoke-mode hash function. Has no effect when
        ``smoke=False``.
    """

    def __init__(
        self,
        *,
        device: Device = "cpu",
        weights_path: Path | None = None,
        base_model: str = "zhihan1996/DNA_bert_3",
        smoke: bool = False,
        seed: int = 0,
    ) -> None:
        if device not in ("cpu", "gpu", "npu"):
            raise ValueError(
                f"device must be 'cpu', 'gpu', or 'npu'; got {device!r}"
            )
        if device == "npu" and not smoke:
            raise DNABERTEpiNpuNotImplementedError(
                "device='npu' real-mode scoring is the v0.4 milestone "
                "(see docs/aie2p-scorer-port-design.md). Use "
                "device='cpu' or device='gpu' for now, or smoke=True "
                "for the torch-free placeholder path."
            )
        self.device = device
        self.weights_path = Path(weights_path) if weights_path else None
        self.base_model = base_model
        self.smoke = smoke
        self.seed = int(seed)

        # Lazy: torch + transformers + the model are loaded on first
        # `score()` call when `smoke=False`. This keeps `import bionpu`
        # cheap for users who never invoke the scorer.
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._torch_device = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, rows: Iterable[CasOFFinderRow]) -> Iterator[ScoreRow]:
        """Score canonical scan rows, yielding :class:`ScoreRow` in input order.

        Identity columns are preserved verbatim — the scorer never
        re-sorts. Sort upstream (in the scan stage) so the
        canonical-row contract holds across the chain.
        """
        rows_list = list(rows)
        if self.smoke:
            yield from self._score_smoke(rows_list)
            return
        yield from self._score_real(rows_list)

    # ------------------------------------------------------------------
    # Smoke backend — deterministic, no torch
    # ------------------------------------------------------------------

    def _score_smoke(self, rows: list[CasOFFinderRow]) -> Iterator[ScoreRow]:
        """Hash-bucket pseudo-random scores. Deterministic per-row.

        Useful as a CI smoke target and as an executable specification
        for the scoring TSV format. Does NOT correlate with real
        off-target probability.
        """
        seed_bytes = self.seed.to_bytes(8, "little", signed=False)
        for r in rows:
            key = (
                f"{r.guide_id}|{r.bulge_type}|{r.crrna}|{r.dna}|"
                f"{r.chrom}|{r.start}|{r.strand}|{r.mismatches}|"
                f"{r.bulge_size}".encode("utf-8")
            )
            digest = hashlib.sha256(seed_bytes + key).digest()
            # Take the first 8 bytes as a uint64; map to [0, 1).
            (uint64,) = struct.unpack("<Q", digest[:8])
            score = uint64 / float(1 << 64)
            yield ScoreRow.from_row(r, score)

    # ------------------------------------------------------------------
    # Real backend — torch + transformers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise DNABERTEpiUnavailableError(
                "DNABERT-Epi real-mode scoring requires `torch` and "
                "`transformers` to be installed. Install them or use "
                "`smoke=True` for a torch-free deterministic placeholder."
            ) from exc

        if self.weights_path is None or not self.weights_path.exists():
            raise DNABERTEpiUnavailableError(
                "DNABERT-Epi real-mode scoring requires a classifier "
                "checkpoint at `weights_path`. Pass `smoke=True` for a "
                "torch-free deterministic placeholder."
            )

        if self.device == "gpu":
            if not torch.cuda.is_available():
                raise DNABERTEpiUnavailableError(
                    "device='gpu' requires a CUDA-visible GPU. "
                    "Use device='cpu' or smoke=True."
                )
            self._torch_device = torch.device("cuda")
        else:
            self._torch_device = torch.device("cpu")

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        # The classifier head topology is defined in :mod:`._head` (a
        # clean-room re-implementation of the no-epi variant of the
        # DNABERTEpiModule from the upstream fork — license-clean
        # because it's our own code under GPL-3.0).
        from ._head import build_no_epi_head

        bert = AutoModel.from_pretrained(self.base_model)
        head = build_no_epi_head(hidden_size=bert.config.hidden_size)
        state = torch.load(self.weights_path, map_location=self._torch_device)
        head.load_state_dict(state)

        self._model = (bert.to(self._torch_device).eval(),
                        head.to(self._torch_device).eval())

    def _score_real(self, rows: list[CasOFFinderRow]) -> Iterator[ScoreRow]:
        self._ensure_model_loaded()
        torch = self._torch
        bert, head = self._model
        device = self._torch_device

        # Inference uses the upstream tokenisation policy: align
        # crrna/dna pair, k=3 k-mer tokens, BERT pair-input mode.
        from ._tokenize import tokenize_pair

        with torch.inference_mode():
            for r in rows:
                ids, mask, types = tokenize_pair(self._tokenizer, r.crrna, r.dna)
                ids = ids.to(device)
                mask = mask.to(device)
                types = types.to(device)
                bert_out = bert(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=types,
                )
                cls = bert_out.last_hidden_state[:, 0, :]
                logits = head(cls)
                # Binary classifier: index 1 = off-target probability.
                prob = torch.softmax(logits, dim=-1)[0, 1].item()
                yield ScoreRow.from_row(r, prob)
