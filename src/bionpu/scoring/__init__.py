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

"""Off-target probability scoring (CRISPR / Cas9).

Public surface for the v0.3 off-target scorer pipeline. Consumes the
canonical scan TSV produced by :mod:`bionpu.scan` and emits an
augmented canonical TSV with a ``score`` column appended.

Backend strategy
----------------

Each scorer publishes a ``device`` axis spanning ``{"cpu", "gpu", "npu"}``
where supported. The intent is *not* that all devices ship at once for
every model — different fabrics make sense for different stages of the
pipeline:

- **CPU** is the universal floor and the byte-equivalence reference.
  Slow but ubiquitous; runs in CI, runs on cloud, runs without
  proprietary drivers.
- **GPU** is the practical scaling layer for transformer-class scorers
  (DNABERT-Epi, CRISPR-BERT, etc.) — most users have one, the stack
  is mature, weights fit VRAM trivially.
- **NPU** (AIE2P) is the differentiator for sustained low-power
  inference. Worth the porting effort only after the GPU path is
  established and the byte-equivalence target is locked in.

The :mod:`bionpu.verify.score` policy module underwrites the contract
that swapping devices does not change the user-visible output beyond
declared numerical tolerance.

Scope (v0.3 alpha)
------------------

DNABERT-Epi (Kimata 2025, no-epi variant) is the first scorer being
wired. The with-epi (BigWig) variant is a follow-up; the AIE2P scorer
backend is a follow-up to the GPU baseline. CRISPR-Net (Lin 2020) is
available as a baseline comparator. See
``docs/model-selection-audit.md`` for the model-selection rationale.
"""

from __future__ import annotations

from .cfd import CFDScorer, aggregate_cfd, cfd_score_pair
from .doench_rs2 import (
    AzimuthNotInstalledError,
    DoenchAzimuthScorer,
    DoenchRS1Scorer,
    doench_rs1_score,
    extract_30mer_context,
)
from .types import ScoreRow, parse_score_tsv, serialize_canonical_score, write_score_tsv

__all__ = [
    "AzimuthNotInstalledError",
    "CFDScorer",
    "DoenchAzimuthScorer",
    "DoenchRS1Scorer",
    "ScoreRow",
    "aggregate_cfd",
    "cfd_score_pair",
    "doench_rs1_score",
    "extract_30mer_context",
    "parse_score_tsv",
    "serialize_canonical_score",
    "write_score_tsv",
]
