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

"""Classifier head for DNABERT-Epi (no-epi variant).

Clean-room re-implementation of the architectural shape described in
the Kimata 2025 paper (PLOS One DOI 10.1371/journal.pone.0335863):
the no-epi branch of ``DNABERTEpiModule`` is just the BERT [CLS]
embedding fed into a small MLP that produces 2-class logits. The
upstream implementation packages this inside a larger module that
also carries the epigenetic encoder + gating MLP; the no-epi branch
is the cls-only path.

This file ships the classifier *topology only*. Trained weights are
not redistributed in the bionpu wheel — users supply their own
checkpoint via ``DNABERTEpiScorer(weights_path=...)``.
"""

from __future__ import annotations


def build_no_epi_head(*, hidden_size: int = 768):
    """Return a ``torch.nn.Module`` mapping BERT [CLS] -> 2-class logits.

    Topology: ``Dropout(0.1) -> Linear(hidden_size, 2)``. This matches
    the no-epi specialisation of the Kimata 2025 classifier head — the
    full module also carries an epi-encoder + gating stack, which we
    omit here.

    The function imports torch lazily so importing
    :mod:`bionpu.scoring` does not pull torch in.
    """
    import torch.nn as nn

    return nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, 2))
