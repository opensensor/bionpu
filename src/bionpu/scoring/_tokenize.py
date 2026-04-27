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

"""DNABERT-3 tokeniser adapter for guide / off-target pairs.

DNABERT-3 expects k=3 k-mer tokens (overlapping triplets of ACGT, no
ambiguity codes). For the off-target prediction task we follow the
Kimata 2025 paper's BERT pair-input policy: the guide RNA goes in
segment A, the candidate DNA in segment B, padded to a fixed length.

This module wraps the upstream tokenizer with a deterministic
to-tensor adapter so the scoring pipeline doesn't have to know the
internal token-id layout. Lazy torch import keeps the module
importable without torch.
"""

from __future__ import annotations

K = 3
MAX_PAIRSEQ_LEN = 24  # crRNA / DNA padded length, matches upstream config.


def _seq_to_kmers(seq: str) -> str:
    """Convert ``ACGT...`` to a space-separated k=3 k-mer string.

    Non-ACGT characters (e.g. ``-`` for hyphen-padded alignments) are
    passed through verbatim — the upstream tokeniser handles ``-``
    tokens as a special-padding alphabet.
    """
    return " ".join(seq[i:i + K] for i in range(len(seq) - K + 1))


def tokenize_pair(tokenizer, crrna: str, dna: str):
    """Tokenise a ``(crrna, dna)`` pair into BERT-pair tensors.

    Returns ``(input_ids, attention_mask, token_type_ids)``, each shape
    ``[1, T]``. Torch is imported lazily.
    """
    crrna_kmers = _seq_to_kmers(crrna)
    dna_kmers = _seq_to_kmers(dna)
    # 2 * (max_pairseq_len - k + 1) + 3 BERT special tokens
    token_max_len = 2 * (MAX_PAIRSEQ_LEN - K + 1) + 3
    enc = tokenizer(
        crrna_kmers,
        dna_kmers,
        padding="max_length",
        truncation=True,
        max_length=token_max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]
