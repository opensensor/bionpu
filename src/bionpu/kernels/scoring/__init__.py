# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# GPL-3.0-or-later. See ../../../LICENSE.

"""DNABERT-Epi AIE2P scorer kernels (v0.4 milestone).

This package collects the AIE2P kernels that compose the BERT body
forward pass for the off-target scorer port. See
``docs/aie2p-scorer-port-design.md`` for the kernel breakdown
(embed_lookup, attn_qkv, attn_softmax_av, attn_output, ffn,
pool, score_head) and the dispatch graph that ties them together.

v0.4-alpha ships scaffolding for the workhorse ``bert_int8_matmul``
op (used by every Linear in the BERT body); building out the other
kernels and the end-to-end dispatch is v0.4-beta+.
"""

from __future__ import annotations

# Eager import to drive NpuOp registration at package import time,
# matching the pattern used by bionpu.kernels.crispr.
from . import bert_int8_matmul as _bert_int8_matmul  # noqa: F401
