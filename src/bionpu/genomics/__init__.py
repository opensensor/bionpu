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

"""High-level genomics primitives composed from kernels under
:mod:`bionpu.kernels.genomics`.

Module layout (one subpackage per CRISPR-shape primitive):

* :mod:`bionpu.genomics.guide_design` — host-side CRISPR guide
  enumeration, PAM filtering, and cheap sequence-quality filters.
* :mod:`bionpu.genomics.guide_workflow` — CPU-only composition of
  guide enumeration and seed-prefiltered off-target discovery.
* :mod:`bionpu.genomics.seed_extend` — minimap2-style seed extraction
  built on top of the v0 minimizer NPU op + a host-side reference
  index. Third silicon-validated CRISPR-shape primitive (after
  ``kmer_count`` and ``minimizer``).
* :mod:`bionpu.genomics.offtarget_seed` — pure-host CRISPR off-target
  seed prefilter for oriented guide/reference candidates.
"""

from bionpu.genomics.guide_design import (
    GuideCandidate,
    GuideFilter,
    canonical_guide_key,
    enumerate_guides,
)
from bionpu.genomics.guide_workflow import (
    GuideDesignResult,
    GuideDesignRun,
    design_guides,
)
from bionpu.genomics.offtarget_seed import (
    GuideRecord,
    OffTargetSeedCandidate,
    encode_seed_2bit,
    prefilter_offtargets,
    reverse_complement,
    seed_mismatch_positions,
)

__all__ = [
    "GuideCandidate",
    "GuideDesignResult",
    "GuideDesignRun",
    "GuideFilter",
    "GuideRecord",
    "OffTargetSeedCandidate",
    "canonical_guide_key",
    "design_guides",
    "encode_seed_2bit",
    "enumerate_guides",
    "prefilter_offtargets",
    "reverse_complement",
    "seed_mismatch_positions",
]
