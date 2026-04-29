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

"""Track B v0 — Prime editor pegRNA design (skeleton).

This is the **first-writer** ``__init__.py`` for ``pe_design``, shipped
with Task T3 of ``track-b-pegrna-design-plan.md`` so subsequent tasks
(T2 edit-spec, T6 enumerator, T7 PE3, T8 ranker, T9 output, T10 CLI,
T11 off-target) can ``from bionpu.genomics.pe_design.types import ...``
without racing on package creation.

The public API surface (``__all__``) is intentionally empty in T3 and
will be extended additively by Wave 2/3+ tasks. The shared types and
constants live in dedicated submodules:

* :mod:`bionpu.genomics.pe_design.types` — shared dataclasses /
  NamedTuples consumed across T2/T4/T5/T6/T7/T8/T9/T11.
* :mod:`bionpu.genomics.pe_design.pegrna_constants` — published
  scaffold sequences + PBS/RTT/nick-distance/edit-length ranges.

Per the long-arc roadmap (``PRDs/PRD-crispr-state-of-the-art-roadmap.md``
§3.2), v0 ships PE2 + PE3 strategies with CPU-only PRIDICT 2.0 scoring
(per Track D's Phase 0 closure, all bf16 transformer scoring is CPU
across Tracks A/B/D/E). Silicon work in v0 is limited to off-target
scan reuse via the locked ``crispr/match_multitile_memtile`` kernel.
"""

from __future__ import annotations

# T3 ships an empty public surface so T6/T7/T8/T9/T10/T11 can each
# extend ``__all__`` additively as their public-API modules land.
__all__: list[str] = []
