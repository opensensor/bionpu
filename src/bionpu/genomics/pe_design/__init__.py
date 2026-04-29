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

"""Track B v0 — Prime editor pegRNA design (public API).

This package implements the long-arc PRD §3.2 prime-editor guide-design
pipeline. The Wave 8 (T10) public surface re-exports:

* :func:`design_prime_editor_guides` — the top-level orchestration
  entry point. Accepts gene symbol (Mode A), target FASTA (Mode B), or
  ``genome="none"`` synbio (Mode C); returns a sorted list of
  :class:`RankedPegRNA`.
* Shared dataclasses + NamedTuples (:class:`RankedPegRNA`,
  :class:`EditSpec`, :class:`PegRNACandidate`,
  :class:`PE3PegRNACandidate`, :class:`OffTargetSite`,
  :class:`PRIDICTScore`, :class:`PegRNAFoldingFeatures`).
* :func:`off_target_scan_for_spacer` — T11's per-spacer adapter; useful
  for scripts that compose the pegRNA pipeline manually.

The CLI shape (``bionpu crispr pe design``) is wired in
:mod:`bionpu.cli`; the standalone argparse subparser lives in
:mod:`bionpu.genomics.pe_design.cli`.

Per the long-arc roadmap (``PRDs/PRD-crispr-state-of-the-art-roadmap.md``
§3.2), v0 ships PE2 + PE3 strategies with CPU-only PRIDICT 2.0 scoring
(per Track D's Phase 0 closure, all bf16 transformer scoring is CPU
across Tracks A/B/D/E). Silicon work in v0 is limited to off-target
scan reuse via the locked ``crispr/match_multitile_memtile`` kernel.

Lazy import policy
------------------

Types and the off_target adapter are imported eagerly because they're
cheap and cycle-free.  ``design_prime_editor_guides`` is exposed via
``__getattr__`` so the heavyweight ``cli.py`` (which pulls in the
PRIDICT 2.0 wrapper) is only loaded on first attribute access.  This
breaks an otherwise-circular import path: ``bionpu.scoring.pridict2``
-> ``pe_design.types`` -> ``pe_design.__init__`` -> ``pe_design.cli``
-> ``bionpu.scoring.pridict2`` (which would still be loading), which
breaks the integration-test entry path.
"""

from __future__ import annotations

from .off_target import off_target_scan_for_spacer
from .types import (
    EditSpec,
    OffTargetSite,
    PE3PegRNACandidate,
    PegRNACandidate,
    PegRNAFoldingFeatures,
    PRIDICTScore,
    RankedPegRNA,
)

__all__: list[str] = [
    # Top-level orchestration (lazy via __getattr__)
    "design_prime_editor_guides",
    # T11 spacer-level adapter (re-exported for script users)
    "off_target_scan_for_spacer",
    # Shared types
    "EditSpec",
    "OffTargetSite",
    "PE3PegRNACandidate",
    "PegRNACandidate",
    "PegRNAFoldingFeatures",
    "PRIDICTScore",
    "RankedPegRNA",
]


def __getattr__(name: str):
    """Lazy attribute resolver for heavyweight imports.

    ``design_prime_editor_guides`` lives in ``cli.py`` which imports
    :class:`bionpu.scoring.pridict2.PRIDICT2Scorer` — eagerly importing
    that here introduces a circular import for callers that load
    :mod:`bionpu.scoring.pridict2` first (e.g.
    ``test_pe_design_integration.py``).
    """
    if name == "design_prime_editor_guides":
        from .cli import design_prime_editor_guides as _fn
        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
