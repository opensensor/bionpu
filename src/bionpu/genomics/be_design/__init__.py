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

"""Track A v0 — Base editor (ABE/CBE) guide design.

Composes the silicon-validated CRISPR-shape primitives into a base
editor design pipeline:

* NEW silicon kernel ``pam_filter_iupac`` enumerates PAM-passing
  positions for an arbitrary IUPAC PAM (Cas9 variant). Single xclbin
  serves every supported variant via runtime-mask header.
* CPU-side composition computes:

  * Edit-window classification (CBE: nt 4-8 from PAM-distal; ABE: 4-7).
  * Bystander-edit count (number of additional editable bases — Cs for
    CBE, As for ABE — within the activity window).

* Existing locked silicon (``crispr/match_multitile_memtile``) provides
  the off-target scan when ``--genome`` is supplied.
* Existing scoring (``bionpu.scoring.cfd``) ranks off-targets.

Public API:

    >>> from bionpu.genomics.be_design import design_base_editor_guides
    >>> guides = design_base_editor_guides(
    ...     target_seq="ACGT...",
    ...     be_variant="BE4max",
    ...     cas9_variant="wt",
    ... )

Phase 1 supported variants (per ``PRDs/PRD-crispr-state-of-the-art-roadmap.md``
§3.1):

* SpCas9 (wild-type, NGG) + ABE7.10 / BE4max.
* SpCas9-NG (NG, padded as NGN for runtime parity).

Phase 2 expansion (8-12 variants; SpRY, SaCas9-KKH, ABE8e, etc.) is
deferred per the long-arc roadmap. The IUPAC kernel itself supports
every variant in Phase 2 today; only ranker tuning + BE-Hive-class
window scoring is deferred.
"""

from __future__ import annotations

from .bystander import bystander_count, enumerate_bystander_edits
from .off_target import OffTargetSite, off_target_scan_for_be_guide
from .pam_variants import (
    BE_VARIANTS,
    CAS9_VARIANTS,
    BaseEditorSpec,
    Cas9PamSpec,
    get_be_spec,
    get_cas9_spec,
)
from .ranker import BaseEditorGuide, composite_be, design_base_editor_guides
from .window_score import (
    AC_WINDOW_BOUNDS_0IDX,
    activity_window_slice,
    activity_window_target_positions,
    target_in_window,
)

__all__ = [
    "AC_WINDOW_BOUNDS_0IDX",
    "BE_VARIANTS",
    "BaseEditorGuide",
    "BaseEditorSpec",
    "CAS9_VARIANTS",
    "Cas9PamSpec",
    "OffTargetSite",
    "activity_window_slice",
    "activity_window_target_positions",
    "bystander_count",
    "composite_be",
    "design_base_editor_guides",
    "enumerate_bystander_edits",
    "get_be_spec",
    "get_cas9_spec",
    "off_target_scan_for_be_guide",
    "target_in_window",
]
