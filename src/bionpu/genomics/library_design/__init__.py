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

"""Track C v0 — Genome-scale pooled CRISPR library design.

Per ``PRDs/PRD-crispr-state-of-the-art-roadmap.md`` §3.3, pooled CRISPR
screen libraries (Brunello, Bassik, Synthego scale) require:

* Per-gene guide selection (typically 4-10 guides/gene) ranked by an
  integrated score (Doench on-target + CFD off-target).
* Global guide-sequence deduplication so the same spacer never appears
  twice across the library.
* Built-in control sets:

  * **Non-targeting** — random 20-nt spacers with no genome match.
  * **Safe-harbor** — canonical published guides (AAVS1, CCR5, ROSA26).
  * **Essential-gene** — canonical published guides (RPS19, RPL15) for
    screen quality control (positive controls).

* TSV emission with columns mappable to vendor pool-design pipelines.

Track C v0 is library-knockout-only (NGG SpCas9) and operates on a
caller-supplied target gene list. Activation (Cas9-VPR), interference
(dCas9-KRAB), and genome-wide enumeration (~20K human genes) are v1
deferrals.

Public API:

    >>> from bionpu.genomics.library_design import design_pooled_library
    >>> guides = design_pooled_library(
    ...     targets=["BRCA1", "TP53", "EGFR"],
    ...     library_type="knockout",
    ...     guides_per_gene=4,
    ...     genome="GRCh38",
    ...     fasta_path="/path/to/grch38.fa",
    ...     n_controls=100,
    ... )

The CLI surface (``bionpu library design ...``) is a thin argparse
wrapper around :func:`design_pooled_library`.
"""

from __future__ import annotations

from .controls import (
    CANONICAL_ESSENTIAL_GENE_GUIDES,
    CANONICAL_SAFE_HARBOR_GUIDES,
    ControlGuide,
    generate_controls,
    generate_non_targeting_controls,
)
from .library_balancer import balance_library, global_dedup
from .output import (
    LIBRARY_TSV_HEADER,
    LibraryGuide,
    assemble_library,
    format_library_tsv,
)
from .per_gene_designer import (
    PerGenePool,
    design_per_gene_pools,
    select_top_guides_for_gene,
)

__all__ = [
    "CANONICAL_ESSENTIAL_GENE_GUIDES",
    "CANONICAL_SAFE_HARBOR_GUIDES",
    "ControlGuide",
    "LIBRARY_TSV_HEADER",
    "LibraryGuide",
    "PerGenePool",
    "assemble_library",
    "balance_library",
    "design_per_gene_pools",
    "design_pooled_library",
    "format_library_tsv",
    "generate_controls",
    "generate_non_targeting_controls",
    "global_dedup",
    "select_top_guides_for_gene",
]


def design_pooled_library(
    *,
    targets,
    library_type: str = "knockout",
    guides_per_gene: int = 4,
    genome: str = "GRCh38",
    fasta_path=None,
    n_controls: int = 1000,
    max_mismatches: int = 4,
    gc_min: float = 25.0,
    gc_max: float = 75.0,
    device: str = "cpu",
    rank_by: str = "crispor",
    pool_oversample: int = 4,
    dedup_strategy: str = "highest_score",
    rng_seed: int = 42,
    silicon_lock_label: str | None = None,
):
    """Run the full Track C v0 library-design pipeline.

    Parameters
    ----------
    targets:
        List of gene symbols. Each must resolve via the Tier 1+
        ``crispr_design`` resolver (BRCA1, TP53, EGFR, etc.).
    library_type:
        One of ``"knockout" | "activation" | "interference"``. Only
        ``"knockout"`` is wired in v0; the others raise
        :class:`NotImplementedError` with a v1 deferral note.
    guides_per_gene:
        Number of guides selected per gene after dedup. Typical
        published libraries use 4-10.
    genome, fasta_path:
        Reference build identifier and FASTA path forwarded to
        :func:`bionpu.genomics.crispr_design.design_guides_for_target`.
    n_controls:
        Number of non-targeting controls to generate. Safe-harbor
        (AAVS1/CCR5/ROSA26) and essential-gene (RPS19/RPL15) controls
        are added in addition with their canonical sequences.
    max_mismatches, gc_min, gc_max, device, rank_by:
        Forwarded to the per-gene designer.
    pool_oversample:
        Per-gene pool size = ``guides_per_gene * pool_oversample``.
        Larger oversampling gives the dedup pass more headroom.
    dedup_strategy:
        ``"highest_score"`` (default) keeps the gene with the higher
        composite score when a guide spacer collides. ``"alphabetical"``
        is deterministic but ignores quality; intended for tests.
    rng_seed:
        Salt for the non-targeting control generator.
    silicon_lock_label:
        Optional diagnostic label forwarded to the silicon lock when
        ``device="npu"``.

    Returns
    -------
    list[LibraryGuide]
        Ordered library: per-gene guides first (sorted by gene then
        rank-within-gene), then controls (non-targeting first,
        safe-harbor next, essential-gene last). ``library_index`` is
        assigned in emission order starting at 1.
    """
    from .pipeline import run_library_pipeline

    return run_library_pipeline(
        targets=list(targets),
        library_type=library_type,
        guides_per_gene=int(guides_per_gene),
        genome=genome,
        fasta_path=fasta_path,
        n_controls=int(n_controls),
        max_mismatches=int(max_mismatches),
        gc_min=float(gc_min),
        gc_max=float(gc_max),
        device=device,
        rank_by=rank_by,
        pool_oversample=int(pool_oversample),
        dedup_strategy=dedup_strategy,
        rng_seed=int(rng_seed),
        silicon_lock_label=silicon_lock_label,
    )
