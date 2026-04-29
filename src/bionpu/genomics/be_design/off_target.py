# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Off-target scan integration for ``bionpu be design`` (Track A v1).

Wraps the locked CRISPR off-target scan (silicon when artifacts are
present; CPU otherwise) plus per-row CFD scoring + aggregation, so the
base-editor ranker can incorporate off-target safety into its
composite score.

Public API:

    >>> from bionpu.genomics.be_design.off_target import (
    ...     OffTargetSite, off_target_scan_for_be_guide,
    ... )
    >>> sites, cfd_agg, n_off = off_target_scan_for_be_guide(
    ...     guide_protospacer="ACGT" * 5,
    ...     pam_seq="AGG",
    ...     genome_path="/path/to/chr22.fa",
    ... )

Per CLAUDE.md:

* The underlying ``crispr/match_multitile_memtile`` silicon kernel is
  invoked through :func:`bionpu.scan.npu_scan` / :func:`cpu_scan`. The
  silicon path is wrapped in ``npu_silicon_lock`` *internally* by
  :func:`_scan_locus_for_offtargets`-style call sites; this module
  follows the same convention via :func:`bionpu.scan.npu_scan` (which
  does not currently take the lock — silicon scope here is the
  multitile_memtile kernel which spawns its own subprocess host_runner
  and which the BE ranker only invokes when the user explicitly
  passes ``--genome <path>``).

* ``--genome none`` (synbio) mode is handled by the caller: this
  module is only invoked when ``genome_path`` is supplied. NaN
  handling for the synbio default lives in the ranker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

__all__ = [
    "OffTargetSite",
    "off_target_scan_for_be_guide",
]


# --------------------------------------------------------------------------- #
# Public dataclass.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OffTargetSite:
    """One off-target hit for a BE guide.

    Attributes:
        chrom: Chromosome / contig name as recorded in the FASTA.
        start: 0-indexed genomic start of the 23-nt window
            (20-nt spacer + 3-nt PAM).
        strand: ``"+"`` (sgRNA matches the top strand) or ``"-"``
            (sgRNA matches the bottom strand).
        spacer_genome: 20-nt off-target genomic spacer (``ACGT``,
            uppercase).
        pam_genome: 3-nt PAM at the off-target site.
        mismatches: spacer-only mismatch count (PAM mismatches always
            count as 0; matches Cas-OFFinder's convention).
        cfd: Doench 2016 CFD score for this site, in ``[0, 1]``.
            Higher = more likely to be cleaved.
    """

    chrom: str
    start: int
    strand: str
    spacer_genome: str
    pam_genome: str
    mismatches: int
    cfd: float


# --------------------------------------------------------------------------- #
# FASTA loader (single-record, mirrors cli._read_first_fasta_record but
# returns name + seq with uppercase, dropping non-ACGT bases via N).
# --------------------------------------------------------------------------- #


def _read_fasta_records(path: Path) -> list[tuple[str, str]]:
    """Read all records from a FASTA. Returns ``[(name, seq), ...]``."""
    if not path.is_file():
        raise FileNotFoundError(f"genome FASTA not found: {path}")
    records: list[tuple[str, str]] = []
    name: str | None = None
    parts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(parts).upper()))
                    parts = []
                name = line[1:].split()[0] if len(line) > 1 else ""
                continue
            if name is None:
                continue
            parts.append(line)
    if name is not None:
        records.append((name, "".join(parts).upper()))
    if not records:
        raise ValueError(f"no FASTA records in {path}")
    return records


# --------------------------------------------------------------------------- #
# Off-target scan + scoring.
# --------------------------------------------------------------------------- #


def off_target_scan_for_be_guide(
    guide_protospacer: str,
    pam_seq: str,
    genome_path: str | Path,
    *,
    max_mismatches: int = 4,
    device: str = "cpu",
    genome_records: Optional[Sequence[tuple[str, str]]] = None,
) -> tuple[list[OffTargetSite], float, int]:
    """Scan a genome for off-target sites near a single BE guide.

    Pipeline (mirrors :func:`bionpu.genomics.crispr_design._scan_locus_for_offtargets`
    + ``_score_off_target_cfd`` for the BE single-guide case):

      1. Dispatch :func:`bionpu.scan.cpu_scan` (or ``npu_scan`` when
         ``device="npu"``) across each FASTA record. The locked
         ``crispr/match_multitile_memtile`` silicon kernel is the
         primary backend for ``device="npu"``; the CPU oracle is
         byte-equivalent.
      2. Filter out the on-target site (``mismatches == 0`` AND
         ``spacer_genome == guide_protospacer``).
      3. Per-site CFD score via
         :func:`bionpu.scoring.cfd.cfd_score_pair`.
      4. Aggregate (sum) the per-site CFDs into a single score per BE
         design (NOT the CRISPOR specificity normalisation —
         we want a *raw sum* for the BE composite, since higher CFD
         sum = more off-target risk).

    Args:
        guide_protospacer: 20-nt ACGT protospacer (5' to 3').
        pam_seq: PAM bases at the on-target site (e.g. ``"AGG"``).
            Used only for the on-target identity check.
        genome_path: Path to a reference FASTA. Multi-record FASTAs
            are scanned record-by-record.
        max_mismatches: Maximum spacer-mismatch threshold. Default 4
            (Cas-OFFinder ceiling).
        device: ``"cpu"`` (default) or ``"npu"``. NPU path requires
            the locked ``crispr/match_multitile_memtile`` artifacts.
        genome_records: Test seam — pass pre-loaded ``[(chrom, seq),
            ...]`` to skip the FASTA read. Useful for unit tests with
            small synthetic genomes.

    Returns:
        ``(sites, cfd_aggregate, off_target_count)``:

        * ``sites``: list of :class:`OffTargetSite` (excluding the
          on-target site itself).
        * ``cfd_aggregate``: sum of per-site CFD scores in ``[0, ~N]``;
          higher = more off-target risk. Returned as ``0.0`` when
          ``len(sites) == 0``. **NaN is reserved for the
          "no scan performed" case** (the caller decides; this
          function always returns a real number).
        * ``off_target_count``: ``len(sites)``.

    Raises:
        ValueError: ``guide_protospacer`` not 20 nt or contains
            non-ACGT bases.
        FileNotFoundError: ``genome_path`` doesn't exist (only when
            ``genome_records`` is None).
    """
    proto = guide_protospacer.upper()
    if len(proto) != 20:
        raise ValueError(
            f"guide_protospacer must be 20 nt; got {len(proto)}"
        )
    if any(c not in "ACGT" for c in proto):
        raise ValueError(
            f"guide_protospacer must be ACGT only; got {proto!r}"
        )
    pam = pam_seq.upper()
    if not pam:
        raise ValueError("pam_seq must be non-empty")

    # Late imports so this module is cheap to load when off-target
    # scoring isn't needed.
    from bionpu.scan import GuideSpec, cpu_scan
    from bionpu.scoring.cfd import cfd_score_pair

    if genome_records is None:
        genome_records = _read_fasta_records(Path(genome_path))

    guide_id = "be_guide"
    spec = GuideSpec(spacer=proto, guide_id=guide_id)

    all_rows = []
    for chrom, seq in genome_records:
        if device == "npu":
            try:
                from bionpu.scan import npu_scan
                rows = npu_scan(
                    chrom=chrom,
                    seq=seq,
                    guides=[spec],
                    pam_template="NGG",
                    max_mismatches=max_mismatches,
                )
            except Exception:  # noqa: BLE001 - downgrade to CPU on any failure
                rows = cpu_scan(
                    chrom=chrom,
                    seq=seq,
                    guides=[spec],
                    pam_template="NGG",
                    max_mismatches=max_mismatches,
                )
        else:
            rows = cpu_scan(
                chrom=chrom,
                seq=seq,
                guides=[spec],
                pam_template="NGG",
                max_mismatches=max_mismatches,
            )
        all_rows.extend(rows)

    # Filter out the on-target site. The on-target is identified by
    # both 0 mismatches AND a spacer match — NOT just 0 mismatches —
    # because a guide can have a perfect match at the locus *and* a
    # second exact match elsewhere (rare but real for non-unique
    # guides; in that case BOTH are off-targets from the perspective
    # of the design caller). For the BE design path the rule is:
    # "exclude one perfect match if and only if its spacer equals the
    # input protospacer"; this conservatively keeps all duplicate
    # exact matches as off-targets.
    sites: list[OffTargetSite] = []
    on_target_dropped = False
    for r in all_rows:
        spacer_genome = r.dna[:20].upper()
        pam_genome = r.dna[20:23].upper()
        # On-target identity: 0 mismatches AND spacer matches the input.
        if (
            r.mismatches == 0
            and spacer_genome == proto
            and not on_target_dropped
        ):
            on_target_dropped = True
            continue
        # Compute per-site CFD via cfd_score_pair (no PAM penalty here
        # — the off-target scan already enforced NGG; non-NGG sites
        # don't reach this path).
        try:
            cfd = cfd_score_pair(
                proto, spacer_genome, pam=None, matrix="doench_2016"
            )
        except (KeyError, ValueError):
            # Defensive fallback: a malformed key (shouldn't happen
            # with NGG-filtered Cas-OFFinder rows) gets a CFD of 0.
            cfd = 0.0
        if math.isnan(cfd) or cfd < 0.0:
            cfd = 0.0
        sites.append(
            OffTargetSite(
                chrom=r.chrom,
                start=int(r.start),
                strand=str(r.strand),
                spacer_genome=spacer_genome,
                pam_genome=pam_genome,
                mismatches=int(r.mismatches),
                cfd=float(cfd),
            )
        )

    # Aggregate: raw sum of per-site CFDs. Higher = more off-target
    # risk. The CRISPOR specificity score (100/(100+sum)) is the
    # right output for the public-facing "CFDspec" column on the
    # CRISPR design path; for BE design we want the raw sum so the
    # ranker can apply a linear penalty (composite_be -= 0.5 * cfd_sum).
    cfd_aggregate = float(sum(s.cfd for s in sites))
    return sites, cfd_aggregate, len(sites)
