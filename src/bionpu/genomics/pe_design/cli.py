# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Track B v0 — ``bionpu crispr pe design`` CLI subcommand (Task T10).

This module exposes both:

* :func:`design_prime_editor_guides` — the in-process Python public API
  that orchestrates the full T2 -> T6 -> T7 -> T8 -> T9 pipeline. Used
  by the CLI wrapper and by external scripts.
* :func:`add_pe_design_subparser` — the argparse subparser registrar.
  :mod:`bionpu.cli` calls this to wire up the third level of the
  ``bionpu crispr pe design`` nesting.
* :func:`run_cli` — the ``func=`` callback the argparse subparser
  dispatches to.

Three operating modes (mirrors the Track A v0 + Wave 1 ``crispr design``
shape):

* **Mode A** (``--target SYMBOL``): resolve gene symbol via
  :func:`bionpu.data.genome_fetcher.resolve_gene_symbol`, fetch the
  flanking window via
  :func:`bionpu.data.genome_fetcher.fetch_genomic_sequence`, then run
  the standard pipeline.
* **Mode B** (``--target-fasta FILE``): read the first FASTA record
  and use it as the target. The edit notation uses the FASTA record
  name as the ``chrom``.
* **Mode C** (``--genome none``): skip off-target scanning entirely.
  Off-target columns become NaN/0 with ``NO_OFF_TARGET_SCAN`` appended
  to ``notes``. Mode C composes with Mode A (gene symbol synbio test)
  or Mode B (FASTA synbio).

Lock discipline (CLAUDE.md non-negotiable): the orchestration is
in-process, so the underlying off-target scan path uses
:mod:`bionpu.dispatch.npu`'s ``_dispatch_lock`` discipline only.
Subprocess wrappers must hold ``npu_silicon_lock`` themselves; T10
inherits this through T11's adapter without adding new lock concerns.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import replace as _dc_replace
from typing import Callable

# These imports are here (rather than lazy inside run_cli) so tests can
# monkey-patch them at the CLI's module level — cheap binding, the heavy
# work is deferred until the scorer / extractor / scanner are actually
# CALLED. The PRIDICT2Scorer ctor is cheap (~ms) per its docstring; only
# `score()` triggers the model load.
from bionpu.data.genome_fetcher import (
    GeneSymbolNotFound,
    fetch_genomic_sequence,
    resolve_gene_symbol,
)
from bionpu.genomics.pe_design.edit_spec import (
    EditTooLargeError,
    RefMismatchError,
    UnsupportedHGVSVariant,
    parse_edit_spec,
)
from bionpu.genomics.pe_design.enumerator import enumerate_pe2_candidates
from bionpu.genomics.pe_design.off_target import off_target_scan_for_spacer
from bionpu.genomics.pe_design.output import write_json, write_tsv
from bionpu.genomics.pe_design.pe3_nicking import select_nicking_guides
from bionpu.genomics.pe_design.pegrna_constants import (
    PBS_LENGTH_MAX,
    PBS_LENGTH_MIN,
    RTT_LENGTH_MAX,
    RTT_LENGTH_MIN,
    SCAFFOLD_VARIANTS,
)
from bionpu.genomics.pe_design.ranker import rank_candidates
from bionpu.genomics.pe_design.types import (
    EditSpec,
    PegRNACandidate,
    RankedPegRNA,
)
from bionpu.scoring.pegrna_folding import compute_folding_features
from bionpu.scoring.pridict2 import PRIDICT2Scorer

__all__ = [
    "design_prime_editor_guides",
    "add_pe_design_subparser",
    "run_cli",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Default ±flank around the gene span we fetch in Mode A. PE3 nicking
# search runs out to 100 bp from the PE2 nick + needs 23 bp of upstream
# protospacer + 3 bp of PAM, so anything ≥150 bp is safe; 200 bp gives
# headroom and matches what the BE design CLI uses for similar windows.
_MODE_A_DEFAULT_FLANKS = 200

# PRIDICT cell-type variants supported by the upstream weights (HCT116
# and U2OS appear in the plan but are NOT supported by the upstream
# checkpoint; see plan §T5 + state/track-b-prereq-probe.md §2.2 for the
# audit). The CLI rejects HCT116 / U2OS at parse time so users see a
# clear error rather than a cryptic upstream raise.
_PRIDICT_CELL_TYPES = ("HEK", "HEK293", "HEK293T", "K562")

# PE3 nicking search window. PRIDICT 2.0's default is 40-90 bp; we widen
# to 40-100 bp to match the T7 default. The CLI does not currently
# expose this as a flag — keeping the surface small for v0.
_PE3_DISTANCE_RANGE = range(40, 101)


# ---------------------------------------------------------------------------
# Mode A / B / C: target resolution
# ---------------------------------------------------------------------------


def _read_first_fasta_record(path: pathlib.Path) -> tuple[str, str]:
    """Read the first record from a FASTA file. Returns ``(name, seq)``.

    Mirrors the helper in :mod:`bionpu.genomics.be_design.cli` so we
    don't have a cross-track import dependency.
    """
    if not path.is_file():
        raise FileNotFoundError(f"target FASTA not found: {path}")
    name: str | None = None
    parts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if name is not None:
                    break
                name = line[1:].split()[0] if len(line) > 1 else ""
                continue
            if name is None:
                continue
            parts.append(line)
    if name is None:
        raise ValueError(f"no FASTA record found in {path}")
    seq = "".join(parts).upper()
    return name, seq


def _resolve_target_mode_a(
    *,
    symbol: str,
    genome: str,
    fasta_path: pathlib.Path | None,
    flanks: int,
) -> tuple[str, str, int]:
    """Mode A: resolve a gene symbol to ``(chrom, sequence, window_start_0b)``.

    ``window_start_0b`` is the 0-based + strand genomic coordinate of
    sequence[0] — the offset the caller subtracts from absolute edit
    coords to translate them into target-window-local offsets the T6
    enumerator consumes.
    """
    coord = resolve_gene_symbol(symbol, genome=genome)
    seq = fetch_genomic_sequence(
        coord, fasta_path=fasta_path, flanks=flanks, genome=genome
    )
    # GeneCoord is 1-based inclusive; the fetched window starts ``flanks``
    # bp 5' of coord.start (in 1-based inclusive coords, that's
    # coord.start - flanks). 0-based equivalent:
    window_start_0b = (coord.start - 1) - flanks
    return coord.chrom, seq, window_start_0b


# ---------------------------------------------------------------------------
# Edit-spec coordinate translation (absolute genomic -> window-local)
# ---------------------------------------------------------------------------


def _translate_edit_to_window_local(
    edit_spec: EditSpec, *, window_start_0b: int, window_length: int
) -> EditSpec:
    """Translate ``edit_spec`` from absolute genomic coords to coords
    that index into the target window passed to the T6 enumerator.

    For Mode B (window starts at offset 0), this is a no-op. For Mode A
    (window starts at coord.start - flanks - 1), we shift start/end by
    the negative window offset.

    Raises :class:`ValueError` if the translated coords fall outside the
    target window — that's a user-error signal (edit position not
    actually inside the gene span + flanks).
    """
    new_start = edit_spec.start - window_start_0b
    new_end = edit_spec.end - window_start_0b
    if new_start < 0 or new_end > window_length:
        raise ValueError(
            f"edit at {edit_spec.chrom}:{edit_spec.start + 1}..{edit_spec.end} "
            f"falls outside the fetched target window "
            f"[{window_start_0b + 1}..{window_start_0b + window_length}]. "
            f"Increase the flanking radius or pass --target-fasta with a "
            f"sequence that covers the edit."
        )
    return _dc_replace(edit_spec, start=new_start, end=new_end)


# ---------------------------------------------------------------------------
# Off-target scan callable factory (Mode C short-circuits)
# ---------------------------------------------------------------------------


def _build_off_target_callable(
    *,
    genome_path: pathlib.Path | None,
) -> Callable[..., tuple[list, float, int]]:
    """Build the per-spacer ``(spacer, *, max_mismatches=N) ->
    (sites, cfd_aggregate, count)`` callable the T8 ranker consumes.

    For Mode C (``genome_path is None``) returns a stub that always
    yields ``([], NaN, 0)`` so the ranker's CFD math composes (a NaN
    aggregate keeps the composite_pridict numeric — see ranker's
    NaN-propagation rules — actually we want 0.0 here so the composite
    survives; the post-pass tags rows with NO_OFF_TARGET_SCAN and
    rewrites cfd_aggregate to NaN/0 explicitly).
    """
    if genome_path is None:
        # Synbio: short-circuit. Returning 0.0 (not NaN) keeps the
        # composite_pridict numeric; the post-pass replaces the field
        # with NaN per the plan §T10 Mode-C contract.
        def _no_op_scan(spacer: str, *, max_mismatches: int = 4):
            return ([], 0.0, 0)
        return _no_op_scan

    def _real_scan(spacer: str, *, max_mismatches: int = 4):
        return off_target_scan_for_spacer(
            spacer, genome_path, max_mismatches=max_mismatches
        )
    return _real_scan


# ---------------------------------------------------------------------------
# Mode-C post-processing: NaN out off-target columns + tag notes
# ---------------------------------------------------------------------------


def _apply_mode_c_synbio_marker(
    rows: list[RankedPegRNA],
) -> list[RankedPegRNA]:
    """Rewrite each row's off-target columns to the Mode-C contract:

    * ``cfd_aggregate_pegrna`` -> NaN
    * ``off_target_count_pegrna`` -> 0 (already 0 from the stub scan)
    * ``cfd_aggregate_nicking`` -> NaN if PE3 else None (left untouched)
    * ``off_target_count_nicking`` -> 0 if PE3 else None
    * ``notes`` -> append ``"NO_OFF_TARGET_SCAN"`` (idempotent — once)
    """
    for row in rows:
        row.cfd_aggregate_pegrna = float("nan")
        row.off_target_count_pegrna = 0
        if row.pe_strategy == "PE3":
            row.cfd_aggregate_nicking = float("nan")
            row.off_target_count_nicking = 0
        if "NO_OFF_TARGET_SCAN" not in row.notes:
            row.notes = tuple(row.notes) + ("NO_OFF_TARGET_SCAN",)
    return rows


# ---------------------------------------------------------------------------
# Target-context construction (PRIDICT 2.0 input format)
# ---------------------------------------------------------------------------


def _build_target_context(
    *,
    target_seq: str,
    edit_spec: EditSpec,
    flanks: int = 100,
) -> str:
    """Construct PRIDICT 2.0's target-context string ``XXX(orig/edit)YYY``.

    PRIDICT 2.0 ingests the target as a string with the edit specified
    inline. We emit ``flanks`` bp of pre-edit sequence on each side
    (clipped to the available window).

    For length-mismatched indels this string is best-effort — PRIDICT
    will run its own enumeration; the context is purely for the model's
    locus-aware features.
    """
    n = len(target_seq)
    s = max(0, edit_spec.start - flanks)
    e = min(n, edit_spec.end + flanks)
    pre = target_seq[s:edit_spec.start]
    post = target_seq[edit_spec.end:e]
    ref = edit_spec.ref_seq or "-"
    alt = edit_spec.alt_seq or "-"
    return f"{pre}({ref}/{alt}){post}"


# ---------------------------------------------------------------------------
# Public Python API
# ---------------------------------------------------------------------------


def design_prime_editor_guides(
    *,
    target_symbol: str | None = None,
    target_fasta: str | pathlib.Path | None = None,
    edit_notation: str,
    strategy: str = "both",
    scaffold_variant: str = "sgRNA_canonical",
    pridict_cell_type: str = "HEK293",
    genome: str | pathlib.Path = "hg38",
    fasta_path: str | pathlib.Path | None = None,
    pbs_lengths: range = range(PBS_LENGTH_MIN, PBS_LENGTH_MAX + 1),
    rtt_lengths: range = range(RTT_LENGTH_MIN, RTT_LENGTH_MAX + 1),
    top_n: int = 20,
    device: str = "cpu",
    low_memory: bool = False,
    max_mismatches: int = 4,
    flanks: int = _MODE_A_DEFAULT_FLANKS,
    use_5folds: bool = False,
) -> list[RankedPegRNA]:
    """Design prime-editor pegRNAs for a single edit specification.

    Parameters
    ----------
    target_symbol:
        Mode A — gene symbol (e.g. ``"VEGFA"``). Mutually exclusive
        with ``target_fasta``.
    target_fasta:
        Mode B — path to a FASTA file. The first record is used; its
        name must match the ``chrom`` field in ``edit_notation``.
    edit_notation:
        Edit specifier in simple (``"C>T at chr1:100"``) or HGVS
        (``"chr1:g.100C>T"``) notation. See
        :func:`bionpu.genomics.pe_design.edit_spec.parse_edit_spec`.
    strategy:
        ``"pe2"``, ``"pe3"``, or ``"both"``. Default ``"both"`` —
        emits PE2 candidates plus all matching PE3-extended candidates.
    scaffold_variant:
        Scaffold name from
        :data:`bionpu.genomics.pe_design.pegrna_constants.SCAFFOLD_VARIANTS`.
        Default ``"sgRNA_canonical"``.
    pridict_cell_type:
        ``"HEK293"`` (default), ``"HEK"``, ``"HEK293T"``, or ``"K562"``.
    genome:
        ``"hg38"`` / ``"hg19"`` (Mode A reference build), a path to a
        FASTA (off-target scan source for Modes A+B), or ``"none"``
        (Mode C — synbio, off-target scan skipped).
    fasta_path:
        Optional explicit reference FASTA for the Mode A
        :func:`fetch_genomic_sequence` call. Defaults to
        ``$BIONPU_GRCH38_FASTA`` then ``data_cache/genomes/grch38/hg38.fa``.
    pbs_lengths / rtt_lengths:
        Inclusive ranges to enumerate. Defaults from
        :mod:`pegrna_constants`.
    top_n:
        Cap on returned ranked rows. Default 20.
    device:
        ``"cpu"`` or ``"npu"`` — forwarded to the off-target adapter.
        v0 only honors ``"cpu"`` (the per-spacer silicon path is filed
        for v1 — see :mod:`bionpu.genomics.pe_design.off_target`).
    low_memory:
        If True, the PRIDICT scorer is used as a context manager and
        torn down after every batch — useful for memory-constrained
        hosts at the cost of extra model-load time on the next call.
    max_mismatches:
        Forwarded to the off-target scan (default 4).
    flanks:
        Mode A flanking radius. Default 200 bp.
    use_5folds:
        Pass through to :class:`PRIDICT2Scorer` — when True, builds the
        5-fold ensemble (~5x slower, more accurate). Default False.

    Returns
    -------
    list[RankedPegRNA]
        Sorted top-N pegRNAs with all T9-schema fields populated.

    Raises
    ------
    ValueError
        Bad input (mutually-exclusive Mode A/B both/none supplied,
        invalid strategy, edit outside fetched window, ...).
    GeneSymbolNotFound
        Mode A: gene symbol unknown (only raised when target_symbol is
        used; target_fasta path is unaffected).
    """
    # ---- Parse + validate strategy ----------------------------------- #
    strategy_norm = strategy.lower()
    if strategy_norm not in {"pe2", "pe3", "both"}:
        raise ValueError(
            f"strategy must be one of 'pe2'|'pe3'|'both'; got {strategy!r}"
        )

    # ---- Mode A vs Mode B target resolution -------------------------- #
    if (target_symbol is None) == (target_fasta is None):
        raise ValueError(
            "exactly one of target_symbol (Mode A) or target_fasta "
            "(Mode B) must be supplied"
        )

    # ---- Mode C? (genome=='none' string) ----------------------------- #
    is_mode_c = isinstance(genome, str) and genome.lower() == "none"

    if target_symbol is not None:
        # Mode A: gene-symbol resolution. Always uses the requested
        # genome build. For Mode A we can't simultaneously be Mode C
        # for the FETCH (we still need a reference FASTA to fetch the
        # gene window); but the OFF-TARGET scan can still be skipped.
        # Resolve build for the fetch.
        if is_mode_c:
            # Mode A + Mode C: use hg38 as the implicit fetch reference.
            # User explicitly opted out of off-target scan via
            # genome=='none'; honor that for the off-target side only.
            fetch_genome = "hg38"
        else:
            fetch_genome = genome if isinstance(genome, str) else "hg38"
        fasta_for_fetch = pathlib.Path(fasta_path) if fasta_path else None
        chrom, target_seq, window_start_0b = _resolve_target_mode_a(
            symbol=target_symbol,
            genome=fetch_genome,
            fasta_path=fasta_for_fetch,
            flanks=int(flanks),
        )
        # Validate the chrom matches the edit notation (best-effort).
        # parse_edit_spec accepts any chrom in the notation; we only
        # require they line up after parsing (below).
    else:
        target_fasta_path = pathlib.Path(target_fasta)
        chrom, target_seq = _read_first_fasta_record(target_fasta_path)
        window_start_0b = 0

    # ---- Parse the edit notation ------------------------------------- #
    # We do NOT pass genome_path for ref-validation here — Mode A would
    # need a chrom-scale FASTA which we don't necessarily have on hand
    # (users may pass --target without --fasta and rely on the fetched
    # window for context). The enumerator's own coordinate-bounds check
    # catches the most common foot-gun.
    edit_spec_global = parse_edit_spec(edit_notation, genome="hg38")

    # Sanity: the chrom in the notation must match the resolved target.
    # Mode B uses the FASTA record name as chrom; Mode A uses the gene's
    # chrom from refGene.
    if edit_spec_global.chrom != chrom:
        raise ValueError(
            f"edit notation references chrom {edit_spec_global.chrom!r}, "
            f"but the resolved target is on {chrom!r}. For Mode A pass "
            f"the genomic chrom (e.g. 'chr1'); for Mode B use the FASTA "
            f"record name."
        )

    # Translate to window-local coords for the T6 enumerator.
    edit_spec = _translate_edit_to_window_local(
        edit_spec_global,
        window_start_0b=window_start_0b,
        window_length=len(target_seq),
    )

    # ---- Off-target source resolution -------------------------------- #
    if is_mode_c:
        off_target_genome_path: pathlib.Path | None = None
    else:
        # ``genome`` is either an alias ("hg38"/"hg19") or a path. For
        # the off-target scan we need a path. If user passed an alias,
        # fall back to the bundled default convention.
        if isinstance(genome, pathlib.Path):
            off_target_genome_path = genome
        elif isinstance(genome, str) and genome.lower() in (
            "hg38", "grch38", "hg19", "grch37"
        ):
            # Use fasta_path if explicit, else the env / default location.
            off_target_genome_path = (
                pathlib.Path(fasta_path)
                if fasta_path
                else _default_grch38_fasta()
            )
            if not off_target_genome_path.is_file():
                raise FileNotFoundError(
                    f"reference FASTA for off-target scan not found at "
                    f"{off_target_genome_path}. Either set "
                    f"$BIONPU_GRCH38_FASTA / pass an explicit fasta_path, "
                    f"or use genome='none' to skip the scan (synbio)."
                )
        else:
            off_target_genome_path = pathlib.Path(genome)
            if not off_target_genome_path.is_file():
                raise FileNotFoundError(
                    f"genome FASTA not found at {off_target_genome_path}; "
                    f"pass 'none' for synbio mode or supply a valid path."
                )
    # `device` reserved for v1 (when the per-spacer silicon path lands).
    del device  # consumed by future silicon path; v0 is CPU-only.

    # ---- Enumerate PE2 candidates ------------------------------------ #
    pe2_candidates = enumerate_pe2_candidates(
        edit_spec,
        target_genome_seq=target_seq,
        scaffold_variant=scaffold_variant,
        pbs_lengths=pbs_lengths,
        rtt_lengths=rtt_lengths,
    )

    # ---- Optionally extend to PE3 ------------------------------------ #
    candidates: list = []
    if strategy_norm in ("pe2", "both"):
        candidates.extend(pe2_candidates)
    if strategy_norm in ("pe3", "both"):
        edit_region = (edit_spec.start, max(edit_spec.end, edit_spec.start + 1))
        for pe2 in pe2_candidates:
            pe3_extensions = select_nicking_guides(
                pe2,
                target_seq,
                distance_range=_PE3_DISTANCE_RANGE,
                edit_region=edit_region,
            )
            candidates.extend(pe3_extensions)

    if not candidates:
        # Empty list is a legitimate outcome (no PAM near the edit /
        # all candidates pruned). Return empty list rather than raising.
        return []

    # ---- Build target_context for PRIDICT ---------------------------- #
    target_context = _build_target_context(
        target_seq=target_seq, edit_spec=edit_spec
    )

    # ---- Build off-target callable ----------------------------------- #
    off_target_fn = _build_off_target_callable(
        genome_path=off_target_genome_path
    )

    # ---- Score + rank (T8) ------------------------------------------- #
    # Construct the PRIDICT scorer. The ctor is cheap (~ms); model
    # weights load lazily on first `.score()` call. ``low_memory`` uses
    # the context-manager teardown to release weights between calls.
    scorer = PRIDICT2Scorer(
        model_variant=pridict_cell_type, use_5folds=use_5folds
    )
    try:
        if low_memory:
            with scorer:
                rows = rank_candidates(
                    candidates=candidates,
                    edit_spec=edit_spec,
                    target_context=target_context,
                    off_target_scan_fn=off_target_fn,
                    scorer=scorer,
                    folding_extractor=compute_folding_features,
                    max_mismatches=int(max_mismatches),
                    top_n=int(top_n),
                )
        else:
            rows = rank_candidates(
                candidates=candidates,
                edit_spec=edit_spec,
                target_context=target_context,
                off_target_scan_fn=off_target_fn,
                scorer=scorer,
                folding_extractor=compute_folding_features,
                max_mismatches=int(max_mismatches),
                top_n=int(top_n),
            )
    finally:
        if not low_memory:
            scorer.close()

    # ---- Mode-C post-processing -------------------------------------- #
    if is_mode_c:
        rows = _apply_mode_c_synbio_marker(rows)

    # Restore the absolute genomic edit_position so the TSV reflects the
    # user-facing coordinate (not the window-local one). edit_spec.start
    # is window-local; the global one is window_start_0b + that.
    abs_pos = window_start_0b + edit_spec.start
    for row in rows:
        row.edit_position = abs_pos

    return rows


# ---------------------------------------------------------------------------
# Default-FASTA helper (kept here so it doesn't pull bionpu.cli into the
# pe_design package import graph).
# ---------------------------------------------------------------------------


def _default_grch38_fasta() -> pathlib.Path:
    import os

    env = os.environ.get("BIONPU_GRCH38_FASTA")
    if env:
        return pathlib.Path(env)
    return pathlib.Path("data_cache/genomes/grch38/hg38.fa")


# ---------------------------------------------------------------------------
# Argparse subparser
# ---------------------------------------------------------------------------


def add_pe_design_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register ``design`` on the ``pe`` sub-subparsers action.

    Called from :mod:`bionpu.cli` after building the ``crispr -> pe``
    sub-subparsers level. Exposed so the top-level CLI can keep its
    argparse construction in one place.
    """
    p = subparsers.add_parser(
        "design",
        help=(
            "Design prime-editor pegRNAs for a target locus and edit "
            "specification. PE2 + PE3 strategies; PRIDICT 2.0 scoring; "
            "ViennaRNA folding; CFD off-target aggregation."
        ),
    )
    # Mode A vs Mode B: exactly one of --target / --target-fasta.
    p.add_argument(
        "--target",
        default=None,
        help=(
            "Gene symbol (Mode A; e.g. VEGFA). Resolves via the bundled "
            "refGene table; the gene span is fetched from --fasta with "
            "±200 bp flanking. Mutually exclusive with --target-fasta."
        ),
    )
    p.add_argument(
        "--target-fasta",
        default=None,
        help=(
            "Path to a FASTA file (Mode B). The first record is used as "
            "the target sequence; the record name must match the chrom "
            "in --edit. Mutually exclusive with --target."
        ),
    )
    p.add_argument(
        "--edit",
        required=True,
        help=(
            "Edit notation. Simple (e.g. 'C>T at chr1:100', "
            "'insAGT at chr1:100', 'del chr1:100..105') or HGVS "
            "(e.g. 'chr1:g.100C>T', 'NM_007294.4:c.5266dupC')."
        ),
    )
    p.add_argument(
        "--strategy",
        default="both",
        choices=["pe2", "pe3", "both"],
        help="PE2, PE3, or both. Default 'both'.",
    )
    p.add_argument(
        "--scaffold",
        default="sgRNA_canonical",
        choices=sorted(SCAFFOLD_VARIANTS),
        help="Scaffold variant. Default 'sgRNA_canonical'.",
    )
    p.add_argument(
        "--pridict-cell-type",
        default="HEK293",
        choices=list(_PRIDICT_CELL_TYPES),
        help=(
            "PRIDICT 2.0 cell-type head. HEK/HEK293/HEK293T are aliases. "
            "Default HEK293. (HCT116/U2OS in the plan are NOT supported "
            "by upstream PRIDICT 2.0 weights and are rejected here.)"
        ),
    )
    p.add_argument(
        "--genome",
        default="hg38",
        help=(
            "Reference for off-target scan. 'hg38'/'hg19' aliases use "
            "$BIONPU_GRCH38_FASTA (or data_cache/genomes/grch38/hg38.fa); "
            "a path is used directly; 'none' = synbio mode (off-target "
            "scan skipped, off-target columns become NaN/0 with "
            "NO_OFF_TARGET_SCAN appended to notes)."
        ),
    )
    p.add_argument(
        "--fasta",
        default=None,
        help=(
            "Override reference FASTA for Mode A's gene-window fetch. "
            "Defaults to $BIONPU_GRCH38_FASTA / "
            "data_cache/genomes/grch38/hg38.fa."
        ),
    )
    p.add_argument(
        "--pbs-min", type=int, default=PBS_LENGTH_MIN,
        help=f"Minimum PBS length. Default {PBS_LENGTH_MIN}.",
    )
    p.add_argument(
        "--pbs-max", type=int, default=PBS_LENGTH_MAX,
        help=f"Maximum PBS length. Default {PBS_LENGTH_MAX}.",
    )
    p.add_argument(
        "--rtt-min", type=int, default=RTT_LENGTH_MIN,
        help=f"Minimum RTT length. Default {RTT_LENGTH_MIN}.",
    )
    p.add_argument(
        "--rtt-max", type=int, default=RTT_LENGTH_MAX,
        help=f"Maximum RTT length. Default {RTT_LENGTH_MAX}.",
    )
    p.add_argument(
        "--top", type=int, default=20,
        help="Top-N ranked pegRNAs. Default 20.",
    )
    p.add_argument(
        "--rank-by",
        default="composite_pridict",
        choices=["composite_pridict"],
        help="Composite-score driver for the top-N sort. Default and "
             "only supported value in v0: 'composite_pridict'.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "npu"],
        help=(
            "Compute device for the off-target scan. v0 honors 'cpu' "
            "only; 'npu' is reserved for v1 silicon path."
        ),
    )
    p.add_argument(
        "--max-mismatches", type=int, default=4,
        help="Maximum mismatches in the off-target scan. Default 4.",
    )
    p.add_argument(
        "--flanks", type=int, default=_MODE_A_DEFAULT_FLANKS,
        help=(
            f"Mode A: ±N bp of flanking around the gene span. Default "
            f"{_MODE_A_DEFAULT_FLANKS}."
        ),
    )
    p.add_argument(
        "--format", default="tsv", choices=["tsv", "json"],
        help="Output format. Default tsv.",
    )
    p.add_argument(
        "--output", default="-",
        help="Output path. '-' (default) writes to stdout.",
    )
    p.add_argument(
        "--low-memory", action="store_true",
        help=(
            "Use the PRIDICT scorer as a context manager and tear down "
            "weights after the run (saves ~95 MB at the cost of extra "
            "model-load time on the next invocation)."
        ),
    )
    p.add_argument(
        "--use-5folds", action="store_true",
        help=(
            "Score with the full PRIDICT 2.0 5-fold ensemble "
            "(~5x slower; more accurate). Default off (fold-1 only)."
        ),
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress the post-run summary on stderr.",
    )
    p.set_defaults(func=run_cli)


def run_cli(args: argparse.Namespace) -> int:
    """Implement ``bionpu crispr pe design ...``."""
    if (args.target is None) == (args.target_fasta is None):
        print(
            "bionpu crispr pe design: pass exactly one of --target SYMBOL "
            "or --target-fasta PATH.",
            file=sys.stderr,
        )
        return 2

    pbs_min = int(args.pbs_min)
    pbs_max = int(args.pbs_max)
    rtt_min = int(args.rtt_min)
    rtt_max = int(args.rtt_max)
    if pbs_min < 1 or pbs_max < pbs_min:
        print(
            f"bionpu crispr pe design: invalid --pbs-min/--pbs-max "
            f"({pbs_min}..{pbs_max}).",
            file=sys.stderr,
        )
        return 2
    if rtt_min < 1 or rtt_max < rtt_min:
        print(
            f"bionpu crispr pe design: invalid --rtt-min/--rtt-max "
            f"({rtt_min}..{rtt_max}).",
            file=sys.stderr,
        )
        return 2

    try:
        rows = design_prime_editor_guides(
            target_symbol=args.target,
            target_fasta=args.target_fasta,
            edit_notation=args.edit,
            strategy=args.strategy,
            scaffold_variant=args.scaffold,
            pridict_cell_type=args.pridict_cell_type,
            genome=args.genome,
            fasta_path=args.fasta,
            pbs_lengths=range(pbs_min, pbs_max + 1),
            rtt_lengths=range(rtt_min, rtt_max + 1),
            top_n=int(args.top),
            device=args.device,
            low_memory=bool(args.low_memory),
            max_mismatches=int(args.max_mismatches),
            flanks=int(args.flanks),
            use_5folds=bool(args.use_5folds),
        )
    except (
        GeneSymbolNotFound,
        FileNotFoundError,
        ValueError,
        RefMismatchError,
        EditTooLargeError,
        UnsupportedHGVSVariant,
    ) as exc:
        print(f"bionpu crispr pe design: {exc}", file=sys.stderr)
        return 2

    # ---- Emit ------------------------------------------------------- #
    out_arg = args.output
    use_stdout = out_arg == "-"
    if args.format == "tsv":
        if use_stdout:
            write_tsv(rows, sys.stdout)
        else:
            out_path = pathlib.Path(out_arg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_tsv(rows, out_path)
    else:  # json
        if use_stdout:
            write_json(rows, sys.stdout)
        else:
            out_path = pathlib.Path(out_arg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(rows, out_path)

    if not bool(getattr(args, "quiet", False)) and not use_stdout:
        target_label = args.target or args.target_fasta
        n_pe2 = sum(1 for r in rows if r.pe_strategy == "PE2")
        n_pe3 = sum(1 for r in rows if r.pe_strategy == "PE3")
        n_synbio = (
            "synbio" if (
                isinstance(args.genome, str) and args.genome.lower() == "none"
            ) else "off-target=" + str(args.genome)
        )
        print(
            f"bionpu crispr pe design: wrote {len(rows)} ranked pegRNAs "
            f"({n_pe2} PE2 + {n_pe3} PE3) to {out_arg} "
            f"(target={target_label}, edit={args.edit!r}, "
            f"strategy={args.strategy}, {n_synbio})",
            file=sys.stderr,
        )

    return 0


# Silence unused-imports warnings on diagnostic helpers used only by
# tests / future callers but loaded via __all__.
_ = math
