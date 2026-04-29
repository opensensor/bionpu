# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""``bionpu be design`` subcommand — Track A v0 base editor guide design.

CLI shape (mirrors ``bionpu crispr design``):

    bionpu be design \\
        --target-fasta brca1.fa \\
        --be-variant BE4max \\
        --cas9-variant wt \\
        --genome none \\
        --top 20 \\
        --output brca1_be_guides.tsv

In v0 we only support ``--target-fasta`` (Mode C; explicit FASTA) and
``--genome none`` (synbio mode — no off-target scan). Mode A (gene
symbol → reference fetch) and full off-target scanning are deferred
to v1+.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from .pam_variants import BE_VARIANTS, CAS9_VARIANTS
from .ranker import BaseEditorGuide, design_base_editor_guides

__all__ = [
    "format_tsv",
    "run_cli",
]


def _read_first_fasta_record(path: pathlib.Path) -> tuple[str, str]:
    """Read the first record from a FASTA file. Returns ``(name, seq)``."""
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
                # No header yet; skip.
                continue
            parts.append(line)
    if name is None:
        raise ValueError(f"no FASTA record found in {path}")
    seq = "".join(parts).upper()
    return name, seq


_TSV_HEADER = (
    "rank\tguide_seq\tpam_seq\ttarget_pos\ttarget_base\t"
    "target_pos_in_protospacer\tin_activity_window\t"
    "bystander_count\tbystander_positions\t"
    "off_target_count\tcfd_aggregate\trank_score\tnotes\n"
)


def format_tsv(guides: list[BaseEditorGuide]) -> str:
    """Format ranked guides as a CRISPOR-style TSV.

    Columns (header first):
      rank, guide_seq, pam_seq, target_pos, target_base,
      target_pos_in_protospacer, in_activity_window, bystander_count,
      bystander_positions (semicolon-separated), off_target_count,
      cfd_aggregate, rank_score, notes

    NaN ``cfd_aggregate`` (synbio mode — no scan performed) renders as
    ``"NaN"``. Numeric values (incl. 0.0) render as ``"%.6f"``.
    """
    import math as _math
    lines = [_TSV_HEADER]
    for i, g in enumerate(guides, 1):
        byst = ";".join(str(p) for p in g.bystander_positions)
        cfd_str = (
            "NaN" if _math.isnan(float(g.cfd_aggregate))
            else f"{g.cfd_aggregate:.6f}"
        )
        lines.append(
            f"{i}\t{g.guide_seq}\t{g.pam_seq}\t{g.target_pos}\t"
            f"{g.target_base}\t{g.target_pos_in_protospacer}\t"
            f"{int(g.in_activity_window)}\t{g.bystander_count}\t{byst}\t"
            f"{g.off_target_count}\t{cfd_str}\t"
            f"{g.rank_score:.6f}\t{g.notes}\n"
        )
    return "".join(lines)


def add_be_design_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``bionpu be design`` on a top-level subparsers action."""
    p = subparsers.add_parser(
        "design",
        help=(
            "Design base editor (ABE/CBE) guides against a target. "
            "Track A v0: SpCas9 wt + SpCas9-NG; BE4max + ABE7.10."
        ),
    )
    # UCSC genome-fetch v1: Mode A (gene symbol) + Mode C (target FASTA).
    # Exactly one of --target / --target-fasta must be supplied.
    p.add_argument(
        "--target",
        default=None,
        help=(
            "Gene symbol (e.g. BRCA1). Resolves to GRCh38 coordinates "
            "via UCSC refGene; the sequence is sliced from --fasta "
            "(or $BIONPU_GRCH38_FASTA / data_cache/genomes/grch38/"
            "hg38.fa). Mutually exclusive with --target-fasta."
        ),
    )
    p.add_argument(
        "--target-fasta",
        default=None,
        help=(
            "Path to a target FASTA. Reads the first record. "
            "Mutually exclusive with --target."
        ),
    )
    p.add_argument(
        "--fasta",
        default=None,
        help=(
            "Reference FASTA for --target gene-symbol resolution. "
            "Defaults to $BIONPU_GRCH38_FASTA or "
            "data_cache/genomes/grch38/hg38.fa."
        ),
    )
    p.add_argument(
        "--flanks",
        type=int,
        default=0,
        help=(
            "Bases of flanking sequence to fetch on either side of "
            "the gene span (only with --target). Default 0."
        ),
    )
    p.add_argument(
        "--be-variant",
        default="BE4max",
        choices=sorted(BE_VARIANTS),
        help="Base editor variant. Default BE4max.",
    )
    p.add_argument(
        "--cas9-variant",
        default="wt",
        choices=sorted(CAS9_VARIANTS),
        help="Cas9 PAM variant. Default 'wt' (NGG).",
    )
    p.add_argument(
        "--genome",
        default="none",
        help=(
            "Reference for off-target scan. Either 'none' (synbio mode; "
            "no off-target scan, v0 default) or a path to a FASTA "
            "(v1; wires the locked crispr/match_multitile_memtile "
            "scan + CFD aggregate scoring per guide)."
        ),
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="Top-N ranked guides. 0 = all. Default 20.",
    )
    p.add_argument(
        "--require-in-window",
        action="store_true",
        help="Drop guides whose target base is outside the activity window.",
    )
    p.add_argument(
        "--use-silicon",
        action="store_true",
        help=(
            "Use the silicon pam_filter_iupac kernel when artifacts "
            "are present; falls back to CPU oracle otherwise."
        ),
    )
    p.add_argument(
        "--output",
        default="-",
        help="Output TSV path. '-' (default) writes to stdout.",
    )
    p.set_defaults(func=run_cli)


def run_cli(args: argparse.Namespace) -> int:
    """Implementation of ``bionpu be design ...``."""
    # UCSC genome-fetch v1: resolve target via either Mode A (gene
    # symbol) or Mode C (target FASTA). Exactly one must be supplied.
    target_arg = getattr(args, "target", None)
    target_fasta_arg = getattr(args, "target_fasta", None)
    if (target_arg is None) == (target_fasta_arg is None):
        print(
            "bionpu be design: pass exactly one of --target SYMBOL "
            "or --target-fasta PATH.",
            file=sys.stderr,
        )
        return 2

    if target_arg is not None:
        # Mode A: gene symbol -> GRCh38 coordinates -> sequence.
        from .target_resolver import GeneSymbolNotFound, resolve_be_target

        fasta_arg = getattr(args, "fasta", None)
        flanks_arg = int(getattr(args, "flanks", 0) or 0)
        try:
            record_name, target_seq, _coord = resolve_be_target(
                target_arg,
                genome="hg38",
                fasta_path=pathlib.Path(fasta_arg) if fasta_arg else None,
                flanks=flanks_arg,
            )
        except GeneSymbolNotFound as exc:
            print(f"bionpu be design: {exc}", file=sys.stderr)
            return 2
        except (ValueError, FileNotFoundError) as exc:
            print(f"bionpu be design: {exc}", file=sys.stderr)
            return 2
    else:
        target_fasta = pathlib.Path(target_fasta_arg)
        try:
            record_name, target_seq = _read_first_fasta_record(target_fasta)
        except (FileNotFoundError, ValueError) as exc:
            print(f"bionpu be design: {exc}", file=sys.stderr)
            return 2

    # v1: --genome is either 'none' (synbio mode) or a path to a FASTA.
    if args.genome == "none":
        genome_arg: pathlib.Path | None = None
    else:
        genome_arg = pathlib.Path(args.genome)
        if not genome_arg.is_file():
            print(
                f"bionpu be design: --genome {args.genome} is not a "
                f"FASTA file. Pass 'none' for synbio mode or supply "
                f"a valid FASTA path.",
                file=sys.stderr,
            )
            return 2

    try:
        guides = design_base_editor_guides(
            target_seq,
            be_variant=args.be_variant,
            cas9_variant=args.cas9_variant,
            genome_path=genome_arg,
            top_n=int(args.top),
            use_silicon=bool(args.use_silicon),
            require_in_window=bool(args.require_in_window),
            notes_prefix=f"target={record_name}",
        )
    except ValueError as exc:
        print(f"bionpu be design: {exc}", file=sys.stderr)
        return 2

    out_text = format_tsv(guides)
    if args.output == "-":
        sys.stdout.write(out_text)
    else:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text)
        print(
            f"bionpu be design: wrote {len(guides)} ranked guides to "
            f"{out_path} (target={record_name}, "
            f"length={len(target_seq):,} bp, "
            f"be_variant={args.be_variant}, "
            f"cas9_variant={args.cas9_variant})",
            file=sys.stderr,
        )
    return 0
