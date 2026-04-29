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
    """
    lines = [_TSV_HEADER]
    for i, g in enumerate(guides, 1):
        byst = ";".join(str(p) for p in g.bystander_positions)
        lines.append(
            f"{i}\t{g.guide_seq}\t{g.pam_seq}\t{g.target_pos}\t"
            f"{g.target_base}\t{g.target_pos_in_protospacer}\t"
            f"{int(g.in_activity_window)}\t{g.bystander_count}\t{byst}\t"
            f"{g.off_target_count}\t{g.cfd_aggregate:.6f}\t"
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
    p.add_argument(
        "--target-fasta",
        required=True,
        help="Path to a target FASTA. Reads the first record.",
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
            "Reference for off-target scan. Currently only 'none' "
            "(synbio mode; no off-target scan) is supported. v1+ "
            "wires the locked match_multitile_memtile kernel."
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
    target_fasta = pathlib.Path(args.target_fasta)
    try:
        record_name, target_seq = _read_first_fasta_record(target_fasta)
    except (FileNotFoundError, ValueError) as exc:
        print(f"bionpu be design: {exc}", file=sys.stderr)
        return 2

    if args.genome != "none":
        print(
            "bionpu be design: only --genome none (synbio mode) is "
            "supported in v0; off-target scan is deferred to v1+. "
            "Re-run with --genome none.",
            file=sys.stderr,
        )
        return 2

    try:
        guides = design_base_editor_guides(
            target_seq,
            be_variant=args.be_variant,
            cas9_variant=args.cas9_variant,
            genome_path=None,  # v0: synbio mode only
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
