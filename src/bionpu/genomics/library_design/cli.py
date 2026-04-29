# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""``bionpu library design`` subcommand — Track C v0 library design.

Mirrors the shape of the sibling ``bionpu crispr design`` and
``bionpu be design`` CLIs:

    bionpu library design \\
        --targets-file gene_list.txt \\
        --library-type knockout \\
        --guides-per-gene 4 \\
        --genome GRCh38 \\
        --controls 1000 \\
        --output library.tsv

In v0 only ``--library-type knockout`` is wired; activation /
interference deferred to v1.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .output import format_library_tsv
from .pipeline import PipelineMetrics, run_library_pipeline

__all__ = [
    "add_library_design_subparser",
    "run_cli",
]


def _read_targets_file(path: pathlib.Path) -> list[str]:
    """Read gene symbols from a file (one per line; ``#`` comments)."""
    if not path.is_file():
        raise FileNotFoundError(f"targets file not found: {path}")
    out: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Allow comma-separated entries on a single line for ergonomics.
            for sym in line.split(","):
                sym = sym.strip()
                if sym:
                    out.append(sym.upper())
    if not out:
        raise ValueError(f"no gene symbols found in {path}")
    return out


def _resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.targets_file:
        return _read_targets_file(pathlib.Path(args.targets_file))
    if args.targets:
        return [s.strip().upper() for s in args.targets.split(",") if s.strip()]
    raise ValueError("must supply either --targets or --targets-file")


def _default_grch38_fasta() -> pathlib.Path:
    """Mirror the sibling default-FASTA helper from ``bionpu.cli``."""
    import os

    env = os.environ.get("BIONPU_GRCH38_FASTA")
    if env:
        return pathlib.Path(env)
    return pathlib.Path("data_cache/genomes/grch38/hg38.fa")


def add_library_design_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``bionpu library design`` on a top-level subparsers action."""
    p = subparsers.add_parser(
        "design",
        help=(
            "Design a pooled CRISPR library for a list of target genes. "
            "Track C v0: knockout libraries (NGG SpCas9); 4-6 guides per "
            "gene; non-targeting + safe-harbor + essential-gene controls."
        ),
    )
    p.add_argument(
        "--targets",
        default=None,
        help=(
            "Comma-separated gene symbols, e.g. 'BRCA1,TP53,EGFR'. "
            "Mutually exclusive with --targets-file (one is required)."
        ),
    )
    p.add_argument(
        "--targets-file",
        default=None,
        help=(
            "Path to a text file with one gene symbol per line. "
            "'#' starts a comment; comma-separated entries on a single "
            "line are accepted."
        ),
    )
    p.add_argument(
        "--library-type",
        default="knockout",
        choices=["knockout", "activation", "interference"],
        help=(
            "Library variant. Track C v0 supports only 'knockout' "
            "(NGG SpCas9); 'activation' and 'interference' raise "
            "NotImplementedError (v1 deferral — different PAM rules + "
            "Cas9-VPR / dCas9-KRAB scoring)."
        ),
    )
    p.add_argument(
        "--guides-per-gene",
        type=int,
        default=4,
        help="Guides per gene. 4-6 is typical (4 = Brunello). Default 4.",
    )
    p.add_argument(
        "--genome",
        default="GRCh38",
        choices=["GRCh38"],
        help="Reference build. Track C v0 supports GRCh38 only.",
    )
    p.add_argument(
        "--fasta",
        default=None,
        help=(
            "Path to the GRCh38 reference FASTA. Defaults to "
            "$BIONPU_GRCH38_FASTA, then to data_cache/genomes/grch38/hg38.fa."
        ),
    )
    p.add_argument(
        "--controls",
        type=int,
        default=1000,
        help=(
            "Number of non-targeting controls to emit. Safe-harbor "
            "(AAVS1/CCR5/ROSA26) + essential-gene (RPS19/RPL15) controls "
            "are added in addition. Default 1000."
        ),
    )
    p.add_argument(
        "--mismatches",
        type=int,
        default=4,
        help="Per-gene off-target max-mismatches. Default 4.",
    )
    p.add_argument(
        "--gc-min",
        type=float,
        default=25.0,
        help="GC%% lower bound. Default 25.",
    )
    p.add_argument(
        "--gc-max",
        type=float,
        default=75.0,
        help="GC%% upper bound. Default 75.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "npu"],
        help=(
            "Compute device for the per-gene off-target scan. Default "
            "'cpu' (numpy oracle). 'npu' wraps every silicon submission "
            "in bionpu.dispatch.npu_silicon_lock per CLAUDE.md."
        ),
    )
    p.add_argument(
        "--rank-by",
        default="crispor",
        choices=["crispor", "bionpu"],
        help=(
            "Composite score driving the per-gene top-N sort + dedup "
            "tiebreak. Default 'crispor' (mirrors CRISPOR CLI)."
        ),
    )
    p.add_argument(
        "--pool-oversample",
        type=int,
        default=4,
        help=(
            "Per-gene pool size = guides_per_gene * pool_oversample. "
            "Larger oversampling gives the dedup pass more headroom. "
            "Default 4."
        ),
    )
    p.add_argument(
        "--dedup-strategy",
        default="highest_score",
        choices=["highest_score", "alphabetical"],
        help=(
            "How to resolve cross-gene guide-spacer collisions. "
            "'highest_score' keeps the higher-composite gene (default). "
            "'alphabetical' is deterministic but quality-blind; intended "
            "for tests."
        ),
    )
    p.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Salt for the deterministic non-targeting control generator.",
    )
    p.add_argument(
        "--silicon-lock-label",
        default=None,
        help=(
            "Optional label written to the NPU silicon lock PID sidecar. "
            "Defaults to `bionpu_library_design:{gene}` per per-gene "
            "design call."
        ),
    )
    p.add_argument(
        "--measurements-out",
        default=None,
        help=(
            "Optional path to write a measurements.json record "
            "(per-gene wall, total wall, dedup count, control counts, "
            "balance summary). Mirrors results/library_design/v0/."
        ),
    )
    p.add_argument(
        "--output",
        default="-",
        help="Output TSV path. '-' (default) writes to stdout.",
    )
    p.set_defaults(func=run_cli)


def run_cli(args: argparse.Namespace) -> int:
    """Implementation of ``bionpu library design ...``."""
    try:
        targets = _resolve_targets(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"bionpu library design: {exc}", file=sys.stderr)
        return 2

    fasta_path = pathlib.Path(args.fasta or _default_grch38_fasta())
    if not fasta_path.is_file():
        print(
            f"bionpu library design: reference FASTA not found at "
            f"{fasta_path!s}. Pass --fasta <path> or set $BIONPU_GRCH38_FASTA.",
            file=sys.stderr,
        )
        return 2

    metrics = PipelineMetrics()
    try:
        rows = run_library_pipeline(
            targets=targets,
            library_type=args.library_type,
            guides_per_gene=int(args.guides_per_gene),
            genome=args.genome,
            fasta_path=fasta_path,
            n_controls=int(args.controls),
            max_mismatches=int(args.mismatches),
            gc_min=float(args.gc_min),
            gc_max=float(args.gc_max),
            device=args.device,
            rank_by=args.rank_by,
            pool_oversample=int(args.pool_oversample),
            dedup_strategy=args.dedup_strategy,
            rng_seed=int(args.rng_seed),
            silicon_lock_label=args.silicon_lock_label,
            metrics=metrics,
        )
    except NotImplementedError as exc:
        print(f"bionpu library design: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"bionpu library design: {exc}", file=sys.stderr)
        return 2

    out_bytes = format_library_tsv(rows)
    if args.output == "-":
        sys.stdout.buffer.write(out_bytes)
    else:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(out_bytes)
        print(
            f"bionpu library design: wrote {len(rows)} rows to "
            f"{out_path} (n_targets={len(targets)}, "
            f"guides_per_gene={args.guides_per_gene}, "
            f"non_targeting={metrics.n_non_targeting}, "
            f"safe_harbor={metrics.n_safe_harbor}, "
            f"essential={metrics.n_essential}, "
            f"under_balanced_genes={metrics.n_under_balanced_genes}, "
            f"dedup_collisions={metrics.dedup_collisions}, "
            f"total_wall={metrics.total_wall_s:.3f}s)",
            file=sys.stderr,
        )

    if args.measurements_out:
        m_path = pathlib.Path(args.measurements_out)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "track": "C",
            "version": "v0",
            "n_targets": len(targets),
            "guides_per_gene": int(args.guides_per_gene),
            "library_type": args.library_type,
            "device": args.device,
            "rank_by": args.rank_by,
            "metrics": metrics.to_json(),
            "n_library_rows": len(rows),
        }
        m_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            f"bionpu library design: wrote measurements to {m_path}",
            file=sys.stderr,
        )

    return 0
