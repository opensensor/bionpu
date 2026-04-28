#!/usr/bin/env python3
# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Producer harness: run ``bionpu crispr design`` for the 20 pinned genes.

Reads ``tests/fixtures/crispor_reference/genes_pinned.json``; emits one
TSV per gene under ``tests/fixtures/crispor_reference/bionpu_output/``;
emits a run-summary JSON capturing wall-clock + counters.

Usage::

    cd bionpu-public
    python tools/run_20gene_validation.py [--device cpu|npu] [--top N]

The complementary CRISPOR-fixture generation is in
``tools/run_20gene_crispor_fixtures.sh`` (which depends on the
external CRISPOR install at the pinned commit
``ed47b7e856010ad0f9f1660872563ef9f736e76c``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _default_repo_root() -> Path:
    # tests live at <repo>/bionpu-public/tests; walk up.
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_20gene_validation")
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument(
        "--genome-fasta", default=None,
        help="Override the GRCh38 FASTA path (default: data_cache/genomes/grch38/hg38.fa)",
    )
    parser.add_argument(
        "--genes", default=None,
        help="Comma-separated subset of pinned symbols to run (default: all 20).",
    )
    args = parser.parse_args(argv)

    repo = _default_repo_root()
    fixtures_dir = repo / "bionpu-public" / "tests" / "fixtures" / "crispor_reference"
    bionpu_out = fixtures_dir / "bionpu_output"
    bionpu_out.mkdir(parents=True, exist_ok=True)
    fasta_path = (
        Path(args.genome_fasta)
        if args.genome_fasta
        else repo / "data_cache" / "genomes" / "grch38" / "hg38.fa"
    )
    if not fasta_path.is_file():
        print(f"genome FASTA not found at {fasta_path}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(repo / "bionpu-public" / "src"))
    from bionpu.genomics.crispr_design import design_guides_for_target

    with (fixtures_dir / "genes_pinned.json").open() as fh:
        meta = json.load(fh)
    pinned = [e["symbol"] for e in meta["genes"]]
    if args.genes:
        wanted = set(args.genes.split(","))
        pinned = [s for s in pinned if s in wanted]

    summary: dict[str, dict[str, object]] = {}
    for sym in pinned:
        out_path = bionpu_out / f"{sym}_bionpu.tsv"
        print(f"==> {sym}", flush=True)
        t0 = time.perf_counter()
        try:
            result = design_guides_for_target(
                target=sym,
                genome="GRCh38",
                fasta_path=fasta_path,
                top_n=args.top,
                device=args.device,
                rank_by="crispor",
            )
            wall = time.perf_counter() - t0
            out_path.write_bytes(result.tsv_bytes)
            summary[sym] = {
                "status": "ok",
                "wall_clock_s": round(wall, 3),
                "guides_returned": len(result.ranked),
                "n_candidates_total": result.n_candidates_total,
                "n_off_target_hits": result.n_off_target_hits,
                "locus_bp": result.target.length,
                "stage_timings_s": result.stage_timings_s,
            }
            print(
                f"    OK in {wall:.2f}s — {len(result.ranked)} guides, "
                f"locus={result.target.length:,} bp",
                flush=True,
            )
        except Exception as exc:
            wall = time.perf_counter() - t0
            summary[sym] = {
                "status": "error",
                "wall_clock_s": round(wall, 3),
                "error_class": type(exc).__name__,
                "error_msg": str(exc),
            }
            print(f"    FAIL ({type(exc).__name__}): {exc}", flush=True)

    summary_path = bionpu_out / "_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSummary: {summary_path}", flush=True)
    ok = sum(1 for v in summary.values() if v["status"] == "ok")
    print(f"  {ok}/{len(summary)} genes succeeded", flush=True)
    return 0 if ok == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
