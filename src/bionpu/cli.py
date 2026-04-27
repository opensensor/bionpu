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

"""bionpu command-line interface.

Subcommands:

- ``bionpu verify crispr <npu.tsv> <ref.tsv>``
  Byte-equality check of an NPU off-target hits TSV against a
  Cas-OFFinder reference TSV.
- ``bionpu verify basecalling <npu.fastq> <ref.fastq>``
  Byte-equality check of an NPU-emitted FASTQ against a Dorado
  reference FASTQ.
- ``bionpu scan ...``
  CRISPR off-target scan (placeholder in v0.1; see
  ``benchmarks/crispr/run_chr.sh`` for the canonical scan pipeline).
- ``bionpu basecall ...``
  Nanopore basecalling (placeholder in v0.1; see
  ``benchmarks/basecalling/run_pod5.sh``).
- ``bionpu bench ...``
  Energy + timing harness (placeholder in v0.1; see
  ``docs/ENERGY_METHODOLOGY.md`` for the methodology).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from . import __version__


def _format_result(label: str, result) -> str:
    lines = [
        f"  result:        {'EQUAL' if result.equal else 'DIVERGENT'}",
        f"  records:       {result.record_count}",
        f"  npu sha256:    {result.npu_sha256}",
        f"  ref sha256:    {result.ref_sha256}",
    ]
    if not result.equal and result.divergences:
        lines.append("  divergences:")
        for div in result.divergences[:8]:
            lines.append(f"    [{div.record_index}] {div.message}")
        if len(result.divergences) > 8:
            lines.append(f"    ... ({len(result.divergences) - 8} more)")
    return f"{label}\n" + "\n".join(lines)


def _cmd_verify_crispr(args: argparse.Namespace) -> int:
    from .verify.crispr import compare_against_cas_offinder

    result = compare_against_cas_offinder(
        args.npu_tsv, args.ref_tsv, max_divergences=args.max_divergences
    )
    print(_format_result(f"verify crispr: {args.npu_tsv} vs {args.ref_tsv}", result))
    return 0 if result.equal else 1


def _cmd_verify_basecalling(args: argparse.Namespace) -> int:
    from .verify.basecalling import compare_against_dorado

    result = compare_against_dorado(
        args.npu_fastq, args.ref_fastq, max_divergences=args.max_divergences
    )
    print(_format_result(
        f"verify basecalling: {args.npu_fastq} vs {args.ref_fastq}", result
    ))
    return 0 if result.equal else 1


def _cmd_scan(args: argparse.Namespace) -> int:
    """Run a CRISPR off-target scan and emit a canonical TSV.

    v0.1: pure-CPU implementation via :func:`bionpu.scan.cpu_scan`.
    v0.2: NPU implementation via :func:`bionpu.scan.npu_scan` (PAM-filter
    kernel dispatched through :mod:`bionpu.dispatch`). The NPU path
    requires the kernel artifacts to be built — see the
    :class:`bionpu.dispatch.npu.NpuArtifactsMissingError` message
    if they are not.
    """
    from .data.canonical_sites import normalize, write_tsv
    from .scan import cpu_scan, npu_scan, parse_guides, read_fasta

    chrom, seq = read_fasta(args.target)
    guides = parse_guides(args.guides)

    print(
        f"bionpu scan [{args.device}]: chrom={chrom!r}, seq={len(seq):,} nt, "
        f"guides={len(guides)}, max_mismatches={args.max_mismatches}",
        file=sys.stderr,
    )

    if args.device == "npu":
        from .dispatch.npu import NpuArtifactsMissingError
        try:
            rows = npu_scan(
                chrom=chrom,
                seq=seq,
                guides=guides,
                pam_template=args.pam,
                max_mismatches=args.max_mismatches,
                op_name=args.op,
            )
        except NpuArtifactsMissingError as exc:
            print(
                f"bionpu scan --device npu: kernel artifacts missing.\n"
                f"  {exc}\n"
                f"\n"
                f"Build the kernel:\n"
                f"  cd src/bionpu/kernels/crispr/pam_filter && make NPU2=1\n"
                f"\n"
                f"Then either copy the produced build/{{early,late}}/final.xclbin\n"
                f"and insts.bin into bionpu/dispatch/_npu_artifacts/ or set\n"
                f"BIONPU_KERNEL_ARTIFACTS_DIR to the directory containing them.",
                file=sys.stderr,
            )
            return 3
    else:
        rows = cpu_scan(
            chrom=chrom,
            seq=seq,
            guides=guides,
            pam_template=args.pam,
            max_mismatches=args.max_mismatches,
        )

    rows = normalize(rows)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(out_path, rows)
    print(
        f"bionpu scan: wrote {len(rows)} hits to {out_path}",
        file=sys.stderr,
    )

    if args.verify:
        from .verify.crispr import compare_against_cas_offinder

        result = compare_against_cas_offinder(out_path, args.verify)
        print(_format_result(
            f"verify crispr: {out_path} vs {args.verify}", result
        ))
        return 0 if result.equal else 1
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """Score canonical scan rows with an off-target probability.

    v0.3 alpha: DNABERT-Epi (no-epi variant) on CPU or GPU. The AIE2P
    backend is a follow-up; the with-epi (BigWig) variant is a
    follow-up. The ``--smoke`` flag exercises the pipeline end-to-end
    without weights or torch (deterministic pseudo-random scores
    keyed by row identity), useful for CI and CLI demonstration.
    """
    from .data.canonical_sites import parse_tsv
    from .scoring.dnabert_epi import DNABERTEpiScorer, DNABERTEpiUnavailableError
    from .scoring.types import write_score_tsv

    rows = parse_tsv(pathlib.Path(args.candidates))
    print(
        f"bionpu score [{args.model}/{args.device}{' smoke' if args.smoke else ''}]: "
        f"{len(rows)} candidate rows from {args.candidates}",
        file=sys.stderr,
    )

    if args.model != "dnabert-epi":
        print(
            f"bionpu score: model {args.model!r} not yet wired; "
            f"only 'dnabert-epi' is available in v0.3 alpha.",
            file=sys.stderr,
        )
        return 2

    weights = pathlib.Path(args.weights) if args.weights else None
    try:
        scorer = DNABERTEpiScorer(
            device=args.device,
            weights_path=weights,
            smoke=args.smoke,
            seed=args.seed,
        )
        scored = list(scorer.score(rows))
    except DNABERTEpiUnavailableError as exc:
        print(f"bionpu score: {exc}", file=sys.stderr)
        return 3

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_score_tsv(out_path, scored)
    print(
        f"bionpu score: wrote {len(scored)} scored rows to {out_path}",
        file=sys.stderr,
    )

    if args.verify:
        from .verify.score import compare_score_outputs

        result = compare_score_outputs(
            out_path,
            args.verify,
            policy=args.verify_policy,
            epsilon=args.verify_epsilon if args.verify_policy == "NUMERIC_EPSILON" else None,
        )
        verdict = "EQUAL" if result.equal else "DIVERGENT"
        print(f"verify score [{result.policy}]: {out_path} vs {args.verify}")
        print(f"  result:        {verdict}")
        print(f"  records:       {result.record_count}")
        print(f"  a sha256:      {result.a_sha256}")
        print(f"  b sha256:      {result.b_sha256}")
        if result.max_abs_diff is not None:
            print(f"  max |a-b|:     {result.max_abs_diff:.6g}")
        if not result.equal and result.divergences:
            print("  divergences:")
            for div in result.divergences[:8]:
                print(f"    [{div.record_index}] {div.message}")
            if len(result.divergences) > 8:
                print(f"    ... ({len(result.divergences) - 8} more)")
        return 0 if result.equal else 1
    return 0


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print(
        f"bionpu {args.cmd}: not yet implemented in v0.1. "
        f"See README.md / docs/REPRODUCE.md for the v0.1 scope.",
        file=sys.stderr,
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bionpu",
        description=(
            "AIE2P-accelerated genomics with reference-equivalence verification."
        ),
    )
    p.add_argument("--version", action="version",
                   version=f"bionpu {__version__}")
    sub = p.add_subparsers(dest="cmd")

    # verify
    p_verify = sub.add_parser(
        "verify",
        help="Byte-equality check of NPU output vs canonical reference.",
    )
    sub_verify = p_verify.add_subparsers(dest="verify_kind")

    p_v_crispr = sub_verify.add_parser(
        "crispr",
        help="Compare NPU off-target TSV vs Cas-OFFinder reference TSV.",
    )
    p_v_crispr.add_argument("npu_tsv", help="NPU-emitted hits TSV")
    p_v_crispr.add_argument("ref_tsv", help="Cas-OFFinder reference TSV")
    p_v_crispr.add_argument("--max-divergences", type=int, default=16)
    p_v_crispr.set_defaults(func=_cmd_verify_crispr)

    p_v_bc = sub_verify.add_parser(
        "basecalling",
        help="Compare NPU-emitted FASTQ vs Dorado reference FASTQ.",
    )
    p_v_bc.add_argument("npu_fastq", help="NPU-emitted FASTQ")
    p_v_bc.add_argument("ref_fastq", help="Dorado reference FASTQ")
    p_v_bc.add_argument("--max-divergences", type=int, default=16)
    p_v_bc.set_defaults(func=_cmd_verify_basecalling)

    # scan
    p_scan = sub.add_parser(
        "scan",
        help="CRISPR off-target scan (CPU in v0.1; NPU is v0.2 scope).",
    )
    p_scan.add_argument(
        "--target",
        required=True,
        help="FASTA file with the target sequence (single record).",
    )
    p_scan.add_argument(
        "--guides",
        required=True,
        help=(
            "Comma-separated list of 20-nt ACGT spacers, OR a path to a "
            "guide-list file (one spacer per line; `id:spacer` allowed; "
            "`#` comments OK)."
        ),
    )
    p_scan.add_argument(
        "--out",
        required=True,
        help="Output canonical TSV path.",
    )
    p_scan.add_argument(
        "--pam",
        default="NGG",
        help="PAM template (only NGG supported in v0.1).",
    )
    p_scan.add_argument(
        "--max-mismatches",
        type=int,
        default=4,
        help="Maximum spacer mismatches.",
    )
    p_scan.add_argument(
        "--device",
        choices=["cpu", "npu"],
        default="cpu",
        help="Compute device. cpu = pure-numpy fallback; npu = AIE2P PAM-filter kernel.",
    )
    p_scan.add_argument(
        "--op",
        default="crispr_pam_filter_early",
        choices=["crispr_pam_filter_early", "crispr_pam_filter_late"],
        help=(
            "NPU op variant (only meaningful with --device npu). "
            "filter-early is the production path; filter-late is a "
            "comparison artifact with the same output bytes but "
            "different on-tile work distribution."
        ),
    )
    p_scan.add_argument(
        "--verify",
        default=None,
        help=(
            "If supplied, compare the scan output byte-equally against this "
            "reference TSV (via bionpu.verify.crispr). Exits 0 on equality, "
            "1 on divergence."
        ),
    )
    p_scan.set_defaults(func=_cmd_scan)

    # score
    p_score = sub.add_parser(
        "score",
        help=(
            "Score canonical scan rows with an off-target probability "
            "(v0.3 alpha: DNABERT-Epi, no-epi, CPU+GPU)."
        ),
    )
    p_score.add_argument(
        "--candidates",
        required=True,
        help="Canonical scan TSV (output of `bionpu scan`).",
    )
    p_score.add_argument(
        "--out",
        required=True,
        help="Output scored canonical TSV.",
    )
    p_score.add_argument(
        "--model",
        default="dnabert-epi",
        choices=["dnabert-epi"],
        help="Scorer model. Only dnabert-epi is wired in v0.3 alpha.",
    )
    p_score.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
        help=(
            "Compute device. cpu = baseline + byte-equivalence reference; "
            "gpu = CUDA-accelerated (requires torch.cuda)."
        ),
    )
    p_score.add_argument(
        "--weights",
        default=None,
        help=(
            "Path to a fine-tuned classifier checkpoint. Required unless "
            "--smoke is set. See docs/model-selection-audit.md for the "
            "current state of trained-weights availability."
        ),
    )
    p_score.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use deterministic pseudo-random scores keyed by row identity. "
            "No torch / weights required; intended for CI and CLI demos."
        ),
    )
    p_score.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Salt for --smoke mode (no effect on real scoring).",
    )
    p_score.add_argument(
        "--verify",
        default=None,
        help=(
            "If supplied, compare the score output against this reference "
            "score TSV under the policy chosen by --verify-policy."
        ),
    )
    p_score.add_argument(
        "--verify-policy",
        default="NUMERIC_EPSILON",
        choices=["NUMERIC_EPSILON", "BITWISE_EXACT"],
        help=(
            "Equivalence policy when --verify is set. "
            "BITWISE_EXACT for deterministic backends (e.g. CPU vs CPU); "
            "NUMERIC_EPSILON for cross-device (CPU vs GPU vs NPU) where "
            "ULP-level deviation is expected."
        ),
    )
    p_score.add_argument(
        "--verify-epsilon",
        type=float,
        default=1e-6,
        help="Per-row score tolerance for NUMERIC_EPSILON. Default 1e-6.",
    )
    p_score.set_defaults(func=_cmd_score)

    # placeholders — scope for v0.2
    for name, help_text in (
        ("basecall", "Nanopore basecalling (v0.2 scope)"),
        ("bench", "Energy + timing harness (v0.2 scope)"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.set_defaults(func=_cmd_not_implemented)

    return p


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.cmd is None:
        p.print_help()
        return 0
    if args.cmd == "verify" and getattr(args, "verify_kind", None) is None:
        # `bionpu verify` with no sub-subcommand — print verify help
        for action in p._subparsers._actions:  # type: ignore[attr-defined]
            if isinstance(action, argparse._SubParsersAction):
                for choice, sub_p in action.choices.items():
                    if choice == "verify":
                        sub_p.print_help()
                        return 0
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
