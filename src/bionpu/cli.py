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

    # placeholders — scope for v0.2
    for name, help_text in (
        ("scan", "CRISPR off-target scan (v0.2 scope)"),
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
