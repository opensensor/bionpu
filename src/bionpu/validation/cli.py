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

"""``bionpu validate`` CLI subcommand — Track F v0.

Supported invocations:

    bionpu validate trim            --reference cutadapt    --fixture synthetic-100
    bionpu validate crispr design   --reference ucsc-pam    --fixture synthetic-pam-injection
    bionpu validate crispr design   --reference cas-offinder --fixture synthetic-pam-injection
    bionpu validate crispr pe design --reference pridict2   --fixture anzalone-hek3
    bionpu validate be design       --reference be-hive     --fixture brca1
    bionpu validate library design  --reference brunello    --fixture synthetic-library-5gene
    bionpu validate all                                       # full matrix
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .agreement import (
    AgreementCheck,
    Verdict,
    matrix_to_json,
    run_full_matrix,
    run_validation,
    write_matrix_json,
)


__all__ = [
    "add_validate_subparser",
    "run_cli",
]


_DEFAULT_OUTPUT = pathlib.Path("results/validate/v0/agreement_matrix.json")


def add_validate_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``bionpu validate`` on a top-level subparsers action."""
    p = subparsers.add_parser(
        "validate",
        help=(
            "Cross-tool validation harness (Track F v0). Compare bionpu "
            "CLIs against canonical reference tools (cutadapt, "
            "Cas-OFFinder, PRIDICT 2.0, ...) and emit an agreement matrix."
        ),
    )
    sub = p.add_subparsers(dest="validate_target")

    # Per-CLI subcommands. The argparse layout mirrors the umbrella
    # CLIs ("crispr design", "crispr pe design", etc.) but every leaf
    # is collapsed to a single command for ergonomics.
    for target_name, helptext in (
        ("trim", "validate bionpu trim vs a reference (cutadapt)"),
        ("crispr-design", "validate bionpu crispr design vs a reference"),
        (
            "crispr-pe-design",
            "validate bionpu crispr pe design vs a reference (PRIDICT 2.0)",
        ),
        ("be-design", "validate bionpu be design vs a reference"),
        ("library-design", "validate bionpu library design vs a reference"),
    ):
        sp = sub.add_parser(target_name, help=helptext)
        sp.add_argument(
            "--reference",
            required=True,
            help=(
                "Reference tool to compare against. Per-target valid "
                "values: trim->cutadapt; crispr-design->{ucsc-pam, "
                "cas-offinder}; crispr-pe-design->pridict2; "
                "be-design->be-hive; library-design->brunello."
            ),
        )
        sp.add_argument(
            "--fixture",
            required=True,
            help="Fixture name, e.g. synthetic-100, synthetic-pam-injection.",
        )
        sp.add_argument(
            "--output",
            default=None,
            help="Optional output JSON path for the single-row agreement check.",
        )
        sp.add_argument(
            "--workspace",
            default=None,
            help="Optional workspace directory (defaults to tempdir).",
        )

    # `bionpu validate all`
    sp_all = sub.add_parser(
        "all",
        help=(
            "Run every cross-check in the v0 plan and emit the full "
            "agreement matrix. Continues past per-row failures."
        ),
    )
    sp_all.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=(
            f"Output JSON path for the agreement matrix. Default: "
            f"{_DEFAULT_OUTPUT}."
        ),
    )
    sp_all.add_argument(
        "--workspace",
        default=None,
        help="Optional workspace directory (defaults to tempdir).",
    )

    p.set_defaults(func=run_cli)


# Map CLI-target argparse name -> bionpu CLI label expected by the
# agreement plan.
_TARGET_TO_CLI = {
    "trim": "trim",
    "crispr-design": "crispr design",
    "crispr-pe-design": "crispr pe design",
    "be-design": "be design",
    "library-design": "library design",
}


def _print_check(check: AgreementCheck) -> None:
    print(
        f"  [{check.verdict.value:7}] {check.bionpu_cli:20s} vs "
        f"{check.reference_tool:14s} ({check.fixture}) "
        f"{('metric=' + str(check.metric) + ' value=' + (f'{check.metric_value:.4f}' if check.metric_value is not None else '-')) if check.metric else ''}",
        file=sys.stderr,
    )
    if check.divergence_summary and check.verdict not in (
        Verdict.PASS,
        Verdict.SKIP,
    ):
        print(f"      {check.divergence_summary}", file=sys.stderr)


def run_cli(args: argparse.Namespace) -> int:
    """Dispatch on the parsed argparse namespace."""
    target = getattr(args, "validate_target", None)
    if target is None:
        # No subcommand — print help via the registered parser.
        # argparse.set_defaults wires func= but if no subcommand was
        # given the dispatch returns 0 with a message.
        print(
            "bionpu validate: pass a target subcommand "
            "(trim, crispr-design, crispr-pe-design, be-design, "
            "library-design, all). See `bionpu validate --help`.",
            file=sys.stderr,
        )
        return 2

    workspace = pathlib.Path(args.workspace) if args.workspace else None

    if target == "all":
        out_path = pathlib.Path(args.output)
        print(
            f"bionpu validate all: running v0 cross-check matrix...",
            file=sys.stderr,
        )
        results = run_full_matrix(workspace=workspace, on_check=_print_check)
        write_matrix_json(results, out_path)
        # Summary
        counts = {v.value: 0 for v in Verdict}
        for c in results:
            counts[c.verdict.value] += 1
        print(
            f"bionpu validate all: wrote {len(results)} rows to {out_path}",
            file=sys.stderr,
        )
        print(
            f"  PASS={counts['PASS']} FAIL={counts['FAIL']} "
            f"DIVERGE={counts['DIVERGE']} SKIP={counts['SKIP']} "
            f"ERROR={counts['ERROR']}",
            file=sys.stderr,
        )
        # Exit code: 0 unless any FAIL or ERROR. SKIP/DIVERGE are not failures.
        if counts["FAIL"] or counts["ERROR"]:
            return 1
        return 0

    cli_label = _TARGET_TO_CLI.get(target)
    if cli_label is None:
        print(
            f"bionpu validate: unknown target {target!r}", file=sys.stderr
        )
        return 2

    try:
        check = run_validation(
            bionpu_cli=cli_label,
            reference=args.reference,
            fixture=args.fixture,
            workspace=workspace,
        )
    except KeyError as exc:
        print(f"bionpu validate: {exc}", file=sys.stderr)
        return 2

    _print_check(check)
    payload = matrix_to_json([check])
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"bionpu validate: wrote single-row report to {out_path}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))

    if check.verdict in (Verdict.FAIL, Verdict.ERROR):
        return 1
    return 0
