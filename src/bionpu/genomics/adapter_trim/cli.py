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

"""Argparse wrapper for ``bionpu trim``.

Subcommand reachable both as ``bionpu trim ...`` (via the umbrella
``bionpu`` CLI) and as a standalone ``python -m
bionpu.genomics.adapter_trim`` entry point through :func:`main`.
"""

from __future__ import annotations

import argparse
import sys

from bionpu.kernels.genomics.primer_scan import (
    SUPPORTED_N_TILES,
    SUPPORTED_P,
    TRUSEQ_P5_ADAPTER,
)

from .trimmer import DEFAULT_BATCH_SIZE, trim_fastq, trim_fastq_batched

__all__ = ["build_parser", "main", "run_cli"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bionpu trim",
        description=(
            "Trim 3' adapters from FASTQ reads using the AIE2P-accelerated "
            "primer_scan v0 silicon kernel. Mirrors `cutadapt -a ADAPTER "
            "--no-indels -e 0` semantics."
        ),
    )
    p.add_argument(
        "--adapter",
        "-a",
        default=TRUSEQ_P5_ADAPTER,
        help=(
            f"Adapter sequence to trim. Length must be one of "
            f"{SUPPORTED_P} (silicon-pinned primer lengths). "
            f"Default: {TRUSEQ_P5_ADAPTER} (TruSeq P5)."
        ),
    )
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input FASTQ path. Gzip auto-detected by .gz suffix.",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output FASTQ path. Gzip auto-detected by .gz suffix.",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "npu"],
        default="npu",
        help=(
            "Compute device. cpu = CPU oracle (find_primer_matches); "
            "npu = AIE2P silicon (BionpuPrimerScan via subprocess "
            "host_runner, npu_silicon_lock-wrapped). Default: npu."
        ),
    )
    p.add_argument(
        "--n-tiles",
        type=int,
        choices=list(SUPPORTED_N_TILES),
        default=4,
        help=(
            f"NPU tile fan-out (ignored on --device cpu). Default: 4. "
            f"Allowed: {list(SUPPORTED_N_TILES)}."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"v1 batched-dispatch size: pack N reads into a single NPU "
            f"dispatch (sentinel-separated stream, Path B). N=1 reproduces "
            f"v0 per-record dispatch (slow). Default: {DEFAULT_BATCH_SIZE}. "
            f"Ignored on --device cpu."
        ),
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress periodic per-1000-read progress reports on stderr.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Also suppress the post-run summary on stderr.",
    )
    return p


def _make_progress_callback(quiet: bool):
    if quiet:
        return None
    def _cb(n: int) -> None:
        print(f"  [bionpu trim] processed {n} reads", file=sys.stderr)
    return _cb


def run_cli(args: argparse.Namespace) -> int:
    """Run the trim CLI with parsed args."""
    op = None
    if args.device == "npu":
        from bionpu.kernels.genomics.primer_scan import (
            SUPPORTED_P as _SP,
            BionpuPrimerScan,
        )
        from bionpu.dispatch.npu import NpuArtifactsMissingError

        adapter_len = len(args.adapter)
        if adapter_len not in _SP:
            print(
                f"bionpu trim: adapter length {adapter_len} not in "
                f"supported set {_SP}. Supply an adapter of length 13/20/25, "
                f"or use --device cpu (oracle path supports any length 1-32).",
                file=sys.stderr,
            )
            return 2
        try:
            op = BionpuPrimerScan(
                primer=args.adapter,
                n_tiles=int(args.n_tiles),
            )
        except ValueError as exc:
            print(f"bionpu trim: {exc}", file=sys.stderr)
            return 2
        if not op.artifacts_present():
            print(
                f"bionpu trim --device npu: silicon artifacts missing for "
                f"{op.name} (P={op.p}, n_tiles={op.n_tiles}) at "
                f"{op.artifact_dir}. "
                f"Build via the kernel Makefile or use --device cpu.",
                file=sys.stderr,
            )
            return 3
    else:
        # CPU path: any 1-32 length adapter is OK for the oracle.
        if not 1 <= len(args.adapter) <= 32:
            print(
                f"bionpu trim --device cpu: adapter length "
                f"{len(args.adapter)} out of supported 1..32 range.",
                file=sys.stderr,
            )
            return 2

    progress = _make_progress_callback(args.no_progress or args.quiet)

    batch_size = int(getattr(args, "batch_size", DEFAULT_BATCH_SIZE))
    if batch_size < 1:
        print(
            f"bionpu trim: --batch-size must be >= 1; got {batch_size}",
            file=sys.stderr,
        )
        return 2

    try:
        if op is not None and batch_size > 1:
            stats = trim_fastq_batched(
                args.in_path,
                args.out_path,
                adapter=args.adapter,
                op=op,
                batch_size=batch_size,
                progress=progress,
            )
        else:
            # CPU path or batch_size==1: fall back to the unbatched
            # iterator (v0 byte-equal).
            stats = trim_fastq(
                args.in_path,
                args.out_path,
                adapter=args.adapter,
                op=op,
                progress=progress,
            )
    except FileNotFoundError as exc:
        print(f"bionpu trim: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"bionpu trim: error during trimming: {exc}", file=sys.stderr)
        raise

    if not args.quiet:
        reads_per_s = stats.n_reads / stats.wall_s if stats.wall_s > 0 else 0.0
        print(
            f"bionpu trim [{args.device}]: {stats.n_reads} reads "
            f"({stats.n_trimmed} trimmed, {stats.n_untrimmed} untrimmed) "
            f"in {stats.wall_s:.3f}s = {reads_per_s:,.0f} reads/s",
            file=sys.stderr,
        )
        print(
            f"  bases in:  {stats.total_bases_in:,}",
            file=sys.stderr,
        )
        print(
            f"  bases out: {stats.total_bases_out:,} "
            f"({stats.total_bases_removed:,} removed)",
            file=sys.stderr,
        )
        if stats.n_silicon_dispatches:
            avg_us = stats.silicon_us / stats.n_silicon_dispatches
            print(
                f"  silicon dispatches: {stats.n_silicon_dispatches} "
                f"(avg {avg_us:.1f} us/read)",
                file=sys.stderr,
            )
        if stats.n_cpu_fallback_reads:
            print(
                f"  cpu fallback (non-ACGT reads): "
                f"{stats.n_cpu_fallback_reads}",
                file=sys.stderr,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return run_cli(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
