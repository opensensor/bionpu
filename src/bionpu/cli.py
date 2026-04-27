"""bionpu command-line interface.

Subcommands:
- ``bionpu scan``     CRISPR off-target scan with optional --verify
- ``bionpu basecall`` Nanopore basecalling with optional --verify
- ``bionpu verify``   Standalone byte-equality check on existing files
- ``bionpu bench``    Energy + timing measurement on either workload

Status: shell. Real subcommands land during the v0.1 extraction.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="bionpu",
        description="AIE2P-accelerated genomics with reference-equivalence verification.",
    )
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("scan", help="CRISPR off-target scan (not yet implemented)")
    sub.add_parser("basecall", help="Nanopore basecalling (not yet implemented)")
    sub.add_parser("verify", help="Byte-equality check (not yet implemented)")
    sub.add_parser("bench", help="Energy + timing measurement (not yet implemented)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.cmd is None:
        p.print_help()
        return 0

    print(
        f"bionpu {args.cmd}: not yet implemented in this v0.1 shell. "
        f"Implementation lands during the extraction from the genetics "
        f"working tree.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
