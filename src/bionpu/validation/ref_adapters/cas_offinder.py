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

"""Cas-OFFinder adapter for Track F v0.

Wraps the locally-built Cas-OFFinder 3.0.0 binary at
``tracks/crispr/reference/cas-offinder-src/build/cas-offinder``.
Per the v0 probe (state/track-f-reference-tool-probe.md), the
binary is OpenCL-backed (CPU + NVIDIA RTX 4070).

For the v0 cross-check we run a zero-mismatch scan: every NGG site
in the target should yield a hit, so the output is equivalent to a
PAM-position list. v1 will broaden to non-zero mismatches with
real off-target panels.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


__all__ = [
    "cas_offinder_installed",
    "cas_offinder_path",
    "run_cas_offinder_pam_scan",
]


# Pinned local build path. The probe doc (state/track-f-reference-
# tool-probe.md §2) records this as the canonical Track F binary.
_PINNED_PATH = Path(
    "/home/matteius/genetics/tracks/crispr/reference/cas-offinder-src/build/cas-offinder"
)


def cas_offinder_path() -> str | None:
    """Resolve the cas-offinder binary path.

    Returns the pinned local path if executable, otherwise None.
    """
    if _PINNED_PATH.is_file() and os.access(_PINNED_PATH, os.X_OK):
        return str(_PINNED_PATH)
    return None


def cas_offinder_installed() -> bool:
    return cas_offinder_path() is not None


def _write_pseudo_genome(workspace: Path, target_seq: str) -> Path:
    """Cas-OFFinder ingests a directory of FASTA files. Build a single-record one."""
    genome_dir = workspace / "pseudo_genome"
    genome_dir.mkdir(parents=True, exist_ok=True)
    fa = genome_dir / "synthetic.fa"
    # Cas-OFFinder reads the first line as the chromosome name and
    # the rest as the sequence. Wrap the sequence in 80-col chunks so
    # the FASTA is canonical.
    lines = ["> synthetic"]
    for i in range(0, len(target_seq), 80):
        lines.append(target_seq[i : i + 80])
    fa.write_text("\n".join(lines) + "\n")
    return genome_dir


def _write_input(
    workspace: Path,
    *,
    genome_dir: Path,
    pam_template: str = "NNNNNNNNNNNNNNNNNNNNNGG",
    query_pattern: str = "NNNNNNNNNNNNNNNNNNNN",
    mismatches: int = 0,
) -> Path:
    """Build the Cas-OFFinder per-run input file.

    Format:
        <line 1>: path to genome file or directory
        <line 2>: PAM template (e.g. NNNNNNNNNNNNNNNNNNNNNGG)
        <line N>: query  N_mismatch  (one per line)
    """
    inp = workspace / "cas_offinder.in"
    body = "\n".join(
        [
            str(genome_dir),
            pam_template,
            f"{query_pattern}NNN {mismatches}",
        ]
    )
    inp.write_text(body + "\n")
    return inp


def run_cas_offinder_pam_scan(
    *,
    target_seq: str,
    workspace: Path,
    timeout_s: float = 120.0,
    spacer_len: int = 20,
) -> list[int]:
    """Run Cas-OFFinder against ``target_seq`` and return forward-strand hit positions.

    The query is a 20-N spacer followed by NGG PAM with zero
    mismatches — equivalent to "find every NGG site". Returns the
    list of 0-based positions on the FORWARD strand using the
    **PAM-start** convention (position of the N in NGG), which
    matches the UCSC oracle and the bionpu PAM scanner.

    Cas-OFFinder reports hit positions as the start of the matched
    23-mer (spacer + PAM). To align with the PAM-start convention
    used elsewhere in Track F we shift each Cas-OFFinder position by
    ``+spacer_len``.
    """
    binary = cas_offinder_path()
    if binary is None:
        raise RuntimeError(f"cas-offinder binary not found at {_PINNED_PATH}")

    workspace = Path(workspace)
    genome_dir = _write_pseudo_genome(workspace, target_seq)
    inp = _write_input(workspace, genome_dir=genome_dir)
    out = workspace / "cas_offinder.out"

    # Use CPU device "C" — does not require CUDA being available
    # post-fork. Cas-OFFinder treats stdout for `-` output paths.
    argv = [binary, str(inp), "C", str(out)]
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cas-offinder exited {proc.returncode}: stderr={proc.stderr.strip()[:400]}"
        )

    # Output format (Cas-OFFinder 3.0.0):
    #   query \t chrom \t pos \t strand \t matched_seq \t mismatches
    positions: list[int] = []
    if not out.exists():
        return positions
    for raw in out.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        # Prefer numeric `position` column; cas-offinder may include
        # bulge columns. Find the first integer-castable column past col 1.
        pos: int | None = None
        strand: str | None = None
        for col in cols[1:]:
            try:
                pos = int(col)
                break
            except ValueError:
                continue
        # Strand column is typically "+" or "-".
        for col in cols:
            if col in ("+", "-"):
                strand = col
                break
        if pos is None:
            continue
        if strand is not None and strand != "+":
            # v0 forward-strand cross-check; skip reverse-strand hits.
            continue
        # Convert cas-offinder's 23-mer-start coord to PAM-start
        # (matches the bionpu / UCSC oracle convention).
        positions.append(pos + spacer_len)
    return sorted(set(positions))
