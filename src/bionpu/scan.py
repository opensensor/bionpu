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

"""Pure-CPU CRISPR off-target scan.

A self-contained numpy implementation of CRISPR off-target search
that doesn't require NPU silicon. Drives :func:`bionpu.cli` 's
``bionpu scan`` subcommand for the v0.1 release.

The output is byte-equal to a Cas-OFFinder run with the same input,
modulo Cas-OFFinder's row-order non-determinism, which is
canonicalised by :mod:`bionpu.data.canonical_sites`. This module's
output passes through ``bionpu.data.canonical_sites.normalize`` before
being written so the byte-equality contract is honoured by
construction.

Limitations
-----------

- Only the NGG PAM template is implemented. IUPAC ambiguity codes
  are future scope.
- No DNA / RNA bulges. Only mismatches.
- Single-threaded numpy. For full-genome scans (3 Gbp), use the NPU
  path (v0.2 scope) or the C++ Cas-OFFinder reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data.canonical_sites import CasOFFinderRow

__all__ = ["GuideSpec", "cpu_scan", "parse_guides", "read_fasta"]


# Two-bit packing: A=0, C=1, G=2, T=3
_BASE_TO_CODE = np.full(256, 4, dtype=np.uint8)  # 4 = "non-ACGT" sentinel
for c, code in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _BASE_TO_CODE[ord(c)] = code
    _BASE_TO_CODE[ord(c.lower())] = code

# Reverse-complement table (same 0..3 codes)
_RC_CODE = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


@dataclass(frozen=True)
class GuideSpec:
    """One guide in the input list — 20 nt ACGT spacer plus an optional ID."""

    spacer: str    # 20-nt ACGT
    guide_id: str  # caller-supplied or auto-generated


def parse_guides(arg: str) -> list[GuideSpec]:
    """Parse the ``--guides`` argument: comma-separated string OR file path.

    File format: one spacer per line, optionally ``id:spacer``. Lines
    starting with ``#`` are comments. All spacers must be 20 nt of ACGT.
    """
    text: str
    if Path(arg).is_file():
        text = Path(arg).read_text()
    else:
        text = arg.replace(",", "\n")
    out: list[GuideSpec] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            gid, spacer = line.split(":", 1)
            gid = gid.strip()
            spacer = spacer.strip()
        else:
            spacer = line
            gid = ""
        spacer = spacer.upper()
        if len(spacer) != 20 or any(c not in "ACGT" for c in spacer):
            raise ValueError(
                f"guide spacer must be 20 nt of ACGT; got {spacer!r}"
            )
        if not gid:
            gid = spacer  # fall back to the spacer itself as ID
        out.append(GuideSpec(spacer=spacer, guide_id=gid))
    if not out:
        raise ValueError(f"no valid guide spacers found in {arg!r}")
    return out


def read_fasta(path: str | Path) -> tuple[str, str]:
    """Read a single-record FASTA. Returns ``(chrom_name, sequence)``.

    The sequence is upper-cased ACGT; non-ACGT bases (N, IUPAC) are
    preserved as-is so downstream non-ACGT detection works.
    """
    p = Path(path)
    chrom = ""
    seq_parts: list[str] = []
    for line in p.read_text().splitlines():
        if line.startswith(">"):
            if chrom:
                # Multi-record FASTA — only first record is consumed.
                break
            chrom = line[1:].split()[0]
        else:
            seq_parts.append(line.strip())
    return chrom, "".join(seq_parts).upper()


def _encode_seq(seq: str) -> np.ndarray:
    """Encode a DNA string as a uint8 array of base codes."""
    return _BASE_TO_CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def _decode_window(arr: np.ndarray) -> str:
    """Inverse of _encode_seq for a single window (codes back to ACGT)."""
    table = np.array([ord("A"), ord("C"), ord("G"), ord("T"), ord("N")],
                     dtype=np.uint8)
    return table[arr].tobytes().decode("ascii")


def _rc(arr: np.ndarray) -> np.ndarray:
    """Reverse-complement a coded array."""
    return _RC_CODE[arr][::-1]


def cpu_scan(
    *,
    chrom: str,
    seq: str,
    guides: Sequence[GuideSpec],
    pam_template: str = "NGG",
    max_mismatches: int = 4,
) -> list[CasOFFinderRow]:
    """Pure-CPU CRISPR off-target scan.

    Args:
        chrom: Chromosome / contig name (recorded verbatim in output rows).
        seq: ACGT[N] sequence to scan.
        guides: Guide list to search for.
        pam_template: PAM motif at the 3' end (only ``"NGG"`` supported).
        max_mismatches: Maximum allowed mismatches in the 20-nt spacer
            region (PAM mismatches always count as 0 in the output's
            ``mismatches`` field, matching Cas-OFFinder's convention).

    Returns:
        List of :class:`bionpu.data.canonical_sites.CasOFFinderRow` —
        one per match. Caller should pass through
        :func:`bionpu.data.canonical_sites.normalize` for byte-equal
        output.
    """
    if pam_template != "NGG":
        raise NotImplementedError(
            f"only NGG PAM is supported; got {pam_template!r}. IUPAC "
            f"ambiguity codes are future scope."
        )

    coded = _encode_seq(seq)
    n = coded.size
    if n < 23:
        return []

    rows: list[CasOFFinderRow] = []
    for g in guides:
        spacer_codes = _encode_seq(g.spacer)
        rc_spacer_codes = _rc(spacer_codes)

        # Forward strand: window at [s, s+23). Spacer = [s, s+20). PAM = [s+20, s+23).
        # PAM is NGG → seq[s+21] == G AND seq[s+22] == G.
        # Mismatches: count spacer_codes != coded[s:s+20].
        positions = np.arange(0, n - 23 + 1)
        windows = np.lib.stride_tricks.sliding_window_view(coded, 23)
        # PAM check: positions where seq[s+21] == G (code 2) and seq[s+22] == G.
        pam_ok_fwd = (windows[:, 21] == 2) & (windows[:, 22] == 2)
        if pam_ok_fwd.any():
            spacer_diffs = windows[pam_ok_fwd, :20] != spacer_codes
            mismatch_counts = spacer_diffs.sum(axis=1)
            keep = mismatch_counts <= max_mismatches
            kept_positions = positions[pam_ok_fwd][keep]
            kept_mismatches = mismatch_counts[keep]
            kept_windows = windows[pam_ok_fwd][keep]
            for s, mm, w in zip(kept_positions, kept_mismatches, kept_windows):
                if (w[:20] == 4).any() or (w[20:23] == 4).any():
                    continue  # non-ACGT in window — Cas-OFFinder skips
                rows.append(
                    CasOFFinderRow(
                        guide_id=g.guide_id,
                        bulge_type="X",
                        crrna=g.spacer + "NGG",
                        dna=_decode_window(w[:23]),
                        chrom=chrom,
                        start=int(s),
                        strand="+",
                        mismatches=int(mm),
                        bulge_size=0,
                    )
                )

        # Reverse strand: same windows, but compare RC(spacer) to seq[s+3:s+23]
        # and require PAM CCN at seq[s:s+3] (CC then anything, since reverse
        # strand's NGG is forward strand's CCN).
        # PAM check: seq[s] == C (code 1) AND seq[s+1] == C.
        pam_ok_rev = (windows[:, 0] == 1) & (windows[:, 1] == 1)
        if pam_ok_rev.any():
            spacer_diffs = windows[pam_ok_rev, 3:23] != rc_spacer_codes
            mismatch_counts = spacer_diffs.sum(axis=1)
            keep = mismatch_counts <= max_mismatches
            kept_positions = positions[pam_ok_rev][keep]
            kept_mismatches = mismatch_counts[keep]
            kept_windows = windows[pam_ok_rev][keep]
            for s, mm, w in zip(kept_positions, kept_mismatches, kept_windows):
                if (w[:23] == 4).any():
                    continue
                # DNA reported is the forward-strand 23-mer (matches
                # Cas-OFFinder convention).
                rows.append(
                    CasOFFinderRow(
                        guide_id=g.guide_id,
                        bulge_type="X",
                        crrna=g.spacer + "NGG",
                        dna=_decode_window(w[:23]),
                        chrom=chrom,
                        start=int(s),
                        strand="-",
                        mismatches=int(mm),
                        bulge_size=0,
                    )
                )

    return rows
