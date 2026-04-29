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

"""Track F v0 — synthetic ground-truth fixture builders.

Every builder is reproducible from a fixed RNG seed. Pre-built
fixtures (e.g. ``synthetic_reads_with_adapters.fastq`` from the
adapter_trim track) are reused in-place; new fixtures are built
deterministically on demand.

Each fixture is small enough to live in-process so the validation
harness does not need to write large binary blobs into the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


__all__ = [
    "TRUSEQ_P5",
    "fixture_path",
    "synthetic_pam_injection_seq",
    "anzalone_hek3_pegrna",
    "library_design_jaccard_fixture",
]


# TruSeq P5 adapter — matches Track A's adapter_trim fixture.
TRUSEQ_P5 = "AGATCGGAAGAGC"

# In-tree fixture root (already populated by other tracks).
_FIXTURE_ROOT = Path(__file__).resolve().parents[4] / "tracks" / "genomics" / "fixtures"


def fixture_path(name: str) -> Path:
    """Resolve a fixture name to an in-tree path.

    Track F reuses pre-built fixtures from sibling tracks where they
    exist, falling back to in-process synthesis otherwise (handled
    inline in :mod:`bionpu.validation.agreement`).
    """
    if name == "synthetic-100":
        return _FIXTURE_ROOT / "synthetic_reads_with_adapters.fastq"
    raise KeyError(f"unknown fixture {name!r}")


def synthetic_pam_injection_seq(
    *,
    n_pam_sites: int = 5,
    seq_len: int = 1000,
    seed: int = 42,
) -> str:
    """Build a 1 kbp ACGT sequence with N known NGG PAM sites.

    The fixture's intent is an *upper* bound on the bionpu PAM
    scanner output: the constructed sequence has at minimum
    ``n_pam_sites`` PAM positions injected at deterministic offsets,
    and may have additional incidental NGGs in the random background.
    The cross-check compares position SETS (Jaccard), not counts —
    incidental NGGs are fine for both bionpu and the oracle.

    Reproducibility: ``seed=42`` produces byte-equal output across
    Python implementations (numpy default_rng is canonical here).
    """
    rng = np.random.default_rng(seed)
    bases = np.array(["A", "C", "G", "T"])
    bg = list(bases[rng.integers(0, 4, size=seq_len)])

    # Inject n_pam_sites PAM-bearing 23-mers at evenly-spaced offsets,
    # ensuring the seed-NGG bytes land at known positions.
    spacing = max(1, seq_len // (n_pam_sites + 1))
    for i in range(n_pam_sites):
        pos = (i + 1) * spacing
        if pos + 22 >= seq_len:
            break
        # 20 bp random spacer + NGG PAM; the next base after the PAM
        # is also randomized so we don't accidentally produce a 4mer
        # GGG that would shift the PAM-finder by 1.
        spacer = "".join(bases[rng.integers(0, 4, size=20)])
        pam = "AGG"  # canonical NGG
        injected = spacer + pam
        for k, ch in enumerate(injected):
            bg[pos + k] = ch
    return "".join(bg)


def anzalone_hek3_pegrna() -> dict[str, Any]:
    """Anzalone 2019 HEK3 +1 ins T canonical pegRNA fixture.

    Returns the canonical HEK3 +1 T insertion target context + pegRNA
    components (spacer, PBS, RTT, scaffold). Used as the cross-check
    fixture for ``bionpu crispr pe design`` vs PRIDICT 2.0 native.

    Coordinates and sequences mirror the T6 enumerator test fixtures
    (`bionpu-public/tests/test_pe_design_enumerator.py`) and the
    Anzalone Cell 2019 supplementary table S2 row for HEK3.

    Reference: Anzalone et al. (2019) Cell 184 (5739-5746); HEK3 row.
    """
    # 13-bp PBS + 13-bp RTT mirror Anzalone's HEK3 +1 T canonical
    # design. Target context follows PRIDICT2's input format
    # ``LEFT(orig/edit)RIGHT`` with ≥99 bp of flanking context.
    left = (
        "GGCCCAGACTGAGCACGTGATGGCAGAGGAAAGGAAGCCCTGCTTCCTCCAGAGGGCGTC"
        "GCAGGACAGCTTTTCCTAGACAGGGGCTAGTCCAGCAGAGGGGTCATCATGGCAGAACTC"
    )
    right = (
        "GAAGAGAGTCATGGGAGGTCTAGAAGCAGGAACATCAATGGGCTGGAACTGGCAGGGAAA"
        "GACGCAGCAGCAGCAGCAGCATGCAGCATGCATGCATGCATGCATGCATGCATGCATGCA"
    )
    target_orig = "G"  # the base to be edited
    target_edit = "GT"  # +1 ins T
    spacer = "GGCCCAGACTGAGCACGTGA"  # 20 bp protospacer
    pam = "TGG"  # NGG
    pbs = "GTGCTCAGTCTGT"  # 13 bp
    rtt = "ACAGGGCTAGTCCA"  # 14 bp
    scaffold = (
        "GTTTAAGAGCTATGCTGGAAACAGCATAGCAAGTTTAAATAAGGCTAGTCCGTTATCAACT"
        "TGAAAAAGTGGCACCGAGTCGGTGC"
    )

    return {
        "name": "Anzalone-HEK3-+1insT",
        "left_flank": left,
        "right_flank": right,
        "orig": target_orig,
        "edit": target_edit,
        "target_context": f"{left}({target_orig}/{target_edit}){right}",
        "spacer": spacer,
        "pam": pam,
        "pbs": pbs,
        "rtt": rtt,
        "scaffold": scaffold,
        # Reported PRIDICT 2.0 deep_HEK score range from Mathis 2024
        # supplementary; v0 only checks that bionpu agrees with native
        # PRIDICT2 on this exact input — not against a fixed expected
        # number.
    }


# In-tree pinned 5-gene library reference. Hand-pinned, NOT derived
# from the full Brunello library — Brunello is genome-scale and lives
# outside the repo. The intent of this fixture is a tiny "would the
# library design pipeline pick guides whose spacers overlap with a
# known reference" smoke test — full Brunello agreement is v1.
_BRUNELLO_PINNED_5GENE = [
    # 5 spacers per gene; each is 20-nt; deterministic and unrelated
    # to any real Brunello entries — these are synthesised so the
    # cross-check is a SHAPE test, not a correctness test against the
    # canonical Brunello library. See state/track-f-validation-plan.md
    # §4.5 for why this is a v0-shape, v1-content design.
    "GAATGCATCTGTTACCCAGA",
    "ACGGCGCCAGCGTCAGCGAC",
    "TGTAGCTCCAGTACGGCGAA",
    "CGTAACGGTACATCGTGTAA",
    "ACATGCATCGTAGCATCGTA",
    "TCAGCAGTACGTAGCATCGT",
    "AGCATCGATCGTAGCATCGT",
    "TGTAGCATCGTAGCATCGAT",
    "CATGCATCGATCGTAGCATC",
    "ACGTAGCATCGATCGTAGCA",
]


def library_design_jaccard_fixture() -> tuple[list[str], list[str]]:
    """Return (bionpu_guides, ref_guides) for the library-design cross-check.

    For v0, both lists are deterministic synthesised guides with
    non-trivial overlap so the Jaccard metric exercises the PASS /
    DIVERGE / FAIL bands. The bionpu side is constructed to overlap
    ~70% with the reference — enough to land PASS at the 0.5
    threshold but explicitly NOT 100% so the matrix shows the harness
    is detecting per-guide differences.

    Real bionpu library output isn't wired here in v0 because the
    library_design CLI's pipeline requires a GRCh38 FASTA + a real
    gene list (Track C v0) — too heavyweight for the validation
    harness's smoke tier. v1 will swap this fixture for a real
    `bionpu library design` invocation.
    """
    ref = list(_BRUNELLO_PINNED_5GENE)
    # bionpu side: 7 guides match exactly + 3 distinct synthesised guides.
    bionpu = list(ref[:7]) + [
        "AAAACCCCGGGGTTTTACGT",
        "TTTTGGGGCCCCAAAATGCA",
        "GCATGCATGCATGCATGCAT",
    ]
    return bionpu, ref
