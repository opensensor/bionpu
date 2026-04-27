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

"""GUIDE-seq off-target dataset fetcher.

Dataset
-------
GUIDE-seq experimental off-target measurements from:

    Tsai SQ, Zheng Z, Nguyen NT, Liebers M, Topkar VV, Thapar V,
    Wyvekens N, Khayter C, Iafrate AJ, Le LP, Aryee MJ, Joung JK.
    (2015). "GUIDE-seq enables genome-wide profiling of off-target
    cleavage by CRISPR-Cas nucleases." *Nature Biotechnology*
    33(2):187-197. doi:10.1038/nbt.3117.

Used by the CRISPR track to validate that the scan + scoring pipeline
finds known off-targets per CRISPR PRD §5.

Pinned artifact
---------------
The pinned artifact is **MOESM22** from the Nature article: an xlsx
workbook containing the supplementary tables - notably the GUIDE-seq
off-target site lists per sgRNA (chromosome, coordinate, strand,
sequence, mismatches, read count). The journal publishes this as a
single sheet bundle.

License
-------
The off-target site lists (sequence + chromosome + read count) are
published as supplementary data alongside the paper under Springer
Nature's standard self-archiving / academic-reuse terms with citation.
We bundle ONLY the machine-readable site lists, not the figures or
text.

Smoke vs full
-------------
The published GUIDE-seq supplementary tables are small (~55 KB - the
data is dense per-row but few sites). Smoke and full are identical;
the smoke flag is preserved for API consistency.

Mirror policy
-------------
Primary: Nature article supplementary (MOESM22). The Springer Nature
``static-content.springer.com`` CDN is the authoritative mirror.

Fallback mirrors (documented for reproducibility writeup; not
auto-tried because the SHA is the contract):

* The Joung lab at MGH historically maintained a project page for
  GUIDE-seq at ``http://www.jounglab.org/``; redistribution of the
  xlsx is not guaranteed across lab-page redesigns. The bundled
  ``guideseq`` Python package on PyPI redistributes the *processed*
  per-cell-line site BED files (different surface; not SHA-comparable
  to the supplementary xlsx).
* PMC archives the article at
  ``https://pmc.ncbi.nlm.nih.gov/articles/PMC4320685/`` but only
  redistributes the PDF supplements, not the xlsx tables.
"""

from __future__ import annotations

from bionpu.data.fetchers import DatasetSpec, register

# Primary URL: Nature Biotechnology article supplementary materials,
# Supplementary Data 22 (MOESM22) - the supplementary tables xlsx.
GUIDE_SEQ_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fnbt.3117/MediaObjects/"
    "41587_2015_BFnbt3117_MOESM22_ESM.xlsx"
)

# pin (fetched 2026-04-25 from the Springer Nature CDN).
GUIDE_SEQ_SHA = (
    "5610680a9920c1d95d48237da670e453e8beb9c718691dda79abbc2bbd5274fb"
)
GUIDE_SEQ_BYTES = 55_294

SPEC = DatasetSpec(
    name="guide_seq",
    kind="crispr-offtarget",
    urls=[GUIDE_SEQ_URL],
    sha256=GUIDE_SEQ_SHA,
    size_bytes=GUIDE_SEQ_BYTES,
    license_name="Nature Biotechnology supplementary (reuse with citation)",
    license_url="https://www.nature.com/articles/nbt.3117",
    citation=(
        "Tsai SQ, Zheng Z, Nguyen NT, et al. (2015). GUIDE-seq enables "
        "genome-wide profiling of off-target cleavage by CRISPR-Cas "
        "nucleases. Nature Biotechnology 33(2):187-197. "
        "doi:10.1038/nbt.3117."
    ),
    # Small enough that smoke == full.
    smoke_subset_url=GUIDE_SEQ_URL,
    smoke_sha256=GUIDE_SEQ_SHA,
    smoke_size_bytes=GUIDE_SEQ_BYTES,
    relpath="crispr/guide_seq/nbt3117_supp_tables.xlsx",
    notes=(
        "GUIDE-seq (Tsai 2015) off-target site lists. Full Nature "
        "Biotech. MOESM22 supplementary-tables xlsx. Used to validate "
        "the CRISPR scan + scoring pipeline. Smoke == full."
    ),
)
register(SPEC)
