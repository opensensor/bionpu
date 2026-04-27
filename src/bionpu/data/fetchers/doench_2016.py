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

"""Doench 2016 guide-activity dataset fetcher.

Dataset
-------
On-target activity measurements for sgRNAs from:

    Doench JG, Fusi N, Sullender M, Hegde M, Vaimberg EW, Donovan KF,
    Smith I, Tothova Z, Wilen C, Orchard R, Virgin HW, Listgarten J,
    Root DE. (2016). "Optimized sgRNA design to maximize activity and
    minimize off-target effects of CRISPR-Cas9." *Nature Biotechnology*
    34(2):184-191. doi:10.1038/nbt.3437.

Used by the CRISPR track for guide-activity scoring validation
.

Pinned artifact
---------------
The pinned artifact is **MOESM8** from the Nature article: a zip
containing 22 ``SuppTables/STable XX *.xlsx`` files. This is the full
published supplementary-data bundle, including:

* STable 17 (Features) - the on-target feature table used by Rule Set 2
  for activity scoring;
* STable 18 (CD33_OffTargetdata) - the off-target measurement set used
  by CFD score derivation;
* STable 21 (Brunello) and STable 22 (Brie) - the human and mouse
  optimized whole-genome libraries;
* and 19 other tables of guide-activity / negative-selection /
  individual-screen data.

Pinning the *whole* zip rather than a single sheet is intentional:
(1) the journal publishes it as a single artifact so downstream
reproducers always cite one URL/SHA, (2) every downstream sgRNA scorer
(Azimuth, DeepCRISPR, CRISPRoff) reaches for one or more of these
tables, and (3) atomic SHA verification on the zip catches partial /
in-flight corruption that would silently slip through if we picked a
single sheet.

License
-------
The supplementary tables (the *data* itself - guide sequences and
measured activity scores) are released alongside the paper under
Springer Nature's standard self-archiving / academic-reuse terms with
citation. We bundle ONLY the data points (sequence + score), not the
figures or text. This is consistent with how every downstream sgRNA
scoring tool (Azimuth, DeepCRISPR, CRISPRoff) reuses these
measurements.

Smoke vs full
-------------
The full supplementary zip is ~128 MB (the file the journal
publishes). For the in-CI smoke variant we reuse the same artifact -
it's the smallest published unit and there is no per-sheet mirror that
would survive checksum pinning. Operators with disk-tight CI should
``--mode full`` once and reuse the cache; smoke and full are the same
URL.

Mirror policy
-------------
Primary: Nature article supplementary (MOESM8). The Springer Nature
``static-content.springer.com`` CDN is the authoritative mirror.

Fallback mirrors (documented for reproducibility writeup; not
auto-tried because the SHA is the contract):

* Broad Institute GPP portal redistributes the *individual* tables in
  TSV form at ``https://portals.broadinstitute.org/gpp/public/`` and
  in the public ``broadinstitute/sgRNA_design`` GitHub mirror; SHA
  will not match the zip pin so use only as a manual reproducibility
  fallback.
* PMC archives the article at
  ``https://pmc.ncbi.nlm.nih.gov/articles/PMC4744125/`` but only
  redistributes the PDF supplements, not the xlsx tables.
"""

from __future__ import annotations

from bionpu.data.fetchers import DatasetSpec, register

# Primary URL: Nature Biotechnology article supplementary materials,
# Supplementary Data 8 (MOESM8) - the supplementary tables zip.
DOENCH_2016_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fnbt.3437/MediaObjects/"
    "41587_2016_BFnbt3437_MOESM8_ESM.zip"
)

# pin (fetched 2026-04-25 from the Springer Nature CDN).
DOENCH_2016_SHA = (
    "50657eb70012fb9b408600621fc03b28c7bac1480060b1f1c4391b86ab4c1488"
)
DOENCH_2016_BYTES = 128_255_968

SPEC = DatasetSpec(
    name="doench_2016",
    kind="crispr-activity",
    urls=[DOENCH_2016_URL],
    sha256=DOENCH_2016_SHA,
    size_bytes=DOENCH_2016_BYTES,
    license_name="Nature Biotechnology supplementary (reuse with citation)",
    license_url="https://www.nature.com/articles/nbt.3437",
    citation=(
        "Doench JG, Fusi N, Sullender M, et al. (2016). Optimized sgRNA "
        "design to maximize activity and minimize off-target effects of "
        "CRISPR-Cas9. Nature Biotechnology 34(2):184-191. "
        "doi:10.1038/nbt.3437."
    ),
    # Smoke == full: the journal publishes a single ~128 MB zip and there
    # is no smaller mirror that would survive checksum pinning. Operators
    # on a disk-tight CI should `--mode full` once and reuse the cache.
    smoke_subset_url=DOENCH_2016_URL,
    smoke_sha256=DOENCH_2016_SHA,
    smoke_size_bytes=DOENCH_2016_BYTES,
    relpath="crispr/doench_2016/nbt3437_supp_tables.zip",
    notes=(
        "Doench 2016 sgRNA on-target activity. Full Nature Biotech. "
        "MOESM8 supplementary-tables zip (22 xlsx files including "
        "STable 17 Features, STable 18 CD33_OffTargetdata, "
        "STable 21 Brunello). Used by guide-activity scoring. "
        "Smoke == full (no smaller published mirror)."
    ),
)
register(SPEC)
