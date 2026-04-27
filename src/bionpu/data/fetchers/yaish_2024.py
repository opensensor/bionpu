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

"""Yaish 2024 packaged off-target datasets fetcher.

Dataset
-------
A consolidated, fold-split package of multiple published off-target
datasets — Lazzarotto 2020 (CHANGE-seq + GUIDE-seq), Tsai 2015,
Chen 2017, Listgarten 2018 — re-formatted into a uniform CSV schema
with pre-computed cross-validation folds. Used by Kimata 2025
DNABERT-Epi and several other papers as the consumable form of these
benchmarks.

    Yaish O., Orenstein Y. (2024). "Generating, modeling and
    evaluating a large-scale set of CRISPR/Cas9 off-target sites
    with bulges." *Nucleic Acids Research* 52(11):e62.
    DOI: 10.1093/nar/gkae270.

Pinned artifact
---------------
``files/datasets.zip`` from the OrensteinLab/CRISPR-Bulge GitHub repo
(SHA-256 ``f892f70b...3eab0``, 524,400,344 bytes / ~500 MiB).
Contains the per-dataset CSVs + per-dataset sgRNA lists + per-dataset
fold-split CSVs that the upstream DNABERT-Epi training pipeline
(``third_party/crispr_dnabert/config.yaml``) consumes. The zip is
served via Git LFS; the fetcher follows the LFS redirect.

License
-------
The CRISPR-Bulge repo is MIT-licensed (see
<https://github.com/OrensteinLab/CRISPR-Bulge/blob/main/LICENSE>).
Each constituent dataset retains its original publication's terms;
all are public-research datasets cited in the upstream paper.

Smoke vs full
-------------
The zip is a single artifact (~500 MiB). There is no smaller
canonical mirror; smoke == full. If disk pressure is a concern, the
fetcher writes once to the cache and the caller extracts on demand.
"""

from __future__ import annotations

from bionpu.data.fetchers import DatasetSpec, register

YAISH_2024_URL = (
    "https://github.com/OrensteinLab/CRISPR-Bulge/raw/main/files/datasets.zip"
)

# pin (fetched 2026-04-27; the GitHub LFS pointer redirected to
# media.githubusercontent.com).
YAISH_2024_SHA = (
    "f892f70ba4ac3b05b03b2171b4ad38746630de08ad630e650f355dd61203eab0"
)
YAISH_2024_BYTES = 524_400_344

SPEC = DatasetSpec(
    name="yaish_2024",
    kind="crispr-validation",
    urls=[YAISH_2024_URL],
    sha256=YAISH_2024_SHA,
    size_bytes=YAISH_2024_BYTES,
    license_name="MIT (CRISPR-Bulge upstream)",
    license_url=(
        "https://github.com/OrensteinLab/CRISPR-Bulge/blob/main/LICENSE"
    ),
    citation=(
        "Yaish O., Orenstein Y. (2024). Generating, modeling and "
        "evaluating a large-scale set of CRISPR/Cas9 off-target "
        "sites with bulges. Nucleic Acids Research 52(11):e62. "
        "doi:10.1093/nar/gkae270."
    ),
    smoke_subset_url=YAISH_2024_URL,
    smoke_sha256=YAISH_2024_SHA,
    smoke_size_bytes=YAISH_2024_BYTES,
    relpath="crispr/yaish_2024/datasets.zip",
    notes=(
        "Yaish 2024 packaged off-target datasets (Lazzarotto 2020 "
        "CHANGE-seq + GUIDE-seq, Tsai 2015, Chen 2017, Listgarten 2018). "
        "Consumed by the upstream DNABERT-Epi training pipeline at "
        "third_party/crispr_dnabert (set the path in its config.yaml). "
        "MIT-licensed; constituent datasets retain their original terms."
    ),
)
register(SPEC)
