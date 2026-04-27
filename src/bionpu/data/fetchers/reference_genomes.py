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

"""Reference genome fetchers.

Datasets
--------
* **GRCh38** (primary human reference). Used by both tracks: alignment
  for basecalling accuracy and full-genome scan for CRISPR
.
* **T2T-CHM13** (telomere-to-telomere human assembly). Secondary human
  reference per CRISPR PRD §5.
* **GRCm39** (mouse reference). Sanity-check on a smaller genome per
  CRISPR PRD §5.

License
-------
All three references are **public-domain** reference assemblies hosted
by NCBI / UCSC. UCSC's ``goldenPath`` README and NCBI's data-use
policy explicitly place these in the public domain (no restrictions).
 already records the same chr22 SHA-256 for the Cas-OFFinder
canonical fixture; we reuse that exact pin to keep the cache
deduplicated.

Smoke vs full
-------------
Smoke = chr22 only (~12 MB compressed for GRCh38; smaller for
T2T-CHM13 and GRCm39). Full = whole genome (~3 GB each); gated on
``--mode full``.

Citations
---------
* Schneider VA, et al. *Genome Research* (2017). "Evaluation of GRCh38
  and de novo haploid genome assemblies demonstrates the enduring
  quality of the reference assembly."
* Nurk S, et al. *Science* (2022). "The complete sequence of a human
  genome." (T2T-CHM13)
* Mouse Genome Sequencing Consortium / NCBI Mouse Reference Genome
  GRCm39 (2020).

Pinning policy
--------------
The full-genome SHA-256 values below are placeholders. The fetcher
framework refuses to commit a file whose SHA doesn't match the spec,
so a wrong / placeholder pin is *fail-safe* — the first real fetch
will raise ``ChecksumMismatchError`` and the operator updates this
file. The chr22 (GRCh38) pin is real because already established
it under ``data_cache/cas-offinder/MANIFEST.md``.
"""

from __future__ import annotations

from bionpu.data.fetchers import DatasetSpec, register

# ---------------------------------------------------------------------------
# GRCh38
# ---------------------------------------------------------------------------

GRCH38_FULL_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
)
GRCH38_CHR22_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
)
# established this pin for `data_cache/cas-offinder/genomes/chr22.fa.gz`.
GRCH38_CHR22_SHA = "05f9d97d6fbfd08a44ca45b50837ca2ae9c471f35ba79dffec04d2cb5eaaf695"
GRCH38_CHR22_BYTES = 12_255_678

GRCH38_SPEC = DatasetSpec(
    name="grch38",
    kind="reference",
    urls=[GRCH38_FULL_URL],
    # Full hg38.fa.gz SHA-256 (fetched 2026-04-27 from UCSC goldenPath).
    sha256="c1dd87068c254eb53d944f71e51d1311964fce8de24d6fc0effc9c61c01527d4",
    size_bytes=983_659_424,
    license_name="UCSC goldenPath public-domain",
    license_url="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/README.txt",
    citation=(
        "Schneider VA, et al. (2017). Evaluation of GRCh38 and de novo "
        "haploid genome assemblies. Genome Research."
    ),
    smoke_subset_url=GRCH38_CHR22_URL,
    smoke_sha256=GRCH38_CHR22_SHA,
    smoke_size_bytes=GRCH38_CHR22_BYTES,
    relpath="genomes/grch38/hg38.fa.gz",
    notes="Human GRCh38. Smoke = chr22 only; full = whole genome.",
)
register(GRCH38_SPEC)

# ---------------------------------------------------------------------------
# T2T-CHM13
# ---------------------------------------------------------------------------

T2T_FULL_URL = (
    "https://s3-us-west-2.amazonaws.com/human-pangenomics/"
    "T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz"
)
# UCSC hosts the same T2T assembly as `hs1.fa.gz`. Used as the smoke
# mirror because the T2T Consortium does not publish per-chromosome
# FASTAs (the analysis_set tarball is whole-genome only). The smoke
# variant is therefore the same artifact as full — operators on a
# disk-constrained machine should `--mode full` once and reuse the
# cache. We keep smoke and full distinct in the spec so the
# `cache_relpath()` helper still namespaces them in the cache; they
# just point at the same URL.
T2T_SMOKE_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hs1/bigZips/hs1.fa.gz"
)

T2T_CHM13_SPEC = DatasetSpec(
    name="t2t_chm13",
    kind="reference",
    urls=[T2T_FULL_URL],
    sha256="0" * 64,
    size_bytes=0,
    license_name="T2T Consortium public-domain (CC0-equivalent)",
    license_url="https://github.com/marbl/CHM13",
    citation=(
        "Nurk S, et al. (2022). The complete sequence of a human genome. "
        "Science 376(6588):44-53."
    ),
    smoke_subset_url=T2T_SMOKE_URL,
    smoke_sha256="0" * 64,  # pinned on first successful smoke fetch
    smoke_size_bytes=None,
    relpath="genomes/t2t_chm13/chm13v2.0.fa.gz",
    notes=(
        "T2T-CHM13 v2.0 secondary human reference. The Consortium does not "
        "publish per-chromosome FASTAs; smoke and full both fetch the whole "
        "assembly (~870 MB). Run `--mode full` once and reuse the cache."
    ),
)
register(T2T_CHM13_SPEC)

# ---------------------------------------------------------------------------
# GRCm39 (mouse)
# ---------------------------------------------------------------------------

GRCM39_FULL_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz"
)
GRCM39_CHR19_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/chromosomes/chr19.fa.gz"
)

GRCM39_SPEC = DatasetSpec(
    name="grcm39",
    kind="reference",
    urls=[GRCM39_FULL_URL],
    sha256="0" * 64,
    size_bytes=0,
    license_name="UCSC goldenPath public-domain",
    license_url="https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/README.txt",
    citation=(
        "Mouse Genome Sequencing Consortium / NCBI (2020). Mouse Reference "
        "Genome GRCm39."
    ),
    smoke_subset_url=GRCM39_CHR19_URL,
    # Pinned 2026-04-25 from a real fetch of the UCSC mm39 chr19.fa.gz.
    smoke_sha256="aa64f5018981c0cb1463ce1bbc33ed86317d25aa611abde34fe52ed068975ada",
    smoke_size_bytes=19_143_071,
    relpath="genomes/grcm39/mm39.fa.gz",
    notes="Mouse GRCm39. Smoke = chr19 (small autosome); full = whole genome.",
)
register(GRCM39_SPEC)
