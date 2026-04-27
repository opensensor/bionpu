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

"""HG002 nanopore POD5 fetcher.

Dataset
-------
HG002 (Genome in a Bottle, NA24385) Oxford Nanopore POD5 reads. Used by
the basecalling track for the streaming smoke, Dorado ONNX
equivalence test, and §3.2 ratification (Phase 2 ).

License
-------
**Oxford Nanopore Technologies Open Data — terms-of-use grant
permissive research / reuse** as documented at
``https://labs.epi2me.io/`` and the AWS Open Data registry record. ONT
explicitly publishes these reads as open data; redistribution is
permitted with attribution. We capture the canonical citation per
the basecalling PRD §5.

The Dorado test-fixture POD5s pinned by Phase 2 (paths under
``nanoporetech/dorado/tests/data/``) ship under Dorado's repository
license — Oxford Nanopore Public License v1.0 (a permissive license
for non-commercial research use; see Dorado's ``LICENCE.txt``). Both
licenses cover redistribution-with-attribution; the POD5 signal data
itself is from ONT's open-data programme (sample id "hg002").

Smoke vs full
-------------
The full HG002 ONT run is hundreds of GB. **Smoke** is a small
multi-read POD5 fixture (~1.6 MB, 3 reads at 5 kHz, R10.4.1 PromethION,
sample_id="hg002") suitable for in-CI streaming smoke and §3.2 fixture
work. The full URL points at AWS Open Data's GIAB 2023.05 PromethION
flowcell directory; in this revision the FULL_URL placeholder remains
because no operator has yet completed a `--mode full` fetch — the
fetcher refuses to write a file with a wrong checksum so the placeholder
is fail-safe (raises ``ChecksumMismatchError`` on first contact and
the operator updates the SHA in the same commit).

URL stability — canonical + alternate mirrors
---------------------------------------------
Canonical (pinned for the smoke):

  https://raw.githubusercontent.com/nanoporetech/dorado/v1.4.0/
  tests/data/single_channel_multi_read_pod5/filtered.pod5

Pinned to the **v1.4.0 git tag** (immutable). If GitHub ever rotates
this artifact (which it does not for tagged content), or the Dorado
project removes the file in a future release, the operator-facing
fallbacks are:

1. **GitHub Codeload** (same content, alternate CDN path):
   ``https://github.com/nanoporetech/dorado/raw/v1.4.0/tests/data/single_channel_multi_read_pod5/filtered.pod5``
2. **AWS Open Data GIAB 2023.05** (R10.4.1 PromethION HG002 flowcells;
   POD5 files are 600 MB-1.4 GB each — slice with ``pod5 subset``):
   ``s3://ont-open-data/giab_2023.05/flowcells/hg002/<flowcell_id>/pod5_pass/``
   Listed via ``aws s3 ls --no-sign-request s3://ont-open-data/giab_2023.05/flowcells/hg002/``
   or the equivalent HTTPS GET on
   ``https://ont-open-data.s3.eu-west-1.amazonaws.com/?list-type=2&prefix=giab_2023.05/flowcells/hg002/``.
3. **AWS Open Data gm24385_2023.12** (same sample, alternate release):
   ``s3://ont-open-data/gm24385_2023.12/`` (single ~96 GB BAM; require
   bambu-style POD5 reconstruction — heavy fallback).

If all three fail, see ``docs/iron-fork-build.md`` (and the TODO in
the orchestrator queue) for an offline-tarball recipe. Phase 2 's
plan log captures the chosen pin's URL + SHA + size for forensic
verification.

Slicing recipe — when the smoke fixture isn't enough
----------------------------------------------------
For the §3.2 ratification target (100-read minimum, 10 K-read ideal),
the Dorado test fixture (3 reads) is below the floor. The path to a
larger fixture is to slice an AWS-hosted PromethION POD5 file::

    BUCKET=ont-open-data
    PREFIX=giab_2023.05/flowcells/hg002
    FLOWCELL=20230424_1302_3H_PAO89685_2264ba8c
    POD5=PAO89685_pass__2264ba8c_afee3a87_0.pod5
    aws s3 cp --no-sign-request \\
        "s3://$BUCKET/$PREFIX/$FLOWCELL/pod5_pass/$POD5" \\
        full.pod5
    pod5 view full.pod5 --ids --output read_ids.txt
    head -n 100 read_ids.txt > smoke_ids.txt
    pod5 subset full.pod5 --ids smoke_ids.txt --output hg002_100reads.pod5
    sha256sum hg002_100reads.pod5  # -> pin

The slicing is operator-driven (not auto-fetched) because the upstream
600 MB POD5 file is too large for in-CI use and the slice's exact SHA
depends on which 100 reads the operator selects + the local pod5
toolchain version. 's reproduction subsection cites the exact
recipe + slice IDs alongside the measurement results.

References
----------
* Genome in a Bottle Consortium HG002 (NA24385) reference samples.
  Zook JM, et al. (2016). Scientific Data 3:160025.
* AWS Open Data: ``s3://ont-open-data/`` (registry entry
  ``ont-open-data``).
* Oxford Nanopore Technologies (2023). "GIAB 2023.05 release —
  HG002 R10.4.1 PromethION reads." https://labs.epi2me.io/
* Dorado v1.4.0 test fixtures (the canonical pin's source):
  https://github.com/nanoporetech/dorado/tree/v1.4.0/tests/data

Pinning policy
--------------
Phase 2 pinned ``smoke_sha256`` to a real SHA-256 from a fetch
of the Dorado v1.4.0 ``single_channel_multi_read_pod5/filtered.pod5``
fixture. The full-bundle ``sha256`` remains a placeholder; operators
upgrading to a real ``--mode full`` run-of-record should update the
SHA in this file in the same commit they fetch the artifact.
"""

from __future__ import annotations

from bionpu.data.fetchers import DatasetSpec, register

# Smoke subset — the canonical pin (Phase 2 , 2026-04-25).
# 3 reads, 5 kHz, R10.4.1 PromethION, sample_id="hg002"; 1,682,120 bytes.
# See module docstring for the URL stability rationale and slicing recipe
# for larger fixtures.
SMOKE_URL = (
    "https://raw.githubusercontent.com/nanoporetech/dorado/v1.4.0/"
    "tests/data/single_channel_multi_read_pod5/filtered.pod5"
)
SMOKE_SHA = "5bc23cac76e8129da645e35e45ceea4a0f383763b1bb35a82866ae758a3394a3"
SMOKE_SIZE = 1_682_120

# Full run: HG002 nanopore reads from AWS Open Data. The full bundle is
# ~hundreds of GB; only fetched when the operator passes ``--mode full``.
# SHA remains a placeholder — fail-safe per the framework's contract.
FULL_URL = (
    "https://s3.amazonaws.com/ont-open-data/"
    "gm24385_2020.09/analysis/r10/basecall/calls.pod5"
)

SPEC = DatasetSpec(
    name="pod5_hg002",
    kind="pod5",
    urls=[FULL_URL],
    sha256="0" * 64,  # placeholder; first --mode full fetch pins this.
    size_bytes=0,
    license_name="ONT Open Data + Oxford Nanopore Public License v1.0",
    license_url="https://labs.epi2me.io/",
    citation=(
        "Oxford Nanopore Technologies (2023). HG002 (NA24385) open-data "
        "POD5 reads, AWS Open Data registry s3://ont-open-data/. "
        "Smoke fixture from nanoporetech/dorado v1.4.0 "
        "tests/data/single_channel_multi_read_pod5/filtered.pod5."
    ),
    smoke_subset_url=SMOKE_URL,
    smoke_sha256=SMOKE_SHA,
    smoke_size_bytes=SMOKE_SIZE,
    relpath="basecalling/hg002.pod5",
    notes=(
        "HG002 ONT POD5. Smoke = Dorado v1.4.0 test fixture "
        "(3 reads, 5 kHz, R10.4.1 PromethION, ~1.6 MB); "
        "full = AWS Open Data ont-open-data bucket (placeholder SHA — "
        "first --mode full fetch updates it). License: ONT Open Data + "
        "ONT Public License v1.0."
    ),
)

register(SPEC)
