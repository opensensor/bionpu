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

"""CRISPRoffT — UTHealth Houston comprehensive CRISPR off-target database.

Dataset
-------
A consolidated database of CRISPR/Cas off-target measurements aggregated
from 74 published studies across 29 experimental techniques. Used by the
PRD-1 Off-Target Predictor v2 work as the verification corpus.

    Wang G., Zhao J., Pavesi M., Vaisvila R., Rohde C., Wang J.,
    Borgesi J., Kim J. H., Pertea M., Salzberg S. L., Yang E. J.,
    Cheng Y., Cheng X., Park J., Liu X. S., Schatz M. C., Hellmann I.,
    Beroukhim R., Lazzarotto C. R., Jiang B., Yan J., Tsai S. Q.,
    Zhao Z. (2025). "CRISPRoffT: comprehensive database of CRISPR/Cas
    off-targets." *Nucleic Acids Research* 53(D1):D914-D924.
    DOI: 10.1093/nar/gkae1025.

Pinned artifact
---------------
``allframe_update_addEpige.txt`` — the full aggregated TSV
(151,429,169 bytes / ~144 MiB, last-modified 2024-10-01). 47 columns,
226,164 guide / off-target pairs spanning 368 unique guides and 22,632
genes; both continuous (``Indel_accu%`` cleavage rate) and binary
(``Identity`` / ``Validation``) labels.

The audit context behind this fetcher (why this dataset, premise
correction vs. the original PRD, license caveats) is documented in
``docs/model-selection-audit.md`` § 6.

License
-------
**CC BY-NC.** The dataset is freely available for non-commercial use
with attribution; the file is **not** redistributable inside a shipped
wheel or commercial bundle. We fetch it on demand into the user's
local cache and keep it out of the repo. Citation must accompany any
published evaluation result that touches this corpus.

Smoke vs full
-------------
The CRISPRoffT homepage publishes one canonical artifact (the
``allframe`` TSV); there is no smaller official mirror that survives
checksum pinning. Smoke == full.

Mirror policy
-------------
Primary: the UTHealth Houston CCSM lab homepage. There is no Zenodo
DOI / FigShare / GitHub release. If the primary URL goes down, the
dataset would need to be re-derived from the cited 74 source studies
(non-trivial; treat the SHA-256 pin as the durable contract here).
"""

from __future__ import annotations

from bionpu.data.fetchers import DatasetSpec, register

CRISPROFFT_URL = (
    "https://ccsm.uth.edu/CRISPRoffT/table_summary/"
    "allframe_update_addEpige.txt"
)

# pin (fetched 2026-04-27 from the UTHealth CCSM CDN; file
# last-modified 2024-10-01).
CRISPROFFT_SHA = (
    "522ac5c6852a86e3c6ca8d5a305fea36a6b4ea0c5a643095d75f36a0491b3899"
)
CRISPROFFT_BYTES = 151_429_169

SPEC = DatasetSpec(
    name="crisprofft",
    kind="crispr-validation",
    urls=[CRISPROFFT_URL],
    sha256=CRISPROFFT_SHA,
    size_bytes=CRISPROFFT_BYTES,
    license_name="CC BY-NC (non-commercial use with attribution)",
    license_url="https://creativecommons.org/licenses/by-nc/4.0/",
    citation=(
        "Wang G. et al. (2025). CRISPRoffT: comprehensive database of "
        "CRISPR/Cas off-targets. Nucleic Acids Research 53(D1):D914-D924. "
        "doi:10.1093/nar/gkae1025."
    ),
    smoke_subset_url=CRISPROFFT_URL,
    smoke_sha256=CRISPROFFT_SHA,
    smoke_size_bytes=CRISPROFFT_BYTES,
    relpath="crispr/crisprofft/allframe_update_addEpige.txt",
    notes=(
        "CRISPRoffT consolidated off-target database (UTHealth CCSM). "
        "47-column TSV, 226,164 guide/off-target pairs, 368 unique "
        "guides, 22,632 genes. Both continuous (Indel_accu%) and binary "
        "(Identity, Validation) labels. CC BY-NC: verification use is "
        "fine with citation; do NOT redistribute the file in a shipped "
        "bundle. Smoke == full (no smaller mirror published)."
    ),
)
register(SPEC)
