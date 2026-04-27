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

"""Public dataset loaders + fetchers.

* ships in-repo smoke fixture loaders (no `data_cache/` dependency).
  Re-exported here for ergonomics.
* ships the public dataset fetcher framework (HG002 POD5, GRCh38 /
  T2T-CHM13 / GRCm39 reference genomes, Doench 2016 guide-activity, and
  GUIDE-seq off-target sets) with checksum verification, license /
  citation in code, and a `data_cache/MANIFEST.md` writer.
"""

from bionpu.data.fetchers import (
    REGISTRY,
    ChecksumMismatchError,
    DatasetSpec,
    Fetcher,
    FetcherError,
    FetcherLockError,
    FetcherNetworkError,
    default_cache_root,
    fetch,
    register,
)
from bionpu.data.load_smoke import (
    PLANTED_GUIDES,
    PLANTED_POSITIONS,
    POD5_UNAVAILABLE_MESSAGE,
    load_smoke_genome,
    load_smoke_pod5,
    smoke_fixtures_root,
)

__all__ = [
    # smoke loaders
    "PLANTED_GUIDES",
    "PLANTED_POSITIONS",
    "POD5_UNAVAILABLE_MESSAGE",
    "load_smoke_genome",
    "load_smoke_pod5",
    "smoke_fixtures_root",
    # fetchers
    "REGISTRY",
    "ChecksumMismatchError",
    "DatasetSpec",
    "Fetcher",
    "FetcherError",
    "FetcherLockError",
    "FetcherNetworkError",
    "default_cache_root",
    "fetch",
    "register",
]
