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

"""In-repo smoke fixture loaders.

These loaders return paths to fixtures committed under
`tracks/{basecalling,crispr}/fixtures/`. They exist so CI (and developer
quick-runs) can execute the full bio pipelines without requiring a populated
`data_cache/`.

Two loaders ship:

* :func:`load_smoke_genome` — synthetic 1 Mbp FASTA with FIXTURE-A guides
  planted at fixed positions. Used by (NumPy CRISPR oracle) and any
  CRISPR smoke test that must not touch `data_cache/`.
* :func:`load_smoke_pod5` — ~10-read POD5 file for the basecalling
  streaming smoke. If the `pod5` package was unavailable when the
  fixture was built (or the file was excluded for size), this raises
  ``FileNotFoundError`` with a stable sentinel message so callers can
  fall back to a real `data_cache/` POD5.

The loaders are intentionally cheap: they perform an existence check and
return a :class:`pathlib.Path`. They do NOT parse the fixture or load it
into memory. Re-export from :mod:`bionpu.data` for ergonomics.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "PLANTED_GUIDES",
    "PLANTED_POSITIONS",
    "POD5_UNAVAILABLE_MESSAGE",
    "load_smoke_genome",
    "load_smoke_pod5",
    "smoke_fixtures_root",
]

# ---- canonical pin (mirrors `tracks/crispr/fixtures/PLANTS.md`) -----------

# FIXTURE-A guide sequences planted into the synthetic 1 Mbp FASTA. Mirrors
# the source-of-truth pin in `tracks/crispr/fixtures/FIXTURE-A.md` and
# `tracks/crispr/fixtures/build_smoke.py`. Re-exported here so
# downstream tasks can import without
# re-typing the values.
PLANTED_GUIDES: tuple[str, ...] = (
    "CACTTGAGCCTCCAAGTAGC",  # g1
    "TTTCCCAGGGCCACTGTTGC",  # g2
    "TCTTAAGTGTTCTCTTTCCG",  # g3
    "ACAGGAGTGAGCCACCAGCC",  # g4
    "ACTGTTTCATTCAGGAACTC",  # g5
)
PLANTED_POSITIONS: tuple[int, ...] = (100_000, 300_000, 500_000, 700_000, 900_000)

POD5_UNAVAILABLE_MESSAGE = "pod5 fixture unavailable in this build"

def _repo_root() -> Path:
    """Resolve the repository root (parent of `bionpu/`)."""
    # bionpu/data/load_smoke.py -> bionpu/data -> bionpu -> <repo>
    return Path(__file__).resolve().parents[2]

def smoke_fixtures_root() -> Path:
    """Return the `tracks/` directory that holds the per-track fixture dirs."""
    return _repo_root() / "tracks"

def load_smoke_genome() -> Path:
    """Path to the synthetic 1 Mbp CRISPR smoke FASTA.

    Returns
    -------
    pathlib.Path
        Absolute path to ``tracks/crispr/fixtures/synthetic_1mbp.fa``.

    Raises
    ------
    FileNotFoundError
        If the fixture has not been built. Run
        ``python tracks/crispr/fixtures/build_smoke.py`` to generate it.
    """
    p = smoke_fixtures_root() / "crispr" / "fixtures" / "synthetic_1mbp.fa"
    if not p.exists():
        raise FileNotFoundError(
            f"synthetic 1 Mbp smoke FASTA not found at {p}. "
            "Generate with `python tracks/crispr/fixtures/build_smoke.py`."
        )
    return p

def load_smoke_pod5() -> Path:
    """Path to the basecalling smoke POD5 file.

    Returns
    -------
    pathlib.Path
        Absolute path to ``tracks/basecalling/fixtures/smoke.pod5``.

    Raises
    ------
    FileNotFoundError
        With the sentinel message ``"pod5 fixture unavailable in this build"``
        if the basecalling fixture was not produced (e.g. the `pod5` package
        was unavailable in the build environment). Callers should treat this
        as "fall back to a real data_cache/ POD5", not as a hard failure.
    """
    p = smoke_fixtures_root() / "basecalling" / "fixtures" / "smoke.pod5"
    if not p.exists():
        raise FileNotFoundError(
            f"{POD5_UNAVAILABLE_MESSAGE}: expected {p}. "
            "Build with `python tracks/basecalling/fixtures/build_smoke.py` "
            "(requires the `pod5` Python package)."
        )
    return p
