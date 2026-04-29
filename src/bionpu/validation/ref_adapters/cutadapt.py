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

"""cutadapt adapter for Track F v0.

Resolves the cutadapt binary on PATH or alongside the running
interpreter (mirrors the helper in
``bionpu-public/tests/test_adapter_trim.py``), then shells out to
``cutadapt -a ADAPTER --no-indels -e 0 -O <P>`` against an input
FASTQ.

Returned trimmed FASTQ is the canonical reference for the
``bionpu trim`` cross-check.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


__all__ = [
    "cutadapt_installed",
    "cutadapt_path",
    "cutadapt_version",
    "build_cutadapt_argv",
    "run_cutadapt",
]


def cutadapt_path() -> str | None:
    """Resolve a cutadapt binary path.

    Searches:
    1. ``$PATH`` via :func:`shutil.which`.
    2. ``Path(sys.executable).parent / "cutadapt"`` for the venv-bundled binary.

    Returns ``None`` if neither candidate exists or is executable.
    """
    on_path = shutil.which("cutadapt")
    if on_path is not None:
        return on_path
    sibling = Path(sys.executable).parent / "cutadapt"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


def cutadapt_installed() -> bool:
    return cutadapt_path() is not None


def cutadapt_version() -> str | None:
    """Return the cutadapt version string or None if missing."""
    p = cutadapt_path()
    if p is None:
        return None
    try:
        out = subprocess.run(
            [p, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception:
        return None
    return out.stdout.strip() or None


def build_cutadapt_argv(
    *,
    cutadapt: str,
    in_path: Path,
    out_path: Path,
    adapter: str,
    min_overlap: int,
) -> list[str]:
    """Construct the canonical cutadapt argv for the trim cross-check."""
    return [
        cutadapt,
        "-a",
        adapter,
        "--no-indels",
        "-e",
        "0",
        "-O",
        str(min_overlap),
        "-o",
        str(out_path),
        str(in_path),
    ]


def run_cutadapt(
    *,
    in_path: Path,
    out_path: Path,
    adapter: str,
    min_overlap: int,
    timeout_s: float = 60.0,
) -> subprocess.CompletedProcess:
    """Run cutadapt and raise :class:`RuntimeError` on failure.

    Returns the :class:`subprocess.CompletedProcess` for the caller's
    inspection (stderr/stdout captured).
    """
    binary = cutadapt_path()
    if binary is None:
        raise RuntimeError(
            "cutadapt not found on PATH or alongside sys.executable"
        )
    argv = build_cutadapt_argv(
        cutadapt=binary,
        in_path=in_path,
        out_path=out_path,
        adapter=adapter,
        min_overlap=min_overlap,
    )
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cutadapt exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc
