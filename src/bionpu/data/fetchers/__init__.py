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

"""Public dataset fetcher framework.

Per  §4.4: each public dataset has a fetcher with checksum
verification, license/citation in code, and a manifest entry. Network
failures must record URL + HTTP status + retry guidance.

Design notes
------------

* **Streaming, atomic writes.** Every fetch downloads to ``<dest>.partial``
  while updating a streaming SHA-256 hasher. On success the file is
  renamed to ``<dest>``; the rename is the atomic commit. If the process
  is killed mid-fetch the cache contains only ``<dest>.partial`` — the
  fetcher will never see a finalised file with a wrong checksum.

* **Resume.** A pre-existing ``.partial`` is resumed via HTTP Range
  when the server advertises ``Accept-Ranges: bytes``. Because we
  cannot retroactively verify that the partial bytes match what the
  server would have served (a server-side rotation would silently
  corrupt the resume), we *re-hash the entire file* after the body is
  complete and compare against the spec's expected SHA-256. A mismatch
  retains a forensic ``.partial.corrupt.<ts>`` for triage and raises.

* **Locking.** A sibling ``<dest>.lock`` file is held with
  ``fcntl.flock(LOCK_EX | LOCK_NB)`` for the duration of the fetch.
  A second process gets a fast ``FetcherLockError`` rather than racing
  on the same partial file.

* **Idempotency.** If ``<dest>`` already exists *and* its SHA-256
  matches ``spec.sha256``, the fetcher is a no-op. A truncated /
  corrupted ``<dest>`` is detected by the same SHA check and re-fetched.

* **Network failures are first-class.** Any non-2xx response or
  ``requests`` exception is wrapped in ``FetcherNetworkError`` with the
  URL, HTTP status (or exception class name), and retry guidance.

* **License/citation in code.** Each ``DatasetSpec`` carries the
  license name, license URL, and a canonical citation; the manifest
  writer copies them per fetch.

The dataset-specific specs live in sibling modules
(``pod5_hg002.py``, ``reference_genomes.py``, ``doench_2016.py``,
``guide_seq.py``) and self-register on import via ``register()``.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

__all__ = [
    "ChecksumMismatchError",
    "DatasetSpec",
    "Fetcher",
    "FetcherError",
    "FetcherLockError",
    "FetcherNetworkError",
    "REGISTRY",
    "default_cache_root",
    "fetch",
    "register",
]

Mode = Literal["smoke", "full"]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class FetcherError(RuntimeError):
    """Base class for all fetcher errors."""

class FetcherLockError(FetcherError):
    """Raised when another process holds the fetcher lockfile."""

class FetcherNetworkError(FetcherError):
    """Raised on HTTP / transport errors. Captures URL + status."""

    def __init__(self, url: str, status: int | None, detail: str) -> None:
        self.url = url
        self.status = status
        self.detail = detail
        if status is not None:
            super().__init__(
                f"network failure fetching {url}: HTTP {status} ({detail}). "
                "Retry once; if persistent, check the dataset's mirror list "
                "in bionpu/data/fetchers/ and consider supplying a local "
                "artifact via --offline."
            )
        else:
            super().__init__(
                f"network failure fetching {url}: {detail}. "
                "Retry once; if persistent, check connectivity and "
                "the dataset's mirror list."
            )

class ChecksumMismatchError(FetcherError):
    """Raised when a downloaded file's SHA-256 doesn't match the spec."""

    def __init__(self, url: str, dest: Path, expected: str, got: str) -> None:
        self.url = url
        self.dest = dest
        self.expected = expected
        self.got = got
        super().__init__(
            f"checksum mismatch for {dest.name} (from {url}): "
            f"expected {expected[:16]}..., got {got[:16]}.... "
            "Forensic copy retained alongside the cache directory."
        )

# ---------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Immutable description of a public dataset.

    Parameters
    ----------
    name:
        Stable lookup key. Must match the module-level register call.
    kind:
        Free-form tag (``"pod5"``, ``"reference"``, ``"crispr-activity"``, …)
        used only for grouping in the manifest.
    urls:
        Ordered list of mirror URLs. Tried in order; first 2xx wins.
    sha256:
        SHA-256 of the *full* (or full-equivalent) artifact.
    size_bytes:
        Expected bytes of the full artifact.
    license_name, license_url, citation:
        Mandatory license / citation metadata copied verbatim into the
        manifest.
    smoke_subset_url, smoke_sha256:
        For datasets large enough that the full artifact is gated on a
        ``--full`` flag, the smoke variant is fetched instead. ``None``
        for already-small datasets where smoke == full.
    smoke_size_bytes:
        Optional; informational only.
    relpath:
        Cache-relative path the artifact is written to. Defaults to
        ``"<kind>/<name>.bin"`` if not provided.
    """

    name: str
    kind: str
    urls: list[str]
    sha256: str
    size_bytes: int
    license_name: str
    license_url: str
    citation: str
    smoke_subset_url: str | None = None
    smoke_sha256: str | None = None
    smoke_size_bytes: int | None = None
    relpath: str | None = None
    notes: str = ""

    def cache_relpath(self, mode: Mode) -> str:
        # Smoke and full live at distinct paths so toggling --mode is
        # never a checksum mismatch.
        base = self.relpath or f"{self.kind}/{self.name}.bin"
        if mode == "smoke":
            stem, _, ext = base.rpartition(".")
            if stem:
                return f"{stem}.smoke.{ext}"
            return base + ".smoke"
        return base

    def url_for(self, mode: Mode) -> str:
        if mode == "smoke":
            if not self.smoke_subset_url:
                # Smoke == full for already-small datasets.
                return self.urls[0]
            return self.smoke_subset_url
        return self.urls[0]

    def expected_sha(self, mode: Mode) -> str:
        if mode == "smoke" and self.smoke_sha256:
            return self.smoke_sha256
        return self.sha256

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, DatasetSpec] = {}

def register(spec: DatasetSpec) -> DatasetSpec:
    """Register a dataset spec under ``spec.name``. Idempotent."""
    REGISTRY[spec.name] = spec
    return spec

# ---------------------------------------------------------------------------
# Cache root
# ---------------------------------------------------------------------------

def default_cache_root() -> Path:
    """Return the default cache root.

    Honours ``BIONPU_DATA_CACHE`` for testing; falls back to
    ``<repo>/data_cache`` (where ``<repo>`` is the parent of
    ``bionpu/``).
    """
    env = os.environ.get("BIONPU_DATA_CACHE")
    if env:
        return Path(env)
    # bionpu/data/fetchers/__init__.py -> bionpu/data/fetchers -> bionpu/data
    # -> bionpu -> <repo>
    return Path(__file__).resolve().parents[3] / "data_cache"

# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

CHUNK = 1 << 16  # 64 KiB streaming chunk

class Fetcher:
    """Stateful fetcher that owns a cache root and serialised manifest."""

    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = Path(cache_root) if cache_root else default_cache_root()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_root / "MANIFEST.md"

    # ----- public API -----

    def fetch(self, spec: DatasetSpec, mode: Mode = "smoke", force: bool = False) -> Path:
        """Fetch ``spec`` into the cache; return the cached path."""
        dest = self._dest_for(spec, mode)
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected_sha = spec.expected_sha(mode)
        url = spec.url_for(mode)

        # Fast-path idempotency: if the dest already exists and matches,
        # skip lock + network entirely.
        if not force and dest.exists():
            got = _sha256_file(dest)
            if got == expected_sha:
                return dest
            # Mismatch — drop the bad file and re-fetch.
            dest.unlink()

        lock_path = Path(str(dest) + ".lock")
        with _exclusive_lock(lock_path):
            # Re-check inside the lock in case a peer just finished.
            if not force and dest.exists():
                got = _sha256_file(dest)
                if got == expected_sha:
                    return dest
                dest.unlink()

            partial = Path(str(dest) + ".partial")
            self._download_to_partial(url, partial, expected_sha)

            # Validate after rename-prep.
            got = _sha256_file(partial)
            if got != expected_sha:
                forensic = partial.with_suffix(
                    partial.suffix + f".corrupt.{int(time.time())}"
                )
                shutil.move(str(partial), str(forensic))
                raise ChecksumMismatchError(url, dest, expected_sha, got)

            os.replace(partial, dest)

        self._append_manifest(spec, mode, dest, expected_sha)
        return dest

    def _dest_for(self, spec: DatasetSpec, mode: Mode = "full") -> Path:
        return self.cache_root / spec.cache_relpath(mode)

    # ----- download -----

    def _download_to_partial(self, url: str, partial: Path, expected_sha: str) -> None:
        """Stream ``url`` into ``partial``, attempting Range-resume if possible.

        Always re-hashes the full file at the end (caller does the
        comparison). If a Range resume produces a wrong SHA we drop the
        partial and retry from zero exactly once.
        """
        # Determine resume strategy.
        resume_offset = 0
        if partial.exists():
            try:
                if _server_supports_ranges(url):
                    resume_offset = partial.stat().st_size
                else:
                    partial.unlink()
            except Exception:
                # Don't trust a fragile partial; restart cleanly.
                with _suppress_oserror():
                    partial.unlink()
                resume_offset = 0

        try:
            self._stream(url, partial, resume_offset)
        except FetcherNetworkError:
            raise
        except Exception as exc:
            raise FetcherNetworkError(url, None, f"{type(exc).__name__}: {exc}") from exc

        # If the resumed prefix was wrong, retry once from zero.
        got = _sha256_file(partial)
        if got != expected_sha and resume_offset > 0:
            with _suppress_oserror():
                partial.unlink()
            self._stream(url, partial, 0)

    def _stream(self, url: str, partial: Path, resume_offset: int) -> None:
        import requests  # local import: keep top-level cheap

        headers = {}
        if resume_offset > 0:
            headers["Range"] = f"bytes={resume_offset}-"

        try:
            with requests.get(url, stream=True, headers=headers, timeout=30) as resp:
                if resp.status_code not in (200, 206):
                    raise FetcherNetworkError(
                        url, resp.status_code, resp.reason or "non-2xx response"
                    )
                # If we asked for a Range and got 200, the server
                # ignored Range — restart the partial file.
                mode = "ab" if (resume_offset > 0 and resp.status_code == 206) else "wb"
                if mode == "wb" and partial.exists():
                    partial.unlink()
                with partial.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        if chunk:
                            fh.write(chunk)
        except FetcherNetworkError:
            raise
        except Exception as exc:
            raise FetcherNetworkError(url, None, f"{type(exc).__name__}: {exc}") from exc

    # ----- manifest -----

    def _append_manifest(
        self, spec: DatasetSpec, mode: Mode, dest: Path, sha: str
    ) -> None:
        if not self.manifest_path.exists():
            self.manifest_path.write_text(_MANIFEST_HEADER)

        rel = dest.relative_to(self.cache_root)
        line = (
            f"| {spec.name} | {mode} | {rel} | `{sha}` | {dest.stat().st_size} | "
            f"[{spec.license_name}]({spec.license_url}) | "
            f"{spec.citation} | "
            f"{datetime.now(tz=UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} |\n"
        )
        with self.manifest_path.open("a") as fh:
            fh.write(line)

# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def fetch(name: str, mode: Mode = "smoke", force: bool = False) -> Path:
    """Convenience wrapper around ``Fetcher().fetch(REGISTRY[name], ...)``."""
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none registered)"
        raise KeyError(f"unknown dataset {name!r}; known: {known}")
    return Fetcher().fetch(REGISTRY[name], mode=mode, force=force)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _server_supports_ranges(url: str) -> bool:
    import requests

    try:
        head = requests.head(url, timeout=15, allow_redirects=True)
    except Exception:
        return False
    if head.status_code >= 400:
        return False
    return head.headers.get("Accept-Ranges", "").lower() == "bytes"

class _suppress_oserror:
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        return et is not None and issubclass(et, OSError)

class _exclusive_lock:
    """Acquire ``fcntl.flock(LOCK_EX | LOCK_NB)`` on ``path``.

    The file is created if missing. On contention, raises
    :class:`FetcherLockError` with a clear retry message. On exit the
    lock is released (the file itself is left in place; harmless).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "ab")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise FetcherLockError(
                    f"another fetcher holds the lock at {self.path}; "
                    "rerun once it completes (or remove the lockfile if you "
                    "are sure no other fetcher is running)."
                ) from exc
            raise
        return self

    def __exit__(self, et, ev, tb):
        if self._fh is not None:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None

_MANIFEST_HEADER = """# `data_cache/` manifest

This file is **append-only**. It records every successful dataset fetch
(via `bionpu data fetch <name>`) so the cache can be audited and
reproduced. The directory above this file is gitignored in its entirety
*except* this manifest.

License names link to canonical license URLs. Citations are the
canonical published citation for each dataset; please cite them in any
derivative publication.

| name | mode | path (rel) | sha256 | bytes | license | citation | fetched_at |
|------|------|------------|--------|-------|---------|----------|------------|
"""

# ---------------------------------------------------------------------------
# Eager imports of the dataset-specific specs (each registers itself).
# Placed at the bottom to avoid circular imports.
# ---------------------------------------------------------------------------

from bionpu.data.fetchers import crisproff as _crisproff  # noqa: E402,F401
from bionpu.data.fetchers import doench_2016 as _doench_2016  # noqa: E402,F401
from bionpu.data.fetchers import guide_seq as _guide_seq  # noqa: E402,F401
from bionpu.data.fetchers import pod5_hg002 as _pod5_hg002  # noqa: E402,F401
from bionpu.data.fetchers import reference_genomes as _reference_genomes  # noqa: E402,F401
from bionpu.data.fetchers import yaish_2024 as _yaish_2024  # noqa: E402,F401

# Mark unused imports as used (for type checkers / ruff F401).
_ = (_doench_2016, _guide_seq, _pod5_hg002, _reference_genomes, field)
