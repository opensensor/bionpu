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

"""CRISPR PAM filter + threshold + sparse-emit op.

Registers ``crispr_pam_filter_early`` and ``crispr_pam_filter_late`` in
:data:`bionpu.dispatch.npu.NPU_OPS` at import time. Both ops produce
**byte-identical sparse hit-list output** (after canonical
normalization) on the same input — they differ only in NPU work
distribution:

* ``crispr_pam_filter_early`` (production): on-tile NGG PAM check on
  Tile A drops ~7/8 of windows before the match tiles compute. Sparse-
  emit on Tile Z applies threshold + emits surviving records via the
  shim DMA-out ring buffer. This is the C-M5 production path.

* ``crispr_pam_filter_late`` (comparison artifact): Tile A passes every
  window through; PAM check happens at Tile Z together with the
  threshold. Same final output bytes, much higher Tile-Z work volume
  (and DMA-out volume on the sparse stream).

Both ops accept the same inputs and return the same shape:

* ``windows_with_pam`` (np.ndarray): shape ``(N_WINDOWS, WINDOW_BYTES_IN)``
  uint8 — 5 spacer bytes + 1 PAM byte per record.
* ``guides_2bit`` (np.ndarray): shape ``(N_GUIDES, SPACER_BYTES)`` uint8.
* ``max_mismatches`` (int): on-tile threshold (default 4).
* Returns ``list[SparseHit]``.

The ``SparseHit`` schema deliberately mirrors :class:`MatchRecord`
from (with optional ``strand``/``chrom`` fields populated by the
caller from the window-index lookup table). This makes 's
``extract_hits`` and 's sparse emit interchangeable for downstream
code (CLI, oracle round-trip, etc.).

Rebuild:
    cd bionpu/kernels/crispr/pam_filter && make NPU2=1 all
    # then vendor build/{early,late}/final.xclbin and the host_runner
    # binary into bionpu/dispatch/_npu_artifacts/crispr_pam_filter_*

If artifacts are absent the op raises :class:`NpuArtifactsMissingError`
the same way / do.

Out of scope for (deferred):

* Match-tile-level PAM-skip branch ( / — for real cycle
  savings on the match tiles, not just emit volume).
* In-tile reverse-complement (host pre-flips; see DESIGN.md §3).
* Memtile-aggregated 4-match-tile fan-in.
* IUPAC PAM ambiguity codes beyond the literal `N` wildcard.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    _xrt_env,  # noqa: PLC2701 — internal helper
    register_npu_op,
)

# Re-export the codec from so callers don't need two import paths.
from bionpu.kernels.crispr.match_singletile import (
    SPACER_BYTES,
    SPACER_LEN,
    encode_2bit,
)

__all__ = [
    "CrisprPamFilterEarly",
    "CrisprPamFilterLate",
    "EMIT_RECORD_BYTES",
    "EMIT_SLOT_BYTES",
    "EMIT_SLOT_RECORDS",
    "GUIDES_PER_TILE",
    "KERNEL_HARDCODED_MAX_MISMATCHES",
    "N_CHUNKS",
    "N_GUIDES",
    "N_MATCH_TILES",
    "N_WINDOWS",
    "NpuArtifactsMissingError",
    "PAM_BYTES",
    "PAM_LEN",
    "infer_slot_bytes_from_blob",
    "SPACER_BYTES",
    "SPACER_LEN",
    "SparseHit",
    "TILE_MEMORY_BREAKDOWN",
    "TILE_MEMORY_BYTES",
    "WINDOW_BYTES_IN",
    "WINDOWS_PER_CHUNK",
    "build_window_record",
    "decode_per_slot_sparse_buffer",
    "decode_sparse_buffer",
    "encode_pam_byte",
    "pam_matches_ngg",
    "pam_matches_ngg_ascii",
]

# --- pinned shape (must match pam_filter.py / runner.cpp / kernel) ---
N_GUIDES = 128
N_WINDOWS = 4096
PAM_LEN = 3       # NGG
PAM_BYTES = 1     # 3 nt × 2 bits / 8 = 0.75 → packed into 1 byte
WINDOW_BYTES_IN = SPACER_BYTES + PAM_BYTES  # 6

# Multi-tile fan-out (unchanged from ).
N_MATCH_TILES = 2
GUIDES_PER_TILE = N_GUIDES // N_MATCH_TILES  # 64

# Sparse-emit record layout (8 bytes — see tile_a_filter.cc / DESIGN.md §4).
EMIT_RECORD_BYTES = 8

# Hardcoded on-tile mismatch threshold compiled into the vendored xclbin's
# Tile-Z runtime sequence (``pam_filter.py`` line 486 — ``emit_kernel(...
# 4, # max_mismatches (host can override post-hoc)``). The IRON runtime
# sequence does not currently accept ``max_mismatches`` as a runtime
# argument, so the silicon path always emits records with mismatch count
# <= 4. The host applies the user-supplied threshold via post-filter for
# values <= 4, and falls back to host emulation when the user requests a
# threshold > 4 (the kernel would silently drop records with mismatch in
# the (4, max_mismatches] range).
KERNEL_HARDCODED_MAX_MISMATCHES: int = 4

# Per-chunk geometry (mirrors ).
WINDOWS_PER_CHUNK = 64
N_CHUNKS = N_WINDOWS // WINDOWS_PER_CHUNK  # 64

# Tile-Z sparse-emit slot geometry — must match `tile_a_filter.cc`'s
# ``EMIT_SLOT_RECORDS`` and ``EMIT_SLOT_BYTES`` (= EMIT_SLOT_RECORDS * 8).
# fix (2026-04-26): bumped 256 -> 1024. The 256 cap silently
# dropped records on chr22's repeat-rich sub-chunks (worst observed
# 508 hits / 64-window sub-chunk; 1004 records dropped over the full
# chr22 run). 1024 keeps slot at 8 KiB, host blob at 64 * 8 KiB =
# 512 KiB per launch. Tile Z writes ONE slot per sub-chunk; the
# runtime sequence drains ``N_CHUNKS = 64`` slots back-to-back into
# the host's sparse-out buffer, so the host blob is laid out as 64 ×
# 8192-byte slots, each starting with a 4-byte little-endian uint32
# record count followed by ``count * EMIT_RECORD_BYTES`` records and
# trailing zero-padding to the slot size.
# See :func:`decode_per_slot_sparse_buffer`.
EMIT_SLOT_RECORDS = 1024
EMIT_SLOT_BYTES = EMIT_SLOT_RECORDS * EMIT_RECORD_BYTES  # 8192

# Artifact paths (populated by `make NPU2=1` + the vendor step).
_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)
_EARLY_DIR = _ART_ROOT / "crispr_pam_filter_early"
_LATE_DIR = _ART_ROOT / "crispr_pam_filter_late"

_KERNEL_NAME = "MLIR_AIE"
_PIPELINE_DEPTH_ENV = "BIONPU_CRISPR_PIPELINE_DEPTH"
_LAUNCH_CHUNKS_ENV = "BIONPU_CRISPR_LAUNCH_CHUNKS"

def _pipeline_depth() -> int:
    raw = os.environ.get(_PIPELINE_DEPTH_ENV)
    if raw is None:
        return 8
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_PIPELINE_DEPTH_ENV} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{_PIPELINE_DEPTH_ENV} must be > 0")
    return value

def _requested_launch_chunks() -> int:
    raw = os.environ.get(_LAUNCH_CHUNKS_ENV)
    if raw is None:
        return 1
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_LAUNCH_CHUNKS_ENV} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{_LAUNCH_CHUNKS_ENV} must be > 0")
    return value

# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #

# 2-bit codes for ACGT (matches / ).
_BASE_2BIT = {ord("A"): 0, ord("C"): 1, ord("G"): 2, ord("T"): 3}

def encode_pam_byte(pam_ascii: str) -> int:
    """Pack a 3-nt ASCII PAM into 1 byte (bits 1:0 / 3:2 / 5:4).

    bits 7:6 are zero padding. ``N`` in the PAM is encoded as 0x0 (the
    same code as A) — Tile A only checks the literal positions, the
    wildcard slot's value is irrelevant.
    """
    if len(pam_ascii) != PAM_LEN:
        raise ValueError(f"PAM must be {PAM_LEN} nt; got {len(pam_ascii)}")
    out = 0
    for i, c in enumerate(pam_ascii):
        if c == "N":
            v = 0
        else:
            v = _BASE_2BIT.get(ord(c.upper()))
            if v is None:
                raise ValueError(f"non-ACGTN PAM base: {c!r}")
        out |= (v & 0x3) << (i * 2)
    return out & 0xFF

def build_window_record(spacer_ascii: str, pam_ascii: str) -> np.ndarray:
    """Build one Tile-A input record: 5 spacer bytes + 1 PAM byte.

    The ``pam_ascii`` is the genomic 3-nt context immediately 3' of the
    spacer. ``N`` in the PAM is encoded with the placeholder 2-bit code
    0x0; Tile A only checks the literal positions of the template
    (positions 1 and 2 must be G for NGG), so the wildcard slot's
    encoded value is irrelevant.
    """
    spacer = encode_2bit(spacer_ascii)
    pam = np.uint8(encode_pam_byte(pam_ascii))
    return np.concatenate([spacer, np.array([pam], dtype=np.uint8)])

def pam_matches_ngg(pam_byte: int) -> bool:
    """Host-side reference for the on-tile ``pam_is_ngg`` check.

    Returns True iff bits 3:2 == 0b10 (G) and bits 5:4 == 0b10 (G).
    Bits 1:0 (N wildcard) are ignored.
    """
    p1 = (pam_byte >> 2) & 0x3
    p2 = (pam_byte >> 4) & 0x3
    return p1 == 0x2 and p2 == 0x2

def pam_matches_ngg_ascii(pam_ascii: str) -> bool:
    """Reference NGG matcher operating on the 3-character ASCII string.

    Equivalent to 's :func:`tracks.crispr.oracle.scanner.matches_pam_template`
    with template ``"NGG"``. Used by tests to prove the on-tile PAM
    check matches the oracle byte-for-byte.
    """
    if len(pam_ascii) != PAM_LEN:
        return False
    return pam_ascii[1] == "G" and pam_ascii[2] == "G"

# --------------------------------------------------------------------------- #
# Tile-memory accounting (per pam_filter.py's IRON lowering — DESIGN.md §5)
# --------------------------------------------------------------------------- #

# Tile A (PAM filter):
_TILE_A_IN_DBL = 2 * WINDOWS_PER_CHUNK * WINDOW_BYTES_IN     # 768
_TILE_A_OUT_DBL = 2 * WINDOWS_PER_CHUNK * SPACER_BYTES        # 640
_TILE_A_PAM_OUT_DBL = 2 * WINDOWS_PER_CHUNK * PAM_BYTES       # 128
_TILE_A_TOTAL = _TILE_A_IN_DBL + _TILE_A_OUT_DBL + _TILE_A_PAM_OUT_DBL  # 1536

# Per-match-tile (unchanged from ):
_MATCH_GUIDES_RESIDENT = N_GUIDES * SPACER_BYTES                  # 640
_MATCH_WINDOWS_DBL = 2 * WINDOWS_PER_CHUNK * SPACER_BYTES         # 640
_MATCH_PARTIAL_OUT_DBL = 2 * WINDOWS_PER_CHUNK * GUIDES_PER_TILE  # 8192
_MATCH_TILE_TOTAL = (
    _MATCH_GUIDES_RESIDENT + _MATCH_WINDOWS_DBL + _MATCH_PARTIAL_OUT_DBL
)  # 9472 B

# Tile Z (joiner + threshold + sparse-emit):
_JOIN_PARTIAL_IN_DBL = N_MATCH_TILES * 2 * WINDOWS_PER_CHUNK * GUIDES_PER_TILE  # 16384
_JOIN_PAM_IN_DBL = 2 * WINDOWS_PER_CHUNK * PAM_BYTES  # 128
# fix: derive from EMIT_SLOT_BYTES (which now reflects
# the bumped EMIT_SLOT_RECORDS = 1024). Old hard-coded 256 * 8 was
# stale after the cap fix.
_EMIT_SLOT_BYTES = EMIT_SLOT_BYTES  # 8192 (per slot, post-fix)
_EMIT_DBL = 2 * _EMIT_SLOT_BYTES    # 16384 (double-buffered tile-DM)
_TILE_Z_TOTAL = _JOIN_PARTIAL_IN_DBL + _JOIN_PAM_IN_DBL + _EMIT_DBL  # 32896

#: per-tile breakdown so MANIFEST and measurements.json can cite it.
TILE_MEMORY_BREAKDOWN: dict[str, int] = {
    "tile_a_filter": _TILE_A_TOTAL,
    "match_tile": _MATCH_TILE_TOTAL,
    "tile_z_emit": _TILE_Z_TOTAL,
}

#: peak per-tile memory in bytes (Tile Z is the worst case).
TILE_MEMORY_BYTES: int = max(TILE_MEMORY_BREAKDOWN.values())

# Sanity: every tile fits the 64 KiB DM cap.
_AIE2P_DM_BUDGET_BYTES = 64 * 1024
assert TILE_MEMORY_BYTES < _AIE2P_DM_BUDGET_BYTES, (
    f"per-tile memory {TILE_MEMORY_BYTES} >= AIE2P DM cap {_AIE2P_DM_BUDGET_BYTES}"
)

# --------------------------------------------------------------------------- #
# Sparse hit record (replaces 's MatchRecord with strand + chrom slots)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class SparseHit:
    """One surviving (guide, window) pair.

    Attributes:
        window_idx: position-in-the-launch-stream window index (0..N_WINDOWS-1).
        guide_idx: row index into the input ``guides_2bit`` (0..N_GUIDES-1).
        mismatches: spacer-only mismatch count, 0..20.
        strand: ``"+"`` or ``"-"`` (host stamps from the launch metadata).
        chrom: contig label (host stamps from the launch metadata).
    """

    window_idx: int
    guide_idx: int
    mismatches: int
    strand: str = "+"
    chrom: str = ""

def decode_sparse_buffer(blob: bytes) -> list[SparseHit]:
    """Decode a SINGLE length-prefixed sparse-record blob.

    Layout::

        bytes 0..3 : uint32 record count (little-endian)
        bytes 4.. : record_count * 8-byte records:
                      window_idx (u16 LE) | guide_idx (u8) | mismatches (u8) | reserved (4)

    This decoder consumes a single-stream blob — useful for
    independently-emitted per-slot blobs and for the
    pre-flattened representation the test fixtures use. For the host-
    side blob produced by the precompiled
    ``crispr_pam_filter_early`` xclbin (which is laid out as
    ``N_CHUNKS = 64`` back-to-back length-prefixed slots — see
    :func:`decode_per_slot_sparse_buffer`), call
    :func:`decode_per_slot_sparse_buffer` instead.
    """
    if len(blob) < 4:
        return []
    n = struct.unpack_from("<I", blob, 0)[0]
    expected = 4 + n * EMIT_RECORD_BYTES
    if len(blob) < expected:
        raise ValueError(
            f"sparse buffer truncated: header says {n} records "
            f"({expected} bytes total) but blob is {len(blob)} bytes"
        )
    out: list[SparseHit] = []
    for i in range(n):
        off = 4 + i * EMIT_RECORD_BYTES
        wnd_lo, wnd_hi, g, mm = struct.unpack_from("<BBBB", blob, off)
        wnd = wnd_lo | (wnd_hi << 8)
        out.append(SparseHit(window_idx=wnd, guide_idx=g, mismatches=mm))
    return out

def infer_slot_bytes_from_blob(blob_len: int, n_slots: int = N_CHUNKS) -> int:
    """Infer per-slot byte size from the on-disk blob length.

    Different vendored xclbins emit different slot capacities (the
    pre-2026-04-26 build emits 256-record / 2048-byte slots; the
    post-2026-04-26 build emits 1024-record / 8192-byte slots — see the
    `EMIT_SLOT_RECORDS` constant in `tile_a_filter.cc`). The blob is
    laid out as ``n_slots`` back-to-back fixed-size slots, so dividing
    the on-disk length by ``n_slots`` recovers the actual slot size.

    Returns:
        Bytes per slot. Used by :func:`decode_per_slot_sparse_buffer`
        when called without an explicit ``slot_bytes`` to auto-tune to
        the vendored artifact.

    Raises:
        ValueError: blob length is not a multiple of ``n_slots`` (would
        indicate a truncated or padded blob).
    """
    if blob_len % n_slots != 0:
        raise ValueError(
            f"blob length {blob_len} is not a multiple of n_slots={n_slots}; "
            "cannot infer slot_bytes — check that the host runner wrote a "
            "complete buffer"
        )
    return blob_len // n_slots


def decode_per_slot_sparse_buffer(
    blob: bytes,
    *,
    n_slots: int = N_CHUNKS,
    slot_bytes: int | None = None,
    slot_records: int | None = None,
    windows_per_slot: int = WINDOWS_PER_CHUNK,
) -> list[SparseHit]:
    """Decode the per-slot sparse-emit layout the AIE2P kernel produces.

    The Tile-Z worker in ``pam_filter.py`` emits ONE
    length-prefixed slot per sub-chunk (``N_CHUNKS = 64`` sub-chunks per
    launch, 64 windows each).  The IRON runtime sequence's
    ``rt.drain(of_sparse.cons(), Out, wait=True)`` writes those slots
    back-to-back into the host's sparse-out buffer, so the on-disk
    layout is::

        slot 0  : bytes              0  .. slot_bytes-1   (count_0 + records_0 + pad)
        slot 1  : bytes      slot_bytes  .. 2*slot_bytes-1
        ...
        slot N-1: bytes (N-1)*slot_bytes .. N*slot_bytes-1

    Each slot is itself length-prefixed (4-byte little-endian count
    followed by ``count * EMIT_RECORD_BYTES`` 8-byte records) with
    trailing zero-padding to ``slot_bytes``.  The kernel hardcodes
    ``chunk_base_window_idx = 0`` (see ``pam_filter.py`` Tile-Z body
    comment), so ``window_idx`` inside each slot is in ``[0,
    windows_per_slot)`` — this decoder reapplies the
    ``slot_index * windows_per_slot`` offset so the returned
    ``SparseHit.window_idx`` is the launch-relative index in
    ``[0, n_slots * windows_per_slot)``, matching what the host-emu
    reference produces.

    The previous host decoder
    (:func:`decode_sparse_buffer`) treated the blob as monolithic and
    therefore only ever saw slot 0's records — the proximate cause of
    .
    The kernel itself is correct: a synthetic 4096-window all-PAM-pass
    input emits 4096 records under this decoder, byte-equal to
    ``_host_emulate_match_and_emit``.

    Args:
        blob: the host-side sparse-out buffer.
        n_slots: number of slots in the blob (default
            ``N_CHUNKS = 64``).
        slot_bytes: bytes per slot. When ``None`` (the default), the
            value is inferred from ``len(blob) // n_slots`` so the
            decoder works with both the pre-2026-04-26 vendored
            artifacts (slot_bytes=2048, 256 records/slot) and the
            post-2026-04-26 rebuilds (slot_bytes=8192, 1024
            records/slot). Pass an explicit value to override.
        slot_records: cap on records per slot. When ``None``, derived
            from ``slot_bytes`` as ``(slot_bytes - 4) //
            EMIT_RECORD_BYTES``. Slots whose count exceeds this are
            truncated to the cap with a silent drop (matches the
            on-tile ``goto done`` short-circuit in
            ``tile_a_filter.cc``).
        windows_per_slot: ``WINDOWS_PER_CHUNK`` for the kernel that
            produced the blob (default ``WINDOWS_PER_CHUNK = 64``).

    Returns:
        list[SparseHit] in slot-major / window-major / guide-major
        order — same order :func:`_host_emulate_match_and_emit` emits
        on the same input, so the two are byte-equal after
        ``bionpu.data.canonical_sites.normalize_file``.
    """
    if slot_bytes is None:
        slot_bytes = infer_slot_bytes_from_blob(len(blob), n_slots)
    if slot_records is None:
        slot_records = max(0, (slot_bytes - 4) // EMIT_RECORD_BYTES)
    out: list[SparseHit] = []
    for s in range(n_slots):
        off = s * slot_bytes
        if off + 4 > len(blob):
            break
        n = struct.unpack_from("<I", blob, off)[0]
        # Cap at slot capacity to mirror the kernel's `goto done` when
        # the ring slot fills.  In practice the kernel writes only
        # `min(n_records, slot_records)` records before short-
        # circuiting; the count header may report the unbounded value
        # depending on where the short-circuit hit relative to the
        # `n_records++`.  Truncating the read matches the records
        # actually present in the slot.
        capped = min(int(n), slot_records)
        slot_base = s * windows_per_slot
        rec_base = off + 4
        for i in range(capped):
            roff = rec_base + i * EMIT_RECORD_BYTES
            wnd_lo, wnd_hi, g, mm = struct.unpack_from("<BBBB", blob, roff)
            wnd = (wnd_lo | (wnd_hi << 8)) + slot_base
            out.append(SparseHit(window_idx=wnd, guide_idx=g, mismatches=mm))
    return out

# --------------------------------------------------------------------------- #
# Host-emulation match: identical to 's _cpu_mismatch_count_matrix
# but operates on (guides_2bit, spacer_bytes_only) extracted from the
# 6-byte windows_in records. Used as the "no NPU" fallback path AND as
# the load-bearing math kernel — produces byte-equal output to the AIE
# kernel by construction (both run the same XOR + popcount arithmetic).
# --------------------------------------------------------------------------- #

def _host_emulate_match_and_emit(
    *,
    guides_2bit: np.ndarray,
    windows_in: np.ndarray,
    pam_template: str,
    max_mismatches: int,
    filter_early: bool,
) -> list[SparseHit]:
    """Run the PAM-filter + match + threshold + sparse-emit pipeline on host.

    This path produces byte-identical output to the on-NPU pipeline by
    construction: it executes the exact same arithmetic the C++ kernel
    does (XOR + 2-bit-pair popcount + 5-byte sum, NGG check, threshold,
    record emit). It is used:

    1. as the CPU fallback when NPU artifacts are not vendored;
    2. as the byte-equality reference inside tests (so the test
       passes whether or not the user has NPU bring-up complete).

    Args:
        guides_2bit: shape ``(N_GUIDES, SPACER_BYTES)`` uint8.
        windows_in: shape ``(N_WINDOWS, WINDOW_BYTES_IN)`` uint8.
        pam_template: ``"NGG"`` (only NGG is implemented; IUPAC is
            future work — see DESIGN.md §8).
        max_mismatches: threshold (Cas-OFFinder convention: inclusive).
        filter_early: True for the production filter-early path; False
            for filter-late. Output is byte-identical either way; the
            arg is retained for parity with the NPU runner so the test
            harness can assert that.

    Returns:
        list[SparseHit] in window-major order, then guide-major within
        a window. (The sort key is documented at the call site; the
        canonical-TSV pipeline applies its own ordering downstream so
        this internal order is for diffing only.)
    """
    if pam_template != "NGG":
        raise NotImplementedError(
            f"only NGG PAM is implemented in v1; got {pam_template!r}"
        )

    # Step 1: PAM check per window — fully vectorized.
    # Layout: bits 1:0=pam[0] (N), 3:2=pam[1] (must be G), 5:4=pam[2] (G).
    pam_bytes = windows_in[:, SPACER_BYTES]  # shape (N_WINDOWS,)
    pam_p1 = (pam_bytes >> 2) & 0x3
    pam_p2 = (pam_bytes >> 4) & 0x3
    pam_pass_mask = (pam_p1 == 0x2) & (pam_p2 == 0x2)  # bool (N_WINDOWS,)

    # Step 2: extract spacer bytes only.
    windows_spacer = windows_in[:, :SPACER_BYTES]

    # Step 3: match arithmetic — same as 's _cpu_mismatch_count_matrix.
    g = guides_2bit.astype(np.uint8)            # (N_GUIDES, 5)
    w = windows_spacer.astype(np.uint8)          # (N_WINDOWS, 5)
    xor = g[:, None, :] ^ w[None, :, :]          # (N_GUIDES, N_WINDOWS, 5)
    m = ((xor | (xor >> 1)) & 0x55).astype(np.uint8)
    c = (m & 0x55) + ((m >> 1) & 0x55)
    c = (c & 0x33) + ((c >> 2) & 0x33)
    c = (c & 0x0F) + ((c >> 4) & 0x0F)
    mismatch_matrix = c.sum(axis=2).astype(np.uint8)  # (N_GUIDES, N_WINDOWS)

    # Step 4: threshold + sparse-emit — fully vectorized.
    # Both filter-early and filter-late produce identical sparse
    # output: filter-early gates at the input, filter-late gates at
    # Tile Z; the surviving records are the same.
    threshold_mask = mismatch_matrix <= max_mismatches      # (N_GUIDES, N_WINDOWS)
    keep_mask = threshold_mask & pam_pass_mask[None, :]      # (N_GUIDES, N_WINDOWS)
    g_idx, w_idx = np.where(keep_mask)
    mm = mismatch_matrix[g_idx, w_idx]
    # Sort window-major (matches the on-tile emit order: outer loop is
    # window, inner loop is guide). lexsort: secondary key first.
    order = np.lexsort((g_idx, w_idx))
    g_idx_sorted = g_idx[order]
    w_idx_sorted = w_idx[order]
    mm_sorted = mm[order]
    return [
        SparseHit(
            window_idx=int(w_idx_sorted[i]),
            guide_idx=int(g_idx_sorted[i]),
            mismatches=int(mm_sorted[i]),
        )
        for i in range(g_idx_sorted.shape[0])
    ]

# --------------------------------------------------------------------------- #
# NpuOp registration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _RunInfo:
    """Timing + provenance for the last NPU run."""

    avg_us: float
    min_us: float
    max_us: float
    n_iters: int
    used_npu: bool
    n_records: int = 0
    n_windows_pam_pass: int = 0  # filter-early: count of windows that
                                  # cleared Tile A's PAM check
    n_windows_total: int = N_WINDOWS  # always N_WINDOWS in v1

def _parse_us(stdout: str, label: str) -> float | None:
    import re

    # Accept both decimal (240123.5) and scientific-notation (6.14401e+06)
    # forms — XRT/iostream defaults flip between them depending on
    # magnitude and locale.
    m = re.search(
        rf"^{label} NPU time:\s*([0-9.eE+-]+)us\.?\s*$",
        stdout,
        re.MULTILINE,
    )
    return float(m.group(1)) if m else None

class _CrisprPamFilterBase(NpuOp):
    """Shared driver for the filter-early and filter-late ops."""

    name: str = ""  # subclasses set
    artifact_dir: Path = Path()  # subclasses set
    filter_early: bool = True  # subclasses set

    def __init__(self) -> None:
        self.last_run: _RunInfo | None = None

    @property
    def _xclbin(self) -> Path:
        return self.artifact_dir / "final.xclbin"

    @property
    def _insts(self) -> Path:
        return self.artifact_dir / "insts.bin"

    @property
    def _binary(self) -> Path:
        return self.artifact_dir / "host_runner"

    def artifacts_present(self) -> bool:
        """True iff all three NPU artifacts are on disk."""
        return all(p.exists() for p in (self._xclbin, self._insts, self._binary))

    def _batch_artifacts(self) -> tuple[Path, Path, Path, int]:
        """Return xclbin, insts, runner, and public chunks per launch."""
        requested = _requested_launch_chunks()
        if requested > 1:
            wide_dir = self.artifact_dir.with_name(
                f"{self.artifact_dir.name}_wide{requested}"
            )
            xclbin = wide_dir / "final.xclbin"
            insts = wide_dir / "insts.bin"
            binary = wide_dir / "host_runner"
            if all(p.exists() for p in (xclbin, insts, binary)):
                return xclbin, insts, binary, requested
        return self._xclbin, self._insts, self._binary, 1

    @staticmethod
    def _validate_inputs(
        guides_2bit: np.ndarray, windows_in: np.ndarray
    ) -> None:
        if not isinstance(guides_2bit, np.ndarray):
            raise TypeError("guides_2bit must be a numpy ndarray")
        if not isinstance(windows_in, np.ndarray):
            raise TypeError("windows_in must be a numpy ndarray")
        if guides_2bit.shape != (N_GUIDES, SPACER_BYTES):
            raise ValueError(
                f"guides_2bit must be shape ({N_GUIDES}, {SPACER_BYTES}); "
                f"got {guides_2bit.shape}"
            )
        if windows_in.shape != (N_WINDOWS, WINDOW_BYTES_IN):
            raise ValueError(
                f"windows_in must be shape ({N_WINDOWS}, {WINDOW_BYTES_IN}); "
                f"got {windows_in.shape}"
            )
        if guides_2bit.dtype != np.uint8:
            raise ValueError(
                f"guides_2bit dtype must be uint8; got {guides_2bit.dtype}"
            )
        if windows_in.dtype != np.uint8:
            raise ValueError(
                f"windows_in dtype must be uint8; got {windows_in.dtype}"
            )

    def _run_npu(
        self,
        guides_2bit: np.ndarray,
        windows_in: np.ndarray,
        max_mismatches: int,
        n_iters: int,
        warmup: int,
    ) -> tuple[bytes, _RunInfo]:
        """Invoke the precompiled host binary and return (sparse_blob, timing)."""
        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as tdir:
            t = Path(tdir)
            guides_bin = t / "guides.bin"
            windows_bin = t / "windows.bin"
            out_bin = t / "out.bin"
            guides_bin.write_bytes(np.ascontiguousarray(guides_2bit).tobytes())
            windows_bin.write_bytes(np.ascontiguousarray(windows_in).tobytes())

            cmd = [
                str(self._binary),
                "-x", str(self._xclbin),
                "-i", str(self._insts),
                "-k", _KERNEL_NAME,
                "--max-mm", str(int(max_mismatches)),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
                "--guides", str(guides_bin),
                "--windows", str(windows_bin),
                "--out", str(out_bin),
            ]
            proc = subprocess.run(  # noqa: S603 — argv strictly controlled
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=120.0,
                env=_xrt_env(),
            )
            if proc.returncode != 0:
                raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker not found",
                )
            avg = _parse_us(proc.stdout, "Avg")
            mn = _parse_us(proc.stdout, "Min")
            mx = _parse_us(proc.stdout, "Max")
            if avg is None or mn is None or mx is None:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] could not parse NPU timing",
                )
            blob = out_bin.read_bytes()
            # n_records: sum of per-slot counts ( fix; the
            # blob is laid out as N_CHUNKS back-to-back length-
            # prefixed slots, NOT as one monolithic length-prefixed
            # stream). The actual slot byte size is inferred from the
            # blob length so we accept both the pre-2026-04-26 (256
            # records / 2048 bytes) and post-2026-04-26 (1024 records /
            # 8192 bytes) vendored artifacts; see
            # :func:`infer_slot_bytes_from_blob`.
            actual_slot_bytes = infer_slot_bytes_from_blob(len(blob), N_CHUNKS)
            actual_slot_records = max(0, (actual_slot_bytes - 4) // EMIT_RECORD_BYTES)
            n_recs = 0
            for s in range(N_CHUNKS):
                off = s * actual_slot_bytes
                if off + 4 > len(blob):
                    break
                slot_n = struct.unpack_from("<I", blob, off)[0]
                n_recs += min(int(slot_n), actual_slot_records)
            info = _RunInfo(
                avg_us=avg, min_us=mn, max_us=mx, n_iters=n_iters,
                used_npu=True, n_records=n_recs,
            )
            return blob, info

    def _run_npu_batch(
        self,
        guides_2bit: np.ndarray,
        windows_batch: Sequence[np.ndarray],
        max_mismatches: int,
    ) -> tuple[list[bytes], _RunInfo]:
        """Invoke one host_runner process for many 4096-window chunks."""
        if not windows_batch:
            return [], _RunInfo(
                avg_us=0.0,
                min_us=0.0,
                max_us=0.0,
                n_iters=0,
                used_npu=True,
                n_records=0,
            )
        for windows_in in windows_batch:
            self._validate_inputs(guides_2bit, windows_in)

        with tempfile.TemporaryDirectory(prefix=f"{self.name}_batch_") as tdir:
            t = Path(tdir)
            guides_bin = t / "guides.bin"
            windows_bin = t / "windows_batch.bin"
            out_bin = t / "out_batch.bin"
            guides_bin.write_bytes(np.ascontiguousarray(guides_2bit).tobytes())
            windows_blob = b"".join(
                np.ascontiguousarray(w).tobytes() for w in windows_batch
            )
            windows_bin.write_bytes(windows_blob)

            xclbin, insts, binary, launch_chunks = self._batch_artifacts()
            cmd = [
                str(binary),
                "-x", str(xclbin),
                "-i", str(insts),
                "-k", _KERNEL_NAME,
                "--max-mm", str(int(max_mismatches)),
                "--windows-batch", str(windows_bin),
                "--out-batch", str(out_bin),
                "--chunks", str(len(windows_batch)),
                "--launch-chunks", str(launch_chunks),
                "--pipeline-depth", str(_pipeline_depth()),
                "--guides", str(guides_bin),
            ]
            proc = subprocess.run(  # noqa: S603 — argv strictly controlled
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=max(120.0, 5.0 + 2.0 * len(windows_batch)),
                env=_xrt_env(),
            )
            if proc.returncode != 0:
                raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker not found",
                )
            avg = _parse_us(proc.stdout, "Avg")
            mn = _parse_us(proc.stdout, "Min")
            mx = _parse_us(proc.stdout, "Max")
            if avg is None or mn is None or mx is None:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] could not parse NPU timing",
                )
            blob = out_bin.read_bytes()
            # Slot byte size is determined by the vendored host_runner /
            # xclbin pair (256-record / 2048-byte for pre-2026-04-26
            # artifacts; 1024-record / 8192-byte for post-2026-04-26
            # rebuilds). Infer from the on-disk length so the host stays
            # compatible with both — the runner allocates a fixed-size
            # SPARSE_OUT_VOL per launch and writes exactly that.
            n_launches = len(windows_batch)
            if blob and (len(blob) % (n_launches * N_CHUNKS) != 0):
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr
                    + f"\n[bionpu] batch output bytes {len(blob)} not "
                    f"divisible by n_launches*N_CHUNKS = "
                    f"{n_launches}*{N_CHUNKS}",
                )
            actual_slot_bytes = (
                len(blob) // (n_launches * N_CHUNKS) if n_launches else 0
            )
            actual_slot_records = max(
                0, (actual_slot_bytes - 4) // EMIT_RECORD_BYTES
            )
            chunk_bytes = N_CHUNKS * actual_slot_bytes
            chunks = [
                blob[i * chunk_bytes : (i + 1) * chunk_bytes]
                for i in range(n_launches)
            ]
            n_recs = 0
            for chunk_blob in chunks:
                for s in range(N_CHUNKS):
                    off = s * actual_slot_bytes
                    slot_n = struct.unpack_from("<I", chunk_blob, off)[0]
                    n_recs += min(int(slot_n), actual_slot_records)
            info = _RunInfo(
                avg_us=avg,
                min_us=mn,
                max_us=mx,
                n_iters=len(windows_batch),
                used_npu=True,
                n_records=n_recs,
            )
            return chunks, info

    def run_batch(
        self,
        *,
        guides_2bit: np.ndarray,
        windows_batch: Sequence[np.ndarray],
        max_mismatches: int = 4,
        force_host: bool = False,
    ) -> list[list[SparseHit]]:
        """Run many chunks through this op, preserving per-chunk hit lists."""
        # See `__call__` for rationale: the vendored xclbin hardcodes the
        # mismatch threshold; widen-via-host fall back when the user asks
        # for more than the kernel can emit.
        kernel_threshold_too_low = max_mismatches > KERNEL_HARDCODED_MAX_MISMATCHES
        if force_host or not self.artifacts_present() or kernel_threshold_too_low:
            return [
                self(
                    guides_2bit=guides_2bit,
                    windows_in=windows_in,
                    max_mismatches=max_mismatches,
                    force_host=True,
                )
                for windows_in in windows_batch
            ]

        blobs, info = self._run_npu_batch(
            guides_2bit, windows_batch, max_mismatches
        )
        out = [decode_per_slot_sparse_buffer(blob) for blob in blobs]
        if max_mismatches < KERNEL_HARDCODED_MAX_MISMATCHES:
            out = [
                [h for h in hits if h.mismatches <= max_mismatches]
                for hits in out
            ]
        n_pam_pass = 0
        for windows_in in windows_batch:
            n_pam_pass += sum(
                1 for w in range(N_WINDOWS)
                if pam_matches_ngg(int(windows_in[w, SPACER_BYTES]))
            )
        self.last_run = _RunInfo(
            avg_us=info.avg_us,
            min_us=info.min_us,
            max_us=info.max_us,
            n_iters=info.n_iters,
            used_npu=True,
            n_records=sum(len(hits) for hits in out),
            n_windows_pam_pass=n_pam_pass,
            n_windows_total=N_WINDOWS * len(windows_batch),
        )
        return out

    def __call__(
        self,
        *,
        guides_2bit: np.ndarray,
        windows_in: np.ndarray,
        max_mismatches: int = 4,
        n_iters: int = 1,
        warmup: int = 0,
        force_host: bool = False,
        **_unused: Any,
    ) -> list[SparseHit]:
        self._validate_inputs(guides_2bit, windows_in)

        # The vendored xclbin compiles ``max_mismatches = 4`` into the
        # runtime sequence (see ``pam_filter.py`` Tile-Z body). The
        # silicon path therefore cannot widen the threshold beyond
        # KERNEL_HARDCODED_MAX_MISMATCHES; for callers requesting a
        # wider threshold we transparently fall back to the host
        # emulator (which produces byte-equal output to silicon at the
        # kernel-supported threshold by construction).
        kernel_threshold_too_low = max_mismatches > KERNEL_HARDCODED_MAX_MISMATCHES

        if force_host or not self.artifacts_present() or kernel_threshold_too_low:
            # Host-emulation path. Produces byte-equal output to the NPU
            # path by construction (same arithmetic). Used both as fallback
            # and as the byte-equality reference for tests.
            pam_bytes = windows_in[:, SPACER_BYTES]
            n_pam_pass = int(
                (
                    (((pam_bytes >> 2) & 0x3) == 0x2)
                    & (((pam_bytes >> 4) & 0x3) == 0x2)
                ).sum()
            )
            hits = _host_emulate_match_and_emit(
                guides_2bit=guides_2bit,
                windows_in=windows_in,
                pam_template="NGG",
                max_mismatches=max_mismatches,
                filter_early=self.filter_early,
            )
            self.last_run = _RunInfo(
                avg_us=0.0,
                min_us=0.0,
                max_us=0.0,
                n_iters=n_iters,
                used_npu=False,
                n_records=len(hits),
                n_windows_pam_pass=n_pam_pass,
            )
            return hits

        blob, info = self._run_npu(
            guides_2bit, windows_in, max_mismatches, n_iters, warmup
        )
        # fix: the on-disk blob is per-slot length-prefixed
        # (N_CHUNKS back-to-back 2048-byte slots, each with its own
        # 4-byte count + records + zero-pad), not a monolithic length-
        # prefixed stream.  Use the per-slot decoder which also
        # reapplies the slot_index * WINDOWS_PER_CHUNK offset to the
        # window index that the kernel writes as a chunk-local 0..63.
        hits = decode_per_slot_sparse_buffer(blob)
        # Apply user-supplied threshold on top of the kernel's hardcoded
        # threshold. The kernel emits records with mismatch <= 4; if the
        # caller requested a stricter threshold (mm <= 0/1/2/3), drop
        # records above it. (For mm > 4 we already redirected to the
        # host emulator above.)
        if max_mismatches < KERNEL_HARDCODED_MAX_MISMATCHES:
            hits = [h for h in hits if h.mismatches <= max_mismatches]
        # Refresh the last_run with PAM-pass count (also derivable from
        # the input bytes; we keep it for symmetry with the host path).
        n_pam_pass = sum(
            1 for w in range(N_WINDOWS)
            if pam_matches_ngg(int(windows_in[w, SPACER_BYTES]))
        )
        self.last_run = _RunInfo(
            avg_us=info.avg_us,
            min_us=info.min_us,
            max_us=info.max_us,
            n_iters=info.n_iters,
            used_npu=True,
            n_records=len(hits),
            n_windows_pam_pass=n_pam_pass,
        )
        return hits

class CrisprPamFilterEarly(_CrisprPamFilterBase):
    """``crispr_pam_filter_early`` — production path.

    On-tile NGG PAM check at Tile A drops PAM-failing windows before the
    match tiles compute. Tile Z applies threshold + emits sparse records.

    See module docstring for the full input/output contract.
    """

    name = "crispr_pam_filter_early"
    artifact_dir = _EARLY_DIR
    filter_early = True

class CrisprPamFilterLate(_CrisprPamFilterBase):
    """``crispr_pam_filter_late`` — comparison artifact.

    Tile A passes every window through. Tile Z does both the PAM check
    AND the threshold + emit. Output bytes are identical to filter-early
    on the same input.
    """

    name = "crispr_pam_filter_late"
    artifact_dir = _LATE_DIR
    filter_early = False

# Register at import time.
register_npu_op("crispr_pam_filter_early", CrisprPamFilterEarly())
register_npu_op("crispr_pam_filter_late", CrisprPamFilterLate())
