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

"""CRISPR PAM filter pktMerge variant.

Retrofits 's filter-early variant using the fork's :class:`PacketFifo`
. Closes the OTHER HALF of
 (Phase 1 documented; Phase 2 closes via PacketFifo).

Architecture target:
    shim ──windows_in (ObjectFifo)── Tile A
                                       │ PacketFifo / direct stream
                                       │ (valid windows only)
                                       ▼
                              match tiles (only see PAM-passing windows)
                                       │ partials ObjectFifo
                                       ▼
                              Tile Z (threshold + sparse-emit) ──→ shim

Key change vs filter-early (the target data-path replacement):
    - v1: Tile A always forwards full chunks; match tiles compute on
      every window, including the ~7/8 that fail NGG (zero-filled).
    - : Tile A emits packets only for PAM-passing windows. The
      packet payload carries original window_idx + spacer bytes, so
      downstream sparse emit preserves global coordinates even though
      invalid windows never reach match tiles.

Predicted: ~7/8 reduction in match-tile cycles, strengthened to ~15/16
for strict NGG. The default dispatch artifact remains the byte-equal
ObjectFifo twin; the opt-in buildable replacement is `DIRECT_STREAM=1`,
which uses Peano AIE2P stream intrinsics rather than ADF stream pointers
or per-packet MLIR stream-op unrolling. Documented in
`results/crispr/c-m5-pktmerge/measurements.json` with verdict.

Output is **byte-identical** to 's filter-early + filter-late after
canonical normalization (the kernel runs the same XOR + popcount + NGG
arithmetic). Determinism: byte-identical across runs.

Op registration:
    ``crispr_pam_filter_pktmerge`` → :class:`CrisprPamFilterPktmerge`

Rebuild:
    cd bionpu/kernels/crispr/pam_filter_pktmerge && make NPU2=1 all
    # direct-stream prototype:
    # make NPU2=1 DIRECT_STREAM=1 build/final.xclbin
    # then vendor build/final.xclbin + insts.bin + host_runner under
    # bionpu/dispatch/_npu_artifacts/crispr_pam_filter_pktmerge/

The default op falls back to the host-emulation byte-equal path the same
way 's variants do when vendored NPU artifacts are unavailable. The
direct-stream xclbin is a promotion/timing candidate, not yet the default
dispatch artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,  # noqa: F401 — re-exported for symmetry
    NpuRunFailed,  # noqa: F401 — re-exported for symmetry
    register_npu_op,
)
from bionpu.kernels.crispr.pam_filter import (
    EMIT_RECORD_BYTES,
    EMIT_SLOT_BYTES,
    EMIT_SLOT_RECORDS,
    GUIDES_PER_TILE,
    N_CHUNKS,
    N_GUIDES,
    N_MATCH_TILES,
    N_WINDOWS,
    PAM_BYTES,
    PAM_LEN,
    SPACER_BYTES,
    SPACER_LEN,
    WINDOW_BYTES_IN,
    WINDOWS_PER_CHUNK,
    SparseHit,
    _CrisprPamFilterBase,
    build_window_record,
    decode_per_slot_sparse_buffer,
    decode_sparse_buffer,
    encode_pam_byte,
    pam_matches_ngg,
    pam_matches_ngg_ascii,
)
from bionpu.kernels.crispr.pam_filter import (
    TILE_MEMORY_BREAKDOWN as _T62_TILE_MEMORY,
)

__all__ = [
    "CrisprPamFilterPktmergeCompactPackets",
    "CrisprPamFilterPktmergeDirectStream",
    "CrisprPamFilterPktmerge",
    "EMIT_RECORD_BYTES",
    "EMIT_SLOT_BYTES",
    "EMIT_SLOT_RECORDS",
    "GUIDES_PER_TILE",
    "HEADER_BYTES",
    "PACKETIZED_WINDOW_INDEX_BYTES",
    "PACKETIZED_WINDOW_PAD_BYTES",
    "PACKETIZED_WINDOW_PAYLOAD_BYTES",
    "N_CHUNKS",
    "N_GUIDES",
    "N_MATCH_TILES",
    "N_WINDOWS",
    "PACKET_ID_VALID",
    "PAM_BYTES",
    "PAM_LEN",
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
    "decode_packetized_valid_windows",
    "host_emulate_packetized_valid_windows",
    "pack_packetized_valid_windows",
    "pktmerge_replacement_contract",
]

# 1-byte packet header (1-bit valid + 7-bit reserved, per AM020 Ch. 2 p. 25
# semantic — though the AXI stream switch routes by 5-bit pkt_id in
# hardware; we use uint8 header_dtype as the canonical filter-early
# documented-tag width).
HEADER_BYTES = 1
PACKET_ID_VALID = 1
PACKET_ID_INVALID = 0

# Replacement PacketFifo ABI. PacketFifo routing ids are assigned per producer
# in the fork, not per packet, so the speedup path cannot route invalid packets
# to a drop sink with PACKET_ID_INVALID. Tile A must simply skip invalid windows.
# The payload keeps the global window index that Tile Z needs for sparse output.
PACKETIZED_WINDOW_INDEX_BYTES = 4
PACKETIZED_WINDOW_PAD_BYTES = 3
PACKETIZED_WINDOW_PAYLOAD_BYTES = (
    PACKETIZED_WINDOW_INDEX_BYTES + SPACER_BYTES + PACKETIZED_WINDOW_PAD_BYTES
)

@dataclass(frozen=True)
class PktmergeReplacementContract:
    """Static contract for the PacketFifo replacement path."""

    packet_id_valid: int
    payload_bytes: int
    index_bytes: int
    spacer_bytes: int
    pad_bytes: int
    strict_ngg_pass_rate: float

    @property
    def match_work_reduction(self) -> float:
        return 1.0 - self.strict_ngg_pass_rate

def pktmerge_replacement_contract() -> PktmergeReplacementContract:
    """Return the corrected PacketFifo ABI contract.

    PacketFifo's current packetflow lowering assigns packet_id per producer.
    Therefore CRISPR filter-early should emit only valid windows, with
    original window_idx in the packet payload, instead of emitting invalid
    packets and expecting fabric-side drop by per-window header.
    """
    return PktmergeReplacementContract(
        packet_id_valid=PACKET_ID_VALID,
        payload_bytes=PACKETIZED_WINDOW_PAYLOAD_BYTES,
        index_bytes=PACKETIZED_WINDOW_INDEX_BYTES,
        spacer_bytes=SPACER_BYTES,
        pad_bytes=PACKETIZED_WINDOW_PAD_BYTES,
        strict_ngg_pass_rate=1.0 / 16.0,
    )

def pack_packetized_valid_windows(
    windows_in: np.ndarray,
    *,
    base_window_idx: int = 0,
) -> np.ndarray:
    """Pack PAM-passing windows into the corrected PacketFifo payload ABI.

    Output is a flat uint8 buffer of 12-byte packets:
    little-endian uint32 original window_idx, 5 spacer bytes, 3 zero pad bytes.
    PAM-failing windows are skipped rather than emitted with a drop header.
    """
    arr = np.asarray(windows_in, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != WINDOW_BYTES_IN:
        raise ValueError(
            f"windows_in must have shape (n, {WINDOW_BYTES_IN}); got {arr.shape}"
        )
    if base_window_idx < 0:
        raise ValueError(f"base_window_idx must be >= 0; got {base_window_idx}")

    packets: list[bytes] = []
    for local_idx, record in enumerate(arr):
        if not pam_matches_ngg(int(record[SPACER_BYTES])):
            continue
        window_idx = base_window_idx + local_idx
        packet = bytearray(PACKETIZED_WINDOW_PAYLOAD_BYTES)
        packet[0:4] = int(window_idx).to_bytes(4, "little", signed=False)
        packet[4 : 4 + SPACER_BYTES] = bytes(record[:SPACER_BYTES])
        packets.append(bytes(packet))
    if not packets:
        return np.zeros((0,), dtype=np.uint8)
    return np.frombuffer(b"".join(packets), dtype=np.uint8).copy()

def decode_packetized_valid_windows(blob: bytes | bytearray | np.ndarray) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Decode the 12-byte PacketFifo payload ABI into indices and spacers."""
    data = np.asarray(blob, dtype=np.uint8).reshape(-1)
    if data.size % PACKETIZED_WINDOW_PAYLOAD_BYTES != 0:
        raise ValueError(
            "packetized valid-window blob length must be a multiple of "
            f"{PACKETIZED_WINDOW_PAYLOAD_BYTES}; got {data.size}"
        )
    n_packets = data.size // PACKETIZED_WINDOW_PAYLOAD_BYTES
    indices = np.zeros((n_packets,), dtype=np.uint32)
    spacers = np.zeros((n_packets, SPACER_BYTES), dtype=np.uint8)
    for i in range(n_packets):
        off = i * PACKETIZED_WINDOW_PAYLOAD_BYTES
        indices[i] = int.from_bytes(bytes(data[off : off + 4]), "little")
        spacers[i] = data[off + 4 : off + 4 + SPACER_BYTES]
        pad = data[
            off + 4 + SPACER_BYTES : off + PACKETIZED_WINDOW_PAYLOAD_BYTES
        ]
        if np.any(pad):
            raise ValueError(f"packet {i} has non-zero pad bytes: {pad.tolist()}")
    return indices, spacers

def host_emulate_packetized_valid_windows(
    *,
    guides_2bit: np.ndarray,
    windows_in: np.ndarray,
    max_mismatches: int,
    base_window_idx: int = 0,
) -> list[SparseHit]:
    """Host reference for the corrected PacketFifo replacement path.

    This runs the same math as the existing host fallback, but the
    intermediate representation is the variable-rate PacketFifo payload:
    only PAM-passing windows are packed, matched, and threshold-emitted.
    """
    guides = np.asarray(guides_2bit, dtype=np.uint8)
    if guides.shape != (N_GUIDES, SPACER_BYTES):
        raise ValueError(
            f"guides_2bit must have shape ({N_GUIDES}, {SPACER_BYTES}); "
            f"got {guides.shape}"
        )
    packet_blob = pack_packetized_valid_windows(
        windows_in,
        base_window_idx=base_window_idx,
    )
    window_indices, spacers = decode_packetized_valid_windows(packet_blob)
    if spacers.shape[0] == 0:
        return []

    xor = guides[:, None, :] ^ spacers[None, :, :]
    pair_diff = ((xor | (xor >> 1)) & 0x55).astype(np.uint8)
    mismatch_counts = np.unpackbits(pair_diff, axis=2).sum(axis=2)

    out: list[SparseHit] = []
    for packet_idx, window_idx in enumerate(window_indices):
        for guide_idx in range(N_GUIDES):
            mismatches = int(mismatch_counts[guide_idx, packet_idx])
            if mismatches <= max_mismatches:
                out.append(
                    SparseHit(
                        window_idx=int(window_idx),
                        guide_idx=guide_idx,
                        mismatches=mismatches,
                    )
                )
    return out

# Tile-memory accounting — the pktmerge variant adds ~64 B for the
# per-chunk header buffer at Tile A but otherwise mirrors 's
# filter-early. The match tiles' memory drops slightly because they
# no longer need a per-chunk window slot when consuming variable-rate
# packets; for simplicity the v1 lowering keeps the same chunk
# geometry so the comparison to is apples-to-apples.
_TILE_A_HEADER_DBL = 2 * WINDOWS_PER_CHUNK * HEADER_BYTES  # 128 B

#: per-tile breakdown — slightly larger Tile A (header buffer); other
#: tiles unchanged from .
TILE_MEMORY_BREAKDOWN: dict[str, int] = {
    "tile_a_pktmerge": _T62_TILE_MEMORY["tile_a_filter"] + _TILE_A_HEADER_DBL,
    "match_tile": _T62_TILE_MEMORY["match_tile"],
    "tile_z_emit": _T62_TILE_MEMORY["tile_z_emit"],
}

#: peak per-tile memory in bytes (Tile Z stays the worst case).
TILE_MEMORY_BYTES: int = max(TILE_MEMORY_BREAKDOWN.values())

# Sanity: every tile fits the 64 KiB DM cap.
_AIE2P_DM_BUDGET_BYTES = 64 * 1024
assert TILE_MEMORY_BYTES < _AIE2P_DM_BUDGET_BYTES, (
    f"per-tile memory {TILE_MEMORY_BYTES} >= AIE2P DM cap {_AIE2P_DM_BUDGET_BYTES}"
)

# Artifact paths (populated by `make NPU2=1` + the vendor step).
_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)
_PKTMERGE_DIR = _ART_ROOT / "crispr_pam_filter_pktmerge"
_PKTMERGE_DIRECT_STREAM_DIR = _ART_ROOT / "crispr_pam_filter_pktmerge_direct_stream"
_PKTMERGE_COMPACT_PACKETS_DIR = (
    _ART_ROOT / "crispr_pam_filter_pktmerge_compact_packets"
)

class CrisprPamFilterPktmerge(_CrisprPamFilterBase):
    """``crispr_pam_filter_pktmerge`` — PacketFifo retrofit.

    Same input/output contract as ``crispr_pam_filter_early`` /
    ``crispr_pam_filter_late``. Output bytes are byte-identical to
    those variants on the same input (canonical-normalized).

    The difference from 's filter-early is **inter-tile**: Tile A
    emits only PAM-passing windows into a :class:`PacketFifo`. The packet
    payload must carry the original window index plus spacer bytes; match
    tiles should never compute on PAM-failing windows. Cross-walk's strict
    NGG prediction: 15/16 reduction in match-tile window work.

    See module docstring for full topology + verdict cross-reference.
    """

    name = "crispr_pam_filter_pktmerge"
    artifact_dir = _PKTMERGE_DIR
    # filter_early=True so the host-emulation path produces the same
    # PAM-pre-filtered hits 's filter-early would; output is
    # byte-equal to both variants by construction (same arithmetic).
    filter_early = True

class CrisprPamFilterPktmergeDirectStream(CrisprPamFilterPktmerge):
    """``crispr_pam_filter_pktmerge_direct_stream`` timing candidate.

    Same host-visible contract as ``crispr_pam_filter_pktmerge``, but its
    artifact directory is populated from ``make NPU2=1 DIRECT_STREAM=1``.
    Keeping this separate lets silicon tests compare the packetized stream
    replacement against the byte-equal default without overwriting the known
    rollback artifact.
    """

    name = "crispr_pam_filter_pktmerge_direct_stream"
    artifact_dir = _PKTMERGE_DIRECT_STREAM_DIR

class CrisprPamFilterPktmergeCompactPackets(CrisprPamFilterPktmerge):
    """``crispr_pam_filter_pktmerge_compact_packets`` timing candidate.

    Same host-visible contract as ``crispr_pam_filter_pktmerge``, but the
    artifact uses a counted compact-packet ObjectFifo between Tile A and the
    match tiles. This keeps the sparse-window packet payload off the single
    direct stream fanout path while preserving the valid-window compaction.
    """

    name = "crispr_pam_filter_pktmerge_compact_packets"
    artifact_dir = _PKTMERGE_COMPACT_PACKETS_DIR

# Register at import time.
register_npu_op("crispr_pam_filter_pktmerge", CrisprPamFilterPktmerge())
register_npu_op(
    "crispr_pam_filter_pktmerge_direct_stream",
    CrisprPamFilterPktmergeDirectStream(),
)
register_npu_op(
    "crispr_pam_filter_pktmerge_compact_packets",
    CrisprPamFilterPktmergeCompactPackets(),
)
