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

"""Pure-CPU CRISPR off-target scan.

A self-contained numpy implementation of CRISPR off-target search
that doesn't require NPU silicon. Drives :func:`bionpu.cli` 's
``bionpu scan`` subcommand for the v0.1 release.

The output is byte-equal to a Cas-OFFinder run with the same input,
modulo Cas-OFFinder's row-order non-determinism, which is
canonicalised by :mod:`bionpu.data.canonical_sites`. This module's
output passes through ``bionpu.data.canonical_sites.normalize`` before
being written so the byte-equality contract is honoured by
construction.

Limitations
-----------

- Only the NGG PAM template is implemented. IUPAC ambiguity codes
  are future scope.
- No DNA / RNA bulges. Only mismatches.
- Single-threaded numpy. For full-genome scans (3 Gbp), use the NPU
  path (v0.2 scope) or the C++ Cas-OFFinder reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data.canonical_sites import CasOFFinderRow

__all__ = [
    "GuideSpec",
    "WindowChunk",
    "build_chunks",
    "cpu_scan",
    "encode_guide_batch",
    "hits_to_canonical_rows",
    "npu_scan",
    "parse_guides",
    "read_fasta",
    "reverse_complement",
]


# Two-bit packing: A=0, C=1, G=2, T=3
_BASE_TO_CODE = np.full(256, 4, dtype=np.uint8)  # 4 = "non-ACGT" sentinel
for c, code in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _BASE_TO_CODE[ord(c)] = code
    _BASE_TO_CODE[ord(c.lower())] = code

# Reverse-complement table (same 0..3 codes)
_RC_CODE = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


@dataclass(frozen=True)
class GuideSpec:
    """One guide in the input list — 20 nt ACGT spacer plus an optional ID."""

    spacer: str    # 20-nt ACGT
    guide_id: str  # caller-supplied or auto-generated


def parse_guides(arg: str) -> list[GuideSpec]:
    """Parse the ``--guides`` argument: comma-separated string OR file path.

    File format: one spacer per line, optionally ``id:spacer``. Lines
    starting with ``#`` are comments. All spacers must be 20 nt of ACGT.
    """
    text: str
    if Path(arg).is_file():
        text = Path(arg).read_text()
    else:
        text = arg.replace(",", "\n")
    out: list[GuideSpec] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            gid, spacer = line.split(":", 1)
            gid = gid.strip()
            spacer = spacer.strip()
        else:
            spacer = line
            gid = ""
        spacer = spacer.upper()
        if len(spacer) != 20 or any(c not in "ACGT" for c in spacer):
            raise ValueError(
                f"guide spacer must be 20 nt of ACGT; got {spacer!r}"
            )
        if not gid:
            gid = spacer  # fall back to the spacer itself as ID
        out.append(GuideSpec(spacer=spacer, guide_id=gid))
    if not out:
        raise ValueError(f"no valid guide spacers found in {arg!r}")
    return out


def read_fasta(path: str | Path) -> tuple[str, str]:
    """Read a single-record FASTA. Returns ``(chrom_name, sequence)``.

    The sequence is upper-cased ACGT; non-ACGT bases (N, IUPAC) are
    preserved as-is so downstream non-ACGT detection works.
    """
    p = Path(path)
    chrom = ""
    seq_parts: list[str] = []
    for line in p.read_text().splitlines():
        if line.startswith(">"):
            if chrom:
                # Multi-record FASTA — only first record is consumed.
                break
            chrom = line[1:].split()[0]
        else:
            seq_parts.append(line.strip())
    return chrom, "".join(seq_parts).upper()


def _encode_seq(seq: str) -> np.ndarray:
    """Encode a DNA string as a uint8 array of base codes."""
    return _BASE_TO_CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def _decode_window(arr: np.ndarray) -> str:
    """Inverse of _encode_seq for a single window (codes back to ACGT)."""
    table = np.array([ord("A"), ord("C"), ord("G"), ord("T"), ord("N")],
                     dtype=np.uint8)
    return table[arr].tobytes().decode("ascii")


def _rc(arr: np.ndarray) -> np.ndarray:
    """Reverse-complement a coded array."""
    return _RC_CODE[arr][::-1]


def cpu_scan(
    *,
    chrom: str,
    seq: str,
    guides: Sequence[GuideSpec],
    pam_template: str = "NGG",
    max_mismatches: int = 4,
) -> list[CasOFFinderRow]:
    """Pure-CPU CRISPR off-target scan.

    Args:
        chrom: Chromosome / contig name (recorded verbatim in output rows).
        seq: ACGT[N] sequence to scan.
        guides: Guide list to search for.
        pam_template: PAM motif at the 3' end (only ``"NGG"`` supported).
        max_mismatches: Maximum allowed mismatches in the 20-nt spacer
            region (PAM mismatches always count as 0 in the output's
            ``mismatches`` field, matching Cas-OFFinder's convention).

    Returns:
        List of :class:`bionpu.data.canonical_sites.CasOFFinderRow` —
        one per match. Caller should pass through
        :func:`bionpu.data.canonical_sites.normalize` for byte-equal
        output.
    """
    if pam_template != "NGG":
        raise NotImplementedError(
            f"only NGG PAM is supported; got {pam_template!r}. IUPAC "
            f"ambiguity codes are future scope."
        )

    coded = _encode_seq(seq)
    n = coded.size
    if n < 23:
        return []

    rows: list[CasOFFinderRow] = []
    for g in guides:
        spacer_codes = _encode_seq(g.spacer)
        rc_spacer_codes = _rc(spacer_codes)

        # Forward strand: window at [s, s+23). Spacer = [s, s+20). PAM = [s+20, s+23).
        # PAM is NGG → seq[s+21] == G AND seq[s+22] == G.
        # Mismatches: count spacer_codes != coded[s:s+20].
        positions = np.arange(0, n - 23 + 1)
        windows = np.lib.stride_tricks.sliding_window_view(coded, 23)
        # PAM check: positions where seq[s+21] == G (code 2) and seq[s+22] == G.
        pam_ok_fwd = (windows[:, 21] == 2) & (windows[:, 22] == 2)
        if pam_ok_fwd.any():
            spacer_diffs = windows[pam_ok_fwd, :20] != spacer_codes
            mismatch_counts = spacer_diffs.sum(axis=1)
            keep = mismatch_counts <= max_mismatches
            kept_positions = positions[pam_ok_fwd][keep]
            kept_mismatches = mismatch_counts[keep]
            kept_windows = windows[pam_ok_fwd][keep]
            for s, mm, w in zip(kept_positions, kept_mismatches, kept_windows):
                if (w[:20] == 4).any() or (w[20:23] == 4).any():
                    continue  # non-ACGT in window — Cas-OFFinder skips
                spacer_genome = _decode_window(w[:20])
                pam_genome = _decode_window(w[20:23])
                rows.append(
                    CasOFFinderRow(
                        guide_id=g.guide_id,
                        bulge_type="X",
                        crrna=g.spacer + "NGG",
                        dna=_format_dna(spacer_genome, g.spacer, pam_genome),
                        chrom=chrom,
                        start=int(s),
                        strand="+",
                        mismatches=int(mm),
                        bulge_size=0,
                    )
                )

        # Reverse strand: same windows, but compare RC(spacer) to seq[s+3:s+23]
        # and require PAM CCN at seq[s:s+3] (CC then anything, since reverse
        # strand's NGG is forward strand's CCN).
        # PAM check: seq[s] == C (code 1) AND seq[s+1] == C.
        pam_ok_rev = (windows[:, 0] == 1) & (windows[:, 1] == 1)
        if pam_ok_rev.any():
            spacer_diffs = windows[pam_ok_rev, 3:23] != rc_spacer_codes
            mismatch_counts = spacer_diffs.sum(axis=1)
            keep = mismatch_counts <= max_mismatches
            kept_positions = positions[pam_ok_rev][keep]
            kept_mismatches = mismatch_counts[keep]
            kept_windows = windows[pam_ok_rev][keep]
            for s, mm, w in zip(kept_positions, kept_mismatches, kept_windows):
                if (w[:23] == 4).any():
                    continue
                # Cas-OFFinder convention: dna for a reverse-strand hit
                # is the reverse-complement 23-mer (spacer + PAM in the
                # orientation matching the guide), with mismatches in
                # the spacer region lowercased.
                rc_w = _rc(w[:23])
                spacer_rc = _decode_window(rc_w[:20])
                pam_rc = _decode_window(rc_w[20:23])
                rows.append(
                    CasOFFinderRow(
                        guide_id=g.guide_id,
                        bulge_type="X",
                        crrna=g.spacer + "NGG",
                        dna=_format_dna(spacer_rc, g.spacer, pam_rc),
                        chrom=chrom,
                        start=int(s),
                        strand="-",
                        mismatches=int(mm),
                        bulge_size=0,
                    )
                )

    return rows


# ---------------------------------------------------------------------------
# NPU scan path — kernel-driven via bionpu.kernels.crispr.pam_filter
# ---------------------------------------------------------------------------
#
# The NPU path packs the input target sequence into the window-record layout
# the PAM-filter kernel expects (5 spacer bytes + 1 PAM byte per window),
# splits the resulting array into `N_WINDOWS`-sized chunks (one per kernel
# launch), dispatches the PAM-filter+match+threshold+sparse-emit op via
# `bionpu.kernels.crispr.pam_filter.CrisprPamFilterEarly`, and decodes the
# returned `SparseHit` records into `CasOFFinderRow`s using the chunk's
# window-position lookup table.

# 2-bit lookup table indexed by ASCII code. Non-ACGT positions get 0xFF so
# the per-window validity check is a single sum-test.
_ASCII_TO_2BIT = np.full(256, 0xFF, dtype=np.uint8)
for _c, _code in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _ASCII_TO_2BIT[ord(_c)] = _code
    _ASCII_TO_2BIT[ord(_c.lower())] = _code

# DNA reverse-complement translation table.
_RC_TABLE = str.maketrans(
    "ACGTacgtNn",
    "TGCAtgcaNn",
)


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of ``seq``.

    Handles upper- and lower-case A/C/G/T and the ambiguity code ``N``.
    Other characters pass through unchanged so a downstream consumer's
    error surfaces rather than producing a silent corrupt complement.
    """
    return seq.translate(_RC_TABLE)[::-1]


@dataclass(frozen=True)
class WindowChunk:
    """One ``N_WINDOWS``-sized chunk handed to a single NPU launch.

    Attributes:
        windows_in: shape ``(N_WINDOWS, WINDOW_BYTES_IN)`` uint8 — what
            Tile A consumes (5 spacer + 1 PAM byte per record).
        positions: list of ``N_WINDOWS`` 0-based forward-strand positions
            that each window slot maps to. For ``-`` strand chunks the
            position is the forward-strand 0-based start of the
            same-genomic-window. A sentinel ``-1`` means "padding —
            ignore in the post-hoc map".
        strand: ``"+"`` or ``"-"``.
        chrom: contig label (stamped into the canonical TSV).
        n_real_windows: number of valid windows in this chunk; the rest
            of the chunk is zero-padded (PAM-failing) records.
    """

    windows_in: np.ndarray
    positions: list[int]
    strand: str
    chrom: str
    n_real_windows: int


def _vectorized_pack_windows(
    seq_str: str,
    *,
    spacer_len: int,
    pam_len: int,
    spacer_bytes: int,
    window_bytes_in: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised 23-byte window builder.

    Returns:
        records: ``(n_valid, window_bytes_in)`` uint8, one row per valid
            all-ACGT 23-nt window.
        positions: ``(n_valid,)`` int64 — 0-based start in ``seq_str``.

    The genome is converted to a uint8 ASCII view; non-ACGT positions
    are flagged with 0xFF in a 2-bit map. A sliding-window validity mask
    is computed in O(N) by cumulative-summing the sentinel flags. Valid
    windows then have their codes extracted and packed into 5 spacer
    bytes + 1 PAM byte using numpy bit ops. For chr22 (~50 M positions)
    this completes in seconds instead of the minutes a per-position
    Python loop takes.
    """
    window_len = spacer_len + pam_len
    seq_u8 = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
    n = seq_u8.shape[0]
    if n < window_len:
        return (
            np.zeros((0, window_bytes_in), dtype=np.uint8),
            np.zeros(0, dtype=np.int64),
        )
    codes = _ASCII_TO_2BIT[seq_u8]                      # values in {0,1,2,3,0xFF}
    is_invalid = (codes == 0xFF).astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(is_invalid)))
    n_starts = n - window_len + 1
    invalid_per_window = (
        cs[window_len : window_len + n_starts] - cs[:n_starts]
    )
    valid_mask = invalid_per_window == 0
    valid_starts = np.flatnonzero(valid_mask).astype(np.int64)
    if valid_starts.shape[0] == 0:
        return (
            np.zeros((0, window_bytes_in), dtype=np.uint8),
            np.zeros(0, dtype=np.int64),
        )

    offsets = np.arange(window_len, dtype=np.int64)
    win_codes = codes[valid_starts[:, None] + offsets[None, :]].astype(np.uint8)
    spacer_codes = win_codes[:, :spacer_len]
    pam_codes = win_codes[:, spacer_len:]

    n_valid = win_codes.shape[0]
    spacer_packed = np.zeros((n_valid, spacer_bytes), dtype=np.uint8)
    for i in range(spacer_len):
        byte_idx = i // 4
        bit_off = (i % 4) * 2
        spacer_packed[:, byte_idx] |= (spacer_codes[:, i] & 0x3) << bit_off

    pam_byte = (
        ((pam_codes[:, 0] & 0x3) << 0)
        | ((pam_codes[:, 1] & 0x3) << 2)
        | ((pam_codes[:, 2] & 0x3) << 4)
    ).astype(np.uint8)

    records = np.empty((n_valid, window_bytes_in), dtype=np.uint8)
    records[:, :spacer_bytes] = spacer_packed
    records[:, spacer_bytes] = pam_byte
    return records, valid_starts


def build_chunks(chrom: str, seq: str) -> list[WindowChunk]:
    """Build forward + reverse-strand window chunks from ``seq``.

    Skips windows containing non-ACGT bases. Pads the trailing chunk
    with all-zero records so every chunk is exactly ``N_WINDOWS`` rows
    (the kernel is shape-pinned). Padding rows are flagged with
    ``positions[w] = -1`` so the post-hoc mapper drops them.
    """
    # Late import — avoids a hard dependency on the kernel package at
    # module load time (so `cpu_scan` can be used standalone).
    from .kernels.crispr.pam_filter import (
        N_WINDOWS,
        PAM_LEN,
        SPACER_BYTES,
        SPACER_LEN,
        WINDOW_BYTES_IN,
    )

    seq_upper = seq.upper()
    window_len = SPACER_LEN + PAM_LEN
    if len(seq_upper) < window_len:
        return []

    fwd_records, fwd_starts = _vectorized_pack_windows(
        seq_upper,
        spacer_len=SPACER_LEN,
        pam_len=PAM_LEN,
        spacer_bytes=SPACER_BYTES,
        window_bytes_in=WINDOW_BYTES_IN,
    )
    rc_seq = reverse_complement(seq_upper)
    rev_records, rev_starts = _vectorized_pack_windows(
        rc_seq,
        spacer_len=SPACER_LEN,
        pam_len=PAM_LEN,
        spacer_bytes=SPACER_BYTES,
        window_bytes_in=WINDOW_BYTES_IN,
    )
    rev_positions = (len(seq_upper) - (rev_starts + window_len)).astype(np.int64)

    chunks: list[WindowChunk] = []
    for records, positions_arr, strand in (
        (fwd_records, fwd_starts, "+"),
        (rev_records, rev_positions, "-"),
    ):
        for i in range(0, records.shape[0], N_WINDOWS):
            slab = records[i : i + N_WINDOWS]
            slab_pos = positions_arr[i : i + N_WINDOWS]
            n_real = slab.shape[0]
            if n_real < N_WINDOWS:
                pad_count = N_WINDOWS - n_real
                pad_recs = np.zeros((pad_count, WINDOW_BYTES_IN), dtype=np.uint8)
                slab = np.concatenate([slab, pad_recs], axis=0)
                pad_pos = np.full(pad_count, -1, dtype=np.int64)
                slab_pos = np.concatenate([slab_pos, pad_pos])
            chunks.append(
                WindowChunk(
                    windows_in=np.ascontiguousarray(slab),
                    positions=slab_pos.tolist(),
                    strand=strand,
                    chrom=chrom,
                    n_real_windows=n_real,
                )
            )
    return chunks


def encode_guide_batch(
    guides: Sequence[GuideSpec],
) -> tuple[np.ndarray, list[GuideSpec | None]]:
    """Encode + pad to the kernel's fixed-size guide batch.

    The PAM-filter kernel is shape-pinned to ``N_GUIDES`` guides per
    launch. We accept fewer than ``N_GUIDES`` and pad with copies of
    the last guide; the mapper drops hits whose ``guide_pad_table``
    entry is ``None``.

    Returns:
        ``(guides_2bit, guide_pad_table)``.
    """
    from .kernels.crispr.match_singletile import SPACER_BYTES, encode_2bit
    from .kernels.crispr.pam_filter import N_GUIDES

    if len(guides) == 0:
        raise ValueError("no guides supplied")
    if len(guides) > N_GUIDES:
        raise NotImplementedError(
            f"v0.2 supports at most {N_GUIDES} guides per launch; "
            f"got {len(guides)}. Multi-batch dispatch is future work."
        )
    out = np.zeros((N_GUIDES, SPACER_BYTES), dtype=np.uint8)
    pad_table: list[GuideSpec | None] = []
    for i in range(N_GUIDES):
        if i < len(guides):
            out[i] = encode_2bit(guides[i].spacer)
            pad_table.append(guides[i])
        else:
            out[i] = encode_2bit(guides[-1].spacer)
            pad_table.append(None)
    return out, pad_table


def _format_dna(spacer_genome: str, spacer_guide: str, pam_genome: str) -> str:
    """Spacer mismatches lowercased; PAM verbatim. Mirrors the Cas-OFFinder convention."""
    chars = []
    for g, s in zip(spacer_genome, spacer_guide, strict=True):
        chars.append(g if g == s else g.lower())
    chars.append(pam_genome)
    return "".join(chars)


def hits_to_canonical_rows(
    *,
    hits,
    chunks: list[WindowChunk],
    chunk_index_per_hit: list[int],
    guide_pad_table: list[GuideSpec | None],
    chrom_seq: str,
    pam_template: str = "NGG",
) -> list[CasOFFinderRow]:
    """Map sparse-hit records → canonical TSV rows.

    For each surviving hit:

    - Look up the chunk it came from (``chunk_index_per_hit[i]``) for
      strand / chrom / position table.
    - Look up the forward-strand position
      (``chunk.positions[hit.window_idx]``); a ``-1`` sentinel means
      the hit fell into a padding slot — drop.
    - Look up the original guide spec (``guide_pad_table[hit.guide_idx]``);
      ``None`` means a padding guide — drop.
    - Decode the genomic 23-mer from the forward sequence (``+`` strand)
      or its RC (``-`` strand).
    - Emit a row mirroring Cas-OFFinder's convention (mismatches
      lowercased in the spacer; PAM verbatim).
    """
    from .kernels.crispr.pam_filter import (
        PAM_LEN,
        SPACER_LEN,
        pam_matches_ngg_ascii,
    )

    window_len = SPACER_LEN + PAM_LEN
    rows: list[CasOFFinderRow] = []
    seq = chrom_seq.upper()
    rc_seq: str | None = None
    for i, hit in enumerate(hits):
        ci = chunk_index_per_hit[i]
        chunk = chunks[ci]
        if hit.window_idx >= len(chunk.positions):
            continue
        position = chunk.positions[hit.window_idx]
        if position < 0:
            continue
        guide = guide_pad_table[hit.guide_idx]
        if guide is None:
            continue

        if chunk.strand == "+":
            window = seq[position : position + window_len]
        else:
            if rc_seq is None:
                rc_seq = reverse_complement(seq)
            rc_off = len(seq) - (position + window_len)
            window = rc_seq[rc_off : rc_off + window_len]

        if len(window) < window_len:
            continue
        spacer_genome = window[:SPACER_LEN]
        pam_genome = window[SPACER_LEN:]
        if not pam_matches_ngg_ascii(pam_genome):
            continue

        dna = _format_dna(spacer_genome, guide.spacer, pam_genome)
        rows.append(
            CasOFFinderRow(
                guide_id=guide.guide_id,
                bulge_type="X",
                crrna=guide.spacer + pam_template,
                dna=dna,
                chrom=chunk.chrom,
                start=position,
                strand=chunk.strand,
                mismatches=hit.mismatches,
                bulge_size=0,
            )
        )
    return rows


def npu_scan(
    *,
    chrom: str,
    seq: str,
    guides: Sequence[GuideSpec],
    pam_template: str = "NGG",
    max_mismatches: int = 4,
    op_name: str = "crispr_pam_filter_early",
) -> list[CasOFFinderRow]:
    """Run the NPU CRISPR off-target scan via the PAM-filter kernel.

    Args:
        chrom: Chromosome / contig name (recorded verbatim).
        seq: ACGT[N] sequence to scan.
        guides: Guide list to search for. ≤ ``N_GUIDES`` per launch
            (the kernel is shape-pinned).
        pam_template: PAM motif (``"NGG"`` only).
        max_mismatches: On-tile threshold passed to the kernel.
        op_name: Either ``"crispr_pam_filter_early"`` (production;
            on-tile NGG check at Tile A drops PAM-failing windows
            before the match tiles compute) or
            ``"crispr_pam_filter_late"`` (comparison artifact; same
            output bytes, different work distribution).

    Returns:
        List of :class:`CasOFFinderRow` — caller normalizes via
        :func:`bionpu.data.canonical_sites.normalize` for byte-equal
        TSV output.

    Raises:
        bionpu.dispatch.npu.NpuArtifactsMissingError: kernel artifacts
            not built. Build via ``make NPU2=1`` in
            ``src/bionpu/kernels/crispr/pam_filter/`` and copy the
            produced ``final.xclbin`` / ``insts.bin`` / ``host_runner``
            into the dispatch artifact tree (or set
            ``BIONPU_KERNEL_ARTIFACTS_DIR``).
    """
    if pam_template != "NGG":
        raise NotImplementedError(
            f"only NGG PAM is supported; got {pam_template!r}."
        )

    # Late imports so this function is a no-op cost when only `cpu_scan`
    # is used.
    from .dispatch.npu import lookup_npu_op
    from .kernels.crispr import pam_filter as _pam  # noqa: F401  — registers ops at import

    op = lookup_npu_op(op_name)

    chunks = build_chunks(chrom, seq)
    if not chunks:
        return []
    guides_2bit, guide_pad_table = encode_guide_batch(guides)

    all_hits: list = []
    chunk_index_per_hit: list[int] = []
    for ci, chunk in enumerate(chunks):
        hits = op(
            windows_in=chunk.windows_in,
            guides_2bit=guides_2bit,
            max_mismatches=max_mismatches,
        )
        all_hits.extend(hits)
        chunk_index_per_hit.extend([ci] * len(hits))

    return hits_to_canonical_rows(
        hits=all_hits,
        chunks=chunks,
        chunk_index_per_hit=chunk_index_per_hit,
        guide_pad_table=guide_pad_table,
        chrom_seq=seq,
        pam_template=pam_template,
    )
