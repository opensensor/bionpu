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

"""K-mer counting NPU op (3 registry entries; ``n_tiles`` constructor arg).

Per ``state/kmer_count_interface_contract.md`` (T1), this module hosts the
:class:`BionpuKmerCount` op class and registers exactly **three**
``NPU_OPS`` entries — one per supported k in ``{15, 21, 31}``. The single
op class accepts ``n_tiles`` via constructor (valid in ``{1, 2, 4, 8}``)
and selects an artifact directory from the 3 x 4 = 12-cell matrix at
``_npu_artifacts/bionpu_kmer_count_k{k}_n{n_tiles}/``. The
``BIONPU_KMER_COUNT_LAUNCH_CHUNKS`` env-var override is consumed by the
:func:`bionpu.kernels.genomics.get_kmer_count_op` helper (T4); the
registry-resident instances themselves carry the contract default
``n_tiles=4``.

Mirrors ``DoradoFastLinearProjectionFusedPerts`` in
``bionpu.kernels.basecalling.linear_projection`` for the
subprocess-vs-pyxrt dispatch routing pattern, and
``_CrisprPamFilterBase`` in ``bionpu.kernels.crispr.pam_filter`` for the
artifact-directory-per-variant + ``_RunInfo`` + per-slot length-prefixed
sparse-emit ABI.

Wire format (per T1 contract):

* Input ``packed_seq`` is a 1-D ``np.uint8`` array carrying packed-2-bit
  DNA (A=00, C=01, G=10, T=11; first base = bits[7:6] of byte 0,
  MSB-first within each byte).
* The host runner streams the input in 4096-byte chunks with
  ``ceil((k-1)/4)`` bytes of overlap so k-mers spanning chunk boundaries
  are not lost.
* Each per-tile core owns a 4096-bucket open-addressed hash table
  (``canonical_u64 + count_u32``, 12 B/record, 48 KiB/table) with an
  emit-on-evict overflow policy at chain length > 8.
* The aggregator drains all per-tile partial emits into a per-slot
  length-prefixed sparse_out ring buffer (1024 records / 16 KiB per
  slot).
* The host post-pass dedups duplicate ``canonical_u64`` entries by
  summing counts, sorts ``(count desc, canonical asc)``, applies
  ``--top`` and ``--threshold``, and emits Jellyfish-FASTA
  (``>count\\nkmer\\n``).

Out-of-scope for v1 (deferred — see ``gaps.yaml``):

* ntHash-based hashing (current implementation is direct 2-bit packing,
  k <= 32).
* Memtile-resident count tables for collision-rate at chr22 scale.
* AIE2P vector-popcount intrinsics on the rolling-canonical update.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.dispatch._pyxrt_helpers import (
    resolve_dispatch_impl,
    run_pyxrt_with_buffers,
)
from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    _xrt_env,  # noqa: PLC2701 — internal helper; mirrors pam_filter
    register_npu_op,
)

__all__ = [
    "BionpuKmerCount",
    "EMIT_RECORD_BYTES",
    "EMIT_SLOT_BYTES",
    "EMIT_SLOT_RECORDS",
    "EVICT_FLAG",
    "HASH_BUCKETS_PER_TILE",
    "KMER_COUNT_DISPATCH_ENV",
    "KMER_MASK_K15",
    "KMER_MASK_K21",
    "KMER_MASK_K31",
    "MAX_TILES",
    "OVERFLOW_THRESHOLD",
    "SEQ_IN_CHUNK_BYTES_BASE",
    "SUPPORTED_K",
    "SUPPORTED_N_TILES",
    "decode_packed_kmer_to_ascii",
    "kmer_mask_for",
    "overlap_bytes_for",
]

# --------------------------------------------------------------------------- #
# Pinned constants — mirror kmer_count_constants.h (T1 contract, byte-equal).
# --------------------------------------------------------------------------- #

#: Supported k-mer widths (one registry entry per value).
SUPPORTED_K: tuple[int, ...] = (15, 21, 31)

#: Supported tile fan-out (n_tiles constructor arg).
SUPPORTED_N_TILES: tuple[int, ...] = (1, 2, 4, 8)

#: Per-k canonical bit masks (bit-equal to the C header).
KMER_MASK_K15: int = (1 << 30) - 1
KMER_MASK_K21: int = (1 << 42) - 1
KMER_MASK_K31: int = (1 << 62) - 1

_KMER_MASKS: dict[int, int] = {
    15: KMER_MASK_K15,
    21: KMER_MASK_K21,
    31: KMER_MASK_K31,
}

#: 4096 buckets * 12 bytes = 48 KiB primary count table per tile.
HASH_BUCKETS_PER_TILE: int = 4096

#: Linear-probe chain length triggering emit-on-evict.
OVERFLOW_THRESHOLD: int = 8

#: 16 bytes / EmitRecord (canonical_u64 + count_u32 + flags_u32).
EMIT_RECORD_BYTES: int = 16

#: 1024 records / sparse_out slot * 16 = 16 KiB.
EMIT_SLOT_RECORDS: int = 1024
EMIT_SLOT_BYTES: int = EMIT_SLOT_RECORDS * EMIT_RECORD_BYTES  # 16384

#: bit 0 of EmitRecord.flags marks a probe-chain eviction.
EVICT_FLAG: int = 1 << 0

#: 4096-byte primary chunk before per-k overlap padding.
SEQ_IN_CHUNK_BYTES_BASE: int = 4096

#: aggregator fan-in cap.
MAX_TILES: int = 8

#: Env var selecting subprocess vs in-process pyxrt dispatch.
KMER_COUNT_DISPATCH_ENV: str = "BIONPU_KMER_COUNT_DISPATCH"

#: Kernel name baked into the xclbin (mirrors CRISPR convention).
_KERNEL_NAME: str = "MLIR_AIE"

# --------------------------------------------------------------------------- #
# Artifact root — same _npu_artifacts/ directory as every other kernel.
# --------------------------------------------------------------------------- #

_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)


def overlap_bytes_for(k: int) -> int:
    """Return ``ceil((k - 1) / 4)`` — the per-k chunk overlap in bytes.

    Matches T1 contract (``SEQ_IN_OVERLAP_K{15,21,31} = {4, 5, 8}``).
    """
    return (int(k) - 1 + 3) // 4


def kmer_mask_for(k: int) -> int:
    """Return the per-k canonical mask. Raises ``ValueError`` for unsupported k."""
    if int(k) not in _KMER_MASKS:
        valid = ", ".join(str(x) for x in SUPPORTED_K)
        raise ValueError(
            f"k={k!r} is not supported; expected one of {{{valid}}}."
        )
    return _KMER_MASKS[int(k)]


def decode_packed_kmer_to_ascii(canonical: int, k: int) -> str:
    """Decode a 2-bit-packed canonical k-mer back to ACGT ASCII.

    Bases are stored MSB-first within the canonical uint64 (highest 2-bit
    pair = base 0). Mirrors the host runner's Jellyfish-FASTA emit.
    """
    if int(k) not in _KMER_MASKS:
        raise ValueError(f"k={k!r} unsupported")
    table = "ACGT"
    mask = _KMER_MASKS[int(k)]
    canonical &= mask
    chars: list[str] = []
    for i in range(int(k) - 1, -1, -1):
        chars.append(table[(canonical >> (2 * i)) & 0x3])
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Timing parser (mirrors linear_projection / pam_filter).
# --------------------------------------------------------------------------- #

# Match plain decimal AND scientific notation (e.g. "1.76837e+06us").
_RE_AVG = re.compile(
    r"^Avg NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MIN = re.compile(
    r"^Min NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)
_RE_MAX = re.compile(
    r"^Max NPU time:\s*([0-9.eE+\-]+)us\.?\s*$", re.MULTILINE
)


def _parse_us_label(stdout: str, label: str) -> float | None:
    m = re.search(
        rf"^{label} NPU time:\s*([0-9.eE+\-]+)us\.?\s*$",
        stdout,
        re.MULTILINE,
    )
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Run info (mirrors pam_filter._RunInfo with kmer-specific counters).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _RunInfo:
    """Timing + provenance for the last NPU run.

    Fields per T1 contract:
        avg_us, min_us, max_us, n_iters, used_npu — same as pam_filter.
        n_records — total emitted records (pre dedup-merge).
        n_evict_records — records with the EVICT_FLAG bit set.
        n_unique_canonical — unique canonical k-mers after host merge.
    """

    avg_us: float
    min_us: float
    max_us: float
    n_iters: int
    used_npu: bool
    n_records: int = 0
    n_evict_records: int = 0
    n_unique_canonical: int = 0


# --------------------------------------------------------------------------- #
# Host-side parsers for the runner output.
# --------------------------------------------------------------------------- #

# Host-side post-pass output is the runner's Jellyfish-FASTA output:
#     >count
#     ACGT...
#     >count
#     ACGT...
# This is the on-disk format that --output writes; T9 parses it back to
# the (canonical_u64, count) tuple list because the contract pins this
# CLI shape (T1, T7).
_BASE_TO_CODE: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}


def _ascii_kmer_to_canonical_u64(kmer: str, k: int) -> int:
    """Pack an ASCII k-mer back to 2-bit MSB-first canonical uint64.

    Inverse of :func:`decode_packed_kmer_to_ascii` (does NOT recompute
    canonical = min(fwd, rc); the runner already wrote the canonical
    representation).
    """
    if len(kmer) != int(k):
        raise ValueError(
            f"kmer length mismatch: got {len(kmer)} expected {k} ({kmer!r})"
        )
    out = 0
    for ch in kmer:
        code = _BASE_TO_CODE.get(ch.upper())
        if code is None:
            raise ValueError(f"non-ACGT base in kmer {kmer!r}: {ch!r}")
        out = (out << 2) | code
    return out & _KMER_MASKS[int(k)]


def _parse_jellyfish_fasta(text: str, k: int) -> list[tuple[int, int]]:
    """Parse Jellyfish-FASTA output to ``[(canonical_u64, count), ...]``.

    Order is preserved (the runner writes count-desc, canonical-asc; T9
    forwards that order to the caller without re-sorting).
    """
    out: list[tuple[int, int]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        i += 1
        if not header:
            continue
        if not header.startswith(">"):
            # Skip stray lines defensively.
            continue
        try:
            count = int(header[1:].strip())
        except ValueError as exc:
            raise NpuRunFailed(
                0,
                text,
                f"[bionpu] could not parse Jellyfish-FASTA count "
                f"header {header!r}",
            ) from exc
        if i >= len(lines):
            raise NpuRunFailed(
                0, text, "[bionpu] Jellyfish-FASTA truncated mid-record"
            )
        kmer_ascii = lines[i].strip()
        i += 1
        canonical = _ascii_kmer_to_canonical_u64(kmer_ascii, k)
        out.append((canonical, count))
    return out


def _decode_emit_blob(
    blob: bytes,
    *,
    n_slots: int,
    slot_bytes: int = EMIT_SLOT_BYTES,
    slot_records: int = EMIT_SLOT_RECORDS,
) -> tuple[list[tuple[int, int, int]], int]:
    """Decode the per-slot length-prefixed sparse_out blob from the kernel.

    Matches the per-slot layout :func:`pam_filter.decode_per_slot_sparse_buffer`
    decodes for CRISPR; widened to 16-byte EmitRecord
    ``(canonical_u64, count_u32, flags_u32)``.

    Returns ``(records, n_evict)`` where ``records`` is a list of
    ``(canonical, count, flags)`` triples in slot order. Used by the
    pyxrt path (the subprocess path reads Jellyfish-FASTA from the
    runner's --output file because the runner already does the
    dedup-merge + sort + topN host-side post-pass).
    """
    records: list[tuple[int, int, int]] = []
    n_evict = 0
    for s in range(n_slots):
        off = s * slot_bytes
        if off + 4 > len(blob):
            break
        n = struct.unpack_from("<I", blob, off)[0]
        capped = min(int(n), slot_records)
        rec_base = off + 4
        for i in range(capped):
            roff = rec_base + i * EMIT_RECORD_BYTES
            if roff + EMIT_RECORD_BYTES > len(blob):
                break
            canonical, count, flags = struct.unpack_from("<QII", blob, roff)
            records.append((int(canonical), int(count), int(flags)))
            if flags & EVICT_FLAG:
                n_evict += 1
    return records, n_evict


def _merge_and_top(
    records: Sequence[tuple[int, int, int]],
    *,
    top_n: int,
    threshold: int,
) -> tuple[list[tuple[int, int]], int]:
    """Dedup-merge by canonical, sum counts, threshold + topN.

    Returns ``(merged, n_unique)`` where ``merged`` is sorted by
    ``(count desc, canonical asc)`` and trimmed to ``top_n`` (``top_n=0``
    is treated as "no top cap; all records >= threshold").
    """
    merged: dict[int, int] = {}
    for canonical, count, _flags in records:
        merged[canonical] = merged.get(canonical, 0) + count
    n_unique = len(merged)
    items = [
        (canonical, count)
        for canonical, count in merged.items()
        if count >= int(threshold)
    ]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    if int(top_n) > 0:
        items = items[: int(top_n)]
    return items, n_unique


# --------------------------------------------------------------------------- #
# Op class.
# --------------------------------------------------------------------------- #


# Kernel-arg slot map for the kmer_count host_runner. Mirrors the
# 3-BO layout in pam_filter (instr, input, output) widened by one for
# the per-tile fan-out — the runner lays them out as
# (opcode, bo_instr, instr_size, bo_input, bo_output, bo_trace).
_ARG_INPUT = 3
_ARG_OUTPUT = 4
_ARG_TRACE = 5


class BionpuKmerCount(NpuOp):
    """K-mer counting on AIE2P (k in {15, 21, 31}, n_tiles in {1, 2, 4, 8}).

    Per ``state/kmer_count_interface_contract.md`` (T1):

    * Three NPU_OPS registry entries, one per supported k. The
      ``n_tiles`` constructor arg selects the artifact directory at
      ``_npu_artifacts/bionpu_kmer_count_k{k}_n{n_tiles}/`` (3 x 4 = 12
      directory matrix; the registry-resident instances all carry
      ``n_tiles=4``).

    * Subprocess dispatch invokes the host_runner under the artifact
      directory with the pinned CLI; the runner does the dedup-merge +
      threshold + topN + Jellyfish-FASTA emit host-side, and T9 parses
      the FASTA back to ``(canonical_u64, count)`` tuples.

    * In-process pyxrt dispatch (when ``BIONPU_KMER_COUNT_DISPATCH=pyxrt``
      or ``_impl='pyxrt'``) uploads the packed input directly via
      :func:`bionpu.dispatch._pyxrt_helpers.run_pyxrt_with_buffers`,
      decodes the per-slot length-prefixed sparse_out blob, then runs
      the dedup-merge + sort + topN locally.
    """

    INPUT_DTYPE = np.uint8
    SUPPORTED_K = SUPPORTED_K
    SUPPORTED_N_TILES = SUPPORTED_N_TILES

    def __init__(self, k: int = 21, n_tiles: int = 4) -> None:
        if int(k) not in SUPPORTED_K:
            valid = ", ".join(str(x) for x in SUPPORTED_K)
            raise ValueError(
                f"BionpuKmerCount: k={k!r} is not supported; expected one of "
                f"{{{valid}}}."
            )
        if int(n_tiles) not in SUPPORTED_N_TILES:
            valid = ", ".join(str(x) for x in SUPPORTED_N_TILES)
            raise ValueError(
                f"BionpuKmerCount: n_tiles={n_tiles!r} is not supported; "
                f"expected one of {{{valid}}}."
            )
        self.k: int = int(k)
        self.n_tiles: int = int(n_tiles)
        # Per-instance name; matches the T1 registry naming pin.
        self.name: str = f"bionpu_kmer_count_k{self.k}"
        self.last_run: _RunInfo | None = None

    # ----- artifact paths ------------------------------------------------- #

    @property
    def artifact_dir(self) -> Path:
        """Per-(k, n_tiles) artifact directory."""
        return _ART_ROOT / f"bionpu_kmer_count_k{self.k}_n{self.n_tiles}"

    @property
    def xclbin(self) -> Path:
        return self.artifact_dir / "final.xclbin"

    @property
    def insts(self) -> Path:
        return self.artifact_dir / "insts.bin"

    @property
    def host_runner(self) -> Path:
        return self.artifact_dir / "host_runner"

    def artifacts_present(self) -> bool:
        """True iff all three NPU artifact leaves exist on disk."""
        return all(
            p.exists() for p in (self.xclbin, self.insts, self.host_runner)
        )

    # ----- dispatch routing ----------------------------------------------- #

    def _resolve_dispatch_impl(self, impl: str | None) -> str:
        """Return ``'subprocess'`` (default) or ``'pyxrt'``.

        Priority: explicit ``impl`` (``_impl=...``), else
        ``BIONPU_KMER_COUNT_DISPATCH``, else ``'subprocess'``.
        """
        return resolve_dispatch_impl(impl, env_var=KMER_COUNT_DISPATCH_ENV)

    # ----- input validation ---------------------------------------------- #

    def _validate_inputs(self, packed_seq: np.ndarray) -> None:
        if not isinstance(packed_seq, np.ndarray):
            raise TypeError(
                "packed_seq must be a numpy.ndarray of dtype uint8"
            )
        if packed_seq.dtype != np.uint8:
            raise ValueError(
                f"packed_seq dtype must be uint8; got {packed_seq.dtype}"
            )
        if packed_seq.ndim != 1:
            raise ValueError(
                f"packed_seq must be 1-D; got shape {packed_seq.shape}"
            )
        # Need at least one full k-mer worth of bases (k bases = ceil(k/4) bytes).
        min_bytes = (2 * self.k + 7) // 8
        if packed_seq.size < min_bytes:
            raise ValueError(
                f"packed_seq too short for k={self.k}: need at least "
                f"{min_bytes} bytes, got {packed_seq.size}"
            )

    # ----- subprocess path ------------------------------------------------ #

    def _run_subprocess(
        self,
        *,
        packed_seq: np.ndarray,
        top_n: int,
        threshold: int,
        n_iters: int,
        warmup: int,
        timeout_s: float,
    ) -> tuple[list[tuple[int, int]], _RunInfo]:
        """Spawn the host_runner; parse Jellyfish-FASTA --output."""
        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as tdir:
            t = Path(tdir)
            input_path = t / "packed_seq.2bit.bin"
            output_path = t / "kmers.jf.fa"
            np.ascontiguousarray(packed_seq).tofile(input_path)

            cmd = [
                str(self.host_runner),
                "-x", str(self.xclbin),
                "-i", str(self.insts),
                "-k", _KERNEL_NAME,
                "--input", str(input_path),
                "--output", str(output_path),
                "--k", str(self.k),
                "--top", str(int(top_n)),
                "--threshold", str(int(threshold)),
                "--launch-chunks", str(self.n_tiles),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
            ]
            proc = subprocess.run(  # noqa: S603 — argv strictly controlled
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
                env=_xrt_env(),
            )
            if proc.returncode != 0:
                raise NpuRunFailed(proc.returncode, proc.stdout, proc.stderr)
            if "PASS!" not in proc.stdout:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] expected 'PASS!' marker missing",
                )
            avg = _parse_us_label(proc.stdout, "Avg")
            mn = _parse_us_label(proc.stdout, "Min")
            mx = _parse_us_label(proc.stdout, "Max")
            if avg is None or mn is None or mx is None:
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr + "\n[bionpu] could not parse NPU timing lines",
                )

            if not output_path.exists():
                raise NpuRunFailed(
                    proc.returncode,
                    proc.stdout,
                    proc.stderr
                    + f"\n[bionpu] runner did not produce --output {output_path}",
                )
            fasta = output_path.read_text()

        kmers = _parse_jellyfish_fasta(fasta, self.k)
        info = _RunInfo(
            avg_us=float(avg),
            min_us=float(mn),
            max_us=float(mx),
            n_iters=int(n_iters),
            used_npu=True,
            n_records=len(kmers),
            n_evict_records=0,  # opaque from the FASTA path
            n_unique_canonical=len(kmers),
        )
        return kmers, info

    # ----- pyxrt path ----------------------------------------------------- #

    def _run_pyxrt(
        self,
        *,
        packed_seq: np.ndarray,
        top_n: int,
        threshold: int,
        n_iters: int,
        warmup: int,
    ) -> tuple[list[tuple[int, int]], _RunInfo]:
        """In-process pyxrt path: upload input BO, drain sparse_out blob, merge."""
        # Per-chunk + per-slot output volume estimate. Mirrors the
        # contract: each chunk produces n_tiles partial slots; the
        # aggregator drains them into ONE sparse_out slot per chunk.
        # n_chunks here is host-conservative (overlap-padded); the
        # kernel uses n_input_bytes to bound its actual loop.
        overlap = overlap_bytes_for(self.k)
        n_bytes = int(packed_seq.size)
        # The input size for 1 chunk including overlap.
        chunk_in = SEQ_IN_CHUNK_BYTES_BASE + overlap
        # Conservative chunk count: ceil(n_bytes / SEQ_IN_CHUNK_BYTES_BASE).
        n_chunks = max(
            1, (n_bytes + SEQ_IN_CHUNK_BYTES_BASE - 1) // SEQ_IN_CHUNK_BYTES_BASE
        )
        # Output blob size: one sparse_out slot per chunk (the aggregator
        # concentrates n_tiles partials into a single slot before drain).
        out_size = n_chunks * EMIT_SLOT_BYTES

        input_bytes = np.ascontiguousarray(packed_seq).tobytes()
        raw_out, avg_us, min_us, max_us = run_pyxrt_with_buffers(
            xclbin_path=self.xclbin,
            insts_path=self.insts,
            in_buffers=[
                (input_bytes, _ARG_INPUT),
                # Trace placeholder: 1 byte (matches the runner C++ which
                # constructs bo_trace with size 1 to keep group_id valid
                # when tracing is disabled).
                (bytes(1), _ARG_TRACE),
            ],
            out_size=out_size,
            out_arg_index=_ARG_OUTPUT,
            n_iters=int(n_iters),
            warmup=int(warmup),
            kernel_name=_KERNEL_NAME,
        )

        records, n_evict = _decode_emit_blob(raw_out, n_slots=n_chunks)
        merged, n_unique = _merge_and_top(
            records, top_n=top_n, threshold=threshold
        )
        info = _RunInfo(
            avg_us=float(avg_us),
            min_us=float(min_us),
            max_us=float(max_us),
            n_iters=int(n_iters),
            used_npu=True,
            n_records=len(records),
            n_evict_records=n_evict,
            n_unique_canonical=n_unique,
        )
        # quiet "chunk_in" (defined for documentation parity with the
        # contract; runner.cpp consumes the same value at the C++ side).
        del chunk_in
        return merged, info

    # ----- entry point ---------------------------------------------------- #

    def __call__(
        self,
        *,
        packed_seq: np.ndarray,
        top_n: int = 1000,
        threshold: int = 1,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 120.0,
        _impl: str | None = None,
        **_unused: Any,
    ) -> list[tuple[int, int]]:
        """Run the kernel and return ``[(canonical_u64, count), ...]``.

        Args:
            packed_seq: 1-D ``np.uint8`` packed-2-bit DNA buffer (T2
                packer output / T3 fixture format). MSB-first within
                each byte; A=00, C=01, G=10, T=11.
            top_n: keep only the top-N most-frequent k-mers; ``0`` means
                "no top cap, return all records >= threshold".
            threshold: drop k-mers with count < ``threshold`` (default
                1 = report all observed k-mers).
            n_iters: number of NPU iterations to time-average.
            warmup: warmup iterations the runner discards before the
                timed window.
            timeout_s: subprocess wall-clock cap.
            _impl: override of ``BIONPU_KMER_COUNT_DISPATCH``.

        Returns:
            ``list[(canonical_kmer_uint64, count)]`` sorted by
            ``(count desc, canonical asc)``.

        Raises:
            NpuArtifactsMissingError: artifact triple (xclbin/insts/host_runner)
                missing for ``(k, n_tiles)``.
            NpuRunFailed: kernel exit non-zero, ``PASS!`` missing, or
                timing parse failed.
        """
        self._validate_inputs(packed_seq)

        impl = self._resolve_dispatch_impl(_impl)

        required: tuple[Path, ...]
        if impl == "pyxrt":
            required = (self.xclbin, self.insts)
        else:
            required = (self.xclbin, self.insts, self.host_runner)
        for p in required:
            if not p.exists():
                raise NpuArtifactsMissingError(
                    f"NPU artifact missing for {self.name} "
                    f"(k={self.k}, n_tiles={self.n_tiles}): {p}. "
                    f"Build via `make NPU2=1 K={self.k} "
                    f"experiment={'production' if self.n_tiles == 1 else f'wide{self.n_tiles}'}"
                    f" all` in this kernel directory, then copy the build "
                    f"outputs to {self.artifact_dir}/."
                )

        if impl == "pyxrt":
            merged, info = self._run_pyxrt(
                packed_seq=packed_seq,
                top_n=top_n,
                threshold=threshold,
                n_iters=n_iters,
                warmup=warmup,
            )
        else:
            merged, info = self._run_subprocess(
                packed_seq=packed_seq,
                top_n=top_n,
                threshold=threshold,
                n_iters=n_iters,
                warmup=warmup,
                timeout_s=timeout_s,
            )

        self.last_run = info
        return merged


# --------------------------------------------------------------------------- #
# Registry (3 entries, one per supported k; n_tiles defaults to 4 per
# T1 contract — the helper get_kmer_count_op() constructs per-call
# instances bound to the requested n_tiles).
# --------------------------------------------------------------------------- #

for _k in SUPPORTED_K:
    register_npu_op(
        f"bionpu_kmer_count_k{_k}",
        BionpuKmerCount(k=_k, n_tiles=4),
    )
del _k
