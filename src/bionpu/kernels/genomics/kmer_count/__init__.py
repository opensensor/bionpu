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

"""K-mer counting NPU op (v0.5 streaming + multi-pass).

Per ``state/kmer_count_interface_contract.md`` (T1) v0.5 — symbols,
ObjectFifo names, constants, and the streaming + multi-pass partition
are pinned by the contract's "v0.5 REDESIGN" section.

This module hosts the :class:`BionpuKmerCount` op class and registers
exactly **three** ``NPU_OPS`` entries — one per supported k in
``{15, 21, 31}``. The op accepts ``n_tiles`` and ``n_passes`` via
constructor and selects an artifact directory at
``_npu_artifacts/bionpu_kmer_count_k{k}_n{n_tiles}_np{n_passes}/``.
The artifact dir contains ``final_p0.xclbin`` ... ``final_p{n_passes-1}.xclbin``
(one xclbin per hash-slice partition) plus a single ``host_runner``.

Wire format (per T1 v0.5):

* Input ``packed_seq`` is a 1-D ``np.uint8`` array carrying packed-2-bit
  DNA (A=00, C=01, G=10, T=11; first base = bits[7:6] of byte 0,
  MSB-first within each byte).
* The host runner streams the input in 4096-byte chunks with per-k
  4-byte-aligned overlap.
* Per chunk × per pass, each tile emits canonical k-mers whose
  ``canonical & ((1 << n_passes_log2) - 1) == pass_idx`` (low-bits
  hash slice). Output blob per pass × per chunk: N_TILES × 32 KiB,
  each ``[uint32 emit_idx][emit_idx × uint64 canonical]``.
* The host runner accumulates ``counts[canonical] += 1`` for every
  emitted canonical across all chunks × all passes, sorts by
  ``(count desc, canonical asc)``, applies ``--top`` + ``--threshold``,
  and emits Jellyfish-FASTA (``>count\\nkmer\\n``).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.dispatch._pyxrt_helpers import (
    resolve_dispatch_impl,
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
    "KMER_COUNT_DISPATCH_ENV",
    "KMER_MASK_K15",
    "KMER_MASK_K21",
    "KMER_MASK_K31",
    "MAX_EMIT_IDX_V05",
    "PARTIAL_OUT_BYTES_V05_PADDED",
    "SEQ_IN_CHUNK_BYTES_BASE",
    "SLICE_HASH_SHIFT",
    "SUPPORTED_K",
    "SUPPORTED_N_PASSES",
    "SUPPORTED_N_TILES",
    "decode_packed_kmer_to_ascii",
    "kmer_mask_for",
    "overlap_bytes_for",
]

# --------------------------------------------------------------------------- #
# Pinned constants — mirror kmer_count_constants.h v0.5 byte-equal.
# --------------------------------------------------------------------------- #

#: Supported k-mer widths (one registry entry per value).
SUPPORTED_K: tuple[int, ...] = (15, 21, 31)

#: Supported tile fan-out (n_tiles constructor arg).
SUPPORTED_N_TILES: tuple[int, ...] = (1, 2, 4, 8)

#: Supported hash-slice partition count (n_passes constructor arg).
SUPPORTED_N_PASSES: tuple[int, ...] = (1, 4, 16)

#: Per-k canonical bit masks (bit-equal to the C header).
KMER_MASK_K15: int = (1 << 30) - 1
KMER_MASK_K21: int = (1 << 42) - 1
KMER_MASK_K31: int = (1 << 62) - 1

_KMER_MASKS: dict[int, int] = {
    15: KMER_MASK_K15,
    21: KMER_MASK_K21,
    31: KMER_MASK_K31,
}

#: 4096-byte primary chunk before per-k overlap padding.
SEQ_IN_CHUNK_BYTES_BASE: int = 4096

#: 32 KiB per pass-slot in the joined output buffer.
PARTIAL_OUT_BYTES_V05_PADDED: int = 32768

#: Max canonicals per pass-slot (kernel caps at this).
MAX_EMIT_IDX_V05: int = 4095

#: Hash-slice shift (0 = use low bits of canonical for slice).
SLICE_HASH_SHIFT: int = 0

#: Env var selecting subprocess vs in-process pyxrt dispatch.
KMER_COUNT_DISPATCH_ENV: str = "BIONPU_KMER_COUNT_DISPATCH"

#: Kernel name baked into the xclbin (mirrors CRISPR convention).
_KERNEL_NAME: str = "MLIR_AIE"


_N_PASSES_LOG2: dict[int, int] = {1: 0, 4: 2, 16: 4}

# --------------------------------------------------------------------------- #
# Artifact root — same _npu_artifacts/ directory as every other kernel.
# --------------------------------------------------------------------------- #

_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)


def overlap_bytes_for(k: int) -> int:
    """Per-k overlap (bytes) — matches IRON Python's
    ``SEQ_OVERLAP_K{15,21,31}`` (pinned to 4-byte-aligned values).
    """
    return {15: 4, 21: 8, 31: 8}[int(k)]


def kmer_mask_for(k: int) -> int:
    """Return the per-k canonical mask. Raises ``ValueError`` for unsupported k."""
    if int(k) not in _KMER_MASKS:
        valid = ", ".join(str(x) for x in SUPPORTED_K)
        raise ValueError(
            f"k={k!r} is not supported; expected one of {{{valid}}}."
        )
    return _KMER_MASKS[int(k)]


def decode_packed_kmer_to_ascii(canonical: int, k: int) -> str:
    """Decode a 2-bit-packed canonical k-mer back to ACGT ASCII."""
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
    """Timing + provenance for the last NPU run."""

    avg_us: float
    min_us: float
    max_us: float
    n_iters: int
    used_npu: bool
    n_records: int = 0
    n_unique_canonical: int = 0
    n_passes: int = 0


# --------------------------------------------------------------------------- #
# Jellyfish-FASTA parser (output back to canonical-u64 tuples).
# --------------------------------------------------------------------------- #

_BASE_TO_CODE: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}


def _ascii_kmer_to_canonical_u64(kmer: str, k: int) -> int:
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
    """Parse Jellyfish-FASTA output to ``[(canonical_u64, count), ...]``."""
    out: list[tuple[int, int]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        i += 1
        if not header:
            continue
        if not header.startswith(">"):
            continue
        try:
            count = int(header[1:].strip())
        except ValueError as exc:
            raise NpuRunFailed(
                0, text,
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


# --------------------------------------------------------------------------- #
# Op class.
# --------------------------------------------------------------------------- #


class BionpuKmerCount(NpuOp):
    """K-mer counting on AIE2P (k ∈ {15, 21, 31}, n_tiles ∈ {1, 2, 4, 8},
    n_passes ∈ {1, 4, 16}).

    Per ``state/kmer_count_interface_contract.md`` v0.5.

    Three NPU_OPS registry entries (one per k). The artifact directory
    is per-(k, n_tiles, n_passes) at
    ``_npu_artifacts/bionpu_kmer_count_k{k}_n{n_tiles}_np{n_passes}/``;
    inside, one ``final_p{i}.xclbin`` + ``insts_p{i}.bin`` pair per
    pass_idx ∈ ``[0, n_passes)`` and one ``host_runner``.
    """

    INPUT_DTYPE = np.uint8
    SUPPORTED_K = SUPPORTED_K
    SUPPORTED_N_TILES = SUPPORTED_N_TILES
    SUPPORTED_N_PASSES = SUPPORTED_N_PASSES

    def __init__(
        self,
        k: int = 21,
        n_tiles: int = 4,
        n_passes: int = 4,
    ) -> None:
        if int(k) not in SUPPORTED_K:
            valid = ", ".join(str(x) for x in SUPPORTED_K)
            raise ValueError(
                f"BionpuKmerCount: k={k!r} not in {{{valid}}}."
            )
        if int(n_tiles) not in SUPPORTED_N_TILES:
            valid = ", ".join(str(x) for x in SUPPORTED_N_TILES)
            raise ValueError(
                f"BionpuKmerCount: n_tiles={n_tiles!r} not in {{{valid}}}."
            )
        if int(n_passes) not in SUPPORTED_N_PASSES:
            valid = ", ".join(str(x) for x in SUPPORTED_N_PASSES)
            raise ValueError(
                f"BionpuKmerCount: n_passes={n_passes!r} not in {{{valid}}}."
            )
        self.k: int = int(k)
        self.n_tiles: int = int(n_tiles)
        self.n_passes: int = int(n_passes)
        self.name: str = f"bionpu_kmer_count_k{self.k}"
        self.last_run: _RunInfo | None = None

    # ----- artifact paths ------------------------------------------------- #

    @property
    def artifact_dir(self) -> Path:
        """Per-(k, n_tiles, n_passes) artifact directory."""
        return (
            _ART_ROOT
            / f"bionpu_kmer_count_k{self.k}_n{self.n_tiles}_np{self.n_passes}"
        )

    def xclbin_for_pass(self, pass_idx: int) -> Path:
        """Path to ``final_p{pass_idx}.xclbin`` inside the artifact dir.

        For backwards compat, if n_passes=1, returns ``final.xclbin``.
        """
        if self.n_passes == 1:
            return self.artifact_dir / "final.xclbin"
        return self.artifact_dir / f"final_p{pass_idx}.xclbin"

    def insts_for_pass(self, pass_idx: int) -> Path:
        if self.n_passes == 1:
            return self.artifact_dir / "insts.bin"
        return self.artifact_dir / f"insts_p{pass_idx}.bin"

    @property
    def xclbin(self) -> Path:
        """Pass-0 xclbin path (used as the runner's --xclbin template)."""
        return self.xclbin_for_pass(0)

    @property
    def insts(self) -> Path:
        return self.insts_for_pass(0)

    @property
    def host_runner(self) -> Path:
        return self.artifact_dir / "host_runner"

    def artifacts_present(self) -> bool:
        """True iff all required NPU artifact leaves exist on disk:
        host_runner + (final_p{i}.xclbin, insts_p{i}.bin) for each pass.
        """
        if not self.host_runner.exists():
            return False
        for p in range(self.n_passes):
            if not self.xclbin_for_pass(p).exists():
                return False
            if not self.insts_for_pass(p).exists():
                return False
        return True

    # ----- dispatch routing ----------------------------------------------- #

    def _resolve_dispatch_impl(self, impl: str | None) -> str:
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
                "-x", str(self.xclbin),                # pass-0 path; runner derives others
                "-i", str(self.insts),
                "-k", _KERNEL_NAME,
                "--input", str(input_path),
                "--output", str(output_path),
                "--k", str(self.k),
                "--top", str(int(top_n)),
                "--threshold", str(int(threshold)),
                "--launch-chunks", str(self.n_tiles),
                "--n-passes", str(self.n_passes),
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
            n_unique_canonical=len(kmers),
            n_passes=self.n_passes,
        )
        return kmers, info

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
        """Run the kernel and return ``[(canonical_u64, count), ...]``."""
        self._validate_inputs(packed_seq)

        impl = self._resolve_dispatch_impl(_impl)
        if impl == "pyxrt":
            # The v0.5 multi-xclbin path is non-trivial via pyxrt; for
            # smoke validation the subprocess path is the gate. pyxrt
            # support tracked as a follow-up.
            raise NotImplementedError(
                "BionpuKmerCount v0.5 pyxrt path not implemented; "
                "use BIONPU_KMER_COUNT_DISPATCH=subprocess (default)."
            )

        # Subprocess path: enforce all required artifacts exist.
        if not self.host_runner.exists():
            raise NpuArtifactsMissingError(
                f"NPU artifact missing for {self.name} "
                f"(k={self.k}, n_tiles={self.n_tiles}, n_passes={self.n_passes}): "
                f"{self.host_runner}. Build via "
                f"`make NPU2=1 K={self.k} n_passes={self.n_passes} "
                f"experiment={'production' if self.n_tiles == 1 else f'wide{self.n_tiles}'} all` "
                f"in this kernel directory."
            )
        for p in range(self.n_passes):
            for need in (self.xclbin_for_pass(p), self.insts_for_pass(p)):
                if not need.exists():
                    raise NpuArtifactsMissingError(
                        f"NPU artifact missing for pass={p}: {need}"
                    )

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
# Registry (3 entries, one per supported k; n_tiles=4, n_passes=4 defaults).
# --------------------------------------------------------------------------- #

for _k in SUPPORTED_K:
    register_npu_op(
        f"bionpu_kmer_count_k{_k}",
        BionpuKmerCount(k=_k, n_tiles=4, n_passes=4),
    )
del _k
