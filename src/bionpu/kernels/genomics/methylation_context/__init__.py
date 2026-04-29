# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

"""Methylation-context scanner NPU op (v0).

The tile scans packed 2-bit DNA and emits sparse cytosine-context
records for ``CG``, ``CHG``, and ``CHH`` on both strands. A minus-strand
cytosine is represented by the forward-reference ``G`` at the same
position.

Binary output format:

``uint64 n_records`` followed by 8-byte records:
``uint32 pos | uint8 strand | uint8 context | uint16 _pad``.

``strand`` is ``0`` for ``+`` and ``1`` for ``-``. ``context`` is
``0=CG``, ``1=CHG``, ``2=CHH``.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bionpu.data.kmer_oracle import unpack_dna_2bit
from bionpu.data.methylation_context_oracle import MethylationContextHit
from bionpu.dispatch._pyxrt_helpers import resolve_dispatch_impl
from bionpu.dispatch.npu import (
    NpuArtifactsMissingError,
    NpuOp,
    NpuRunFailed,
    _xrt_env,  # noqa: PLC2701
    register_npu_op,
)

__all__ = [
    "BionpuMethylationContext",
    "CONTEXT_TO_CODE",
    "MAX_EMIT_IDX",
    "METHYLATION_CONTEXT_DISPATCH_ENV",
    "PARTIAL_OUT_BYTES_PADDED",
    "RECORD_BYTES",
    "SEQ_IN_CHUNK_BYTES_BASE",
    "SUPPORTED_SEQ_CHUNK_BYTES_BASE",
    "SUPPORTED_N_CHUNKS_PER_LAUNCH",
    "SUPPORTED_N_TILES",
]

SUPPORTED_N_TILES: tuple[int, ...] = (1, 2, 4, 8)
SUPPORTED_N_CHUNKS_PER_LAUNCH: tuple[int, ...] = (1, 2, 4, 8)
SUPPORTED_SEQ_CHUNK_BYTES_BASE: tuple[int, ...] = (1024, 2048, 4096)
SEQ_IN_CHUNK_BYTES_BASE: int = 4096
PARTIAL_OUT_BYTES_PADDED: int = 32768
RECORD_BYTES: int = 8
MAX_EMIT_IDX: int = 4094
METHYLATION_CONTEXT_DISPATCH_ENV: str = "BIONPU_METHYLATION_CONTEXT_DISPATCH"
CONTEXT_TO_CODE: dict[str, int] = {"CG": 0, "CHG": 1, "CHH": 2}
_CODE_TO_CONTEXT: tuple[str, ...] = ("CG", "CHG", "CHH")
_KERNEL_NAME: str = "MLIR_AIE"

_ART_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dispatch"
    / "_npu_artifacts"
)


@dataclass(frozen=True)
class _RunInfo:
    avg_us: float
    min_us: float
    max_us: float
    n_iters: int
    used_npu: bool
    n_records: int = 0


def _parse_us_label(stdout: str, label: str) -> float | None:
    m = re.search(
        rf"^{label} NPU time:\s*([0-9.eE+\-]+)us\.?\s*$",
        stdout,
        re.MULTILINE,
    )
    return float(m.group(1)) if m else None


def _motif_for_hit(seq: str, pos: int, strand: str, context: str) -> str:
    if context == "CG":
        return "CG"
    if strand == "+":
        return seq[pos : pos + 3]
    rc = str.maketrans("ACGT", "TGCA")
    return "C" + seq[pos - 1].translate(rc) + seq[pos - 2].translate(rc)


def _parse_records_blob(path: Path, *, seq: str) -> list[MethylationContextHit]:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 8:
        raise NpuRunFailed(
            0, "",
            f"[bionpu] binary output blob {path} too small "
            f"({raw.size} bytes; expected >=8)",
        )
    n_records = int(np.frombuffer(raw[:8], dtype=np.uint64)[0])
    expected_bytes = 8 + n_records * RECORD_BYTES
    if raw.size < expected_bytes:
        raise NpuRunFailed(
            0, "",
            f"[bionpu] binary output blob {path} truncated: "
            f"n_records={n_records}, bytes={raw.size}, need={expected_bytes}",
        )
    if n_records == 0:
        return []
    rec_dt = np.dtype([
        ("pos", "<u4"),
        ("strand", "u1"),
        ("context", "u1"),
        ("pad", "<u2"),
    ])
    arr = np.frombuffer(raw[8:expected_bytes], dtype=rec_dt)
    out: list[MethylationContextHit] = []
    for pos_raw, strand_raw, context_raw, _pad in arr:
        pos = int(pos_raw)
        strand = "+" if int(strand_raw) == 0 else "-"
        context_i = int(context_raw)
        if context_i >= len(_CODE_TO_CONTEXT):
            raise NpuRunFailed(
                0, "",
                f"[bionpu] invalid methylation context code {context_i}",
            )
        context = _CODE_TO_CONTEXT[context_i]
        out.append(
            MethylationContextHit(
                pos=pos,
                strand=strand,
                context=context,
                motif=_motif_for_hit(seq, pos, strand, context),
            )
        )
    return out


class BionpuMethylationContext(NpuOp):
    """CG/CHG/CHH methylation-context scan on AIE2P."""

    INPUT_DTYPE = np.uint8
    SUPPORTED_N_TILES = SUPPORTED_N_TILES
    SUPPORTED_N_CHUNKS_PER_LAUNCH = SUPPORTED_N_CHUNKS_PER_LAUNCH

    def __init__(
        self,
        *,
        n_tiles: int = 4,
        n_chunks_per_launch: int = 1,
        seq_chunk_bytes_base: int = SEQ_IN_CHUNK_BYTES_BASE,
    ) -> None:
        if int(n_tiles) not in SUPPORTED_N_TILES:
            valid = ", ".join(str(x) for x in SUPPORTED_N_TILES)
            raise ValueError(
                f"BionpuMethylationContext: n_tiles={n_tiles!r} "
                f"not in {{{valid}}}."
            )
        if int(n_chunks_per_launch) not in SUPPORTED_N_CHUNKS_PER_LAUNCH:
            valid = ", ".join(str(x) for x in SUPPORTED_N_CHUNKS_PER_LAUNCH)
            raise ValueError(
                "BionpuMethylationContext: n_chunks_per_launch="
                f"{n_chunks_per_launch!r} not in {{{valid}}}."
            )
        if int(seq_chunk_bytes_base) not in SUPPORTED_SEQ_CHUNK_BYTES_BASE:
            valid = ", ".join(str(x) for x in SUPPORTED_SEQ_CHUNK_BYTES_BASE)
            raise ValueError(
                "BionpuMethylationContext: seq_chunk_bytes_base="
                f"{seq_chunk_bytes_base!r} not in {{{valid}}}."
            )
        self.n_tiles = int(n_tiles)
        self.n_chunks_per_launch = int(n_chunks_per_launch)
        self.seq_chunk_bytes_base = int(seq_chunk_bytes_base)
        self.name = "bionpu_methylation_context"
        self.last_run: _RunInfo | None = None

    @property
    def artifact_dir(self) -> Path:
        base = f"bionpu_methylation_context_n{self.n_tiles}"
        if self.seq_chunk_bytes_base != SEQ_IN_CHUNK_BYTES_BASE:
            base = f"{base}_c{self.seq_chunk_bytes_base}"
        if self.n_chunks_per_launch == 1:
            return _ART_ROOT / base
        return _ART_ROOT / f"{base}_b{self.n_chunks_per_launch}"

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
        return all(p.exists() for p in (self.xclbin, self.insts, self.host_runner))

    def _resolve_dispatch_impl(self, impl: str | None) -> str:
        return resolve_dispatch_impl(
            impl, env_var=METHYLATION_CONTEXT_DISPATCH_ENV
        )

    def _validate_inputs(self, packed_seq: np.ndarray, n_bases: int) -> None:
        if not isinstance(packed_seq, np.ndarray):
            raise TypeError("packed_seq must be a numpy.ndarray of dtype uint8")
        if packed_seq.dtype != np.uint8:
            raise ValueError(
                f"packed_seq dtype must be uint8; got {packed_seq.dtype}"
            )
        if packed_seq.ndim != 1:
            raise ValueError(
                f"packed_seq must be 1-D; got shape {packed_seq.shape}"
            )
        expected = (int(n_bases) + 3) // 4
        if int(n_bases) < 0:
            raise ValueError(f"n_bases must be >= 0; got {n_bases}")
        if packed_seq.size != expected:
            raise ValueError(
                f"packed_seq has {packed_seq.size} bytes but "
                f"n_bases={n_bases} requires {expected}"
            )

    def _run_subprocess(
        self,
        *,
        packed_seq: np.ndarray,
        n_bases: int,
        top_n: int,
        n_iters: int,
        warmup: int,
        timeout_s: float,
    ) -> tuple[list[MethylationContextHit], _RunInfo]:
        from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

        with tempfile.TemporaryDirectory(prefix=f"{self.name}_") as tdir:
            t = Path(tdir)
            input_path = t / "packed_seq.2bit.bin"
            output_path = t / "methylation_contexts.bin"
            np.ascontiguousarray(packed_seq).tofile(input_path)
            cmd = [
                str(self.host_runner),
                "-x", str(self.xclbin),
                "-i", str(self.insts),
                "-k", _KERNEL_NAME,
                "--input", str(input_path),
                "--n-bases", str(int(n_bases)),
                "--output", str(output_path),
                "--output-format", "binary",
                "--launch-chunks", str(self.n_tiles),
                "--n-chunks-per-launch", str(self.n_chunks_per_launch),
                "--top", str(int(top_n)),
                "--iters", str(int(n_iters)),
                "--warmup", str(int(warmup)),
            ]
            with npu_silicon_lock():
                proc = subprocess.run(  # noqa: S603
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
                    + f"\n[bionpu] runner did not produce {output_path}",
                )
            seq = unpack_dna_2bit(packed_seq, int(n_bases))
            records = _parse_records_blob(output_path, seq=seq)

        info = _RunInfo(
            avg_us=float(avg),
            min_us=float(mn),
            max_us=float(mx),
            n_iters=int(n_iters),
            used_npu=True,
            n_records=len(records),
        )
        return records, info

    def __call__(
        self,
        *,
        packed_seq: np.ndarray,
        n_bases: int,
        top_n: int = 0,
        n_iters: int = 1,
        warmup: int = 0,
        timeout_s: float = 600.0,
        _impl: str | None = None,
        **_unused: Any,
    ) -> list[MethylationContextHit]:
        self._validate_inputs(packed_seq, n_bases)

        impl = self._resolve_dispatch_impl(_impl)
        if impl == "pyxrt":
            raise NotImplementedError(
                "BionpuMethylationContext pyxrt path not implemented; "
                "use BIONPU_METHYLATION_CONTEXT_DISPATCH=subprocess."
            )
        if not self.host_runner.exists():
            raise NpuArtifactsMissingError(
                f"NPU artifact missing for {self.name}: {self.host_runner}. "
                "Build the methylation_context kernel artifacts first."
            )
        for need in (self.xclbin, self.insts):
            if not need.exists():
                raise NpuArtifactsMissingError(f"NPU artifact missing: {need}")

        records, info = self._run_subprocess(
            packed_seq=packed_seq,
            n_bases=int(n_bases),
            top_n=top_n,
            n_iters=n_iters,
            warmup=warmup,
            timeout_s=timeout_s,
        )
        self.last_run = info
        return records


register_npu_op("bionpu_methylation_context", BionpuMethylationContext(n_tiles=4))
