from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bionpu.data.kmer_oracle import unpack_dna_2bit
from bionpu.data.methylation_context_oracle import find_methylation_contexts

ROOT = Path(__file__).resolve().parents[3]
packed_path = Path('/home/matteius/genetics/tracks/genomics/fixtures/chr22.2bit.bin')
artifact = ROOT / 'src/bionpu/dispatch/_npu_artifacts/bionpu_methylation_context_n4'
out_path = Path(__file__).resolve().parent / 'silicon_chr22.bin'
summary_path = Path(__file__).resolve().parent / 'measurements.json'

packed = np.fromfile(packed_path, dtype=np.uint8)
n_bases = int(packed.size * 4)

cmd = [
    str(artifact / 'host_runner'),
    '-x', str(artifact / 'final.xclbin'),
    '-i', str(artifact / 'insts.bin'),
    '-k', 'MLIR_AIE',
    '--input', str(packed_path),
    '--n-bases', str(n_bases),
    '--output', str(out_path),
    '--output-format', 'binary',
    '--launch-chunks', '4',
    '--n-chunks-per-launch', '1',
    '--iters', '1',
    '--warmup', '0',
]

t0 = time.perf_counter()
proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=900)
silicon_wall_s = time.perf_counter() - t0
if proc.returncode != 0:
    raise SystemExit(proc.stderr + '\n' + proc.stdout)

raw = np.fromfile(out_path, dtype=np.uint8)
n_records = int(np.frombuffer(raw[:8], dtype=np.uint64)[0]) if raw.size >= 8 else 0
rec_dt = np.dtype([('pos', '<u4'), ('strand', 'u1'), ('context', 'u1'), ('pad', '<u2')])
arr = np.frombuffer(raw[8:8 + n_records * 8], dtype=rec_dt) if n_records else np.array([], dtype=rec_dt)
context_names = np.array(['CG', 'CHG', 'CHH'])
silicon_counts = Counter()
strand_counts = Counter()
if n_records:
    ctx_vals, ctx_n = np.unique(arr['context'], return_counts=True)
    for c, n in zip(ctx_vals, ctx_n):
        silicon_counts[str(context_names[int(c)])] = int(n)
    strand_vals, strand_n = np.unique(arr['strand'], return_counts=True)
    for s, n in zip(strand_vals, strand_n):
        strand_counts['+' if int(s) == 0 else '-'] = int(n)

# CPU oracle count; materializes hits for exact count parity with the reference oracle.
t1 = time.perf_counter()
seq = unpack_dna_2bit(packed, n_bases)
hits = find_methylation_contexts(seq)
oracle_wall_s = time.perf_counter() - t1
oracle_counts = Counter(h.context for h in hits)
oracle_strand_counts = Counter(h.strand for h in hits)

matched = n_records == len(hits) and silicon_counts == oracle_counts and strand_counts == oracle_strand_counts
summary = {
    'fixture': str(packed_path),
    'packed_bytes': int(packed.size),
    'n_bases': n_bases,
    'artifact_dir': str(artifact),
    'host_runner_stdout': proc.stdout,
    'host_runner_stderr': proc.stderr,
    'silicon_wall_s': silicon_wall_s,
    'silicon_records': n_records,
    'silicon_counts_by_context': dict(silicon_counts),
    'silicon_counts_by_strand': dict(strand_counts),
    'oracle_wall_s': oracle_wall_s,
    'oracle_records': len(hits),
    'oracle_counts_by_context': dict(oracle_counts),
    'oracle_counts_by_strand': dict(oracle_strand_counts),
    'byte_equal_by_counts': matched,
    'record_deficit': int(len(hits) - n_records),
    'record_recovery': float(n_records / len(hits)) if hits else 1.0,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
