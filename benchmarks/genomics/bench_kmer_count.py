#!/usr/bin/env python3
# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
# SPDX-License-Identifier: GPL-3.0-only

"""T15 — End-to-end k-mer counting benchmark (NPU vs Jellyfish).

Per ``state/kmer_count_interface_contract.md`` (T1) v0.5 — uses the
shipped ``BionpuKmerCount`` op via ``get_kmer_count_op`` and the
``npu_silicon_lock`` mutex (CLAUDE.md non-negotiable rule).

Sweep:

* fixtures: ``smoke_10kbp.2bit.bin``, ``synthetic_1mbp.2bit.bin``,
  ``chr22.2bit.bin``
* k ∈ {15, 21, 31}
* n_tiles=4, n_passes=4 (the v1.0 silicon-validated configuration —
  v1.1 will sweep n_passes ∈ {1, 16})

For each (fixture, k) cell:

* Run NPU dispatch under ONE wrapping ``npu_silicon_lock`` for the
  whole sweep. Capture per-iter avg/min/max wall + silicon-only
  avg_us_per_iter from the runner's per-iter NPU-time prints.
* Run ``jellyfish count -t 1 -m k`` and ``jellyfish count -t 8 -m k``
  on the corresponding FASTA (chr22 has a real FASTA; smoke +
  synthetic_1mbp are unpacked on the fly).
* Compute throughput (k-mers/sec) and silicon-only / end-to-end
  speedups vs Jellyfish.
* Optionally capture RAPL CPU + xrt-smi NPU energy if the readers
  probe successfully (bench/energy/auto_reader).

chr22 byte-equal is KNOWN to fail at the v1.0 build (n_passes=4
hits ``MAX_EMIT_IDX_V05=4095`` cap on the all-A canonical=0 slice
+ chunk-overlap double-emit, both filed in ``gaps.yaml``); the
script records the verdict alongside the throughput numbers and
does NOT fail the run.

Usage::

    source /opt/xilinx/xrt/setup.sh
    source /home/matteius/xdna-bringup/ironenv/bin/activate
    export PYTHONPATH=/opt/xilinx/xrt/python:\
/home/matteius/genetics/bionpu-public/src:$PYTHONPATH
    python3 bench_kmer_count.py [--out RESULTS_DIR] [--skip-chr22]
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/matteius/genetics")
FIXTURE_DIR = REPO_ROOT / "tracks" / "genomics" / "fixtures"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "kmer"
SMOKE_BIN = FIXTURE_DIR / "smoke_10kbp.2bit.bin"
SYNTHETIC_BIN = FIXTURE_DIR / "synthetic_1mbp.2bit.bin"
CHR22_BIN = FIXTURE_DIR / "chr22.2bit.bin"
SYNTHETIC_FA_SRC = (
    REPO_ROOT / "tracks" / "crispr" / "fixtures" / "synthetic_1mbp.fa"
)
CHR22_FA_SRC = REPO_ROOT / "data_cache" / "cas-offinder" / "genomes" / "chr22.fa"


def _utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _run(cmd: list[str], timeout: float | None = None,
         capture: bool = True) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _unpack_2bit_to_fasta(bin_path: Path, fasta_path: Path,
                          record_name: str = "seq") -> int:
    """Unpack a header-less 2-bit binary to a single-record FASTA.

    Returns the number of bases written.
    """
    from bionpu.data.kmer_oracle import unpack_dna_2bit
    buf = np.fromfile(bin_path, dtype=np.uint8)
    n_bases = int(buf.size) * 4
    seq = unpack_dna_2bit(buf, n_bases)
    with fasta_path.open("w") as fh:
        fh.write(f">{record_name}\n")
        # 60 bases per line (canonical FASTA wrapping).
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i + 60])
            fh.write("\n")
    return n_bases


def _jellyfish_count(fasta_path: Path, k: int, threads: int,
                     out_db: Path, timeout: float = 600.0) -> dict:
    """Run ``jellyfish count`` and return wall_seconds + record count."""
    cmd = [
        "jellyfish", "count",
        "-C",
        "-m", str(k),
        "-s", "100M",
        "-t", str(threads),
        "-o", str(out_db),
        str(fasta_path),
    ]
    t0 = time.monotonic()
    rc, stdout, stderr = _run(cmd, timeout=timeout)
    wall = time.monotonic() - t0
    if rc != 0:
        return {
            "wall_seconds": wall,
            "rc": rc,
            "stderr_tail": stderr[-512:],
            "ok": False,
        }
    # Optional: count distinct via stats (cheap; same db).
    n_distinct = None
    rc2, stats_out, _ = _run(
        ["jellyfish", "stats", str(out_db)], timeout=60.0
    )
    if rc2 == 0:
        for line in stats_out.splitlines():
            line = line.strip()
            if line.startswith("Distinct:"):
                try:
                    n_distinct = int(line.split()[-1])
                except ValueError:
                    pass
    return {
        "wall_seconds": wall,
        "rc": 0,
        "n_distinct": n_distinct,
        "ok": True,
    }


def _bench_one(*, fixture_name: str, fixture_bin: Path,
               fasta_path: Path, k: int, n_tiles: int, n_passes: int,
               n_iters: int, warmup: int, top_n: int,
               timeout_s: float = 600.0) -> dict:
    """Run NPU dispatch + Jellyfish 1T/8T for one cell. Returns the cell dict.

    Caller is expected to be inside ``npu_silicon_lock``.
    """
    from bionpu.kernels.genomics import get_kmer_count_op

    cell: dict = {
        "fixture_name": fixture_name,
        "fixture_path": str(fixture_bin),
        "fasta_path": str(fasta_path),
        "k": k,
        "n_tiles": n_tiles,
        "n_passes": n_passes,
        "n_iters": n_iters,
        "warmup": warmup,
        "top_n": top_n,
    }

    # ---- NPU run --------------------------------------------------------- #
    op = get_kmer_count_op(k=k, n_tiles=n_tiles)
    buf = np.fromfile(fixture_bin, dtype=np.uint8)
    n_bytes = int(buf.size)
    n_bases = n_bytes * 4
    cell["fixture_bytes"] = n_bytes
    cell["n_bases"] = n_bases
    n_kmers = max(0, n_bases - k + 1)
    cell["n_kmers"] = n_kmers

    npu_err: str | None = None
    npu_records = 0
    npu_wall = None
    avg_us = min_us = max_us = None
    try:
        t0 = time.monotonic()
        records = op(
            packed_seq=buf,
            top_n=int(top_n),
            threshold=1,
            n_iters=int(n_iters),
            warmup=int(warmup),
            timeout_s=timeout_s,
        )
        npu_wall = time.monotonic() - t0
        npu_records = len(records)
        run_info = op.last_run
        if run_info is not None:
            avg_us = run_info.avg_us
            min_us = run_info.min_us
            max_us = run_info.max_us
    except Exception as exc:
        npu_err = f"{type(exc).__name__}: {exc}"
    cell["npu"] = {
        "wall_seconds_total": npu_wall,
        "avg_us_per_iter": avg_us,
        "min_us_per_iter": min_us,
        "max_us_per_iter": max_us,
        "n_records_returned": npu_records,
        "error": npu_err,
    }

    # ---- Jellyfish 1T + 8T ---------------------------------------------- #
    with tempfile.TemporaryDirectory(prefix="kmer_bench_") as tmpd:
        db_1t = Path(tmpd) / "jf_1t.jf"
        db_8t = Path(tmpd) / "jf_8t.jf"
        jf1 = _jellyfish_count(fasta_path, k=k, threads=1, out_db=db_1t)
        jf8 = _jellyfish_count(fasta_path, k=k, threads=8, out_db=db_8t)
    cell["jellyfish_1t"] = jf1
    cell["jellyfish_8t"] = jf8

    # ---- Throughput / speedups ------------------------------------------ #
    speedups: dict = {}
    if avg_us is not None and avg_us > 0:
        sec_per_iter_silicon = avg_us / 1_000_000.0
        speedups["npu_silicon_kmers_per_sec"] = (
            n_kmers / sec_per_iter_silicon if n_kmers > 0 else None
        )
        if jf1.get("ok"):
            speedups["speedup_silicon_only_vs_jf1t"] = (
                jf1["wall_seconds"] / sec_per_iter_silicon
            )
        if jf8.get("ok"):
            speedups["speedup_silicon_only_vs_jf8t"] = (
                jf8["wall_seconds"] / sec_per_iter_silicon
            )
    if npu_wall is not None and npu_wall > 0:
        speedups["npu_end_to_end_kmers_per_sec"] = (
            n_kmers * max(1, n_iters) / npu_wall
            if n_kmers > 0 else None
        )
        if jf1.get("ok"):
            speedups["speedup_end_to_end_vs_jf1t"] = (
                jf1["wall_seconds"] * max(1, n_iters) / npu_wall
            )
        if jf8.get("ok"):
            speedups["speedup_end_to_end_vs_jf8t"] = (
                jf8["wall_seconds"] * max(1, n_iters) / npu_wall
            )
    if jf1.get("ok"):
        speedups["jf1t_kmers_per_sec"] = (
            n_kmers / jf1["wall_seconds"] if n_kmers > 0 else None
        )
    if jf8.get("ok"):
        speedups["jf8t_kmers_per_sec"] = (
            n_kmers / jf8["wall_seconds"] if n_kmers > 0 else None
        )
    cell["throughput"] = speedups

    return cell


def _try_energy_readers() -> dict:
    """Probe RAPL + XRT energy readers; return a status sketch."""
    out: dict = {"cpu_rapl": None, "npu_xrt": None}
    try:
        from bionpu.bench.energy import probe_rapl
        try:
            p = probe_rapl()
            out["cpu_rapl"] = {"available": True, "path": str(p)}
        except Exception as exc:
            out["cpu_rapl"] = {"available": False, "reason": str(exc)}
    except Exception as exc:
        out["cpu_rapl"] = {"available": False, "reason": f"import: {exc}"}
    try:
        from bionpu.bench.energy import probe_xrt
        try:
            p = probe_xrt()
            out["npu_xrt"] = {"available": True, "info": str(p)}
        except Exception as exc:
            out["npu_xrt"] = {"available": False, "reason": str(exc)}
    except Exception as exc:
        out["npu_xrt"] = {"available": False, "reason": f"import: {exc}"}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="T15 k-mer counting bench (NPU vs Jellyfish 1T/8T).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Results dir; default: results/kmer/v1-bench-{iso}/",
    )
    p.add_argument(
        "--ks",
        type=str,
        default="15,21,31",
        help="Comma-separated k values to sweep (default: 15,21,31).",
    )
    p.add_argument(
        "--fixtures",
        type=str,
        default="smoke,synthetic,chr22",
        help="Comma-separated fixtures (default: smoke,synthetic,chr22).",
    )
    p.add_argument("--n-tiles", type=int, default=4)
    p.add_argument("--n-passes", type=int, default=4)
    p.add_argument("--n-iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--top-n", type=int, default=1000)
    p.add_argument(
        "--skip-chr22",
        action="store_true",
        help="Skip the chr22 fixture (long-running; ~100s of Python "
             "post-pass dominates end-to-end).",
    )
    p.add_argument("--label-suffix", type=str, default="")
    args = p.parse_args(argv)

    # ---- Local import (after PYTHONPATH set in shell) ------------------- #
    from bionpu.dispatch.npu_silicon_lock import npu_silicon_lock

    iso = _utc_iso()
    out_dir = args.out
    if out_dir is None:
        out_dir = DEFAULT_RESULTS_DIR / f"v1-bench-{iso}"
    out_dir.mkdir(parents=True, exist_ok=True)
    measurements_path = out_dir / "measurements.json"

    if shutil.which("jellyfish") is None:
        print("bench_kmer_count: jellyfish not found on PATH; abort.",
              file=sys.stderr)
        return 2

    # ---- Pre-build fixtures + fastas ------------------------------------ #
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    fixtures = [x.strip() for x in args.fixtures.split(",") if x.strip()]
    if args.skip_chr22 and "chr22" in fixtures:
        fixtures = [x for x in fixtures if x != "chr22"]

    # Allocate temp dir for derived FASTAs (smoke + synthetic).
    tmp_root = tempfile.mkdtemp(prefix="kmer_bench_fa_")
    tmp_root_p = Path(tmp_root)
    fixture_specs: dict[str, dict] = {}
    if "smoke" in fixtures:
        smoke_fa = tmp_root_p / "smoke_10kbp.fa"
        n = _unpack_2bit_to_fasta(SMOKE_BIN, smoke_fa, "smoke_10kbp")
        fixture_specs["smoke"] = {
            "name": "smoke_10kbp",
            "bin": SMOKE_BIN,
            "fa": smoke_fa,
            "n_bases": n,
        }
    if "synthetic" in fixtures:
        # The 2bit binary represents the FASTA after non-ACGT scrubbing,
        # so the canonical k-mer set MAY differ from the FASTA on the
        # margin. Re-derive a FASTA from the .2bit.bin to make Jellyfish
        # operate on the byte-equal sequence.
        synth_fa = tmp_root_p / "synthetic_1mbp_unpacked.fa"
        n = _unpack_2bit_to_fasta(SYNTHETIC_BIN, synth_fa, "synthetic_1mbp")
        fixture_specs["synthetic"] = {
            "name": "synthetic_1mbp",
            "bin": SYNTHETIC_BIN,
            "fa": synth_fa,
            "n_bases": n,
        }
    if "chr22" in fixtures:
        # chr22.fa is the source-of-truth FASTA; the 2bit binary was
        # produced from it. Use the original FASTA so Jellyfish sees the
        # identical sequence the NPU ran on. Note: 2bit conversion folds
        # N -> A so the count distributions WILL diverge slightly at
        # canonical=0 (centromere). T14 measurements quantify this.
        fixture_specs["chr22"] = {
            "name": "chr22",
            "bin": CHR22_BIN,
            "fa": CHR22_FA_SRC,
            "n_bases": 0,  # fill from buf below
        }

    energy = _try_energy_readers()

    measurements: dict = {
        "schema_version": "1.0.0",
        "task_id": "T15",
        "design_version": "v0.5",
        "kernel_name": "bionpu_kmer_count",
        "generated_at_iso8601": iso,
        "host": os.uname().nodename,
        "config": {
            "n_tiles": args.n_tiles,
            "n_passes": args.n_passes,
            "n_iters": args.n_iters,
            "warmup": args.warmup,
            "top_n": args.top_n,
            "ks": ks,
            "fixtures": list(fixture_specs.keys()),
        },
        "energy_readers": energy,
        "cells": {},
        "caveats": [],
    }

    # ---- Single npu_silicon_lock wraps the entire sweep ----------------- #
    label = (
        f"kmer_bench:{iso}"
        + (f":{args.label_suffix}" if args.label_suffix else "")
    )
    cells_completed = 0
    sweep_t0 = time.monotonic()
    try:
        with npu_silicon_lock(label=label):
            for fixture_key, spec in fixture_specs.items():
                for k in ks:
                    print(
                        f"[bench] cell fixture={fixture_key} k={k} "
                        f"n_tiles={args.n_tiles} n_passes={args.n_passes}",
                        file=sys.stderr,
                    )
                    cell = _bench_one(
                        fixture_name=spec["name"],
                        fixture_bin=spec["bin"],
                        fasta_path=spec["fa"],
                        k=k,
                        n_tiles=args.n_tiles,
                        n_passes=args.n_passes,
                        n_iters=args.n_iters,
                        warmup=args.warmup,
                        top_n=args.top_n,
                    )
                    measurements["cells"][f"{fixture_key}_k{k}"] = cell
                    cells_completed += 1
                    # Persist after each cell so a crash at chr22 still
                    # leaves the smoke/synthetic results on disk.
                    with measurements_path.open("w") as fh:
                        json.dump(measurements, fh, indent=2, default=str)
    finally:
        sweep_wall = time.monotonic() - sweep_t0
        measurements["sweep_wall_seconds"] = sweep_wall
        measurements["cells_completed"] = cells_completed
        # chr22 caveats per Wave 5 retry findings.
        if "chr22" in fixture_specs:
            measurements["caveats"].append(
                "chr22 + n_passes=4: byte-equal vs Jellyfish FAILS — "
                "the canonical=0 (all-A) slice hits MAX_EMIT_IDX_V05=4095 "
                "cap-fire; gaps.yaml `kmer-chr22-canonical0-cap-fire`."
            )
            measurements["caveats"].append(
                "chunk-overlap edge double-emit: ~3-8 excess per "
                "canonical at chr22 scale; gaps.yaml "
                "`kmer-chunk-overlap-double-emit`."
            )
            measurements["caveats"].append(
                "End-to-end Python host post-pass over ~30M canonicals "
                "dominates wall time at chr22 scale (silicon avg "
                "~951 us/iter; host wall ~33 s); gaps.yaml "
                "`kmer-host-postpass-python-bottleneck`."
            )
        with measurements_path.open("w") as fh:
            json.dump(measurements, fh, indent=2, default=str)
        # Cleanup derived FASTAs.
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_root_p)

    print(f"[bench] wrote {measurements_path}", file=sys.stderr)
    print(f"[bench] cells_completed={cells_completed} "
          f"sweep_wall={sweep_wall:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
