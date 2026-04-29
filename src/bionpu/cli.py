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

"""bionpu command-line interface.

Subcommands:

- ``bionpu verify crispr <npu.tsv> <ref.tsv>``
  Byte-equality check of an NPU off-target hits TSV against a
  Cas-OFFinder reference TSV.
- ``bionpu verify basecalling <npu.fastq> <ref.fastq>``
  Byte-equality check of an NPU-emitted FASTQ against a Dorado
  reference FASTQ.
- ``bionpu scan ...``
  CRISPR off-target scan (placeholder in v0.1; see
  ``benchmarks/crispr/run_chr.sh`` for the canonical scan pipeline).
- ``bionpu basecall ...``
  Nanopore basecalling (placeholder in v0.1; see
  ``benchmarks/basecalling/run_pod5.sh``).
- ``bionpu bench ...``
  Energy + timing harness (placeholder in v0.1; see
  ``docs/ENERGY_METHODOLOGY.md`` for the methodology).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import __version__


def _format_result(label: str, result) -> str:
    lines = [
        f"  result:        {'EQUAL' if result.equal else 'DIVERGENT'}",
        f"  records:       {result.record_count}",
        f"  npu sha256:    {result.npu_sha256}",
        f"  ref sha256:    {result.ref_sha256}",
    ]
    if not result.equal and result.divergences:
        lines.append("  divergences:")
        for div in result.divergences[:8]:
            lines.append(f"    [{div.record_index}] {div.message}")
        if len(result.divergences) > 8:
            lines.append(f"    ... ({len(result.divergences) - 8} more)")
    return f"{label}\n" + "\n".join(lines)


def _cmd_verify_crispr(args: argparse.Namespace) -> int:
    from .verify.crispr import compare_against_cas_offinder

    result = compare_against_cas_offinder(
        args.npu_tsv, args.ref_tsv, max_divergences=args.max_divergences
    )
    print(_format_result(f"verify crispr: {args.npu_tsv} vs {args.ref_tsv}", result))
    return 0 if result.equal else 1


def _cmd_verify_basecalling(args: argparse.Namespace) -> int:
    from .verify.basecalling import compare_against_dorado

    result = compare_against_dorado(
        args.npu_fastq, args.ref_fastq, max_divergences=args.max_divergences
    )
    print(_format_result(
        f"verify basecalling: {args.npu_fastq} vs {args.ref_fastq}", result
    ))
    return 0 if result.equal else 1


def _cmd_scan(args: argparse.Namespace) -> int:
    """Run a CRISPR off-target scan and emit a canonical TSV.

    v0.1: pure-CPU implementation via :func:`bionpu.scan.cpu_scan`.
    v0.2: NPU implementation via :func:`bionpu.scan.npu_scan` (PAM-filter
    kernel dispatched through :mod:`bionpu.dispatch`). The NPU path
    requires the kernel artifacts to be built — see the
    :class:`bionpu.dispatch.npu.NpuArtifactsMissingError` message
    if they are not.
    """
    from .data.canonical_sites import normalize, write_tsv
    from .scan import cpu_scan, npu_scan, parse_guides, read_fasta

    chrom, seq = read_fasta(args.target)
    guides = parse_guides(args.guides)

    print(
        f"bionpu scan [{args.device}]: chrom={chrom!r}, seq={len(seq):,} nt, "
        f"guides={len(guides)}, max_mismatches={args.max_mismatches}",
        file=sys.stderr,
    )

    if args.device == "npu":
        from .dispatch.npu import NpuArtifactsMissingError
        try:
            rows = npu_scan(
                chrom=chrom,
                seq=seq,
                guides=guides,
                pam_template=args.pam,
                max_mismatches=args.max_mismatches,
                op_name=args.op,
            )
        except NpuArtifactsMissingError as exc:
            print(
                f"bionpu scan --device npu: kernel artifacts missing.\n"
                f"  {exc}\n"
                f"\n"
                f"Build the kernel:\n"
                f"  cd src/bionpu/kernels/crispr/pam_filter && make NPU2=1\n"
                f"\n"
                f"Then either copy the produced build/{{early,late}}/final.xclbin\n"
                f"and insts.bin into bionpu/dispatch/_npu_artifacts/ or set\n"
                f"BIONPU_KERNEL_ARTIFACTS_DIR to the directory containing them.",
                file=sys.stderr,
            )
            return 3
    else:
        rows = cpu_scan(
            chrom=chrom,
            seq=seq,
            guides=guides,
            pam_template=args.pam,
            max_mismatches=args.max_mismatches,
        )

    rows = normalize(rows)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(out_path, rows)
    print(
        f"bionpu scan: wrote {len(rows)} hits to {out_path}",
        file=sys.stderr,
    )

    if args.verify:
        from .verify.crispr import compare_against_cas_offinder

        result = compare_against_cas_offinder(out_path, args.verify)
        print(_format_result(
            f"verify crispr: {out_path} vs {args.verify}", result
        ))
        return 0 if result.equal else 1
    return 0


def _cmd_design(args: argparse.Namespace) -> int:
    """Enumerate CRISPR guides and run the host-side seed prefilter."""
    from .genomics import GuideFilter, design_guides
    from .scan import read_fasta

    target_chrom, target_seq = read_fasta(args.target)
    ref_chrom, ref_seq = read_fasta(args.reference)
    pams = tuple(args.pam or ["NGG"])

    guide_filter = GuideFilter(
        min_gc=args.min_gc,
        max_gc=args.max_gc,
        max_n_fraction=args.max_n_fraction,
        max_homopolymer=args.max_homopolymer,
        min_entropy_bits=args.min_entropy_bits,
    )
    run = design_guides(
        target_seq,
        {ref_chrom or "reference": ref_seq},
        chrom=target_chrom or "target",
        pam_templates=pams,
        guide_filter=guide_filter,
        seed_length=args.seed_length,
        max_seed_mismatches=args.max_seed_mismatches,
        include_reverse=not args.forward_only,
        pam_aware=not args.ignore_reference_pam,
    )

    payload = {
        "target": {
            "chrom": target_chrom or "target",
            "bases": len(target_seq),
        },
        "reference": {
            "chrom": ref_chrom or "reference",
            "bases": len(ref_seq),
        },
        "pam_templates": list(pams),
        "seed_length": args.seed_length,
        "max_seed_mismatches": args.max_seed_mismatches,
        "n_candidates": len(run.candidates),
        "n_passing": len(run.passing_guides),
        "n_rejected": len(run.rejected_guides),
        "seed_hit_count": run.seed_hit_count,
        "guides": [_design_result_to_json(result) for result in run.results],
        "rejected": [
            _guide_candidate_to_json(guide) for guide in run.rejected_guides
        ],
    }

    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out == "-":
        sys.stdout.write(body)
    else:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body)
        print(
            f"bionpu design: wrote {len(run.results)} passing guides and "
            f"{run.seed_hit_count} seed hits to {out_path}",
            file=sys.stderr,
        )
    return 0


def _guide_candidate_to_json(guide) -> dict[str, object]:
    return {
        "guide_id": guide.guide_id,
        "spacer": guide.spacer,
        "pam": guide.pam,
        "chrom": guide.chrom,
        "window_start": guide.window_start,
        "spacer_start": guide.spacer_start,
        "strand": guide.strand,
        "canonical_key": guide.canonical_key,
        "gc": guide.gc,
        "n_fraction": guide.n_frac,
        "max_homopolymer": guide.max_homopolymer,
        "low_complexity": guide.low_complexity,
        "passes_filters": guide.passes_filters,
        "rejection_reasons": list(guide.rejection_reasons),
    }


def _design_result_to_json(result) -> dict[str, object]:
    guide = _guide_candidate_to_json(result.guide)
    guide.update(
        {
            "seed_hit_count": result.seed_hit_count,
            "exact_seed_hit_count": result.exact_seed_hit_count,
            "mismatched_seed_hit_count": result.mismatched_seed_hit_count,
            "reference_names": list(result.reference_names),
            "off_targets": [
                {
                    "ref_name": hit.ref_name,
                    "position": hit.position,
                    "strand": hit.strand,
                    "seed": hit.seed,
                    "target_seed": hit.target_seed,
                    "seed_mismatches": hit.seed_mismatches,
                    "seed_mismatch_positions": list(hit.seed_mismatch_positions),
                    "pam": hit.pam,
                }
                for hit in result.off_targets
            ],
        }
    )
    return guide


def _cmd_score(args: argparse.Namespace) -> int:
    """Score canonical scan rows with an off-target probability.

    v0.3 alpha: DNABERT-Epi (no-epi variant) on CPU or GPU. The AIE2P
    backend is a follow-up; the with-epi (BigWig) variant is a
    follow-up. The ``--smoke`` flag exercises the pipeline end-to-end
    without weights or torch (deterministic pseudo-random scores
    keyed by row identity), useful for CI and CLI demonstration.
    """
    from .data.canonical_sites import parse_tsv
    from .scoring.dnabert_epi import (
        DNABERTEpiNpuNotImplementedError,
        DNABERTEpiScorer,
        DNABERTEpiUnavailableError,
    )
    from .scoring.types import write_score_tsv

    rows = parse_tsv(pathlib.Path(args.candidates))
    print(
        f"bionpu score [{args.model}/{args.device}{' smoke' if args.smoke else ''}]: "
        f"{len(rows)} candidate rows from {args.candidates}",
        file=sys.stderr,
    )

    if args.model != "dnabert-epi":
        print(
            f"bionpu score: model {args.model!r} not yet wired; "
            f"only 'dnabert-epi' is available in v0.3 alpha.",
            file=sys.stderr,
        )
        return 2

    weights = pathlib.Path(args.weights) if args.weights else None
    passport = pathlib.Path(args.passport_dir) if args.passport_dir else None
    try:
        scorer = DNABERTEpiScorer(
            device=args.device,
            weights_path=weights,
            passport_dir=passport,
            smoke=args.smoke,
            seed=args.seed,
        )
        scored = list(scorer.score(rows))
    except DNABERTEpiUnavailableError as exc:
        print(f"bionpu score: {exc}", file=sys.stderr)
        return 3
    except DNABERTEpiNpuNotImplementedError as exc:
        print(f"bionpu score: {exc}", file=sys.stderr)
        return 4

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_score_tsv(out_path, scored)
    print(
        f"bionpu score: wrote {len(scored)} scored rows to {out_path}",
        file=sys.stderr,
    )

    if args.verify:
        from .verify.score import compare_score_outputs

        result = compare_score_outputs(
            out_path,
            args.verify,
            policy=args.verify_policy,
            epsilon=args.verify_epsilon if args.verify_policy == "NUMERIC_EPSILON" else None,
        )
        verdict = "EQUAL" if result.equal else "DIVERGENT"
        print(f"verify score [{result.policy}]: {out_path} vs {args.verify}")
        print(f"  result:        {verdict}")
        print(f"  records:       {result.record_count}")
        print(f"  a sha256:      {result.a_sha256}")
        print(f"  b sha256:      {result.b_sha256}")
        if result.max_abs_diff is not None:
            print(f"  max |a-b|:     {result.max_abs_diff:.6g}")
        if not result.equal and result.divergences:
            print("  divergences:")
            for div in result.divergences[:8]:
                print(f"    [{div.record_index}] {div.message}")
            if len(result.divergences) > 8:
                print(f"    ... ({len(result.divergences) - 8} more)")
        return 0 if result.equal else 1
    return 0


def _cmd_score_quantize(args: argparse.Namespace) -> int:
    """Calibrate DNABERT-Epi to INT8 + write quantization passport.

    Produces the AIE2P-targeted quant passport (raw INT8 weight
    arrays + per-channel scales + per-tensor activation scales) that
    the v0.4 NPU scorer kernels consume. See
    docs/aie2p-scorer-port-design.md for the full pipeline.
    """
    from .scoring.quantize import calibrate_dnabert_epi

    weights = pathlib.Path(args.weights)
    out_dir = pathlib.Path(args.out_dir)
    try:
        passport = calibrate_dnabert_epi(
            weights_path=weights,
            out_dir=out_dir,
            calibration_corpus_id=args.calibration_id,
            n_samples_target=args.n_samples,
        )
    except RuntimeError as exc:
        print(f"bionpu score-quantize: {exc}", file=sys.stderr)
        return 3
    print(
        f"bionpu score-quantize: wrote passport for {len(passport.weights)} "
        f"linear layers + {len(passport.activations)} activation scales "
        f"(n_calib={passport.n_calibration_samples}) to {out_dir}",
        file=sys.stderr,
    )
    return 0


def _cmd_score_extract_head(args: argparse.Namespace) -> int:
    """Convert an upstream DNABERT-Epi checkpoint into a bionpu head dict.

    The upstream training pipeline saves the full ``DNABERTEpiModule``
    state dict (BERT body + epi-encoder + gating MLP + classifier).
    bionpu's no-epi scorer only needs the classifier head's
    ``Linear(768, 2)`` weights; this command extracts them into a
    state_dict file that ``bionpu score --weights ...`` can load.
    """
    try:
        import torch
    except ImportError:
        print(
            "bionpu score-extract-head: torch is required for this command "
            "(install with `pip install torch`).",
            file=sys.stderr,
        )
        return 3

    from .scoring._extract_head import ExtractError, extract_no_epi_head

    upstream_path = pathlib.Path(args.upstream_checkpoint)
    out_path = pathlib.Path(args.out)
    try:
        bionpu_state = extract_no_epi_head(upstream_path)
    except ExtractError as exc:
        print(f"bionpu score-extract-head: {exc}", file=sys.stderr)
        return 4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bionpu_state, out_path)
    print(
        f"bionpu score-extract-head: wrote {out_path} "
        f"(weight shape {tuple(bionpu_state['1.weight'].shape)}, "
        f"bias shape {tuple(bionpu_state['1.bias'].shape)})",
        file=sys.stderr,
    )
    return 0


def _cmd_bench_probe(args: argparse.Namespace) -> int:
    """Probe per-device energy readers on this host.

    Cheap, side-effect free. Reports AVAILABLE / UNAVAILABLE for
    RAPL (CPU), nvidia-smi (GPU), xrt-smi (NPU) per
    docs/POWER_DOMAINS.md. Stub fallback is a documented operating
    mode, not an error — exit 0 always; ``--require-all-real``
    exits 1 if any reader falls back to a stub.
    """
    import json

    from .bench.probe import probe_readers

    report = probe_readers()
    if args.format == "json":
        print(json.dumps(report.to_json(), indent=2))
    else:
        print(f"bench probe: {report.hostname} ({report.platform_str})")
        for r in report.readers:
            tag = "AVAILABLE" if r.available else "UNAVAILABLE"
            print(f"  {r.device:<3}  {tag:<11}  source={r.source}")
            print(f"       detail: {r.detail}")
    if args.require_all_real and not report.all_real():
        return 1
    return 0


def _cmd_bench_scan(args: argparse.Namespace) -> int:
    """Time + measure energy of a `bionpu scan` invocation.

    Wraps the existing :func:`bionpu.scan.cpu_scan` /
    :func:`bionpu.scan.npu_scan` paths in
    :class:`bionpu.bench.harness.TimedRun` with the best-available
    energy reader for the chosen device. Emits a single
    ``measurements.json`` record matching ``bench/schema.json``.

    The scan output TSV is written alongside (so byte-equivalence
    can be verified against a reference); the bench JSON carries
    timing / energy / RSS / VRAM-peak only.
    """
    import json

    from .bench.energy import auto_reader
    from .bench.harness import TimedRun
    from .data.canonical_sites import normalize, write_tsv
    from .scan import cpu_scan, npu_scan, parse_guides, read_fasta

    chrom, seq = read_fasta(args.target)
    guides = parse_guides(args.guides)

    reader = auto_reader(args.device)
    print(
        f"bionpu bench scan [{args.device}]: chrom={chrom!r}, "
        f"seq={len(seq):,} nt, guides={len(guides)}, "
        f"reader={type(reader).__name__}",
        file=sys.stderr,
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path = pathlib.Path(args.bench_out) if args.bench_out else None

    with TimedRun(
        track="crispr",
        op="scan",
        device=args.device,
        energy_reader=reader,
    ) as run:
        run.record_units(bases=len(seq), guides=len(guides))
        if args.device == "npu":
            from .dispatch.npu import NpuArtifactsMissingError
            try:
                rows = npu_scan(
                    chrom=chrom, seq=seq, guides=guides,
                    pam_template=args.pam, max_mismatches=args.max_mismatches,
                )
            except NpuArtifactsMissingError as exc:
                print(f"bionpu bench scan: {exc}", file=sys.stderr)
                return 3
        else:
            rows = cpu_scan(
                chrom=chrom, seq=seq, guides=guides,
                pam_template=args.pam, max_mismatches=args.max_mismatches,
            )
        rows = normalize(rows)

    write_tsv(out_path, rows)
    measurement = run.measurements.to_json()  # type: ignore[union-attr]
    measurement["accuracy"] = {"hits": len(rows)}

    if bench_path is not None:
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        bench_path.write_text(json.dumps(measurement, indent=2) + "\n")
        print(f"bionpu bench scan: wrote {bench_path}", file=sys.stderr)
    else:
        print(json.dumps(measurement, indent=2))
    return 0


def _cmd_kmer_count(args: argparse.Namespace) -> int:
    """Count canonical k-mers from a packed-2-bit DNA binary (T16).

    Two device paths share an output formatter so byte-equality
    between CPU oracle (T2 ``count_kmers_canonical``) and NPU silicon
    (T9 ``BionpuKmerCount`` via ``get_kmer_count_op``) is verifiable
    on the smoke fixture.

    Wire format (input): packed-2-bit MSB-first bytes per T1's
    contract — the T3 fixture builder produces these with no header
    or trailer, so ``n_bases = file_size_bytes * 4`` is exact for the
    standard fixtures (smoke / synthetic_1mbp / chr22).

    Output (Jellyfish-FASTA per Q5): interleaved
    ``>count\\nACGT-kmer\\n`` records sorted by
    ``(count desc, canonical asc)`` matching T7 runner format.
    """
    import numpy as np

    from .data.kmer_oracle import count_kmers_canonical, unpack_dna_2bit
    from .kernels.genomics import (
        KMER_COUNT_VALID_K,
        KMER_COUNT_VALID_N_TILES,
        get_kmer_count_op,
    )
    from .kernels.genomics.kmer_count import decode_packed_kmer_to_ascii

    # n=8 is in the helper's KMER_COUNT_VALID_N_TILES but the v1 PRD
    # deferral (gaps.yaml `kmer-n8-memtile-dma-cap`) drops it from the
    # CLI-allowed set. Reject explicitly with a clear error before the
    # helper would happily accept it.
    if int(args.launch_chunks) == 8:
        print(
            "bionpu kmer-count: --launch-chunks 8 is rejected: deferred for "
            "v1 (gaps.yaml `kmer-n8-memtile-dma-cap` — n=8 exceeds the "
            "memtile DMA capacity in the current toolchain). Allowed "
            "values: 1, 2, 4.",
            file=sys.stderr,
        )
        return 2

    # argparse `choices` already constrains k and launch_chunks; this
    # is a defensive belt-and-braces against future arg typo.
    if int(args.k) not in KMER_COUNT_VALID_K:
        print(
            f"bionpu kmer-count: unsupported --k {args.k!r}; expected one "
            f"of {KMER_COUNT_VALID_K}.",
            file=sys.stderr,
        )
        return 2
    if int(args.launch_chunks) not in KMER_COUNT_VALID_N_TILES:
        print(
            f"bionpu kmer-count: unsupported --launch-chunks "
            f"{args.launch_chunks!r}; expected one of "
            f"{KMER_COUNT_VALID_N_TILES}.",
            file=sys.stderr,
        )
        return 2

    input_path = pathlib.Path(args.input)
    if not input_path.is_file():
        print(
            f"bionpu kmer-count: input file not found: {input_path}",
            file=sys.stderr,
        )
        return 2

    # Load the packed-2-bit binary. T3 fixture builder produces
    # header-less output; n_bases = file_size_bytes * 4 (full bytes,
    # no slack).
    buf = np.fromfile(input_path, dtype=np.uint8)
    n_bases = int(buf.size) * 4

    print(
        f"bionpu kmer-count [{args.device}]: input={input_path} "
        f"({buf.size:,} bytes / {n_bases:,} bases), k={args.k}, "
        f"top={args.top}, threshold={args.threshold}, "
        f"launch_chunks={args.launch_chunks}",
        file=sys.stderr,
    )

    # ---- compute (canonical, count) records ------------------------------ #
    if args.device == "npu":
        from .dispatch.npu import NpuArtifactsMissingError
        from .dispatch.npu_silicon_lock import npu_silicon_lock

        try:
            op = get_kmer_count_op(k=int(args.k), n_tiles=int(args.launch_chunks))
        except (KeyError, ValueError) as exc:
            print(f"bionpu kmer-count: {exc}", file=sys.stderr)
            return 3

        label = (
            args.silicon_lock_label
            or f"bionpu_kmer_count:k{args.k}_n{args.launch_chunks}"
        )
        try:
            with npu_silicon_lock(label=label):
                records = op(
                    packed_seq=buf,
                    top_n=int(args.top),
                    threshold=int(args.threshold),
                )
        except NpuArtifactsMissingError as exc:
            print(
                f"bionpu kmer-count --device npu: kernel artifacts missing.\n"
                f"  {exc}",
                file=sys.stderr,
            )
            return 3
    elif args.device == "cpu":
        # Decode the packed input back to ACGT for the oracle. The
        # task warns that chr22 is 50 MB as a string — for fixtures of
        # that size users should prefer --device npu. Smoke + 1 Mbp
        # fixture decode quickly.
        seq = unpack_dna_2bit(buf, n_bases)
        counts = count_kmers_canonical(seq, int(args.k))
        # Match T9's host post-pass: threshold + sort + topN.
        items = [
            (canonical, count)
            for canonical, count in counts.items()
            if count >= int(args.threshold)
        ]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        if int(args.top) > 0:
            items = items[: int(args.top)]
        records = items
    else:  # pragma: no cover — argparse `choices` already constrains
        print(
            f"bionpu kmer-count: unknown --device {args.device!r}",
            file=sys.stderr,
        )
        return 2

    # ---- format Jellyfish-FASTA output ----------------------------------- #
    lines: list[str] = []
    for canonical, count in records:
        kmer_ascii = decode_packed_kmer_to_ascii(int(canonical), int(args.k))
        lines.append(f">{int(count)}")
        lines.append(kmer_ascii)
    body = "".join(line + "\n" for line in lines)

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body)
        print(
            f"bionpu kmer-count: wrote {len(records)} k-mer records to "
            f"{out_path}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(body)
    return 0


def _cmd_trim(args: argparse.Namespace) -> int:
    """Trim 3' adapters from FASTQ reads (bionpu trim v0).

    First production composition built on top of the
    silicon-validated CRISPR-shape primitives. Wraps
    :class:`bionpu.kernels.genomics.primer_scan.BionpuPrimerScan` v0
    with FASTQ I/O and per-read trim post-processing. Mirrors
    ``cutadapt -a ADAPTER --no-indels -e 0`` semantics:

    * Forward-strand exact match anywhere in the read.
    * Trim at the leftmost match position (drop adapter + 3' tail).
    * Per-read silicon dispatch (subprocess; npu_silicon_lock-wrapped).
    * Reads with non-ACGT bases fall back to the CPU oracle so
      cutadapt's "non-ACGT resets the rolling state" semantic is
      preserved.

    See :mod:`bionpu.genomics.adapter_trim` for the in-process API.
    """
    from .genomics.adapter_trim.cli import run_cli

    return run_cli(args)


def _cmd_crispr_design(args: argparse.Namespace) -> int:
    """End-to-end CRISPR guide design (PRD-guide-design-on-xdna v0.2 Tier 1).

    Wires the five-stage pipeline (target resolution -> PAM scan ->
    off-target scan -> on/off-target scoring -> rank+emit) for
    Mode A (gene symbol) input and TSV output. See
    :func:`bionpu.genomics.crispr_design.design_guides_for_target` for
    the in-process API and the Tier 1 scope-narrowing notes.
    """
    from .genomics.crispr_design import (
        DEFAULT_GC_MAX,
        DEFAULT_GC_MIN,
        DEFAULT_MAX_MISMATCHES,
        DEFAULT_TOP_N,
        GeneNotFoundError,
        design_guides_for_target,
        format_result_json,
    )

    target_fasta_path = pathlib.Path(args.target_fasta) if args.target_fasta else None
    fasta_path = pathlib.Path(args.fasta or _default_grch38_fasta())
    if target_fasta_path is None and not fasta_path.is_file():
        print(
            f"bionpu crispr design: reference FASTA not found at "
            f"{fasta_path!s}. Pass --fasta <path-to-grch38.fa> to override "
            f"(Tier 1 default looks at "
            f"data_cache/genomes/grch38/hg38.fa).",
            file=sys.stderr,
        )
        return 2

    try:
        result = design_guides_for_target(
            target=args.target,
            genome=args.genome,
            fasta_path=fasta_path,
            top_n=int(args.top),
            max_mismatches=int(args.mismatches),
            gc_min=float(args.gc_min),
            gc_max=float(args.gc_max),
            device=args.device,
            rank_by=args.rank_by,
            silicon_lock_label=args.silicon_lock_label,
            target_fasta_path=target_fasta_path,
        )
    except GeneNotFoundError as exc:
        print(f"bionpu crispr design: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"bionpu crispr design: {exc}", file=sys.stderr)
        return 2

    out_bytes = (
        format_result_json(result)
        if args.format == "json"
        else result.tsv_bytes
    )

    if args.output and args.output != "-":
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(out_bytes)
        print(
            f"bionpu crispr design: wrote {len(result.ranked)} ranked "
            f"guides as {args.format} to {out_path} "
            f"(target={result.target.gene} "
            f"{result.target.chrom}:{result.target.start}-{result.target.end}, "
            f"locus={result.target.length:,} bp, "
            f"candidates={result.n_candidates_total}, "
            f"off_target_hits={result.n_off_target_hits})",
            file=sys.stderr,
        )
        for stage, secs in result.stage_timings_s.items():
            print(f"  stage {stage}: {secs:.3f}s", file=sys.stderr)
    else:
        sys.stdout.buffer.write(out_bytes)
    return 0


def _default_grch38_fasta() -> pathlib.Path:
    """Tier 1 default reference path.

    Resolves to ``$BIONPU_GRCH38_FASTA`` if set, else the in-tree cache
    path ``data_cache/genomes/grch38/hg38.fa``. Production callers
    should pass ``--fasta`` explicitly; this default is a developer
    convenience.
    """
    import os

    env = os.environ.get("BIONPU_GRCH38_FASTA")
    if env:
        return pathlib.Path(env)
    return pathlib.Path("data_cache/genomes/grch38/hg38.fa")


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print(
        f"bionpu {args.cmd}: not yet implemented in v0.1. "
        f"See README.md / docs/REPRODUCE.md for the v0.1 scope.",
        file=sys.stderr,
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bionpu",
        description=(
            "AIE2P-accelerated genomics with reference-equivalence verification."
        ),
    )
    p.add_argument("--version", action="version",
                   version=f"bionpu {__version__}")
    sub = p.add_subparsers(dest="cmd")

    # verify
    p_verify = sub.add_parser(
        "verify",
        help="Byte-equality check of NPU output vs canonical reference.",
    )
    sub_verify = p_verify.add_subparsers(dest="verify_kind")

    p_v_crispr = sub_verify.add_parser(
        "crispr",
        help="Compare NPU off-target TSV vs Cas-OFFinder reference TSV.",
    )
    p_v_crispr.add_argument("npu_tsv", help="NPU-emitted hits TSV")
    p_v_crispr.add_argument("ref_tsv", help="Cas-OFFinder reference TSV")
    p_v_crispr.add_argument("--max-divergences", type=int, default=16)
    p_v_crispr.set_defaults(func=_cmd_verify_crispr)

    p_v_bc = sub_verify.add_parser(
        "basecalling",
        help="Compare NPU-emitted FASTQ vs Dorado reference FASTQ.",
    )
    p_v_bc.add_argument("npu_fastq", help="NPU-emitted FASTQ")
    p_v_bc.add_argument("ref_fastq", help="Dorado reference FASTQ")
    p_v_bc.add_argument("--max-divergences", type=int, default=16)
    p_v_bc.set_defaults(func=_cmd_verify_basecalling)

    # scan
    p_scan = sub.add_parser(
        "scan",
        help="CRISPR off-target scan (CPU in v0.1; NPU is v0.2 scope).",
    )
    p_scan.add_argument(
        "--target",
        required=True,
        help="FASTA file with the target sequence (single record).",
    )
    p_scan.add_argument(
        "--guides",
        required=True,
        help=(
            "Comma-separated list of 20-nt ACGT spacers, OR a path to a "
            "guide-list file (one spacer per line; `id:spacer` allowed; "
            "`#` comments OK)."
        ),
    )
    p_scan.add_argument(
        "--out",
        required=True,
        help="Output canonical TSV path.",
    )
    p_scan.add_argument(
        "--pam",
        default="NGG",
        help="PAM template (only NGG supported in v0.1).",
    )
    p_scan.add_argument(
        "--max-mismatches",
        type=int,
        default=4,
        help="Maximum spacer mismatches.",
    )
    p_scan.add_argument(
        "--device",
        choices=["cpu", "npu"],
        default="cpu",
        help="Compute device. cpu = pure-numpy fallback; npu = AIE2P PAM-filter kernel.",
    )
    p_scan.add_argument(
        "--op",
        default="crispr_pam_filter_early",
        choices=["crispr_pam_filter_early", "crispr_pam_filter_late"],
        help=(
            "NPU op variant (only meaningful with --device npu). "
            "filter-early is the production path; filter-late is a "
            "comparison artifact with the same output bytes but "
            "different on-tile work distribution."
        ),
    )
    p_scan.add_argument(
        "--verify",
        default=None,
        help=(
            "If supplied, compare the scan output byte-equally against this "
            "reference TSV (via bionpu.verify.crispr). Exits 0 on equality, "
            "1 on divergence."
        ),
    )
    p_scan.set_defaults(func=_cmd_scan)

    # design
    p_design = sub.add_parser(
        "design",
        help="CRISPR guide enumeration plus CPU seed-off-target prefilter.",
    )
    p_design.add_argument(
        "--target",
        required=True,
        help="FASTA file with the target sequence to enumerate guides from.",
    )
    p_design.add_argument(
        "--reference",
        required=True,
        help="FASTA file with the reference sequence to seed-prefilter against.",
    )
    p_design.add_argument(
        "--out",
        required=True,
        help="Output JSON path, or '-' for stdout.",
    )
    p_design.add_argument(
        "--pam",
        action="append",
        default=None,
        help="PAM template to accept. May be repeated. Defaults to NGG.",
    )
    p_design.add_argument(
        "--seed-length",
        type=int,
        default=12,
        help="PAM-proximal seed length used for off-target prefiltering.",
    )
    p_design.add_argument(
        "--max-seed-mismatches",
        type=int,
        default=0,
        help="Maximum seed mismatches retained by the prefilter.",
    )
    p_design.add_argument(
        "--min-gc",
        type=float,
        default=0.40,
        help="Minimum guide-spacer GC fraction.",
    )
    p_design.add_argument(
        "--max-gc",
        type=float,
        default=0.70,
        help="Maximum guide-spacer GC fraction.",
    )
    p_design.add_argument(
        "--max-n-fraction",
        type=float,
        default=0.0,
        help="Maximum allowed N fraction in a guide spacer.",
    )
    p_design.add_argument(
        "--max-homopolymer",
        type=int,
        default=4,
        help="Maximum allowed same-base run in a guide spacer.",
    )
    p_design.add_argument(
        "--min-entropy-bits",
        type=float,
        default=1.2,
        help="Minimum guide-spacer Shannon entropy in bits.",
    )
    p_design.add_argument(
        "--forward-only",
        action="store_true",
        help="Only enumerate forward-strand target candidates.",
    )
    p_design.add_argument(
        "--ignore-reference-pam",
        action="store_true",
        help="Seed-prefilter reference windows without checking adjacent PAMs.",
    )
    p_design.set_defaults(func=_cmd_design)

    # score
    p_score = sub.add_parser(
        "score",
        help=(
            "Score canonical scan rows with an off-target probability "
            "(v0.3 alpha: DNABERT-Epi, no-epi, CPU+GPU)."
        ),
    )
    p_score.add_argument(
        "--candidates",
        required=True,
        help="Canonical scan TSV (output of `bionpu scan`).",
    )
    p_score.add_argument(
        "--out",
        required=True,
        help="Output scored canonical TSV.",
    )
    p_score.add_argument(
        "--model",
        default="dnabert-epi",
        choices=["dnabert-epi"],
        help="Scorer model. Only dnabert-epi is wired in v0.3 alpha.",
    )
    p_score.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu", "npu"],
        help=(
            "Compute device. cpu = baseline + byte-equivalence reference; "
            "gpu = CUDA-accelerated (requires torch.cuda); "
            "npu = AIE2P silicon (v0.4 milestone — currently raises "
            "DNABERTEpiNpuNotImplementedError; see "
            "docs/aie2p-scorer-port-design.md)."
        ),
    )
    p_score.add_argument(
        "--weights",
        default=None,
        help=(
            "Path to a fine-tuned classifier checkpoint. Required for "
            "--device cpu / gpu unless --smoke is set."
        ),
    )
    p_score.add_argument(
        "--passport-dir",
        default=None,
        help=(
            "Directory produced by `bionpu score-quantize` (contains "
            "passport.json + weights.npz). Required for --device npu."
        ),
    )
    p_score.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use deterministic pseudo-random scores keyed by row identity. "
            "No torch / weights required; intended for CI and CLI demos."
        ),
    )
    p_score.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Salt for --smoke mode (no effect on real scoring).",
    )
    p_score.add_argument(
        "--verify",
        default=None,
        help=(
            "If supplied, compare the score output against this reference "
            "score TSV under the policy chosen by --verify-policy."
        ),
    )
    p_score.add_argument(
        "--verify-policy",
        default="NUMERIC_EPSILON",
        choices=["NUMERIC_EPSILON", "BITWISE_EXACT"],
        help=(
            "Equivalence policy when --verify is set. "
            "BITWISE_EXACT for deterministic backends (e.g. CPU vs CPU); "
            "NUMERIC_EPSILON for cross-device (CPU vs GPU vs NPU) where "
            "ULP-level deviation is expected."
        ),
    )
    p_score.add_argument(
        "--verify-epsilon",
        type=float,
        default=1e-6,
        help="Per-row score tolerance for NUMERIC_EPSILON. Default 1e-6.",
    )
    p_score.set_defaults(func=_cmd_score)

    # score-extract-head — upstream-checkpoint → bionpu-head converter
    p_xhead = sub.add_parser(
        "score-extract-head",
        help=(
            "Convert an upstream DNABERT-Epi state-dict checkpoint into "
            "a bionpu-loadable no-epi head state-dict (the bridge "
            "between third_party/crispr_dnabert training output and "
            "`bionpu score --weights`)."
        ),
    )
    p_xhead.add_argument(
        "--upstream-checkpoint",
        required=True,
        help="Path to the upstream DNABERT-Epi state_dict (.pt).",
    )
    p_xhead.add_argument(
        "--out",
        required=True,
        help="Output path for the bionpu head state_dict.",
    )
    p_xhead.set_defaults(func=_cmd_score_extract_head)

    # score-quantize — INT8 calibration + AIE2P quantization passport
    p_quant = sub.add_parser(
        "score-quantize",
        help=(
            "Calibrate a fine-tuned DNABERT-Epi state-dict to INT8 "
            "and write the AIE2P-targeted quantization passport "
            "(raw INT8 weights + per-channel scales + per-tensor "
            "activation scales). v0.4 milestone — feeds the NPU "
            "scorer kernels."
        ),
    )
    p_quant.add_argument(
        "--weights",
        required=True,
        help="Path to the fine-tuned classifier head state-dict.",
    )
    p_quant.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for passport.json + weights.npz.",
    )
    p_quant.add_argument(
        "--calibration-id",
        default="synthetic_seed42_n128",
        help=(
            "Stable identifier for the calibration corpus, recorded "
            "in the passport. Default uses the deterministic "
            "synthetic corpus baked into the calibrator."
        ),
    )
    p_quant.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="Cap on calibration samples consumed.",
    )
    p_quant.set_defaults(func=_cmd_score_quantize)

    # bench
    p_bench = sub.add_parser(
        "bench",
        help=(
            "Energy + timing harness. `bench probe` reports per-device "
            "energy-reader availability; `bench scan` wraps a CRISPR scan "
            "in the timed harness and emits a measurements.json record."
        ),
    )
    sub_bench = p_bench.add_subparsers(dest="bench_kind")

    p_b_probe = sub_bench.add_parser(
        "probe",
        help=(
            "Report per-device energy-reader availability "
            "(RAPL / nvidia-smi / xrt-smi). Side-effect-free."
        ),
    )
    p_b_probe.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format. Default: human-readable text.",
    )
    p_b_probe.add_argument(
        "--require-all-real", action="store_true",
        help="Exit 1 if any reader falls back to a stub (CI gate).",
    )
    p_b_probe.set_defaults(func=_cmd_bench_probe)

    p_b_scan = sub_bench.add_parser(
        "scan",
        help="Wrap a `bionpu scan` invocation in the timed/energy harness.",
    )
    p_b_scan.add_argument("--target", required=True, help="FASTA target.")
    p_b_scan.add_argument(
        "--guides", required=True,
        help="Comma-separated 20-nt spacers, or a guide-list file path.",
    )
    p_b_scan.add_argument("--out", required=True, help="Canonical scan TSV path.")
    p_b_scan.add_argument(
        "--bench-out", default=None,
        help=(
            "measurements.json path. If omitted, the JSON is printed to stdout."
        ),
    )
    p_b_scan.add_argument(
        "--device", choices=["cpu", "npu"], default="cpu",
        help="Compute device for the scan stage.",
    )
    p_b_scan.add_argument("--pam", default="NGG", help="PAM template.")
    p_b_scan.add_argument("--max-mismatches", type=int, default=4)
    p_b_scan.set_defaults(func=_cmd_bench_scan)

    # kmer-count — canonical k-mer counting on CPU (oracle) or NPU silicon.
    p_kmer = sub.add_parser(
        "kmer-count",
        help=(
            "Count canonical k-mers from a packed-2-bit DNA binary "
            "(CPU oracle via T2 count_kmers_canonical or NPU silicon "
            "via T9 BionpuKmerCount). Output is Jellyfish-FASTA "
            "(>count\\nkmer\\n) sorted by (count desc, canonical asc)."
        ),
    )
    p_kmer.add_argument(
        "input",
        help=(
            "Path to a packed-2-bit DNA binary (T3 fixture format: "
            "MSB-first 2-bit packing, A=00, C=01, G=10, T=11; "
            "n_bases = file_size_bytes * 4)."
        ),
    )
    p_kmer.add_argument(
        "--k", type=int, choices=[15, 21, 31], default=21,
        help="K-mer width. Default: 21.",
    )
    p_kmer.add_argument(
        "--output", default=None,
        help="Output Jellyfish-FASTA path. Default: stdout.",
    )
    p_kmer.add_argument(
        "--top", type=int, default=1000,
        help="Keep only the top-N most-frequent k-mers (0 = no cap). Default: 1000.",
    )
    p_kmer.add_argument(
        "--threshold", type=int, default=1,
        help="Drop k-mers with count below this. Default: 1.",
    )
    p_kmer.add_argument(
        "--device", choices=["cpu", "npu"], default="npu",
        help="Compute device. cpu = T2 numpy oracle; npu = AIE2P silicon. Default: npu.",
    )
    p_kmer.add_argument(
        "--launch-chunks", type=int, choices=[1, 2, 4, 8], default=4,
        help=(
            "Tile fan-out for the NPU dispatch (ignored on --device cpu). "
            "Default: 4. Note: n=8 is rejected at runtime per the v1 "
            "deferral (gaps.yaml `kmer-n8-memtile-dma-cap`)."
        ),
    )
    p_kmer.add_argument(
        "--silicon-lock-label", default=None,
        help=(
            "Optional label written to the NPU silicon lock PID sidecar "
            "(/tmp/bionpu-npu-silicon.pid) for diagnostic. Defaults to "
            "`bionpu_kmer_count:k{K}_n{N}`."
        ),
    )
    p_kmer.set_defaults(func=_cmd_kmer_count)

    # trim — adapter trimming (bionpu trim v0; primer_scan composition).
    # The arg flags mirror the standalone CLI in
    # bionpu.genomics.adapter_trim.cli so users can swap freely.
    p_trim = sub.add_parser(
        "trim",
        help=(
            "Trim 3' adapters from FASTQ reads using AIE2P silicon "
            "(primer_scan v0 composition). cutadapt -a compatible."
        ),
    )
    p_trim.add_argument(
        "--adapter", "-a",
        default="AGATCGGAAGAGC",
        help=(
            "Adapter sequence to trim. Length must be 13, 20, or 25 "
            "(silicon-pinned primer lengths). Default: AGATCGGAAGAGC "
            "(TruSeq P5)."
        ),
    )
    p_trim.add_argument(
        "--in", dest="in_path", required=True,
        help="Input FASTQ path. Gzip auto-detected by .gz suffix.",
    )
    p_trim.add_argument(
        "--out", dest="out_path", required=True,
        help="Output FASTQ path. Gzip auto-detected by .gz suffix.",
    )
    p_trim.add_argument(
        "--device", choices=["cpu", "npu"], default="npu",
        help=(
            "Compute device. cpu = CPU oracle reference; "
            "npu = AIE2P silicon (subprocess host_runner; "
            "npu_silicon_lock-wrapped). Default: npu."
        ),
    )
    p_trim.add_argument(
        "--n-tiles", type=int, choices=[1, 2, 4, 8], default=4,
        help="NPU tile fan-out (ignored on --device cpu). Default: 4.",
    )
    p_trim.add_argument(
        "--batch-size", type=int, default=1024,
        help=(
            "v1 batched-dispatch size: pack N reads into a single NPU "
            "dispatch (Path B sentinel-separated stream). 1 = v0 per-record "
            "dispatch. Default: 1024 (sized to clear the 5K reads/s gate "
            "given the ~100ms subprocess-dispatch floor). "
            "Ignored on --device cpu."
        ),
    )
    p_trim.add_argument(
        "--no-progress", action="store_true",
        help="Suppress periodic progress reports on stderr.",
    )
    p_trim.add_argument(
        "--quiet", action="store_true",
        help="Also suppress the post-run summary on stderr.",
    )
    p_trim.set_defaults(func=_cmd_trim)

    # crispr — umbrella for the end-to-end guide design wrapper
    # (PRD-guide-design-on-xdna v0.2). The two-level shape (`bionpu
    # crispr design ...`) leaves room for siblings (`bionpu crispr
    # screen`, `bionpu crispr verify`, ...) without further reshuffles.
    p_crispr = sub.add_parser(
        "crispr",
        help=(
            "CRISPR end-to-end guide design wrapper (Tier 1: BRCA1 only, "
            "locus-scope off-target scan)."
        ),
    )
    sub_crispr = p_crispr.add_subparsers(dest="crispr_kind")

    p_c_design = sub_crispr.add_parser(
        "design",
        help=(
            "Enumerate candidate guides for a gene, coordinate range, or FASTA target, score on/off-"
            "target activity, and emit a ranked TSV."
        ),
    )
    p_c_design.add_argument(
        "--target",
        required=True,
        help=(
            "Gene symbol (Mode A), 1-based inclusive coordinate range "
            "(Mode B, e.g. chr17:43044295-43125483), or, with "
            "--target-fasta, only the output label."
        ),
    )
    p_c_design.add_argument(
        "--genome",
        default="GRCh38",
        choices=["GRCh38", "none"],
        help="Reference build. Use 'none' with --target-fasta.",
    )
    p_c_design.add_argument(
        "--fasta",
        default=None,
        help=(
            "Path to the GRCh38 reference FASTA. Defaults to "
            "$BIONPU_GRCH38_FASTA, then to the in-tree cache "
            "data_cache/genomes/grch38/hg38.fa."
        ),
    )
    p_c_design.add_argument(
        "--target-fasta",
        default=None,
        help=(
            "Mode C: design guides against the first record in a local "
            "target FASTA using local coordinates. Does not require GRCh38."
        ),
    )
    p_c_design.add_argument(
        "--top",
        type=int,
        default=10,  # Tier 1 brief; PRD §3.1 default is 20.
        help="Emit the top-N ranked guides. Default 10 (Tier 1).",
    )
    p_c_design.add_argument(
        "--mismatches",
        type=int,
        default=4,
        help="Maximum off-target mismatches considered. Default 4.",
    )
    p_c_design.add_argument(
        "--gc-min",
        type=float,
        default=25.0,
        help="GC%% lower bound for the LOW_GC advisory flag. Default 25.",
    )
    p_c_design.add_argument(
        "--gc-max",
        type=float,
        default=75.0,
        help="GC%% upper bound for the HIGH_GC advisory flag. Default 75.",
    )
    p_c_design.add_argument(
        "--rank-by",
        default="crispor",
        choices=["crispor", "bionpu"],
        help=(
            "Composite score that drives the top-N sort. Default "
            "'crispor' (mimic CRISPOR's CLI). 'bionpu' uses the "
            "v1 baseline linear re-weight of RS1+CFD (PRD §7.1 Q5)."
        ),
    )
    p_c_design.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "npu"],
        help=(
            "Compute device for the PAM/off-target scan stages. Default "
            "'cpu' (numpy reference path) for Tier 1; 'npu' wraps every "
            "silicon submission in bionpu.dispatch.npu_silicon_lock per "
            "CLAUDE.md non-negotiable, and requires the precompiled "
            "CRISPR xclbins under bionpu/dispatch/_npu_artifacts/."
        ),
    )
    p_c_design.add_argument(
        "--silicon-lock-label",
        default=None,
        help=(
            "Optional label written to the NPU silicon lock PID sidecar "
            "(/tmp/bionpu-npu-silicon.pid). Defaults to "
            "`bionpu_crispr_design:{gene}`."
        ),
    )
    p_c_design.add_argument(
        "--format",
        default="tsv",
        choices=["tsv", "json"],
        help="Output format. Default: tsv.",
    )
    p_c_design.add_argument(
        "--output",
        default="-",
        help="Output path. '-' (default) writes to stdout.",
    )
    p_c_design.set_defaults(func=_cmd_crispr_design)

    # crispr -> pe -> design — Track B v0 prime-editor pegRNA design
    # (PRD-crispr-state-of-the-art-roadmap §3.2). The 3-level shape
    # (`bionpu crispr pe design ...`) sits as a sibling of
    # `bionpu crispr design` so the existing nuclease-design CLI is
    # preserved unchanged. `pe` is a namespace for prime-editing-only
    # subcommands; future siblings (`bionpu crispr pe screen`, etc.)
    # can land here without further nesting churn.
    p_c_pe = sub_crispr.add_parser(
        "pe",
        help=(
            "Prime editor pegRNA tools (Track B v0). See "
            "`bionpu crispr pe design --help` for the design subcommand."
        ),
    )
    sub_c_pe = p_c_pe.add_subparsers(dest="pe_kind")
    from .genomics.pe_design.cli import add_pe_design_subparser
    add_pe_design_subparser(sub_c_pe)

    # be — Track A v0 base editor design (PRD-crispr-state-of-the-art-roadmap §3.1).
    # The two-level shape (`bionpu be design ...`) leaves room for siblings
    # (`bionpu be score`, `bionpu be batch`, ...) without further reshuffles.
    p_be = sub.add_parser(
        "be",
        help=(
            "Base editor (ABE/CBE) design. Track A v0: SpCas9 wt + "
            "SpCas9-NG; BE4max + ABE7.10."
        ),
    )
    sub_be = p_be.add_subparsers(dest="be_kind")

    from .genomics.be_design.cli import add_be_design_subparser
    add_be_design_subparser(sub_be)

    # library — Track C v0 pooled CRISPR library design
    # (PRD-crispr-state-of-the-art-roadmap §3.3). Two-level shape
    # (`bionpu library design ...`) leaves room for `bionpu library
    # validate`, `bionpu library balance`, etc.
    p_lib = sub.add_parser(
        "library",
        help=(
            "Genome-scale pooled CRISPR library design. Track C v0: "
            "knockout libraries (NGG SpCas9); per-gene-list scope; "
            "non-targeting + safe-harbor + essential-gene controls."
        ),
    )
    sub_lib = p_lib.add_subparsers(dest="library_kind")
    from .genomics.library_design.cli import add_library_design_subparser
    add_library_design_subparser(sub_lib)

    # placeholders — scope for v0.3+
    for name, help_text in (
        ("basecall", "Nanopore basecalling (v0.2+ scope)"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.set_defaults(func=_cmd_not_implemented)

    return p


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.cmd is None:
        p.print_help()
        return 0
    # Print sub-subcommand help when invoked without one.
    for parent_cmd, kind_attr in (
        ("verify", "verify_kind"),
        ("bench", "bench_kind"),
        ("crispr", "crispr_kind"),
        ("be", "be_kind"),
        ("library", "library_kind"),
    ):
        if args.cmd == parent_cmd and getattr(args, kind_attr, None) is None:
            for action in p._subparsers._actions:  # type: ignore[attr-defined]
                if isinstance(action, argparse._SubParsersAction):
                    for choice, sub_p in action.choices.items():
                        if choice == parent_cmd:
                            sub_p.print_help()
                            return 0
            return 0
    # 3-level: `bionpu crispr pe` without a `design` subcommand prints
    # the pe-level help. The pe_kind attribute is None when the
    # ``design`` child wasn't supplied.
    if (
        args.cmd == "crispr"
        and getattr(args, "crispr_kind", None) == "pe"
        and getattr(args, "pe_kind", None) is None
    ):
        for action in p._subparsers._actions:  # type: ignore[attr-defined]
            if isinstance(action, argparse._SubParsersAction):
                crispr_parser = action.choices.get("crispr")
                if crispr_parser is None:
                    continue
                for sub_action in crispr_parser._subparsers._actions:  # type: ignore[attr-defined]
                    if isinstance(sub_action, argparse._SubParsersAction):
                        pe_parser = sub_action.choices.get("pe")
                        if pe_parser is not None:
                            pe_parser.print_help()
                            return 0
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
