# pe_design — Prime-Editor pegRNA design (Track B v0)

## 1. Strategic context

Track B is the prime-editor (PE) leg of the long-arc CRISPR roadmap
(PRDs/PRD-crispr-state-of-the-art-roadmap.md §3.2). v0 ships nucleotide-level
pegRNA design as a host-side composition over published kernels: real
PRIDICT 2.0 efficiency scoring (CPU; MIT-licensed, runtime-dep-only per
restrictive-license-model-policy), real ViennaRNA folding (CPU), and the
locked CRISPR off-target silicon kernels reused via a thin adapter onto the
existing `crispr_design` scan path. Track D Phase 0 closure (PRIDICT-class
transformer scoring stays CPU-only for v0) means there is no AIE2P silicon
work in this v0 ship; the public API surface (`device="cpu|npu"` flag,
`off_target_scan_for_spacer` callable) is locked so the v1 silicon path can
drop in behind the same boundaries without churning T8/T10. PRIDICT 2.0
turned out to be MIT-licensed (T1 finding, surprise vs. the
restrictive-license assumption), so the runtime-dep posture is comfortable;
a future research-spike retrain-from-scratch path remains open per the
restrictive-license memory.

## 2. Architecture

Modules and dataflow:

```
  EditSpec ──► enumerator.py ──► [PegRNACandidate]
  (T2)         (T6: PE2,                       │
                both strands,                  ▼
                PAM-recreation               pe3_nicking.py (T7)
                + Pol-III pruning)           (PE3PegRNACandidate)
                                              │
                                              ▼
                                    pegrna_folding.py (T4) ──┐
                                    pridict2.py (T5) ────────┤
                                    off_target.py (T11) ─────┤
                                                              ▼
                                                       ranker.py (T8)
                                                       (composite_pridict)
                                                              │
                                                              ▼
                                                       output.py (T9)
                                                       (32-col TSV + JSON)
                                                              │
                                                              ▼
                                                       cli.py (T10)
                                                       (Modes A/B/C)
```

| Module | Role |
|---|---|
| `pegrna_constants.py` (T3) | Scaffold variants (Anzalone 2019, Nelson 2022 evopreQ1/tevopreQ1, cr772 placeholder); PBS 8-15 nt, RTT 10-30 nt, PE3 distance 40-100 bp, MAX_EDIT_LENGTH_BP=50. |
| `types.py` (T3) | Shared dataclasses: EditSpec, PegRNAFoldingFeatures, PRIDICTScore, PegRNACandidate, PE3PegRNACandidate, OffTargetSite, RankedPegRNA. |
| `edit_spec.py` (T2) | Simple + HGVS parsers; minus-strand alt-allele complement; length-cap + RefMismatchError + UnsupportedHGVSVariant. |
| `enumerator.py` (T6) | PE2 candidate generator (both strands, NGG PAM, deterministic sort by `(chrom, nick_site, strand, pbs_length, rtt_length, scaffold_variant)`). |
| `pe3_nicking.py` (T7) | Opposite-strand NGG within 40-100 bp; off-bystander filter; PE3PegRNACandidate output. |
| `bionpu.scoring.pegrna_folding` (T4) | ViennaRNA wrapper: MFE + structure + PBS pairing prob + scaffold disruption. |
| `bionpu.scoring.pridict2` (T5) | PRIDICT 2.0 wrapper: HEK + K562 heads, scaffold-OOD flag, target-context cache, score_arbitrary_pegrna + component-triple match (T12 fix). |
| `off_target.py` (T11) | Thin delegation to `crispr_design._scan_locus_for_offtargets` + `cfd.aggregate_cfd`. Sits ABOVE the lock layer (CLAUDE.md). |
| `ranker.py` (T8) | Composite scoring + NaN-safe sort + content-addressed pegrna_id. |
| `output.py` (T9) | 32-column TSV + JSON; round-trip stable; NaN→"NA"/null. |
| `cli.py` (T10) | `bionpu crispr pe design`; Modes A/B/C; 3-level subparser nesting. |

## 3. PE2/PE3 algorithm

**PE2 enumeration (T6):**
For each candidate spacer position with NGG PAM near the edit site, on **both
strands** (running the enumerator on the genome's `+` then again on
`reverse_complement(target)` and translating sigma-local nicks back via
`nick_site_plus = N - nick_site_sigma`):

1. Identify nick site: 17 nt downstream of PAM (Cas9-H840A nicks 3 nt 5' of
   PAM on the protospacer-paired strand).
2. **PBS** (Primer Binding Site): reverse-complement of 8-15 nt 5' of the
   nick on the unedited strand.
3. **RTT** (Reverse-Transcription Template): reverse-complement of (10-30 nt
   downstream of nick + the desired edit + flanking).
4. **Full pegRNA**: spacer + scaffold + RTT + PBS (3' to 5' of the RNA).
5. **rt_product_seq**: post-edit genomic sequence the RT enzyme produces;
   sanity-checked byte-for-byte against the desired edit.

**Pruning rules (in order):**
1. Pol-III TTTT in user-controlled regions (spacer, RTT, PBS, plus 3-nt
   junctions with scaffold) — the canonical scaffold's terminal `UUUU` is
   *not* counted (intentional terminator hairpin).
2. GC% of the spacer outside [25, 75]%.
3. **PAM-recreation**: drop only when the edit's coordinate range overlaps
   the PAM bases AND the post-edit PAM still matches NGG (narrowed from the
   literal "drop any post-edit NGG" reading to avoid pruning the typical
   case where the edit doesn't touch the PAM).
4. RTT length < edit_size + 5 (RT must extend past the edit by ≥5 nt).

**PE3 nicking selection (T7):**
For each PE2 candidate, walk the **opposite strand** for NGG PAMs within
`distance_range` (default 40-100 bp) of the PE2 nick site (in `+` coords).
Distance is `abs(pe3_nick_site_plus - pe2.nick_site)`. Optional
`edit_region` filter rejects nicks landing inside the edit window.

## 4. Wire format

32-column TSV (locked one-way: dataclass field re-orders are tolerated; a
column-order change in `TSV_HEADER` trips a header-mismatch on read-back of
older TSVs). JSON export mirrors the same fields as a single
`json.dumps(payload, indent=2)` document (NOT NDJSON). NaN → `"NA"` in TSV,
`null` in JSON; None → `""` in TSV, `null` in JSON; the `notes` field is
`","`-joined in TSV and a list in JSON. Schema reference and round-trip
contract live in `output.py`.

## 5. Validation strategy

| Gate | Result | Notes |
|---|---|---|
| T6 enumerator vs Anzalone 2019 HEK3 byte-equal cross-check | PASS | Spacer `GGCCCAGACTGAGCACGTGA`, PAM `TGG`, 13-nt PBS `CGUGCUCAGUCUG`, RTT for `+1 ins T` byte-equal. |
| T12 integration tests with REAL PRIDICT calls | 4/4 PASS | Anzalone 2019 + Mathis 2024 README oracle within ±5% + PE3 + Mode C synbio. |
| T13 smoke (5 targets × 3 modes + determinism) | 5/5 PASS | Max per-target wall 92.62 s vs 30 min budget; BRCA1 Mode-A 3-run TSV byte-identical (`96a1193cb7f9...`); overall wall 340.8 s. |
| Full pe_design suite | 56+ tests | constants 12 + edit_spec 13 + enumerator 9 + pe3_nicking 4 + pegrna_folding 5 + ranker 6 + off_target 4 + output 3 + cli 5 + integration 4. |
| Cumulative bionpu suite | ≥110+ tests | Zero regressions. |

## 6. Critical T10/T12 architectural decision: scaffold-invariant component-triple lookup

T6's enumerator ships pegRNAs with the **Anzalone 2019 wt-scaffold body**
(`GUUUU...UUUU` with the Pol-III terminator hairpin). PRIDICT 2.0's internal
`pegRNAfinder` bakes in the **Chen 2013 F+E optimised variant**
(`GUUUC...GGUGC`, no terminator). A full assembled-pegRNA-string lookup into
PRIDICT's scoring CSV therefore misses for every T6 candidate.

The fix (T12, "Option A"): T5's `_score_one` falls back to a **component
triple** `(spacer, PBSrevcomp, RTrevcomp)` against PRIDICT's CSV columns
`(Spacer-Sequence, PBSrevcomp, RTrevcomp)`. The triple is **scaffold-
invariant** — PRIDICT's own model takes the components as input features
regardless of scaffold body. T8's ranker forwards
`spacer_dna=cand.spacer_seq, pbs_dna=cand.pbs_seq, rtt_dna=cand.rtt_seq`
into every `score()` call (with a `TypeError` fallback to the legacy 4-arg
signature so test stubs still work). T5 also exposes
`score_arbitrary_pegrna(*, spacer, pbs, rtt, target_context, ...)` as a
direct entry-point for callers outside the enumerator.

Option B (cache pegRNAfinder output and flag-as-not-in-cache for misses)
was rejected because it would have left T6's both-strand and
PAM-recreation-pruning candidates as `PEGRNA_NOT_ENUMERATED_BY_PRIDICT`
whenever they fell outside PRIDICT's enumeration overlap.

## 7. v1 deferrals (filed)

- **AIE2P silicon transformer port** (gated on Track D / Track E
  microbenchmark; PRIDICT 2.0 stays CPU until the 700 µs/dispatch primer-
  scan floor is characterized at conv-stem-shape inputs).
- **Multi-edit pegRNAs** (Mathis 2024 supports up to ~50 bp; v0 caps at
  single-base + small indels via MAX_EDIT_LENGTH_BP=50).
- **PEgg / TwinPE / PE5 advanced strategies** (PE2 + PE3 only in v0).
- **HGVS inversion / structural-variant support** (raises
  `UnsupportedHGVSVariant` today).
- **pegRNA library design** (multi-target × multi-edit; Track B + Track C
  composition, v1+).
- **HDR repair-template alternative path** (NHEJ-side, separate track).
- **Cell-type variants HCT116 / U2OS** (PRIDICT 2.0 ships HEK + K562 heads
  only — T5 finding; HCT116/U2OS rejected at construction with v1-retrain
  hint).
- **Paralog-aware composite filtering** (T13 HBB / HBD / HBG1 / HBG2
  finding: dense off-target hits across paralogs produce composite NaN).
- **Order-sheet generation** (TSV ships as v0; vendor-specific synthesis
  ordering is v1).
- **Real-time interactive design** (CLI batch only in v0).
- **Full-genome off-target scan** (T13 v0 deviation: ±500 kbp slice for
  Mode A budget control).
- **Transcript `c.NNN` → genomic-pos full per-exon CDS map** (T2 best-
  effort: refGene TSV ships only `(txStart, txEnd)`; v0 treats `c.NNN` as
  1-based offset on the transcript's strand without intron skipping).
- **Mode B → on-the-fly genome resolution** (Mode B requires a user FASTA
  window today; v1 lifts to genome-fetcher integration).
- **`resolve_gene_symbol_from_refseq_id` O(N) prefix-walk** (currently ~360
  refGene entries; trivial today, indexed lookup is v1 hygiene).

## 8. Reproducibility

```bash
# env (one-time)
source /home/$USER/xdna-bringup/ironenv/bin/activate
pip install -e bionpu-public[pe]    # ViennaRNA + PRIDICT 2.0 transitive deps
# PRIDICT 2.0 cloned to third_party/PRIDICT2 (commit c133c35); MIT-licensed,
# trained_models/ (95 MB) ships in-tree.

# run
python -m bionpu.cli crispr pe design \
    --target BRCA1 \
    --edit "G>T at chr17:43044395" \
    --strategy both \
    --genome hg38 \
    --top 5 \
    --output -
```

Mode A = gene symbol + hg38 fetch (Track B v0 scopes off-target to ±500 kbp
slice). Mode B = user FASTA window + same window for off-target. Mode C =
synbio plasmid + `--genome none` (off-target columns NaN, notes carry
`NO_OFF_TARGET_SCAN`). Determinism: 3 runs of identical input produce
byte-identical TSV output (T13 gate).
