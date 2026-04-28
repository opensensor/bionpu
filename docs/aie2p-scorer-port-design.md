# DNABERT-Epi AIE2P scorer port — design

PRD-1 v0.4 follow-up. The CPU + GPU backends in
[`bionpu.scoring.dnabert_epi`](../src/bionpu/scoring/dnabert_epi.py)
are working on real trained weights with byte-equivalence proven
across CPU↔GPU under NUMERIC_EPSILON. This document scopes the
remaining backend — AIE2P NPU silicon — and lays out the kernel
breakdown, dispatch graph, and verification plan.

## Constraints

**Model.** DNABERT-3 base (BERT-base topology) + a 2-class
classifier head:

| Layer | Shape | Params | INT8 size |
|---|---|---|---|
| Word embedding (vocab=4101 × 768) | 4101 × 768 | 3.15 M | 3.0 MB |
| Position embedding (512 × 768) | 512 × 768 | 0.39 M | 0.4 MB |
| Token-type embedding (2 × 768) | 2 × 768 | ~0 | ~0 |
| LayerNorm + dropout | — | 0.001 M | — |
| **Embeddings total** | — | **3.55 M** | **3.4 MB** |
| Per attention block: Q, K, V, O (each 768 × 768) | 768 × 768 × 4 | 2.36 M | 2.25 MB |
| Per FFN: intermediate (768 × 3072) + output (3072 × 768) | (768·3072 + 3072·768) | 4.72 M | 4.5 MB |
| Per LayerNorm + biases | — | 0.005 M | — |
| **Per layer** | — | **7.09 M** | **6.75 MB** |
| 12 layers | — | **85.05 M** | **81 MB** |
| Pooler (768 × 768) | 768 × 768 | 0.59 M | 0.56 MB |
| Classifier (Dropout → Linear 768 × 2) | 768 × 2 | 0.002 M | ~0 |
| **Total** | — | **89.20 M** | **85 MB** |

**AIE2P (Strix) on-chip capacity:**
- 8 × 4 = 32 compute tiles per AIE-ML partition
- Each compute tile: 64 KB DM + 16 KB PM
- 4 memtiles × 512 KB = 2 MB total memtile
- Total fast memory: ~4 MB
- **The full model does not fit on-chip** (85 MB INT8 ≫ 4 MB)
- Per-layer (~7 MB INT8) does not fit a single memtile either
- Largest single matmul (FFN1, 768 × 3072) = 2.3 MB INT8 → fits memtile,
  exceeds tile DM by 36×

**Conclusion:** weight-streaming pipeline. DDR ↔ memtile ↔ tile, one
layer (or one matmul within a layer) at a time.

## Inference contract

Input: a single (crrna 24nt, dna 24nt) pair, tokenized via DNABERT-3's
3-mer WordPiece tokenizer. Sequence length: 47 tokens
(`2 * (24-3+1) + 3` for CLS / SEP / SEP). Token type IDs distinguish
the two segments. Embedded as 47 × 768 FP16 / INT8.

Output: scalar in [0, 1] = `softmax(classifier(pool(bert(x))))[1]`.

For batch inference: rows accumulate into chunks; each chunk runs as
one forward pass. Reasonable initial chunk: 1 row (latency-optimised),
then expand to e.g. 32 once weight-streaming is amortised.

## Kernel breakdown

The port is decomposed into units roughly matching the basecalling
track's existing kernel granularity:

### Tier 1 — embedding lookup (one-shot per inference)
- **`bert_embed_lookup`** — gather word + position + token-type
  embeddings, sum, LayerNorm. Input: 47 token IDs (int32). Output:
  47 × 768 FP16. Weight footprint: ~3.5 MB (entire embedding table)
  — fits memtile if we slice by vocab range, otherwise lives in DDR
  and gathers one row at a time.

### Tier 2 — transformer block (×12, identical topology)
- **`bert_attn_qkv`** — three 47×768 @ 768×768 GEMMs producing
  Q, K, V tensors. Streamed weights from DDR. Approximate cost:
  3 × 0.24 MFLOP per layer.
- **`bert_attn_softmax_av`** — `attn = softmax(Q·Kᵀ / √d_k) · V`.
  47 × 47 × 12 (heads) score matrix; reduction-heavy, fits on tile.
- **`bert_attn_output`** — 47×768 @ 768×768 + LayerNorm + residual.
- **`bert_ffn`** — 47×768 @ 768×3072 (GeLU) @ 3072×768 + LayerNorm
  + residual. **Bandwidth-bound**: per-layer FFN weights are 4.5 MB
  INT8 streamed from DDR per inference. At PCIe Gen4 x4 (~7 GB/s)
  this is ~0.65 ms / layer.

### Tier 3 — pooler + classifier (one-shot at end)
- **`bert_pool`** — 1×768 @ 768×768 + tanh.
- **`bionpu_score_head`** — 1×768 @ 768×2 + softmax. Already runnable
  on AIE2P trivially (output of a single matmul fits one tile).

### Tier 4 — reference & quant

- **`bionpu.quant`** — INT8 quantisation passport for each weight
  matrix. Per-channel weight scaling, per-tensor activation scaling.
  Calibration: ~1000 random off-target candidate rows from the held-
  out fold's test partition. **Do this before kernel work** — a
  dispatch graph is meaningless if the quantised model's CPU-FP32
  reference doesn't match the un-quantised CPU-FP32 reference within
  the 1e-3 NUMERIC_EPSILON the harness checks.

## Dispatch graph

```
DDR (host)         Memtile (NPU)         Compute tile (NPU)
─────────          ─────────             ─────────────
embeddings ──────► (gather slice) ───► [embed_lookup_tile]
weights/L0 ──────► W_qkv (2.25MB)  ───► [attn_qkv_tile]
                                        [attn_softmax_av_tile]
weights/L0 ──────► W_attn_out + LN ──► [attn_output_tile]
weights/L0 ──────► W_ffn (4.5MB)   ───► [ffn_tile]
                                        ↓
                                       (residual + LN cached on tile)
                                        ↓
weights/L1 ──────► (next layer; pipeline overlap)
…
weights/L11 done
weights/pool ────► [pool_tile]
weights/cls  ────► [score_head_tile] ──► softmax → score (host)
```

Pipeline overlap: while compute tile processes layer N, memtile DMA
fetches weights for layer N+1. That hides ~half the DDR→memtile
latency assuming 4.5 MB / layer at 7 GB/s ≈ 0.65 ms layer-fetch and
~0.5-1 ms / layer compute.

Estimated single-token latency: 12 × ~1 ms = ~12 ms / token. At batch
1, throughput-bound by weight bandwidth, not compute. Larger batches
amortise the per-layer weight fetch.

## Verification harness

Same NUMERIC_EPSILON contract that's already proving CPU↔GPU
equivalence:

```
bionpu score  --candidates scan.tsv --device cpu --weights X.pt --out cpu.tsv
bionpu score  --candidates scan.tsv --device npu --weights X.pt --out npu.tsv \
              --verify cpu.tsv --verify-policy NUMERIC_EPSILON --verify-epsilon 1e-3
```

Tolerance budget:
- FP16 GEMM order divergence: ~1e-6 per layer, 12 layers compounding
  worst-case ~1e-4
- INT8 quantization-vs-FP32 error: 1e-3 to 1e-2 depending on
  calibration coverage
- **Recommended NPU vs CPU eps: 5e-3** — generous enough to absorb
  INT8 rounding, tight enough to surface real divergence

If the eps needs to be looser than 1e-2 to pass, the quantization
passport is the suspect, not the AIE2P silicon.

## Concrete first task

Before any kernel work, **port DNABERT-3 base inference to AIE2P
host-emulation only** (no silicon). The bionpu kernel toolchain has
a host-emulation fallback path that runs the same dispatch graph on
the CPU using the same byte-level kernel binaries; if NPU silicon is
unavailable the harness exercises the path via that fallback.

This validates:
1. The dispatch graph compiles (i.e., the IRON Python topology lowers
   cleanly to MLIR-AIE)
2. The kernel C++ produces the right outputs on identical inputs
   (CPU emulation matches FP32 reference within 1e-6)
3. The bionpu.dispatch glue handles weight-streaming, batch=1
   inference, and softmax post-processing

Once that lands, the silicon dispatch is a deployment problem
(loading the produced xclbins via `bionpu.dispatch.npu`) rather than
an algorithmic correctness problem.

## Scope of v0.4

Minimal viable port:
1. `bionpu.quant` calibration + INT8 weight extraction for the
   trained DNABERT-Epi (4 hours) — **DONE 2026-04-28**
2. `bert_int8_matmul` head specialization (M=47, K=768, N=2),
   single-tile, scalar — **DONE 2026-04-28**, byte-equivalent on silicon
3. End-to-end dispatch glue in `bionpu.scoring.dnabert_epi` for
   `device='npu'` (head only; BERT body still on torch CPU) —
   **DONE 2026-04-28**, NUMERIC_EPSILON@0.05 EQUAL vs CPU
4. `bert_int8_matmul` qkvo specialization (N=768) — **NEXT** (multi-tile
   + K-chunking, see § below)
5. `bert_int8_matmul` ffn1/ffn2 specializations (N=3072, K=3072) —
   DDR-streamed weights (~5 days)
6. `bert_attn_softmax_av` IRON Python kernel (1 week)
7. `bert_embed_lookup` + `bert_pool` IRON Python kernels (1 week)
8. Wire all kernels into a single `bionpu score --device npu` end-
   to-end forward pass (no torch BERT body fallback) (3 days)
9. NUMERIC_EPSILON@5e-3 byte-equivalence vs CPU reference for the
   full forward pass (1 day)
10. Energy + latency benchmarking via `bionpu bench score`
    (already works; needs `bionpu bench score` CLI extension) (2 days)

Total estimate: ~5-6 weeks of focused work for one engineer.

## v0.4-beta: scaling bert_int8_matmul from N=2 to N=768 (qkvo)

The v0.4-alpha head kernel is single-tile with full-K-resident weight
+ full-K-resident input. That recipe does not scale to qkvo / ffn —
the weights blow past tile DM (64 KB) and even memtile (512 KB).

**Per-tile DM budget at qkvo shape (M=47, K=768, N=768):**

| variant | per-tile w | per-tile x_chunk | per-tile y_i32 acc | total | fits 64KB? |
|---|---|---|---|---|---|
| 1 tile, full K, N=768 | 576 KB | 36 KB | 144 KB | ~756 KB | no — by 11.5× |
| 4 tiles, full K, N=192/tile | 144 KB | 36 KB | 36 KB | 216 KB | no |
| 4 tiles, K_chunk=128, N=192/tile | 24 KB | 6 KB | 36 KB | 66 KB | barely no |
| 4 tiles, K_chunk=64, N=192/tile | 12 KB | 3 KB | 36 KB | 51 KB | **yes** |
| 4 tiles, K_chunk=64, N=192/tile, m_chunk=24 | 12 KB | 3 KB | 18 KB | 33 KB | yes (margin) |

**v0.4-beta topology:** 4 compute tiles, N-axis fan-out (each tile
owns 192 of 768 output channels), K-chunked along the reduction
axis (K_chunk=64 → 12 chunks per launch), int32 partial accumulator
held resident on each tile across K-chunks; final fused-scale +
INT8 saturate happens once at the end of the K loop. Memtile holds
the full per-tile weight slice (192 × 768 = 144 KB int8, fits a
single 512 KB memtile with margin); DMA streams K-slices from
memtile to compute tile each chunk.

```
shim ──x── memtile ──┬─→ compute_0 (N[0..191],   K-chunked)  ──y_p0──┐
                     ├─→ compute_1 (N[192..383], K-chunked) ──y_p1── memtile (concat) ──y── shim
                     ├─→ compute_2 (N[384..575], K-chunked) ──y_p2──┤
                     └─→ compute_3 (N[576..767], K-chunked) ──y_p3──┘
shim ──w── memtile (resident, 576 KB) ──192 KB slice/tile──→ each compute
shim ──s── (broadcast, tiny) ──→ all compute tiles
```

The combined-scale array (768 fp32 = 3 KB) broadcasts to all four
compute tiles. The output is concatenated by the memtile into a
single shim-DMA-out blob.

**Implementation notes:**
- Same `bionpu_int8_matmul.cc` symbol body, parameterised on M / K /
  N / N_per_tile / K_chunk via Makefile macros.
- New IRON-Python `--variant qkvo` flag emits the multi-tile
  topology with the four compute-tile workers.
- Host runner stays the same shape: file-backed x.bin / ws.bin /
  y.bin; the host pre-stages w + scales into ws.bin.

**Estimated effort:** 2-3 days for first build attempt + iteration.
The IRON Python topology is the big lift; the kernel C++ is
incremental over the v0.4-alpha scalar path.

## v0.4-rc: scaling to N=3072 (ffn1/ffn2)

ffn1: M × 768 → M × 3072 — weight = 768 × 3072 = 2.3 MB int8.
Single memtile is 512 KB, so the weight no longer fits memtile-
resident. **DDR-streamed weight** is required: shim DMA fetches
K_chunk × N_chunk slices from DDR each iteration; memtile becomes
a 1-deep relay buffer.

ffn2: same shape transposed (M × 3072 → M × 768), same DMA budget.

This is the bandwidth-bound bottleneck of the port. At PCIe Gen4 x4
(~7 GB/s) and 4.7 MB / layer FFN weight at INT8, every BERT layer
costs ~0.65 ms of weight-fetch latency, summing to ~8 ms for 12
layers. That's the floor on per-token inference latency for the
DNABERT-Epi NPU port.

**Estimated effort:** 4-5 days of focused work.

## What this PR does NOT do

- BigWig epigenetic feature stage (with-epi variant) — separate
  follow-up. The no-epi variant is the AIE2P port target.
- Multi-tile cascade optimisations — the basecalling track's
  cascade work could apply but is orthogonal; first port lands
  without it.
- Model distillation to DistilBERT-shape (45 M params) — a separate
  optimization once the port works at full 89 M.
