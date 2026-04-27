# — Design rationale

## Why this kernel exists

's ship reduced from PRD §4.2's original "4 match tiles × 32 guides"
sketch to "2 match tiles × 64 guides" because aiecc rejected the 4-into-1
joiner topology with:

```
<unknown>:0: error: 'aie.tile' op number of input DMA channel exceeded!
```

That's filed as in `bionpu/kernels/crispr/match_multitile/
`. Original interpretation: hardware silicon ceiling (compute
tile = 2 input DMA channels per AM020 Ch. 2 p. 27). That's correct as a
**compute-tile** statement.

The AM020 cross-walk (`docs/.md` §)
revealed the canonical AIE-ML escape hatch: **memtile-mediated
aggregation**. AM020 Ch. 5 p. 74 documents:

> Memtile DMA: 6 MM2S + 6 S2MM channels per memtile. Channels 0..3 can
> access east/west neighbour memtile memory.

That's 3× the compute-tile fan-in budget — and it's exactly the
canonical AIE-ML pattern (Figure 22 + 23, "Dataflow Mappings 1/2/3"):
producers fan into memtile, memtile reorganises, memtile fans out.

## Concrete IRON primitive used

`mlir-aie/programming_examples/basic/vector_reduce_max/
single_column_designs/vector_reduce_max_memtile.py` exposes the pattern:

```python
outC = ObjectFifo(joined_ty, name="outC")
outC.prod().join(
    of_offsets,
    obj_types=[partial_ty] * n_cores,
    names=[f"memC{i}" for i in range(n_cores)],
)
```

`outC.prod().join(...)` returns one producer-side FIFO per core; each
worker writes its slot into memtile, memtile reorganises into the
contiguous `joined_ty` via 5D DMA address generation, and memtile MM2S
streams the joined buffer to whatever the `outC.cons()` is wired to
(here: shim DMA out via `rt.drain(of_out.cons(), Out, wait=True)`).

The complementary primitive (`.split()`) is the producer-side fan-out;
we don't need it because our broadcasts (guides, windows) are 1:N
already.

## Why this works where 4-into-1-on-compute-tile didn't

The placement constraint that fired in was per-tile DMA-channel
count. With memtile aggregation:

| target              | input DMA channels needed | budget |
|---------------------|---------------------------|--------|
| Memtile (this work) | 4 (one per match tile)    | 6      |
| Compute tile | 4                          | 2 (FAIL) |

Memtile MM2S → shim is one channel. Total memtile S2MM = 4, MM2S = 1.
Both within budget.

## Throughput hypothesis

 baseline (chr22 × 128 guides, n_iters=10, warmup=2):
- Wall-clock per launch: **240.18 ms** (subprocess + XRT setup dominates).
- Kernel-only: **10379.7 µs** average per iteration.
- Throughput: **394 616 windows/sec**.
- Per-launch geometry: 64 chunks × (2 match tiles × 64 guides each).

 prediction:
- 2× match-tile parallelism on the compute side → kernel-only could
  approach 700 K windows/sec or ~5200 µs/iter.
- Memtile aggregation overhead serialises 4 channels into 1 DDR
  read/write per chunk → likely absorbs some of the 2× ceiling.
- **Realistic target: 130-180 ms wall-clock; 600-700 K windows/sec.**

The actual measurement is the load-bearing RQ4 datum: if memtile S2MM
serialisation absorbs most of the match-tile parallelism gain, that's a
publishable finding (next wall is memtile bandwidth, not compute-tile
DMA).

## Byte-equality non-negotiable

The math is verbatim from / :

```
for each 5-byte (guide, window):
  for each byte:
    xor = g ^ w
    m = (xor | (xor >> 1)) & 0x55
    sum += popcount(m)
  out[w * G + g] = sum   # range 0..20
```

The only architectural difference is **fan-in width** (4 tiles × 32
guides vs 2 tiles × 64 guides). The output dense `(N_GUIDES, N_WINDOWS)`
matrix is bit-for-bit identical. Tests assert `np.array_equal` against:
1. 's output (the load-bearing comparison),
2. NumPy oracle,
3. Cas-OFFinder canonical TSV.

## What's NOT in this kernel

* PAM filtering
* Threshold + emit on-tile
* Slide-by-1 windowing on-tile
* Genome-scale scan
* INT8 vector intrinsics

This kernel is **purely the fan-in retrofit** — same dense-matrix
output contract as .

## AIE2P-vs-AM020 caveats this design surfaces

AM020 documents AIE-ML (= AIE2). Our chip is AIE2P. Per the cross-walk's
"Caveats specific to our chip" section:

* **Memtile size** — AM020 says 512 KiB on AIE-ML. AIE2P unverified at
  the kernel layer. Our memtile footprint (32 KiB) is well below either
  estimate, so this design is robust to a 4× AIE2P memtile reduction.
* **Memtile DMA channel count** — AM020 says 6 + 6. We use 4 + 1, well
  within either estimate. If AIE2P has fewer channels (unlikely but
  possible), we still fit.
* **5D address generation** — AM020 documents memtile DMA as 5D vs
  compute-tile 4D. IRON's `.join()` operator abstracts this; we
  don't directly invoke 5D primitives. If the IRON lowering for
  `.join()` doesn't fully exploit 5D (e.g. only emits 4D access
  patterns), that's filed as a gap entry in ``.
