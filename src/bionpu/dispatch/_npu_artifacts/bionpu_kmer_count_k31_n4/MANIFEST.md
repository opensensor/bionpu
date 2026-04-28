# bionpu_kmer_count_k31_n4

K-mer count NPU artifact for k=31, n_tiles=4.

## Identity

- kernel_name: `bionpu_kmer_count`
- k: 31
- n_tiles: 4
- registry_name: `bionpu_kmer_count_k31` (per T1 contract; `n_tiles`
  is a constructor arg on the shared op class, not a registry name)
- build_date: 2026-04-28T08:43:37+00:00
- host: matteius-ProArt-P16-H7606WI-H7606WI

## Op contract (per state/kmer_count_interface_contract.md)

- input shape: 1-D `np.uint8` packed-2-bit DNA (A=00, C=01, G=10, T=11;
  first base = bits[7:6] of byte 0, MSB-first within each byte).
- chunk size: 4104 bytes (4096 base + per-k overlap; k=15 overlap=4,
  k=21 overlap=8, k=31 overlap=8 — all 4-byte aligned per the T11 fix
  for aiecc dma_bd alignment).
- output: list of `(canonical_u64, count)` tuples, sorted
  `(count desc, canonical asc)`.
- registry: `bionpu_kmer_count_k31` (selected by `get_kmer_count_op`
  helper; the registry-resident instance defaults to `n_tiles=4`).
- wire format: see `state/kmer_count_interface_contract.md` (revised
  2026-04-28: memtile-aggregated combine + per-k chunk align + 1024
  buckets-per-tile, codified in the topology revision section).

## Artifact files

| File           | Size (bytes)   | sha256                                                          |
|----------------|---------------:|-----------------------------------------------------------------|
| final.xclbin   | 43353   | f50f0f94c501fa4dd7dc52e272f4b6af5f14064da4fd19bad72d3ed64d40af27 |
| insts.bin      | 300    | c2e789b8407e289bceb3915d5b305e77bf4009ab2f8874d206340747af5798eb  |
| host_runner    | 58176     | bb9e85b189aa180e74670c9cb167361cd5ea6725ed1e3a1f9abc3c862afb4620   |

## Build provenance

- Source: `bionpu-public/src/bionpu/kernels/genomics/kmer_count/`
- Wave-3 fixes (commits in bionpu-public):
  - `3b028d8` — Fix A (k=21 chunk align 5->8) + Fix B (HASH_BUCKETS_PER_TILE 4096->1024)
  - `480a88c` — Fix C (memtile-aggregated combine topology in IRON Python)
  - `c7a2c44` — KMER_K_ACTIVE macro gating + per-K artifact suffix + final tile-DM fit
- Build invocation:

      make NPU2=1 K=31 experiment=wide4 seq=10000 all
      cp build/final_k31_n4_L10000.xclbin <here>/final.xclbin
      cp build/insts_k31_n4_L10000.bin    <here>/insts.bin
      cp bionpu_kmer_count                <here>/host_runner

