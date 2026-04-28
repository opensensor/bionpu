# bionpu_kmer_count_k15_n2

K-mer count NPU artifact for k=15, n_tiles=2.

## Identity

- kernel_name: `bionpu_kmer_count`
- k: 15
- n_tiles: 2
- registry_name: `bionpu_kmer_count_k15` (per T1 contract; `n_tiles`
  is a constructor arg on the shared op class, not a registry name)
- build_date: 2026-04-28T08:43:37+00:00
- host: matteius-ProArt-P16-H7606WI-H7606WI

## Op contract (per state/kmer_count_interface_contract.md)

- input shape: 1-D `np.uint8` packed-2-bit DNA (A=00, C=01, G=10, T=11;
  first base = bits[7:6] of byte 0, MSB-first within each byte).
- chunk size: 4100 bytes (4096 base + per-k overlap; k=15 overlap=4,
  k=21 overlap=8, k=31 overlap=8 — all 4-byte aligned per the T11 fix
  for aiecc dma_bd alignment).
- output: list of `(canonical_u64, count)` tuples, sorted
  `(count desc, canonical asc)`.
- registry: `bionpu_kmer_count_k15` (selected by `get_kmer_count_op`
  helper; the registry-resident instance defaults to `n_tiles=4`).
- wire format: see `state/kmer_count_interface_contract.md` (revised
  2026-04-28: memtile-aggregated combine + per-k chunk align + 1024
  buckets-per-tile, codified in the topology revision section).

## Artifact files

| File           | Size (bytes)   | sha256                                                          |
|----------------|---------------:|-----------------------------------------------------------------|
| final.xclbin   | 25417   | 0144290135c3764f58820ff4ecce2df6886651754736b536b87a6fa9d8cdbfa2 |
| insts.bin      | 300    | b6e71d07e53c38e6841e8909248cba8b882127dfe75979c562db338f10eca249  |
| host_runner    | 58176     | bb9e85b189aa180e74670c9cb167361cd5ea6725ed1e3a1f9abc3c862afb4620   |

## Build provenance

- Source: `bionpu-public/src/bionpu/kernels/genomics/kmer_count/`
- Wave-3 fixes (commits in bionpu-public):
  - `3b028d8` — Fix A (k=21 chunk align 5->8) + Fix B (HASH_BUCKETS_PER_TILE 4096->1024)
  - `480a88c` — Fix C (memtile-aggregated combine topology in IRON Python)
  - `c7a2c44` — KMER_K_ACTIVE macro gating + per-K artifact suffix + final tile-DM fit
- Build invocation:

      make NPU2=1 K=15 experiment=wide2 seq=10000 all
      cp build/final_k15_n2_L10000.xclbin <here>/final.xclbin
      cp build/insts_k15_n2_L10000.bin    <here>/insts.bin
      cp bionpu_kmer_count                <here>/host_runner

