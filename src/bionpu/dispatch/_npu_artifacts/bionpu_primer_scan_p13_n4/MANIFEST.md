# bionpu_primer_scan_p13_n4 — silicon artifacts

v0 silicon-validated (P=13, n_tiles=4) primer/adapter scan kernel.
Path B (runtime primer canonical via 24-byte chunk header).

| field           | value                                            |
|-----------------|--------------------------------------------------|
| primer length   | P = 13 (TruSeq P5 default at runtime)            |
| n_tiles         | 4 (broadcast topology)                           |
| build seq       | 10000                                            |
| MAX_EMIT_IDX    | 2046 / 32 KiB pass-slot                          |
| header          | 24 bytes (v0 extends kmer_count's 8-byte prefix) |

See `kernels/genomics/primer_scan/DESIGN.md` and `MANIFEST.md` for
build invocations and shape pins.

## SHA-256 sums

d2a99c7401483ca61783bfe96bd33e63024c5f1d4987837a0614ecb11348abb7  final.xclbin
a3c044b606e43974e0cbed481fd5bdb3dce4e914c94c9dff1aab8a9fe84b5008  insts.bin
63328c6b2f12fbdaf7c5d3a1b04ffc68ad259ab64b5d7846be7fac36db597e86  host_runner
