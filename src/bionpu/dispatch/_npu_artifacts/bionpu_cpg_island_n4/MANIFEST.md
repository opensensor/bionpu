# bionpu_cpg_island_n4 — artifact manifest

Built artifacts for the v0 CpG-island candidate kernel at `n_tiles=4`.

| file | size (bytes) | sha256 |
| --- | ---: | --- |
| `final.xclbin` | 18537 | `d391de5c387ee75603c0f0b7d63d27c30b025012ee3f0180244427f55e8d51eb` |
| `insts.bin` | 300 | `71dd6bda155d8f4ab163a6686d929500221bc1b06008d4eb163c7228a46b4c3e` |
| `host_runner` | 52624 | `0b1916d9ec753802b9ca70d23fe7cb1761d1b084a791c240b699bf149de81c99` |

* `build_date`: 2026-04-28
* `target`: AIE2P (NPU2)
* `compiler`: Peano (llvm-aie)
* Validated: silicon byte-equal vs CPU oracle on all-CG 1000 bp
  (`[(0, 1000)]`, 801 candidate starts) and all-A 1000 bp (`[]`).
