# bionpu_methylation_context_n4_b4_c1024 — artifact manifest

Batched record-safe artifacts for the v0 methylation-context scanner at
`n_tiles=4`, `seq_chunk_bytes=1024`, `n_chunks_per_launch=4`.

| file | size (bytes) | sha256 |
| --- | ---: | --- |
| `final.xclbin` | 17385 | `c6c6be13200b7856874c48acb63d95bf6e50a00ad8e8a97663734052f411098d` |
| `insts.bin` | 300 | `1c9864e266008fc9519a9e4414c0aa4719ca4fb2f234d064a9c8af46296721c0` |
| `host_runner` | 57144 | `ec7f4e4a278a140937b030551008530c8de53a54ce27444493a693ded45408f9` |

* `build_date`: 2026-04-29
* `target`: AIE2P (NPU2)
* `compiler`: Peano (llvm-aie)
* Build: `make NPU2=1 experiment=wide4 seq_chunk_bytes=1024 n_chunks_per_launch=4 all`
* Validated: full chr22 record-equal vs CPU oracle
  (`18,406,838` records, no cap-fire).
