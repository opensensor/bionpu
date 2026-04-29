# bionpu_methylation_context_n4_b8_c1024 — artifact manifest

Batched record-safe artifacts for the v0 methylation-context scanner at
`n_tiles=4`, `seq_chunk_bytes=1024`, `n_chunks_per_launch=8`.

| file | size (bytes) | sha256 |
| --- | ---: | --- |
| `final.xclbin` | 18409 | `dd97a26db968ef63b4fa189a6afad85bcd718bbbe58ec1684998a7fb4912b70e` |
| `insts.bin` | 300 | `ccab0805719126447ed21df6477043021aa246d145e324f26f23bb0342532e52` |
| `host_runner` | 57144 | `e4795e319516de15fb5f5029af735558cef69a6d6305803de8dd50b0cfd18085` |

* `build_date`: 2026-04-29
* `target`: AIE2P (NPU2)
* `compiler`: Peano (llvm-aie)
* Build: `make NPU2=1 experiment=wide4 seq_chunk_bytes=1024 n_chunks_per_launch=8 all`
* Validated: full chr22 record-equal vs CPU oracle
  (`18,406,838` records, no cap-fire).
* Host runner skips global sort/unique when chunk-owned records are
  already monotonic.
