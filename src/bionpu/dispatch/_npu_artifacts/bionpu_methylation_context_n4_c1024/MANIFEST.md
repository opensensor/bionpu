# bionpu_methylation_context_n4_c1024 — artifact manifest

Record-safe built artifacts for the v0 methylation-context scanner at
`n_tiles=4` with `seq_chunk_bytes=1024`.

| file | size (bytes) | sha256 |
| --- | ---: | --- |
| `final.xclbin` | 17257 | `1169a5b2cd323aeed5ef67b9ac19c68e33971c727bc0db3b8ddedc0fc39a31fa` |
| `insts.bin` | 300 | `dbe3be1a99cdbef50cffcd1fd806e041febbada7e491ae55e30b1c3e1455cc6c` |
| `host_runner` | 57144 | `ec7f4e4a278a140937b030551008530c8de53a54ce27444493a693ded45408f9` |

* `build_date`: 2026-04-28
* `target`: AIE2P (NPU2)
* `compiler`: Peano (llvm-aie)
* Build: `make NPU2=1 experiment=wide4 seq_chunk_bytes=1024 all`
* Validated: full chr22 record-equal vs CPU oracle
  (`18,406,838` records, no cap-fire).
