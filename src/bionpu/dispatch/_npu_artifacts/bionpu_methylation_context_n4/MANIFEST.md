# bionpu_methylation_context_n4 — artifact manifest

Built artifacts for the v0 methylation-context scanner at `n_tiles=4`.

| file | size (bytes) | sha256 |
| --- | ---: | --- |
| `final.xclbin` | 17257 | `cb96649ac9dbf40b3e247ee031556362819029d8f8a2be7141b878fe48ece9e7` |
| `insts.bin` | 300 | `2495ccf971ab71327df2d82e0612a17ad513bb28abb36f673e60a7c40ff3c98d` |
| `host_runner` | 57144 | `f02f0400b25e055f2be9b0157e5bba1638c9e8a8575dd26ff352a6966d0f70d7` |

* `build_date`: 2026-04-28
* `target`: AIE2P (NPU2)
* `compiler`: Peano (llvm-aie)
* Build: `make NPU2=1 experiment=wide4 all`
* Validated: silicon byte-equal vs CPU oracle on mixed CG/CHG/CHH
  smoke and all-A smoke via `tests/test_methylation_context_correctness.py`.
