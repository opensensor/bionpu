# bionpu_minimizer_k15_w10_n4 — artifact manifest

Built artifacts for the v0 minimizer kernel at `(k=15, w=10, n_tiles=4)`.

| file          | size (bytes) | sha256                                                              |
|---------------|--------------|---------------------------------------------------------------------|
| `final.xclbin` |       24489 | `8c8c53d111ad91b643a3e8d18a4fc9b516930337a1a6065b6d67081633ec3644`  |
| `insts.bin`   |          300 | `a3c044b606e43974e0cbed481fd5bdb3dce4e914c94c9dff1aab8a9fe84b5008`  |
| `host_runner` |       109872 | `a116e44e149b7ff552132a74c67aa701a6fa8fc1a1943c65e8cea0e1a9900f3d`  |

* `build_date`: 2026-04-28T12:30:07Z
* `host`: matteius-ProArt-P16-H7606WI
* `target`: AIE2P (NPU2)
* `compiler`: Peano (llvm-aie)
* Validated: silicon byte-equal vs CPU oracle on `smoke_10kbp.2bit.bin`
  (1983/1983 records).
* chr22 top-1000 byte-equal vs oracle: 1000/1000 (full byte-equal
  blocked by per-chunk emit cap; see kernel `gaps.yaml` —
  `minimizer-emit-cap-saturation`).
