# bionpu_minimizer_k21_w11_n4 — artifact manifest

Built artifacts for the v0 minimizer kernel at `(k=21, w=11, n_tiles=4)`.

| file          | size (bytes) | sha256                                                              |
|---------------|--------------|---------------------------------------------------------------------|
| `final.xclbin` |       24489 | `47549fd560845d8df0016da5b6f5e4847977bb1f2904677db4be7946221883ca`  |
| `insts.bin`   |          300 | `a3c044b606e43974e0cbed481fd5bdb3dce4e914c94c9dff1aab8a9fe84b5008`  |
| `host_runner` |       109872 | `a116e44e149b7ff552132a74c67aa701a6fa8fc1a1943c65e8cea0e1a9900f3d`  |

* `build_date`: 2026-04-28T12:30:07Z
* `host`: matteius-ProArt-P16-H7606WI
* `target`: AIE2P (NPU2)
* `compiler`: Peano (llvm-aie)
* Validated: silicon byte-equal vs CPU oracle on `smoke_10kbp.2bit.bin`
  (1836/1836 records).
