# bionpu_tandem_repeat_n4 — artifact manifest

Built artifacts for the v0 short tandem repeat (STR) kernel at
`n_tiles=4`.

| file | size (bytes) | sha256 |
| --- | ---: | --- |
| `final.xclbin` | 21865 | `a17be29183a34a3583fc0af33d94a5622ac466797ae7f3824f3f3c8f8d31c01e` |
| `insts.bin` | 300 | `6603b3a4853fd65e9a5bf66c4a47c78da0274bcb13ec89304c8f4ab5cf047157` |
| `host_runner` | 58128 | `6e80eee0a965bb2c77c30956f2ee49b1241557b22bedd0e263c314d8e3bb07e0` |

* `build_date`: 2026-04-28
* `target`: AIE2P (NPU2)
* `compiler`: Peano (llvm-aie)
* Constants: `MIN_PERIOD=1, MAX_PERIOD=6, MIN_COPIES=5`
* Validation: silicon byte-equal vs CPU oracle
  (`bionpu.data.tandem_repeat_oracle.find_tandem_repeats`).
  - synthetic-inject (892 bases, 7 records): PASS byte-equal
  - smoke (10 kbp, 26 records): PASS byte-equal
  - chr22 (50.8 Mbp, 289548 silicon vs 289551 oracle records): partial
    pass — 10 record diffs (≤ v0 threshold of 10; see gaps.yaml entry
    `tandem-repeat-overlap-merge-phase-shift`)
* chr22 e2e wall: 9.4 s (silicon-only dispatch ~7.8 s)
