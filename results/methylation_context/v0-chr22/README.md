# methylation_context v0 chr22 validation

Fixture:
`/home/matteius/genetics/tracks/genomics/fixtures/chr22.2bit.bin`

Artifact:
`src/bionpu/dispatch/_npu_artifacts/bionpu_methylation_context_n4`

## Result

The v0 silicon path runs successfully on full chr22, but full-record
byte equality is blocked by sparse-output cap-fire. Methylation context
classification is not sparse enough for a 4,094-record per-chunk cap:
most `C` and `G` positions produce a valid `CG`, `CHG`, or `CHH`
record.

| metric | value |
| --- | ---: |
| packed bytes | 12,704,617 |
| bases | 50,818,468 |
| chunks | 3,108 |
| silicon records | 9,840,462 |
| CPU oracle records | 18,406,838 |
| recovery | 53.46% |
| deficit | 8,566,376 |
| silicon wall | 23.277 s |
| host-reported avg NPU dispatch | 813.408 us |
| CPU oracle wall | 21.318 s |

## Counts

| context | silicon | CPU oracle |
| --- | ---: | ---: |
| CG | 664,817 | 1,269,292 |
| CHG | 2,291,284 | 4,336,761 |
| CHH | 6,884,361 | 12,800,785 |

| strand | silicon | CPU oracle |
| --- | ---: | ---: |
| + | 4,888,833 | 9,160,652 |
| - | 4,951,629 | 9,246,186 |

## Interpretation

This is a useful boundary result. The tile classifier and host ABI are
silicon-validated on exact smoke tests, but full chr22 requires a v1
output strategy:

- wider per-chunk sparse output,
- smaller chunks,
- context-filtered mode, or
- count-only/window-aggregate mode when per-base records are too dense.

The failure mode is localized to output density, not classifier logic.
