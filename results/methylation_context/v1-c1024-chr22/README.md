# methylation_context v1 c1024 chr22 validation

Fixture:
`/home/matteius/genetics/tracks/genomics/fixtures/chr22.2bit.bin`

Artifact:
`src/bionpu/dispatch/_npu_artifacts/bionpu_methylation_context_n4_c1024`

## Result

Reducing the sequence chunk size from 4096 bytes to 1024 bytes fixes the
full chr22 record-mode cap-fire while preserving the original output
semantics.

| metric | value |
| --- | ---: |
| packed bytes | 12,704,617 |
| bases | 50,818,468 |
| chunks | 12,505 |
| silicon records | 18,406,838 |
| CPU oracle records | 18,406,838 |
| recovery | 100.00% |
| deficit | 0 |
| record-equal | true |
| silicon wall | 154.771 s |
| host-reported avg NPU dispatch | 386.744 us |
| CPU oracle wall | 21.330 s |

## Counts

| context | silicon | CPU oracle |
| --- | ---: | ---: |
| CG | 1,269,292 | 1,269,292 |
| CHG | 4,336,761 | 4,336,761 |
| CHH | 12,800,785 | 12,800,785 |

| strand | silicon | CPU oracle |
| --- | ---: | ---: |
| + | 9,160,652 | 9,160,652 |
| - | 9,246,186 | 9,246,186 |

## Interpretation

The failure in the original 4096-byte chunk artifact was output
density, not classifier logic. A 2048-byte chunk variant recovered
96.9% but still hit the 4,094-record cap in dense regions. The
1024-byte variant removes cap-fire on chr22 and gives exact record
parity at the cost of more dispatches.

The next throughput fix is batching multiple 1024-byte chunks per
launch once the record-safe contract is pinned.
