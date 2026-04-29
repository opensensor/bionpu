# methylation_context v1 c1024 b4 chr22 validation

Artifact:
`src/bionpu/dispatch/_npu_artifacts/bionpu_methylation_context_n4_b4_c1024`

## Result

The `b4` artifact preserves full chr22 record equality while reducing
the number of launches from 12,505 to 3,127. End-to-end wall improves
only slightly, which shows dense output parsing/sort/write is now the
dominant cost.

| metric | value |
| --- | ---: |
| batches | 3,127 |
| records | 18,406,838 |
| record-equal | true |
| record recovery | 100.00% |
| silicon wall | 150.818 s |
| avg NPU dispatch | 1,057.99 us |

Counts match the CPU oracle exactly for `CG`, `CHG`, `CHH`, `+`, and
`-`.
