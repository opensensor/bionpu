# methylation_context v1 c1024 b8 chr22 validation

Artifact:
`src/bionpu/dispatch/_npu_artifacts/bionpu_methylation_context_n4_b8_c1024`

## Result

The `b8` artifact preserves full chr22 record equality while reducing
the number of launches from 12,505 to 1,564. End-to-end wall is nearly
identical to `b4`, so batching alone is not the next major lever.

| metric | value |
| --- | ---: |
| batches | 1,564 |
| records | 18,406,838 |
| record-equal | true |
| record recovery | 100.00% |
| silicon wall | 150.647 s |
| avg NPU dispatch | 1,845.39 us |

Counts match the CPU oracle exactly for `CG`, `CHG`, `CHH`, `+`, and
`-`.
