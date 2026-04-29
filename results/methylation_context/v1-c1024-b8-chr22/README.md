# methylation_context v1 c1024 b8 chr22 validation

Artifact:
`src/bionpu/dispatch/_npu_artifacts/bionpu_methylation_context_n4_b8_c1024`

## Result

The `b8` artifact preserves full chr22 record equality while reducing
the number of launches from 12,505 to 1,564. The host runner now filters
chunk-overlap records into strict owned intervals, observes monotonic
output, skips the global sort/unique pass, and streams binary records
directly to the output file. End-to-end runner wall dropped by ~41x
versus the prior vector-materialized `b8` path.

| metric | value |
| --- | ---: |
| batches | 1,564 |
| records | 18,406,838 |
| record-equal | true |
| record recovery | 100.00% |
| silicon wall | 3.675 s |
| avg NPU dispatch | 1,545.51 us |

Counts match the CPU oracle exactly for `CG`, `CHG`, `CHH`, `+`, and
`-`.
