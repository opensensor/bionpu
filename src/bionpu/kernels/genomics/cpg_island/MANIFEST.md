# bionpu_cpg_island — source manifest

Sliding-window Gardiner-Garden CpG-island candidate detection on AIE2P.

| file | purpose |
| --- | --- |
| `cpg_island.py` | IRON Python lowering. Broadcasts packed DNA chunks to 1/2/4/8 tiles and joins sparse candidate-position output. |
| `cpg_island_tile.cc` | AIE2P C++ tile kernel. Emits `uint32` candidate window-start positions passing fixed-point GC and obs/exp thresholds. |
| `cpg_island_constants.h` | Wire-format, threshold, chunk, and sparse-output constants. |
| `runner.cpp` | XRT host runner. Plans overlapped chunks, dispatches silicon, parses sparse output, and writes candidate positions. |
| `__init__.py` | Python `BionpuCpgIsland` op wrapper and `NPU_OPS` registration. |
| `Makefile` | Build recipe for AIE2P xclbin, instruction binary, and host runner. |

## Wire Contract

| constant | value |
| --- | ---: |
| `CI_W` | 200 |
| `CI_GC_NUM / CI_GC_DEN` | 1 / 2 |
| `CI_OE_NUM / CI_OE_DEN` | 3 / 5 |
| `CI_SEQ_IN_CHUNK_BYTES_BASE` | 4096 |
| `CI_SEQ_IN_OVERLAP` | 52 |
| `CI_HEADER_BYTES` | 8 |
| `CI_PARTIAL_OUT_BYTES_PADDED` | 32768 |
| `CI_RECORD_BYTES` | 4 |
| `CI_MAX_EMIT_IDX` | 8190 |

The kernel emits candidate window starts. The Python op merges
contiguous candidate runs into `(start, end)` islands with the same
run-length criterion as `bionpu.data.cpg_oracle`.
