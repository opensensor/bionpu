# bionpu_tandem_repeat — source manifest

Short tandem repeat (STR) detection on AIE2P. v0 detects mono- through
hexa-nucleotide repeats with at least 5 consecutive copies, on the
forward strand only.

| file | purpose |
| --- | --- |
| `tandem_repeat.py` | IRON Python lowering. Broadcasts packed DNA chunks to 1/2/4/8 tiles and joins sparse STR-record output. |
| `tandem_repeat_tile.cc` | AIE2P C++ tile kernel. Per-period streak counters; emits 16-byte STR records on streak break / end-of-chunk. |
| `tandem_repeat_constants.h` | Wire-format, period bounds, chunk geometry, sparse-output constants. |
| `runner.cpp` | XRT host runner. Plans overlapped chunks, dispatches silicon, parses sparse output, applies oracle-style dedup. |
| `__init__.py` | Python `BionpuTandemRepeat` op wrapper and `NPU_OPS` registration. |
| `Makefile` | Build recipe for AIE2P xclbin, instruction binary, and host runner. |

## Wire Contract

| constant | value |
| --- | ---: |
| `TR_MIN_PERIOD` | 1 |
| `TR_MAX_PERIOD` | 6 |
| `TR_MIN_COPIES` | 5 |
| `TR_SEQ_IN_CHUNK_BYTES_BASE` | 4096 |
| `TR_SEQ_IN_OVERLAP` | 12 |
| `TR_HEADER_BYTES` | 8 |
| `TR_PARTIAL_OUT_BYTES_PADDED` | 32768 |
| `TR_RECORD_BYTES` | 16 |
| `TR_MAX_EMIT_IDX` | 2046 |

The kernel emits multiple overlapping streak records (one per period
that satisfied threshold). The Python op + C++ runner apply
oracle-style dedup so the final result matches
`bionpu.data.tandem_repeat_oracle.find_tandem_repeats`.

## Per-cell artifact directories

Built artifacts live under
`bionpu-public/src/bionpu/dispatch/_npu_artifacts/bionpu_tandem_repeat_n{n_tiles}/`.
Each artifact directory contains `final.xclbin`, `insts.bin`, and
`host_runner`. v0 silicon-validates `n_tiles=4`.

## Build invocation (from this directory)

```bash
source /opt/xilinx/xrt/setup.sh
source /home/$USER/xdna-bringup/ironenv/bin/activate
export MLIR_AIE_DIR=/home/$USER/genetics/third_party/mlir-aie
export PEANO_INSTALL_DIR=/home/$USER/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie

make NPU2=1 experiment=wide4 seq=10000 all
```
