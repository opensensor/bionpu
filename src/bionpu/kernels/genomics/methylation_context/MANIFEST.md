# bionpu_methylation_context — source manifest

CG/CHG/CHH methylation-context scanning on AIE2P. v0 emits sparse
cytosine-context records on both strands; minus-strand cytosines are
represented by forward-reference `G` positions.

| file | purpose |
| --- | --- |
| `methylation_context.py` | IRON Python lowering. Broadcasts packed DNA chunks to 1/2/4/8 tiles and joins sparse context-record output. |
| `methylation_context_tile.cc` | AIE2P C++ tile kernel. Per-base CG/CHG/CHH classifier with sparse emit. |
| `methylation_context_constants.h` | Wire-format, chunk geometry, context-code, and sparse-output constants. |
| `runner.cpp` | XRT host runner. Plans overlapped chunks, dispatches silicon, parses sparse output, sorts/dedups records. |
| `__init__.py` | Python `BionpuMethylationContext` op wrapper and `NPU_OPS` registration. |
| `Makefile` | Build recipe for AIE2P xclbin, instruction binary, and host runner. |

## Wire Contract

| constant | value |
| --- | ---: |
| `MC_SEQ_IN_CHUNK_BYTES_BASE` | 4096 |
| `MC_SEQ_IN_OVERLAP` | 4 |
| `MC_HEADER_BYTES` | 8 |
| `MC_PARTIAL_OUT_BYTES_PADDED` | 32768 |
| `MC_RECORD_BYTES` | 8 |
| `MC_MAX_EMIT_IDX` | 4094 |

Output record:

`uint32 pos | uint8 strand | uint8 context | uint16 pad`

`strand`: `0=+`, `1=-`

`context`: `0=CG`, `1=CHG`, `2=CHH`

## Build invocation

```bash
source /opt/xilinx/xrt/setup.sh
source /home/$USER/xdna-bringup/ironenv/bin/activate
export MLIR_AIE_DIR=/home/$USER/genetics/third_party/mlir-aie
export PEANO_INSTALL_DIR=/home/$USER/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie

make NPU2=1 experiment=wide4 all
```

The record-safe chr22 artifact uses smaller chunks:

```bash
make NPU2=1 experiment=wide4 seq_chunk_bytes=1024 all
```
