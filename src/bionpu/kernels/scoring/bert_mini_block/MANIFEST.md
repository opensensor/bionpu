# bert_mini_block — source manifest (PRD-dnabert-epi step 0.3)

AIE2P softmax + LayerNorm kernels for one BERT-mini transformer block.
Phase 0 step 0.3 of `PRDs/PRD-dnabert-epi-on-xdna.md` v0.1.3 (§3.8).

## Status (as of 2026-04-28)

- `bert_mini_attention_softmax` — silicon-validated; bf16; structurally
  correct vs PyTorch FP32 (MSE 1.2e-6, row sum ~0.995).
- `bert_mini_layer_norm` — silicon-validated; bf16; structurally correct
  vs PyTorch FP32 (MSE 9.9e-5; row mean ~0; row std ~1).

Bit-equal compare against the numpy reference fails because the numpy
reference applies bf16 round-trip after every step while silicon
accumulates in `aie::accum<accfloat>` (higher precision). PyTorch FP32
is the correct ground-truth comparator. See
`state/phase0/bert_mini_block_step0_3_status.json` finding `AUX-2`.

## Files

| name | role |
|---|---|
| `bert_mini_block.cc` | AIE2P C++ kernel — both softmax and LN entry points. Adapted from upstream `mlir-aie/aie_kernels/aie2p/{softmax,layer_norm}.cc` (Apache-2.0 WITH LLVM-exception). |
| `bert_mini_softmax.py` | IRON-Python lowering for per-row softmax (NUM_ROWS=M·NUM_HEADS=188 row dispatches inside one range_ loop). |
| `bert_mini_layer_norm.py` | IRON-Python lowering for per-row LN. γ‖β packed into one shim slab to fit NPU2Col1's 2-MM2S budget. |
| `Makefile` | Builds `softmax` and `layer_norm` xclbins. `make NPU2=1 all`. |
| `__init__.py` | Python `NpuOp` registration: `bert_mini_attention_softmax`, `bert_mini_layer_norm`. Numpy reference fallback when silicon artifacts are missing. |

## Op contracts

### `bert_mini_attention_softmax`

| arg | shape | dtype | layout |
|---|---|---|---|
| `scores` | (NUM_ROWS, PAD) | f32 | row-major; tail [M..PAD-1] pre-filled with -65000 sentinel; pre-scaled by 1/√head_dim |
| Returns | (NUM_ROWS, PAD) | f32 | softmax(scores), bf16 round-trip applied |

Default shape: NUM_ROWS = M·NUM_HEADS = 47·4 = 188; PAD = 64.

### `bert_mini_layer_norm`

| arg | shape | dtype | layout |
|---|---|---|---|
| `x`     | (M, HIDDEN) | f32 | row-major activations (residual already added on host) |
| `gamma` | (HIDDEN,) | f32 | per-channel scale |
| `beta`  | (HIDDEN,) | f32 | per-channel bias |
| Returns | (M, HIDDEN) | f32 | gamma·((x-mean)/√(var+ε)) + beta, bf16 round-trip applied |

Default shape: M=47, HIDDEN=256.

## Build

```bash
source /opt/xilinx/xrt/setup.sh
source /home/$USER/xdna-bringup/ironenv/bin/activate
export MLIR_AIE_DIR=/home/$USER/genetics/third_party/mlir-aie
export PEANO_INSTALL_DIR=/home/$USER/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie
make NPU2=1 all
```

Artifacts are then copied to
`bionpu/dispatch/_npu_artifacts/bert_mini_attention_softmax/` and
`bionpu/dispatch/_npu_artifacts/bert_mini_layer_norm/` for the
`NpuOp` dispatcher to pick up.

## Tile-DM budget

Both kernels run on a single AIE2P CoreTile:

### softmax

| resident | bytes | % of 64 KiB |
|---|---|---|
| in row (PAD bf16, depth=2) | 64×2×2 = 256 | 0.4% |
| out row (PAD bf16, depth=2) | 64×2×2 = 256 | 0.4% |
| stack | ~1024 | 1.6% |
| **total** | **~1.5 KB** | **2.4%** |

### layer_norm

| resident | bytes | % of 64 KiB |
|---|---|---|
| in row (HIDDEN bf16, depth=2) | 256×2×2 = 1024 | 1.6% |
| γ‖β slab (2·HIDDEN bf16) | 2×256×2 = 1024 | 1.6% |
| out row (HIDDEN bf16, depth=2) | 256×2×2 = 1024 | 1.6% |
| stack | ~1024 | 1.6% |
| **total** | **~4 KB** | **6.3%** |

## Out-of-scope auxiliaries (surfaced for follow-up)

Per PRD-dnabert-epi step 0.3 prompt: "If Q·Kᵀ or scores·V can't be done
with shape overrides on the existing `bert_int8_matmul.cc` and require
a new IRON lowering, **stop and report**." See `state/phase0/
bert_mini_block_step0_3_status.json` for full findings; summary:

- `bert_int8_matmul` head-variant kernel symbol is hardcoded to
  `bert_int8_matmul_2`. Q·Kᵀ at N=47 and scores·V at N=64 need new
  C++ symbols.
- scores·V K=47 doesn't satisfy K%K_CHUNK==0 for the existing qkvo
  K_CHUNK=8; K=47's only divisors are {1, 47}.
- `bert_int8_matmul/__init__.py` hardcodes `_K=768`, `_QKVO_N=768`,
  `_FFN_GROUP_N=768`. hidden=256 dispatch needs either parameterization
  or a separate kernel directory.
- GELU upstream kernel (`mlir-aie/.../gelu.cc`) exists but step-0.3
  budget didn't allow authoring it.

These are tracked as scope split into step 0.3b for the v0.1.4 PRD
update.
