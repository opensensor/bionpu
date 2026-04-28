# CRISPR-Net composite NPU artifacts (T6.5 / C-M7)

**Op name:** `crispr_net_score`
**Source:** `bionpu/kernels/crispr/crispr_net/__init__.py`
**Architecture basis:** Lin et al. 2020 (Peppags/CRISPR-Net, MIT) —
license review at `tracks/crispr/LICENSES/REVIEW.md`.

## v1 contract

CRISPR-Net's architecture decomposes entirely into ops that already
have NPU lowerings in this repo:

| Layer (CRISPR-Net)         | Existing NPU lowering surface             |
|---                         |---                                        |
| Conv1D k=5, in=7, out=10   | T4.2's `dorado_fast_conv_stem` (k=5 family) |
| BiLSTM hidden=15 ×2        | T6.4-A's `dorado_fast_lstm_cell_bf16`      |
| Dense(80) → Dense(20) → Dense(1) | T6.4-A's `dorado_fast_linear_projection` |

So **no new xclbin is shipped with T6.5**. The composite chains the
above existing ops via `bionpu.dispatch.dispatch` when shape-matched,
and falls back to a faithful **bf16-envelope host emulation** when
shape-mismatched (the gap entries in
`bionpu/kernels/crispr/crispr_net/gaps.yaml` document this).

The bf16 envelope reproduces the live NPU's per-element numerical
narrowing at every kernel boundary (matmul-input, gate writeback,
recurrent state writeback) so the per-site numerical bound vs the
FP32 reference is the same as a live-NPU run would produce.

## Files

This directory ships only the MANIFEST. There is no `final.xclbin`,
`insts.bin`, or `host_runner` because v1 has no new xclbin.

If a future iteration ships a shape-parametric T=24 Conv1D + LSTM
xclbin (resolution of G-T6.5-001/002), the artifacts go here:

* `final.xclbin`
* `insts.bin`
* `host_runner`

The op surface is forwards-compatible — `CrisprNetCompositeOp.is_live_npu_path_available()`
returns False in v1 and would flip True once those land.
