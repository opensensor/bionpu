# Postmortem — `linear_projection` stage 3 fused-perts (B1 / B2 / B3b)

This is the closing narrative for the DESIGN-fusion.md stage-3
implementation. The spec lives in `DESIGN-fusion.md`; this doc only
records what happened on silicon and points the next reader at the
integration knobs.

## Headline

Silicon-validated **58.2x** speedup over the per-group baseline — an
outcome ~4x better than the stage-3 estimate of 4x. Stage 3 is now
encroaching on stage-4 territory in absolute wall-time.

| variant       | wall-time (avg/iter) | max-abs vs ref       | gate     |
|---------------|----------------------|----------------------|----------|
| per-group     | **563.49 ms**        | 1.19e-6 (vs FP32)    | (n/a)    |
| fused-perts   | **9.68 ms**          | 2.92e-2 (vs bf16)    | < 5e-2   |
| **speedup**   | **58.2x**            | 3.00e-2 (cross)      |          |

Source measurements: `results/basecalling/b-m6-fused-perts/measurements.json`
(B2 silicon validation, 2026-04-28).

## Why stage 3 outperformed the 141 ms target

DESIGN-fusion.md estimated stage 3 at ~141 ms (4x speedup over
the per-group baseline's ~564 ms). Actual silicon measurement: 9.68
ms. The estimate is off by ~14x — substantively, not measurement
noise.

The conservative ingredient was `_call_overhead_us`. The estimate
modelled per-call dispatch overhead as ~422 us per call x 1336 calls
= 564 ms wall (i.e. 100% of per-group wall time). Stage 3 reduces the
call count to 334 (one per timestep, kernel walks the 4 OC groups
internally), which would give 422 us x 334 = 141 ms IF the per-call
overhead were fixed. Silicon shows it's not — the fused kernel does
material work between dispatches that amortises the overhead. The
334-call kernel-internal-loop hits ~29 us/call, not 422 us/call.

Practical implication: stage 4 (the "few ms" full-kernel-fusion floor
the design doc envisioned) buys far less marginal speedup than
estimated. Stage 3 is the new realistic upper bound until / unless a
profile run identifies a new bottleneck.

## Encoder integration

The encoder pipeline (v0.2 scope, not yet landed) selects between
the per-group and fused-perts artifacts via env var, plumbed through
a one-call helper at the basecalling-package top level.

* Env var: **`BIONPU_DORADO_LINEAR_PROJECTION_VARIANT`** in
  `{'per_group' (default for now), 'fused_perts'}`.
* Helper:
  `bionpu.kernels.basecalling.get_linear_projection_op()` returns the
  registered NpuOp picked by the env var.
* Default is `per_group` until the encoder pipeline is cut over;
  flipping to `fused_perts` is a one-line env-var change.
* Both ops remain registered in `bionpu.dispatch.npu.NPU_OPS`;
  callers can also look them up by op-name directly.

## Multi-dispatch benchmark

The single-dispatch B2 run (`state/b2-silicon-validation.py`)
validates *correctness* at one input shape; the multi-dispatch B3b
benchmark validates that the speedup *holds* under encoder-realistic
load (many chunks, shared weight, single silicon-lock acquire).

Path: `bionpu-public/benchmarks/basecalling/bench_linear_projection_block.py`

Validation gate (B3b done criteria): `--chunks 8` produces a
measurements JSON at
`results/basecalling/b-m7-linear-projection-block-<iso>/measurements.json`
with no NaN/Inf and bf16 max-abs drift under 5e-2.

## What's NOT closed

* **stage 4** (full-kernel single-dispatch) is open but de-prioritised —
  the marginal speedup over stage 3 has gotten much smaller in light
  of B2 results.
* **48-KiB ObjectFifo lowering** (`B1-fused-perts-objectfifo-48k-chunk`)
  is still open in `gaps.yaml` — silicon ran clean but we never
  silicon-confirmed whether the IRON ObjectFifo single-element
  acquire produces a single shim DMA -> tile-DM transfer or a chunked
  one. The 9.68 ms wall-time is consistent with single-shot transfer,
  but a profile run would close the loop on this.
* **encoder pipeline driver** (v0.2) is not in the repo yet; the
  helper above is scaffolding for when it lands.

## See also

* `DESIGN-fusion.md` — the spec; this doc is its postmortem only.
* `gaps.yaml` — open items including the encoder-integration env-var
  hook (B3b).
* `tests/test_linear_projection_fused_correctness.py` — correctness
  regressions (build-clean) for both variants.
* `tests/test_basecalling_dispatch_helper.py` — env-var dispatch
  helper tests.
* `state/b2-silicon-validation.py` — single-dispatch silicon
  validation script that produced
  `results/basecalling/b-m6-fused-perts/measurements.json`.
