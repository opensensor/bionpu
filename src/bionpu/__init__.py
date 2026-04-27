"""bionpu: AIE2P-accelerated genomics with reference-equivalence verification.

Top-level package. Subpackages:

- :mod:`bionpu.kernels`  — AIE2P MLIR-AIE kernels (CRISPR + basecalling)
- :mod:`bionpu.dispatch` — NPU silicon dispatch + serialisation lock
- :mod:`bionpu.verify`   — Byte-equality harness (public API)
- :mod:`bionpu.bench`    — Energy + timing measurement
- :mod:`bionpu.data`     — Reference / fixture fetchers
- :mod:`bionpu.quant`    — Quantisation helpers (FP32 → bf16 / int8)

See the project README for usage and architectural overview.
"""

__version__ = "0.1.0.dev0"
