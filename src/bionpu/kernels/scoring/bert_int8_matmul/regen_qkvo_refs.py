#!/usr/bin/env python3
"""Regenerate qkvo silicon-byte-equal reference fixtures.

Writes ``xs.bin``, ``w.bin``, and ``y_qkvo_ref.bin`` from the canonical
Python packers + emulator. Use after a ``K_CHUNK`` or shape change
invalidates the on-disk fixtures.

Background: commit 8f74723 (2026-04-28) flipped ``K_CHUNK`` from 64 to
8 in the runner wire format but did not regenerate the on-disk fixtures.
The canonical host-runner compare command in ``MANIFEST.md`` therefore
aborts with ``size mismatch`` until the fixtures are refreshed. This
script is the refresh.

Usage::

    python3 regen_qkvo_refs.py --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bionpu.kernels.scoring.bert_int8_matmul import (
    _emulate,
    _pack_qkvo_w,
    _pack_qkvo_xs,
)

_M, _K, _N = 47, 768, 768


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.integers(-128, 127, size=(_M, _K), dtype=np.int8)
    w = rng.integers(-128, 127, size=(_N, _K), dtype=np.int8)
    scales_combined = (
        rng.uniform(0.5, 1.5, size=_N + 1).astype(np.float32) * 1e-5
    )

    xs = _pack_qkvo_xs(x, scales_combined)
    w_packed = _pack_qkvo_w(np.ascontiguousarray(w))
    y_ref = _emulate(x, w, scales_combined, expected_n=_N)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "xs.bin").write_bytes(xs)
    (out / "w.bin").write_bytes(w_packed)
    (out / "y_qkvo_ref.bin").write_bytes(y_ref.astype(np.int8).tobytes())

    print(
        f"regenerated xs.bin ({len(xs)} B), w.bin ({len(w_packed)} B), "
        f"y_qkvo_ref.bin ({y_ref.size} B) — seed={args.seed}, "
        f"M={_M} K={_K} N={_N}"
    )


if __name__ == "__main__":
    main()
