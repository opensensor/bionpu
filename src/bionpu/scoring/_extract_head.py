# bionpu — AIE2P-accelerated genomics with reference-equivalence verification.
# Copyright (C) 2026 OpenSensor / Matt Davis <matt@opensensor.io>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Bridge between upstream DNABERT-Epi checkpoints and bionpu's head.

The upstream training pipeline (``third_party/crispr_dnabert``) saves
the full ``DNABERTEpiModule`` state dict — BERT body + epi-encoder +
gating MLP + classifier. bionpu's :mod:`._head` ships only the no-epi
classifier head (``Dropout → Linear(768, 2)``), and the BERT body is
loaded separately from Hugging Face. This module converts an upstream
checkpoint's classifier-head weights into a bionpu-loadable
state_dict.

Why a separate file
-------------------

Keeps the upstream-specific state_dict-key knowledge (which evolves
across upstream commits / fork branches) isolated from the runtime
scorer. The bionpu wheel doesn't need this module at inference time;
it's only used at checkpoint-conversion time.

License-clean: this code is GPL-3.0 (bionpu's own), reads only the
weight tensor names + values from a state_dict, and never imports
upstream code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "ExtractError",
    "extract_no_epi_head",
]


class ExtractError(RuntimeError):
    """Raised when the upstream checkpoint can't be converted."""


# Upstream key candidates for the no-epi classifier head's
# ``Linear(768, 2)`` weights. Multiple shapes covered because the
# upstream module uses ``nn.Sequential(Dropout, Linear)`` and the
# checkpoint may carry either the named field (``classifier.1``) or
# a flattened name (``classifier``).
_WEIGHT_CANDIDATES: tuple[tuple[str, str], ...] = (
    # (weight key, bias key)
    ("classifier.1.weight", "classifier.1.bias"),
    ("classifier.weight", "classifier.bias"),
    ("module.classifier.1.weight", "module.classifier.1.bias"),
    ("module.classifier.weight", "module.classifier.bias"),
)

# The bionpu head module is ``nn.Sequential(Dropout, Linear)``; PyTorch
# names the Linear's weights ``"1.weight"`` / ``"1.bias"``.
_BIONPU_KEYS = ("1.weight", "1.bias")


def extract_no_epi_head(
    upstream_state: dict[str, Any] | Path,
    *,
    expected_hidden_size: int = 768,
    expected_num_classes: int = 2,
) -> dict[str, Any]:
    """Convert an upstream DNABERT-Epi state_dict into a bionpu head dict.

    Parameters
    ----------
    upstream_state:
        Either a ``state_dict``-shaped mapping or a path to a file
        produced by ``torch.save(state, path)``.
    expected_hidden_size:
        Expected ``in_features`` of the classifier ``Linear``. The
        no-epi specialisation has no concatenated epi-features, so
        ``in_features`` is exactly the BERT hidden size (768 for
        DNABERT-3 base).
    expected_num_classes:
        Expected ``out_features`` (2 for binary off-target
        classification — index 1 is "off-target").

    Returns
    -------
    dict
        State-dict shaped ``{"1.weight": tensor[2,768], "1.bias": tensor[2]}``
        that ``DNABERTEpiScorer(weights_path=...)`` can ``load_state_dict``
        into the bionpu head from :mod:`._head`.

    Raises
    ------
    ExtractError
        If no recognised classifier-head keys are present, or the
        tensor shape doesn't match the no-epi expectation (which
        usually means the checkpoint was trained WITH epi features,
        in which case bionpu's no-epi head is the wrong target).
    """
    if isinstance(upstream_state, (str, Path)):
        try:
            import torch
        except ImportError as exc:
            raise ExtractError(
                "extracting from a checkpoint file requires `torch` to "
                "be installed; pass an in-memory state_dict instead "
                "if torch is unavailable."
            ) from exc
        state = torch.load(Path(upstream_state), map_location="cpu", weights_only=True)
    else:
        state = upstream_state

    if not isinstance(state, dict):
        raise ExtractError(
            f"expected a state_dict-shaped mapping; got {type(state).__name__}"
        )

    weight = bias = None
    matched_keys: tuple[str, str] | None = None
    for w_key, b_key in _WEIGHT_CANDIDATES:
        if w_key in state and b_key in state:
            weight = state[w_key]
            bias = state[b_key]
            matched_keys = (w_key, b_key)
            break

    if weight is None or bias is None:
        keys_seen = sorted(state.keys())
        # Hint at most-likely upstream layout when none matched.
        head_like = [k for k in keys_seen if "classifier" in k.lower()]
        raise ExtractError(
            f"no recognised classifier-head keys in upstream state_dict. "
            f"Tried: {[w for w, _ in _WEIGHT_CANDIDATES]}. "
            f"Classifier-like keys observed: {head_like or 'none'}. "
            f"If this is an unfamiliar upstream branch, add a key pair "
            f"to bionpu/scoring/_extract_head._WEIGHT_CANDIDATES."
        )

    # Shape check — prevents loading a with-epi checkpoint into the
    # no-epi head (the with-epi classifier's in_features is
    # 768 + 256 * len(using_epi_data), not 768).
    w_shape = tuple(getattr(weight, "shape", ()))
    if len(w_shape) != 2 or w_shape != (expected_num_classes, expected_hidden_size):
        raise ExtractError(
            f"classifier weight at {matched_keys[0]!r} has shape {w_shape}; "
            f"expected ({expected_num_classes}, {expected_hidden_size}). "
            f"This usually means the upstream checkpoint was trained WITH "
            f"epigenetic features (using_epi_data was non-empty); the "
            f"bionpu no-epi head can't consume that. Train with "
            f"`--using_epi_data ''` (no epi) or wait for the with-epi "
            f"variant of bionpu.scoring."
        )
    b_shape = tuple(getattr(bias, "shape", ()))
    if b_shape != (expected_num_classes,):
        raise ExtractError(
            f"classifier bias at {matched_keys[1]!r} has shape {b_shape}; "
            f"expected ({expected_num_classes},)"
        )

    return {_BIONPU_KEYS[0]: weight, _BIONPU_KEYS[1]: bias}
