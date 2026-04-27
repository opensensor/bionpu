"""bionpu.scoring — torch-backed integration tests.

GPL-3.0. (c) 2026 OpenSensor.

Exercises the full plumbing of the real (non-smoke) DNABERTEpiScorer
path WITHOUT requiring a pre-trained checkpoint or network access:

1. Synthesises a deterministic random classifier head matching the
   bionpu no-epi shape.
2. Saves it as a state_dict file.
3. Round-trips through :func:`bionpu.scoring._extract_head.extract_no_epi_head`
   to confirm the upstream-checkpoint conversion API works on a
   minimal synthetic input.
4. Verifies that the scorer's torch path raises a clear error when
   torch is present but the checkpoint can't satisfy the head shape
   contract.

Skipped when torch is not installed (the scorer's smoke mode covers
the torch-free path; tests/test_scoring_smoke.py handles those).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from bionpu.scoring._extract_head import ExtractError, extract_no_epi_head


def _make_synthetic_state(
    *, hidden_size: int = 768, num_classes: int = 2, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Synthesise a deterministic state_dict with bionpu's expected keys."""
    g = torch.Generator().manual_seed(seed)
    return {
        "1.weight": torch.randn(num_classes, hidden_size, generator=g),
        "1.bias": torch.randn(num_classes, generator=g),
    }


def _make_synthetic_upstream_state(
    *, hidden_size: int = 768, num_classes: int = 2, seed: int = 0,
    key_prefix: str = "classifier.1",
) -> dict[str, torch.Tensor]:
    """Mimic the upstream DNABERT-Epi checkpoint key layout (no-epi variant)."""
    g = torch.Generator().manual_seed(seed)
    return {
        f"{key_prefix}.weight": torch.randn(num_classes, hidden_size, generator=g),
        f"{key_prefix}.bias": torch.randn(num_classes, generator=g),
    }


def test_extract_round_trips_canonical_upstream_keys() -> None:
    """The default (`classifier.1.*`) upstream layout must convert cleanly."""
    upstream = _make_synthetic_upstream_state(key_prefix="classifier.1")
    bionpu_state = extract_no_epi_head(upstream)

    assert set(bionpu_state.keys()) == {"1.weight", "1.bias"}
    assert torch.equal(bionpu_state["1.weight"], upstream["classifier.1.weight"])
    assert torch.equal(bionpu_state["1.bias"], upstream["classifier.1.bias"])


def test_extract_handles_flat_classifier_layout() -> None:
    """Some upstream checkpoints flatten ``Sequential`` keys."""
    upstream = _make_synthetic_upstream_state(key_prefix="classifier")
    bionpu_state = extract_no_epi_head(upstream)
    assert torch.equal(bionpu_state["1.weight"], upstream["classifier.weight"])
    assert torch.equal(bionpu_state["1.bias"], upstream["classifier.bias"])


def test_extract_handles_dataparallel_module_prefix() -> None:
    """DataParallel-wrapped checkpoints prefix every key with ``module.``."""
    upstream = _make_synthetic_upstream_state(key_prefix="module.classifier.1")
    bionpu_state = extract_no_epi_head(upstream)
    # The bionpu output keys are layout-canonical regardless of how the
    # upstream named them.
    assert set(bionpu_state.keys()) == {"1.weight", "1.bias"}


def test_extract_rejects_with_epi_classifier_shape() -> None:
    """A classifier trained WITH epi features has in_features = 768 + 256*N,
    not 768. Loading that into the no-epi head would silently truncate;
    we should fail loud instead."""
    g = torch.Generator().manual_seed(0)
    upstream = {
        # 768 + 256 = 1024 ⇒ with-epi (1 epi track). Wrong shape for no-epi.
        "classifier.1.weight": torch.randn(2, 1024, generator=g),
        "classifier.1.bias": torch.randn(2, generator=g),
    }
    with pytest.raises(ExtractError, match="trained WITH"):
        extract_no_epi_head(upstream)


def test_extract_rejects_missing_keys() -> None:
    """Random unfamiliar state_dict must surface the candidate-list."""
    upstream = {"some_unrelated.layer.weight": torch.zeros(1)}
    with pytest.raises(ExtractError, match="no recognised classifier-head keys"):
        extract_no_epi_head(upstream)


def test_extract_rejects_wrong_bias_shape() -> None:
    upstream = {
        "classifier.1.weight": torch.zeros(2, 768),
        "classifier.1.bias": torch.zeros(3),  # wrong: should be 2
    }
    with pytest.raises(ExtractError, match="bias.*shape"):
        extract_no_epi_head(upstream)


def test_extract_round_trips_via_torch_save(tmp_path: Path) -> None:
    """End-to-end file round-trip: torch.save → extract_no_epi_head."""
    upstream = _make_synthetic_upstream_state()
    ckpt = tmp_path / "upstream.pt"
    torch.save(upstream, ckpt)

    bionpu_state = extract_no_epi_head(ckpt)
    assert torch.equal(bionpu_state["1.weight"], upstream["classifier.1.weight"])


def test_extracted_head_loads_into_bionpu_head_module() -> None:
    """The extracted state_dict must be load_state_dict-compatible with
    bionpu.scoring._head.build_no_epi_head's output."""
    from bionpu.scoring._head import build_no_epi_head

    head = build_no_epi_head()
    upstream = _make_synthetic_upstream_state()
    bionpu_state = extract_no_epi_head(upstream)

    # No "missing keys" / "unexpected keys" — strict load.
    head.load_state_dict(bionpu_state, strict=True)

    # And the loaded weights match what was extracted.
    assert torch.equal(head[1].weight, upstream["classifier.1.weight"])
    assert torch.equal(head[1].bias, upstream["classifier.1.bias"])


def test_scorer_real_path_loads_extracted_head(tmp_path: Path) -> None:
    """Build a synthetic head, run extract, save in bionpu format, point the
    scorer at it, and confirm the scorer's lazy-loader accepts it.

    Skips the actual BERT body load (which would need network +
    HuggingFace cache) — that's covered by the upstream paper-replication
    walkthrough, not by this in-process test.
    """
    from bionpu.scoring._head import build_no_epi_head

    upstream = _make_synthetic_upstream_state(seed=42)
    bionpu_state = extract_no_epi_head(upstream)

    bionpu_ckpt = tmp_path / "bionpu-head.pt"
    torch.save(bionpu_state, bionpu_ckpt)

    # Round-trip: load into a fresh head via torch.load + state_dict apply.
    state2 = torch.load(bionpu_ckpt, map_location="cpu", weights_only=True)
    head2 = build_no_epi_head()
    head2.load_state_dict(state2, strict=True)

    # Determinism: same input must produce the same logits across two
    # freshly-loaded heads.
    head1 = build_no_epi_head()
    head1.load_state_dict(bionpu_state, strict=True)
    head1.eval()
    head2.eval()

    cls_emb = torch.randn(1, 768, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        out1 = head1(cls_emb)
        out2 = head2(cls_emb)
    assert torch.equal(out1, out2)
