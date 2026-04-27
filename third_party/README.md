# `third_party/`

External research repositories pulled in as git submodules so that
`bionpu` can drive their training / evaluation pipelines to reproduce
published baselines and produce model checkpoints that
[`bionpu.scoring`](../src/bionpu/scoring/) can consume.

These are **upstream tools, not bionpu code.** `bionpu`'s own scoring
modules contain clean-room re-implementations of the architectural
shapes described in the upstream papers and never `import` from
anything under `third_party/`. The submodules exist solely so that
the upstream authors' training/eval pipelines are pinned, fetchable
on `git submodule update --init`, and reproducible at a known commit.

## Submodules

### `crispr_dnabert/`

* **Upstream:** `kimatakai/CRISPR_DNABERT`
  ([fork pinned](https://github.com/opensensor/CRISPR_DNABERT) at
  `opensensor/CRISPR_DNABERT`)
* **Paper:** Kimata K. *et al.* (2025), "Improved CRISPR/Cas9
  Off-target Prediction with DNABERT and Epigenetic Features,"
  *PLOS One*. DOI:
  [10.1371/journal.pone.0335863](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335863).
* **Pinned commit:** see `git submodule status third_party/crispr_dnabert`.
* **Used by:** [`bionpu.scoring.dnabert_epi`](../src/bionpu/scoring/dnabert_epi.py)
  consumes a fine-tuned classifier checkpoint produced by the
  upstream training pipeline. The clean-room classifier head in
  [`bionpu/scoring/_head.py`](../src/bionpu/scoring/_head.py) matches
  the no-epi specialisation of the upstream `DNABERTEpiModule`.
* **Reproduction recipe:** see
  [`docs/reproduce-dnabert-epi.md`](../docs/reproduce-dnabert-epi.md).

## License posture

Upstream repos may carry their own licenses (or, for the DNABERT-Epi
fork specifically, no `LICENSE` file at the time of vendoring). They
are **not** redistributed inside the `bionpu` wheel; they live as
git submodules that users explicitly opt in to via
`git submodule update --init`. `bionpu` itself remains GPL-3.0
end-to-end and contains no upstream-licensed code.

If the upstream license posture matters for your downstream use,
consult each submodule's repo directly. We track license-grant
follow-ups separately from the bring-up work.
