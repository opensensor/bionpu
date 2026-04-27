# PRD-1 model-selection audit

> Status: Week 0 of PRD-1 ("Off-Target Predictor v2 on AIE2P"). This
> document is the input to the week-2 model-decision milestone.
> Findings are factual extracts from public sources (papers, GitHub
> repos, dataset homepages) — the decision itself is not made here.

## TL;DR

**All four candidate models we audited fail the PRD's commercial-use
license gate** (none ship a LICENSE file; under default copyright that
means all rights reserved). **The PRD's core premise also has factual
errors** that materially change the candidate set. The dataset
(CRISPRoffT) is downloadable but **CC BY-NC** — fine for verification,
not redistributable in a shipped artifact.

This document recommends a pivot in the PRD before any code lands.
See [§ 7 Recommendation](#7-recommendation).

## 1. Premise correction

The PRD opens with:

> *"The 2025 Cao et al. benchmark established that newer transformer-class
> models — CRISPR-BERT, DNABERT-Epi, CRISPR-SGRU — match or exceed
> CRISPR-Net on the CRISPRoffT validation set."*

This is **inaccurate** on three counts, verified against the Cao 2025
*Small Methods* paper (DOI [10.1002/smtd.202500122][cao2025], PubMed
40468633):

1. The Cao 2025 CRISPRoffT benchmark evaluates **six** models —
   CRISPR-Net, CRISPR-IP, R-CRISPR, CRISPR-M, CrisprDNT, Crispr-SGRU.
   CRISPR-BERT and DNABERT-Epi are **not** in the lineup.
2. The Cao 2025 abstract's verdict is *"no model consistently
   outperforms other models across all scenarios; CRISPR-Net,
   R-CRISPR, and Crispr-SGRU show strong overall performance"* —
   they are co-leaders, not Crispr-SGRU singularly leading.
3. The paper that does compare CRISPR-BERT and DNABERT-Epi is
   Mukai/Kimata 2025 ([PLOS One DOI 10.1371/journal.pone.0335863][mukai2025])
   — and it evaluates on **CHANGE-seq / GUIDE-seq / TTISS**, not on
   CRISPRoffT.

So the PRD's named candidates and dataset are misaligned. Either:

- **Keep CRISPRoffT as the validation corpus** → candidate set should
  be R-CRISPR + Crispr-SGRU (the actual CRISPRoffT performers in
  Cao 2025).
- **Keep CRISPR-BERT / DNABERT-Epi as candidates** → validation
  corpus should be CHANGE-seq / GUIDE-seq subsets, not CRISPRoffT.

[cao2025]: https://onlinelibrary.wiley.com/doi/10.1002/smtd.202500122
[mukai2025]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335863

## 2. CRISPR-BERT (Luo et al. 2024)

| Field | Finding |
|---|---|
| Repo | https://github.com/BrokenStringx/CRISPR-BERT |
| Paper | Luo Y. et al., "Interpretable CRISPR/Cas9 off-target activities with mismatches and indels prediction using BERT," *Computers in Biology and Medicine* 169:107932 (2024). [PubMed 38199209](https://pubmed.ncbi.nlm.nih.gov/38199209/). |
| License | **NO LICENSE FILE.** GitHub `/license` API returns 404. Default copyright = all rights reserved. **HARD GATE FAILURE.** |
| Parameters | ~10M (BERT-tiny `uncased_L-2_H-256_A-4` = 2 layers × 256 hidden × 4 heads = ~4.4M, plus multi-scale Conv2D heads + 2× BiGRU(40) + Dense layers). Shipped weights `weight/I1.h5` = 40.8 MB at FP32. **Fits AIE2P.** |
| Pre-trained weights | Yes for one dataset (I1) shipped in repo as 40.8 MB `.h5`. None for the other six datasets. Also requires external BERT-tiny checkpoint from `google-research/bert` not in repo. |
| Training code | **Not present** — only `model_test.py` for eval. Training loop must be reconstructed. |
| CRISPRoffT evaluation | **Never evaluated on CRISPRoffT.** Original paper uses seven internal datasets (I1, I2, HEK293t, K562, II4, II5, II6). |
| Verdict | **Blocked by license.** Even if licensed, no CRISPRoffT target exists to validate against ±0.005. |

There is also a separate `OSsari/CrisprBERT` (Sari 2024) — different
architecture, smaller (~1M params), also no LICENSE file. Same
verdict.

## 3. DNABERT-Epi (Kimata et al. 2025)

| Field | Finding |
|---|---|
| Repo | https://github.com/kimatakai/CRISPR_DNABERT |
| Paper | Kimata K., "Improved CRISPR/Cas9 Off-target Prediction with DNABERT and Epigenetic Features," *PLOS One* (2025). DOI [10.1371/journal.pone.0335863][mukai2025]. |
| License | **NO LICENSE FILE.** Default copyright. **HARD GATE FAILURE.** Upstream DNABERT-3 weights on HF (`zhihan1996/DNA_bert_3`) also have no declared license on the HF page. |
| Parameters | ~86M (BERT-base topology — 12 layers × 768 hidden × 12 heads + epigenetic-MLP head). **EXCEEDS the PRD's 50M ceiling.** No smaller distilled variant offered. |
| Pre-trained weights | **DNABERT-Epi weights not released.** User must fine-tune from `zhihan1996/DNA_bert_3` themselves. |
| Training code | Present (`src/run_model.py`, `src/run_preprocess.py`). |
| CRISPRoffT evaluation | **Never evaluated on CRISPRoffT.** Paper evaluates only on Lazzarotto 2020 GUIDE-seq (T-cell), 14-fold CV. ROC-AUC 0.9857 ± 0.0124, PR-AUC 0.5501 ± 0.0673. |
| Inference-time epigenetic features | **Required.** ATAC-seq + H3K4me3 + H3K27ac, 500 bp window × 50 bins = 300-dim feature vector. Cell-type-specific tracks (paper used GSM4498611 / GSM4495703 / GSM4495711 — Lazzarotto T-cell). Paper's own limitation: *"validation … was limited to a single GUIDE-seq dataset due to the lack of publicly available, matched epigenetic data for other off-target datasets."* |
| Verdict | **Triple block:** license + size + inference-time BigWig fetch dependency. Pipeline integration would need a feature-fetcher stage between PAM scan and scoring, plus cell-type-config burden the CLI doesn't support. |

## 4. R-CRISPR (Niu et al. 2021)

| Field | Finding |
|---|---|
| Repo | **None.** No code repository on GitHub or Code Ocean. The paper's data-availability statement points only to Lin 2020's CRISPR-Net Code Ocean capsule for *training data*, not R-CRISPR code. |
| Paper | Niu R. et al., "R-CRISPR: A Deep Learning Network to Predict Off-Target Activities … using RepVGG and BiLSTM," *Genes* 12(12):1878 (2021). DOI [10.3390/genes12121878](https://doi.org/10.3390/genes12121878), [PMC8702036](https://pmc.ncbi.nlm.nih.gov/articles/PMC8702036/). CC BY 4.0 on the paper text only. |
| License | No code license (no code published). Paper text is CC BY 4.0 — covers the architectural description, not an implementation. |
| Parameters | Estimated ~150K from the paper's architecture description (40 Conv kernels + 40 RepVGG blocks + BiLSTM(15) + Dense). Tiny — fits AIE2P trivially. |
| Pre-trained weights | **None.** Not in any public location. |
| Training code | **Not released.** Architecture must be reconstructed clean-room from the paper. |
| CRISPRoffT performance | Cao 2025 lists R-CRISPR as a co-leader on CRISPRoffT, but **per-model AUROC/PRAUC numbers are paywalled** — Wiley returned 403 to WebFetch; PMC mirror under embargo until 2026-06-04. Cannot extract the ±0.005 target without institutional access. Original Niu 2021 paper reports AUROC 0.991 / AUPRC 0.319 on GUIDE_II and AUROC 0.976 / AUPRC 0.460 on CIRCLE 5-fold CV — those are the only concrete numbers without paywall. |
| Verdict | **Reimplementation-only path.** No code, no weights → would need a clean-room rebuild from the CC BY 4.0 paper description. Architecture is small enough that this is tractable (~150K params, RepVGG + BiLSTM(15)) but adds weeks. |

## 5. Crispr-SGRU (Zhang et al. 2024)

| Field | Finding |
|---|---|
| Repo | https://github.com/BrokenStringx/Crispr-SGRU |
| Paper | Zhang G. et al., "Crispr-SGRU: Prediction of CRISPR/Cas9 Off-Target Activities with Mismatches and Indels Using Stacked BiGRU," *International Journal of Molecular Sciences* 25(20):10945 (2024). DOI [10.3390/ijms252010945](https://doi.org/10.3390/ijms252010945), [PMC11507390](https://pmc.ncbi.nlm.nih.gov/articles/PMC11507390/). |
| License | **NO LICENSE FILE in the repo.** Verified: `LICENSE` / `LICENSE.md` / `LICENSE.txt` all return 404; `/license` API returns "Not Found"; no metadata declaration. **HARD GATE FAILURE on the code.** Paper itself is CC BY 4.0 (covers text/architecture, not the shipped code or `weights/*.h5`). |
| Parameters | ~408K (Inception block + BiGRU(30/20/10) + Dense(128/64/2)). Dense(2880→128) dominates (368K of 408K). **Fits AIE2P trivially.** |
| Pre-trained weights | **Yes** — shipped in `weights/` subdir, ~1.69 MB Keras `.h5` per dataset, 5-fold per benchmark (`CHANGEseq_1.h5` … `_5.h5`, also `BE3`, `II5`, `II6`, `hek`, `k562`). License-blocked from redistribution. |
| Training code | Present (`Train/MODEL.py`, `Train/Encoder_sgRNA_off.py`). Repo hygiene is poor — `model_test.py` has unresolved git-merge markers and a broken import (won't run as-shipped without minor fixup). |
| Architecture | 24×7 binary input → Inception (4× parallel Conv2D, kernels {1,2,3,5}) → BiGRU(30) → BiGRU(20) → BiGRU(10) → concat → Dense(128) → Dense(64) → Dense(2 sigmoid). 24 timesteps × small hidden dims = clean AIE2P map; the only compute-heavy piece is the Dense(2880→128) layer. |
| CRISPRoffT performance | Same paywall problem as R-CRISPR. Cao 2025 lists Crispr-SGRU as a co-leader; per-model number unverifiable without institutional access. The IJMS 2024 paper itself does NOT evaluate on CRISPRoffT — eight internal benchmarks (HEK293t/K562/BE3/II5/II6/I1/I2/CHANGE-seq) only. |
| Verdict | **Blocked by license** on the code; reimplementation possible from the CC BY 4.0 paper description (~408K params, well-defined architecture). Best-fit for AIE2P among the four if the license is resolved. |

## 6. CRISPRoffT dataset

| Field | Finding |
|---|---|
| Hosting | https://ccsm.uth.edu/CRISPRoffT/ (UTHealth Houston). Direct download (verified HTTP 200, last-modified 2024-10-01): `https://ccsm.uth.edu/CRISPRoffT/table_summary/allframe_update_addEpige.txt` (151,429,169 bytes / ~144 MiB). |
| Paper | Wang G. et al., "CRISPRoffT: comprehensive database of CRISPR/Cas off-targets," *Nucleic Acids Research* 53(D1):D914-D924 (2025). DOI [10.1093/nar/gkae1025](https://doi.org/10.1093/nar/gkae1025). |
| Format | TSV (`.txt` extension). 47 columns. 226,164 guide / off-target pairs; 8,840 experimentally validated off-targets; 368 unique guides; 22,632 genes; 85 Cas/gRNA combinations. Aggregated from 74 published studies across 29 experimental techniques. |
| Labels | Both — continuous `Indel_accu%` cleavage rate AND binary `Identity` (ON/OFF) + `Validation` (TRUE/FALSE). |
| Train / val split | **No canonical split.** It's a database, not a benchmark partition. The verification harness must define its own split; recommend leave-one-sgRNA-out to match Cao 2025 + Crispr-SGRU's reported methodology. |
| License | **CC BY-NC.** Non-commercial verification use is fine with attribution. **Cannot be redistributed** in a shipped wheel or commercial bundle. |
| Mirror status | No Zenodo DOI, no GitHub release, no FigShare. Single web-UI download channel. **Pin SHA-256 of the 151,429,169-byte TSV** in our reproducibility lock-file. |

## 7. Recommendation

The PRD as written cannot proceed without revision. Three concrete
blockers:

1. **All four candidate models lack permissive licenses.** Direct-use
   path requires either (a) explicit written grants from each author
   or (b) clean-room reimplementation from the published architecture
   description. CC BY 4.0 on a paper covers the text/figures, not the
   code.
2. **No model has been evaluated on CRISPRoffT in a publicly-citable
   form** that gives us a concrete ±0.005 AUROC target. The Cao 2025
   numbers exist but are paywalled and PMC-embargoed until June 2026.
3. **CRISPRoffT is CC BY-NC** — fine for our verification harness, but
   blocks any commercial product distribution that bakes the dataset
   into a shipped artifact.

Three options for revising the PRD, in order of conservativeness:

### Option A — Stay with CRISPR-Net as the headline model

Keep CRISPR-Net (`JasonLinjc/CRISPR-Net`, MIT-style license, used by
the Cao 2025 benchmark as a baseline) as the production scorer.
Frame the v2 work as "CRISPR-Net on AIE2P" rather than "transformer
v2." Lower ambition, fully unblocked, ships clean.

### Option B — Clean-room Crispr-SGRU

Pick Crispr-SGRU as the v2 candidate (best AIE2P fit per the audit:
408K params, well-defined GRU stack, IJMS paper is CC BY 4.0).
Reimplement from the paper, train from public datasets (HEK293t /
K562 / CHANGE-seq), establish our own CRISPRoffT baseline numbers
(since Cao 2025's are paywalled). Adds ~3-4 weeks vs. the original
PRD's week-2 model-decision milestone.

### Option C — Get institutional access to Cao 2025 + email authors

In parallel: (a) get the Cao 2025 PDF via institutional library access
to extract per-model CRISPRoffT numbers, and (b) email the Crispr-SGRU
+ R-CRISPR authors requesting MIT/Apache-2.0 written grants. If both
come back clean, proceed with the original PRD targeting Crispr-SGRU
or R-CRISPR. If either fails, fall back to Option A or B.

### Default path if no decision

Continue the week-0 infrastructure work (verify-score API, bench CLI,
CRISPRoffT fetcher, CRISPR-Net wired as baseline) — those are
independent of model selection and useful regardless. Hold the
week-2 model decision pending revised PRD.
