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

"""Track B v0 — PRIDICT 2.0 scoring wrapper (Mathis 2024).

Wraps the upstream PRIDICT 2.0 (`Mathis et al. 2024, Nature
Biotechnology <https://doi.org/10.1038/s41587-024-02153-y>`_) editing
efficiency predictor for use as a CPU-only scorer inside the
``bionpu crispr pe design`` pipeline.

Architecture
------------

PRIDICT 2.0 is **batch-native**: a single
:class:`pridict.pridictv2.predict_outcomedistrib.PRIEML_Model` runs
inference over a DataFrame of pegRNA features (built upstream by
``primesequenceparsing`` + ``pegRNAfinder``). Per Track B's T1 probe
(:doc:`state/track-b-prereq-probe.md` §4), the natural batch unit is
"all pegRNAs enumerated for a single target locus" — a target context
in PRIDICT2's ``XXX(orig/edit)YYY`` format with ≥100 bp of flanking
sequence.

This wrapper:

* Lazy-imports the upstream package (soft-gates with
  :class:`PRIDICTNotInstalledError` when the import fails).
* Lazy-loads the trained model weights ONCE per
  :class:`PRIDICT2Scorer` instance via PRIEML_Model's
  ``build_retrieve_models`` API — held in memory until
  ``__exit__`` / ``close()`` releases them (T13 ``--low-memory``
  hook).
* Caches PRIDICT's per-target enumeration DataFrame in-process,
  keyed by ``(target_context, model_variant)`` — repeated scoring
  calls on the same target reuse the enumeration.
* Caches per-pegRNA scores keyed by
  ``(pegrna_canonical, scaffold_variant, model_variant, target_context)``.

The underlying batch entry is exposed as ``score_batch`` so the T8
ranker can score N pegRNAs in one DataLoader cycle when they share
a target context. The plan §T5 batch-vs-single throughput note: T1
measured ~1 ms/pegRNA batched (765 pegRNAs in 0.9 s wall on CPU
after model load), vs ~50-200 ms when each pegRNA forces a fresh
target enumeration.

Soft-gating policy
------------------

Per ``CLAUDE.md`` and ``restrictive-license-model-policy.md``, this
wrapper treats PRIDICT 2.0 as a **runtime dep** only — the upstream
git checkout lives at ``third_party/PRIDICT2/`` and must be on
``PYTHONPATH``. We do not vendor the trained weights into
``bionpu-public/``.

Cell-type variants
------------------

The PRIDICT 2.0 trained models ship two cell-type heads: ``HEK`` and
``K562`` (per ``pridict2_pegRNA_design.py:get_cell_types``). The plan
§T5 references HCT116 and U2OS as additional variants — these are
NOT supported by the upstream weights and will raise
:class:`ValueError` at scorer construction time.

Synonyms accepted at construction:

* ``"HEK"``, ``"HEK293"``, ``"HEK293T"`` -> column
  ``PRIDICT2_0_editing_Score_deep_HEK``
* ``"K562"`` -> column ``PRIDICT2_0_editing_Score_deep_K562``
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from typing import TYPE_CHECKING

from bionpu.genomics.pe_design.types import PRIDICTScore

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "PRIDICT2Scorer",
    "PRIDICTNotInstalledError",
]


# ----------------------------------------------------------------------
# Errors + capability gate
# ----------------------------------------------------------------------


class PRIDICTNotInstalledError(ImportError):
    """Raised when the upstream PRIDICT 2.0 package cannot be imported.

    Per Track B T5 spec: the wrapper soft-gates rather than crashing
    the whole CLI. Callers (T8 ranker / T10 CLI) catch this and
    surface a help-text install hint to the user.
    """


# Module-level helper kept as a function (not inlined) so tests can
# monkeypatch the loader to simulate a missing upstream package.
def _load_prieml_model_class():
    """Lazy-import :class:`PRIEML_Model` from the upstream PRIDICT 2.0
    checkout. Wrapped so tests can monkeypatch the resolution path
    without poking at ``sys.modules`` plumbing.

    Returns
    -------
    tuple[type, types.ModuleType]
        ``(PRIEML_Model_class, design_script_module)`` — the design
        script module exposes the heavyweight ``pegRNAfinder`` +
        ``load_pridict_model`` helpers we delegate to for target-level
        enumeration.

    Raises
    ------
    ImportError
        If the ``pridict`` package or ``pridict2_pegRNA_design`` module
        cannot be imported. Callers wrap this in
        :class:`PRIDICTNotInstalledError`.
    """
    from pridict.pridictv2.predict_outcomedistrib import (  # noqa: F401
        PRIEML_Model,
    )

    # Also load the heavyweight design driver (``pegRNAfinder`` etc.).
    # The upstream module uses ``mp.set_start_method("spawn", force=True)``
    # on import which would clobber the parent process's start method;
    # we guard against that by capturing+restoring the value if it has
    # already been set.
    import torch.multiprocessing as mp

    prior_start_method = None
    try:
        prior_start_method = mp.get_start_method(allow_none=True)
    except RuntimeError:
        prior_start_method = None

    # The upstream design script lives at the repo root; users put
    # ``third_party/PRIDICT2/`` on PYTHONPATH per Track B T1's probe.
    pridict2_design = importlib.import_module("pridict2_pegRNA_design")

    # If the upstream import set the spawn method but the parent had
    # something else, we don't try to undo it — just log via a comment.
    # The behavior matches running ``pridict2_pegRNA_design.py`` directly.
    _ = prior_start_method

    return PRIEML_Model, pridict2_design


# ----------------------------------------------------------------------
# Cell-type variant resolution
# ----------------------------------------------------------------------


# Map user-facing variant names to the column suffix PRIDICT 2.0
# emits in its prediction CSV. PRIDICT 2.0 ships HEK + K562 only;
# HCT116 / U2OS from the plan are not supported by upstream weights.
_VARIANT_TO_PRIDICT_CELLTYPE: dict[str, str] = {
    "HEK": "HEK",
    "HEK293": "HEK",
    "HEK293T": "HEK",
    "K562": "K562",
}

_UNSUPPORTED_VARIANTS: frozenset[str] = frozenset({"HCT116", "U2OS"})


def _resolve_cell_type(model_variant: str) -> str:
    """Map a user-facing model variant to PRIDICT 2.0's cell-type
    column suffix, or raise if the variant is not available.
    """
    norm = model_variant.strip()
    if norm in _UNSUPPORTED_VARIANTS:
        raise ValueError(
            f"model_variant={model_variant!r} is listed in the Track B "
            f"plan §T5 but is NOT shipped by upstream PRIDICT 2.0 "
            f"trained weights. Available cell-type variants: "
            f"{sorted(set(_VARIANT_TO_PRIDICT_CELLTYPE))}. To support "
            f"{model_variant} a retrain-from-scratch on cell-type-"
            f"specific library data would be required (filed as a "
            f"Track B v1 follow-on)."
        )
    if norm not in _VARIANT_TO_PRIDICT_CELLTYPE:
        raise ValueError(
            f"Unknown model_variant={model_variant!r}. Available: "
            f"{sorted(set(_VARIANT_TO_PRIDICT_CELLTYPE))} (HEK/HEK293/"
            f"HEK293T are aliases for the same head)."
        )
    return _VARIANT_TO_PRIDICT_CELLTYPE[norm]


# ----------------------------------------------------------------------
# Canonical pegRNA hashing (cache key)
# ----------------------------------------------------------------------


def _canonical_pegrna(pegrna_seq: str) -> str:
    """Return the canonical RNA string for cache-keying.

    PRIDICT 2.0 emits pegRNAs as DNA strings (``ACGT``) in its
    ``pegRNA`` column even though the molecule itself is RNA. The
    enumerator (T6) emits RNA strings (``ACGU``). We canonicalise to
    DNA-letter case so cache hits work across both producers.
    """
    return pegrna_seq.upper().replace("U", "T")


# ----------------------------------------------------------------------
# Scaffold-OOD detection
# ----------------------------------------------------------------------


_CANONICAL_SCAFFOLD_NAME = "sgRNA_canonical"


# ----------------------------------------------------------------------
# PRIDICT2Scorer
# ----------------------------------------------------------------------


class PRIDICT2Scorer:
    """In-process PRIDICT 2.0 scoring wrapper for the pegRNA design
    pipeline.

    Parameters
    ----------
    model_variant:
        Cell-type variant. Accepts ``"HEK"`` / ``"HEK293"`` /
        ``"HEK293T"`` (all alias the HEK head) or ``"K562"``. Default
        ``"HEK293"``. Per upstream constraints, ``"HCT116"`` and
        ``"U2OS"`` raise :class:`ValueError` at construction time.
    use_5folds:
        When ``True``, build the ensemble of 5 trained model folds
        (~5x slower load + inference; matches PRIDICT 2.0's
        ``--use_5folds`` flag). Default ``False`` (fold-1 only).
    work_dir:
        Optional directory where the upstream ``pegRNAfinder``
        scratch CSV is written. Defaults to a tempdir per scorer
        instance, deleted on ``close()``.

    Examples
    --------
    Single-pegRNA scoring (drives the test-suite path)::

        with PRIDICT2Scorer(model_variant="HEK293") as scorer:
            score = scorer.score(
                pegrna_seq=pegrna,
                scaffold_variant="sgRNA_canonical",
                target_context="GCC...A(C/T)GTG...TGT",  # PRIDICT2 fmt
            )
            print(score.efficiency)  # 0-100 scale

    Batched scoring (drives the T8 ranker path)::

        scores = scorer.score_batch(
            pegrna_seqs=[p1, p2, p3],
            scaffold_variants=["sgRNA_canonical"] * 3,
            target_contexts=[ctx, ctx, ctx],  # all share one target
        )

    Notes
    -----
    The plan §T5 spec mentions a ``folding_features`` parameter for
    consistency with T4's pipeline. PRIDICT 2.0 does NOT consume
    externally-computed folding features — its model includes its own
    in-house feature engineering. The wrapper accepts and IGNORES
    ``folding_features`` if supplied (preserving the call signature
    T8 expects).
    """

    def __init__(
        self,
        model_variant: str = "HEK293",
        *,
        use_5folds: bool = False,
        work_dir: str | None = None,
    ) -> None:
        # Validate up-front so callers fail fast on bad config.
        self.model_variant = model_variant
        self.cell_type = _resolve_cell_type(model_variant)
        self.use_5folds = use_5folds
        self._work_dir = work_dir
        self._tmp_handle = None  # tempfile.TemporaryDirectory if we own it

        # Load the upstream module + lazy-instantiate the model.
        try:
            self._prieml_class, self._pridict2_design = _load_prieml_model_class()
        except ImportError as exc:
            raise PRIDICTNotInstalledError(
                f"PRIDICT 2.0 not importable: {exc}. Install via Track B "
                f"T1 prereq — clone "
                f"https://github.com/uzh-dqbm-cmi/PRIDICT2 to "
                f"third_party/PRIDICT2/ and add it to PYTHONPATH "
                f"(see state/track-b-prereq-probe.md §2.2)."
            ) from exc

        # Trained-model weights are loaded lazily on first scoring call
        # so construction is cheap (~ms). ``_models_lst_dict`` is the
        # ``models_lst_dict`` PRIDICT 2.0's ``deeppridict`` consumes.
        self._models_lst_dict = None
        self._closed = False

        # Per-target enumeration cache: target_context -> DataFrame.
        # Each DataFrame has the full pegdataframe + score columns.
        self._enum_cache: dict[str, "pd.DataFrame"] = {}
        self._enumeration_calls = 0  # counter for cache-hit tests

        # Per-pegRNA score cache (after scaffold-OOD note application).
        # Key: (pegrna_canonical, scaffold_variant, model_variant,
        #       target_context).
        self._score_cache: dict[tuple[str, str, str, str], PRIDICTScore] = {}
        self._cache_hits = 0  # counter for cache-hit tests
        # Re-entrant lock so nested score()/score_batch() calls in the
        # same thread don't deadlock; concurrent threads serialise on
        # PRIDICT inference (the model + DataLoader plumbing is not
        # thread-safe).
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Context-manager protocol (T13 --low-memory hook)
    # ------------------------------------------------------------------

    def __enter__(self) -> "PRIDICT2Scorer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the ~95 MB of trained-model weights from memory.

        Idempotent. After ``close()``, calling ``score()`` will
        re-trigger model load (no error). The wrapper also tears down
        the scratch tempdir if it owns one.
        """
        with self._lock:
            self._models_lst_dict = None
            self._enum_cache.clear()
            # Don't drop the score-cache: it's a deterministic
            # (pegrna, scaffold, variant, target) -> score map and
            # re-entry can keep using it.
            if self._tmp_handle is not None:
                try:
                    self._tmp_handle.cleanup()
                except OSError:
                    pass
                self._tmp_handle = None
            self._closed = True

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score(
        self,
        pegrna_seq: str,
        *,
        scaffold_variant: str = _CANONICAL_SCAFFOLD_NAME,
        target_context: str,
        folding_features=None,
    ) -> PRIDICTScore:
        """Score a single pegRNA.

        Parameters
        ----------
        pegrna_seq:
            The complete pegRNA sequence (spacer + scaffold + RTT +
            PBS), 5' -> 3'. Accepts both DNA-letter (``ACGT``) and
            RNA-letter (``ACGU``) forms; cache key is canonicalised
            to DNA letters.
        scaffold_variant:
            Scaffold variant name from
            :data:`bionpu.genomics.pe_design.pegrna_constants.SCAFFOLD_VARIANTS`.
            When ``!= "sgRNA_canonical"``, the result's ``notes``
            tuple includes ``"SCAFFOLD_OUT_OF_DISTRIBUTION"``.
        target_context:
            The target locus in PRIDICT 2.0 input format
            (``XXX(orig/edit)YYY``) with ≥100 bp of flanking sequence.
        folding_features:
            ViennaRNA features from T4. PRIDICT 2.0 does not consume
            external folding features, so this is accepted for API
            consistency with T8 and IGNORED.

        Returns
        -------
        PRIDICTScore
            Score with ``efficiency`` on 0-100 scale, ``edit_rate`` on
            0-1 scale, ``confidence`` on 0-1 scale (currently a
            placeholder of 1.0 since PRIDICT 2.0 does not expose an
            inference-time uncertainty head; filed as v1 follow-on),
            and ``notes`` tuple with any OOD / failure flags.
        """
        del folding_features  # accepted for T8 API consistency; ignored.

        with self._lock:
            return self._score_one(pegrna_seq, scaffold_variant, target_context)

    def score_batch(
        self,
        pegrna_seqs: list[str],
        *,
        scaffold_variants: list[str] | None = None,
        target_contexts: list[str],
        folding_features_list=None,
    ) -> list[PRIDICTScore]:
        """Score a batch of pegRNAs.

        The implementation groups pegRNAs by ``target_context`` and
        runs the upstream enumeration ONCE per unique target — this
        is the throughput lever per T1's batch-API probe.

        Parameters
        ----------
        pegrna_seqs:
            List of pegRNA RNA sequences.
        scaffold_variants:
            Optional list of scaffold-variant names, one per pegRNA.
            Defaults to ``"sgRNA_canonical"`` for all.
        target_contexts:
            List of target contexts (PRIDICT2 format strings), one per
            pegRNA. Pegs with the same target context share one
            upstream enumeration.
        folding_features_list:
            Accepted for API consistency; ignored.

        Returns
        -------
        list[PRIDICTScore]
            One score per input pegRNA, in input order.
        """
        del folding_features_list  # accepted for API consistency; ignored.

        if len(pegrna_seqs) != len(target_contexts):
            raise ValueError(
                f"score_batch: pegrna_seqs ({len(pegrna_seqs)}) and "
                f"target_contexts ({len(target_contexts)}) length mismatch"
            )
        if scaffold_variants is None:
            scaffold_variants = [_CANONICAL_SCAFFOLD_NAME] * len(pegrna_seqs)
        if len(scaffold_variants) != len(pegrna_seqs):
            raise ValueError(
                f"score_batch: scaffold_variants ({len(scaffold_variants)}) "
                f"and pegrna_seqs ({len(pegrna_seqs)}) length mismatch"
            )

        out: list[PRIDICTScore] = []
        with self._lock:
            for pegrna_seq, scaffold_variant, target_context in zip(
                pegrna_seqs, scaffold_variants, target_contexts, strict=True
            ):
                out.append(
                    self._score_one(pegrna_seq, scaffold_variant, target_context)
                )
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_one(
        self,
        pegrna_seq: str,
        scaffold_variant: str,
        target_context: str,
    ) -> PRIDICTScore:
        """Score a single pegRNA, with both target-level and
        per-pegRNA caching.
        """
        canonical = _canonical_pegrna(pegrna_seq)
        cache_key = (canonical, scaffold_variant, self.model_variant, target_context)

        # Score-level cache hit.
        if cache_key in self._score_cache:
            self._cache_hits += 1
            return self._score_cache[cache_key]

        # Target-level enumeration (cached per target_context).
        try:
            df = self._enumerate_for_target(target_context)
        except Exception as exc:  # pragma: no cover - hard error path
            score = PRIDICTScore(
                efficiency=float("nan"),
                edit_rate=float("nan"),
                confidence=float("nan"),
                notes=("PRIDICT_FAILED", f"enumeration_failed:{type(exc).__name__}"),
            )
            self._score_cache[cache_key] = score
            return score

        # Match the enumerated row by canonical pegRNA sequence.
        if "pegRNA" not in df.columns or len(df) == 0:
            score = PRIDICTScore(
                efficiency=float("nan"),
                edit_rate=float("nan"),
                confidence=float("nan"),
                notes=("PRIDICT_FAILED", "empty_enumeration"),
            )
            self._score_cache[cache_key] = score
            return score

        df_canonical = df["pegRNA"].astype(str).apply(_canonical_pegrna)
        match = df.loc[df_canonical == canonical]
        if len(match) == 0:
            score = PRIDICTScore(
                efficiency=float("nan"),
                edit_rate=float("nan"),
                confidence=float("nan"),
                notes=(
                    "PRIDICT_FAILED",
                    "PEGRNA_NOT_ENUMERATED_BY_PRIDICT",
                ),
            )
            self._score_cache[cache_key] = score
            return score

        # Take the first match (PRIDICT may emit duplicates for
        # equivalent pegRNAs; the first row is the highest-ranked).
        row = match.iloc[0]
        eff_col = f"PRIDICT2_0_editing_Score_deep_{self.cell_type}"
        eff_pct = float(row[eff_col])
        # PRIDICT 2.0 reports ``pred_averageedited * 100`` as the
        # efficiency column. Edit-rate (0-1) is the unscaled head;
        # we recover it as efficiency / 100. Confidence is a
        # placeholder until we plumb through ensemble variance.
        edit_rate = eff_pct / 100.0

        notes: tuple[str, ...] = ()
        if scaffold_variant != _CANONICAL_SCAFFOLD_NAME:
            notes = ("SCAFFOLD_OUT_OF_DISTRIBUTION",)

        score = PRIDICTScore(
            efficiency=eff_pct,
            edit_rate=edit_rate,
            # 1.0 placeholder. v1: derive from
            # ``PRIDICT2_0_editing_Score_deep_*`` ensemble variance
            # when ``use_5folds=True``.
            confidence=1.0,
            notes=notes,
        )
        self._score_cache[cache_key] = score
        return score

    def _ensure_models_loaded(self) -> None:
        """Lazy-load the trained-model weights on first inference.

        Idempotent. Cleared by :meth:`close`.
        """
        if self._models_lst_dict is None:
            run_ids = list(range(5)) if self.use_5folds else [0]
            self._models_lst_dict = self._pridict2_design.load_pridict_model(
                run_ids=run_ids
            )

    def _ensure_work_dir(self) -> str:
        """Return a writable scratch dir for upstream CSV output."""
        if self._work_dir is not None:
            os.makedirs(self._work_dir, exist_ok=True)
            return self._work_dir
        if self._tmp_handle is None:
            import tempfile

            self._tmp_handle = tempfile.TemporaryDirectory(
                prefix="bionpu-pridict2-"
            )
        return self._tmp_handle.name

    def _enumerate_for_target(self, target_context: str) -> "pd.DataFrame":
        """Run PRIDICT 2.0's full enumeration for one target context.

        Returns the upstream ``pegdataframe`` (with score columns) as
        a pandas DataFrame. Cached per ``target_context``.

        Implementation: reuses the upstream ``pegRNAfinder`` work
        function in-process (no subprocess fork). We give it a stub
        queue object (only ``put`` is called) and read the resulting
        CSV from the work_dir.
        """
        if target_context in self._enum_cache:
            return self._enum_cache[target_context]

        self._ensure_models_loaded()
        work_dir = self._ensure_work_dir()
        seq_name = f"bionpu_pridict_seq_{self._enumeration_calls}"

        # Stub queue: pegRNAfinder's only contract is queue.put((idx,
        # status_or_exception)). We capture into a list so we can
        # surface upstream errors as PRIDICTNotInstalledError /
        # ValueError.
        captured: list[tuple[int, object]] = []

        class _StubQueue:
            def put(self, item):  # noqa: D401 - inline stub
                captured.append(item)

        # pegRNAfinder accepts dfrow as either a pandas Series or a
        # row-like with named fields. A plain dict satisfies the
        # ``dfrow['editseq']`` / ``dfrow['sequence_name']`` access
        # pattern.
        dfrow = {"editseq": target_context, "sequence_name": seq_name}

        # Suppress the upstream's spammy progress prints by
        # temporarily redirecting stdout via the env var the upstream
        # honours for TF (TF is not loaded here but we still get
        # tqdm output that is fine to keep).
        self._pridict2_design.pegRNAfinder(
            dfrow=dfrow,
            models_list=self._models_lst_dict,
            queue=_StubQueue(),
            pindx=self._enumeration_calls,
            pred_dir=work_dir,
            nicking=False,
            ngsprimer=False,
        )
        self._enumeration_calls += 1

        # Surface upstream errors. captured[0] is (pindx,
        # 'Prediction successful!') OR (pindx, exception_instance).
        if not captured:
            raise RuntimeError(
                f"PRIDICT 2.0 enumeration produced no status for "
                f"target_context={target_context!r}"
            )
        _, status = captured[0]
        if isinstance(status, BaseException):
            raise RuntimeError(
                f"PRIDICT 2.0 enumeration raised: {type(status).__name__}: "
                f"{status}"
            ) from status

        csv_path = os.path.join(work_dir, f"{seq_name}_pegRNA_Pridict_full.csv")
        if not os.path.exists(csv_path):
            raise RuntimeError(
                f"PRIDICT 2.0 enumeration completed but CSV not found at "
                f"{csv_path}; status={status!r}"
            )

        import pandas as pd

        df = pd.read_csv(csv_path)
        self._enum_cache[target_context] = df
        return df


# ----------------------------------------------------------------------
# Optional: ensure the upstream PRIDICT2 checkout is on PYTHONPATH.
# ----------------------------------------------------------------------
# When users follow Track B's T1 setup
# (``state/track-b-prereq-probe.md`` §2.2), they export
# ``PYTHONPATH=$REPO/third_party/PRIDICT2:$REPO/bionpu-public/src``
# manually. Some CI / test paths skip the export — we add a
# best-effort fallback that injects the standard path if the env var
# is missing AND the directory exists. This is opt-out via the
# ``BIONPU_PRIDICT2_AUTO_PATH=0`` env var (set by callers who want a
# clean failure mode).
def _maybe_inject_pridict2_path() -> None:
    if os.environ.get("BIONPU_PRIDICT2_AUTO_PATH", "1") == "0":
        return
    if importlib.util.find_spec("pridict2_pegRNA_design") is not None:
        return  # already importable; nothing to do
    # Walk up the bionpu-public source tree to find the repo root.
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        candidate = os.path.join(here, "third_party", "PRIDICT2")
        if os.path.isdir(candidate) and os.path.isfile(
            os.path.join(candidate, "pridict2_pegRNA_design.py")
        ):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent


_maybe_inject_pridict2_path()
