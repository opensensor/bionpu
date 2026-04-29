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

"""Track F v0 — cross-tool validation harness.

The :func:`run_validation` entry point dispatches a single
``(bionpu_cli, reference_tool, fixture)`` triple to its concrete
adapter under :mod:`bionpu.validation.ref_adapters`, runs both sides,
and emits an :class:`AgreementCheck` with a PASS / FAIL / SKIP /
DIVERGE / ERROR verdict.

The :func:`run_full_matrix` helper iterates over the v0 cross-check
plan (`state/track-f-validation-plan.md` §4) and emits a list of
``AgreementCheck``s plus a JSON-serialisable matrix.

This module is *evidence-producer, not gate* — divergences surfaced
here are research surface for v1, not failures of v0.
"""

from __future__ import annotations

from .agreement import (
    AgreementCheck,
    Verdict,
    matrix_to_json,
    run_full_matrix,
    run_validation,
)

__all__ = [
    "AgreementCheck",
    "Verdict",
    "matrix_to_json",
    "run_full_matrix",
    "run_validation",
]
