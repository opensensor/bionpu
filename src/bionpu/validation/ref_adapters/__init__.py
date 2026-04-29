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

"""Reference-tool adapters for Track F v0.

Each adapter exports an ``installed()`` predicate + a ``run_*``
invocation helper. The agreement-matrix engine
(:mod:`bionpu.validation.agreement`) uses ``installed()`` to decide
SKIP vs run, and ``run_*`` to actually invoke the external tool.
"""

from __future__ import annotations
