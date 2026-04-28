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

"""High-level genomics primitives composed from kernels under
:mod:`bionpu.kernels.genomics`.

Module layout (one subpackage per CRISPR-shape primitive):

* :mod:`bionpu.genomics.seed_extend` — minimap2-style seed extraction
  built on top of the v0 minimizer NPU op + a host-side reference
  index. Third silicon-validated CRISPR-shape primitive (after
  ``kmer_count`` and ``minimizer``).
"""
