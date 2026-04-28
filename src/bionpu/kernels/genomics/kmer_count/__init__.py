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

"""Stub for the k-mer counting kernel package.

Per ``state/kmer_count_interface_contract.md`` (T1), this module will
host the :class:`BionpuKmerCount` :class:`NpuOp` subclass and the three
``register_npu_op`` calls (one per supported k in ``{15, 21, 31}``). T9
replaces this stub with the real implementation. T4 only ships an empty
package init so :func:`bionpu.kernels.genomics.get_kmer_count_op` can
import it without ImportError.
"""
