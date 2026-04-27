"""pytest root conftest — wire `import bionpu` to the public bionpu-public/ tree.

The internal ``genetics/bionpu/`` package has been promoted to a separate
public repository at https://github.com/opensensor/bionpu (cloned at
``./bionpu-public/``). This conftest puts ``bionpu-public/src`` on
sys.path so ``import bionpu`` resolves to the public package during
tests, while ``import bionpu_internal`` continues to resolve to this
repo's private extensions (currently just ``bionpu_internal.report``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BIONPU_PUBLIC_SRC = _HERE / "bionpu-public" / "src"

if _BIONPU_PUBLIC_SRC.is_dir() and str(_BIONPU_PUBLIC_SRC) not in sys.path:
    # Insert FIRST so it wins over any stale install on the active env.
    sys.path.insert(0, str(_BIONPU_PUBLIC_SRC))
