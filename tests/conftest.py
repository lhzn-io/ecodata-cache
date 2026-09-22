"""Make the src-layout packages importable for the test session.

The project has no [build-system], so `uv sync` never installs `ecodata_cache`
(src/) or `ecodata_serve` (service/) into the venv. Tests import both directly.
Putting the two directories on sys.path here means `uv run python -m pytest`
works from a fresh shell without a PYTHONPATH export.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("src", "service"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
