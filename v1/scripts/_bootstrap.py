"""Put `src/` on the import path so entry scripts run from a fresh clone.

Only needed before `make install` has done an editable install. Importing this module
is a no-op once the package is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
