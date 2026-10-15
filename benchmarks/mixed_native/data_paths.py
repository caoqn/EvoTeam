from __future__ import annotations

import os
import sys
from pathlib import Path


def mixed_data_root(project_root: str | Path) -> Path:
    configured = os.environ.get("MIXED_DATA_ROOT", "").strip()
    root = Path(configured).expanduser() if configured else Path(project_root)
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root
