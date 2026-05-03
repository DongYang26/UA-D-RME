from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_root() -> None:
    """Ensure local packages resolve when scripts are executed by file path."""
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
