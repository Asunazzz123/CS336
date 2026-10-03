"""Resolve test imports against this Assignment 1 checkout."""

import sys
from pathlib import Path

# Both assignments provide cs336_basics; prefer this checkout for test imports.
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root in sys.path:
    sys.path.remove(_project_root)
sys.path.insert(0, _project_root)
