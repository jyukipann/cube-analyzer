"""Make source modules importable from tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "source"
CUBE_DIR = SOURCE_DIR / "cube"

for d in [str(SOURCE_DIR), str(CUBE_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)
