"""Make the cube modules importable from tests."""
import sys
from pathlib import Path

CUBE_DIR = Path(__file__).resolve().parent.parent / "source" / "cube"
if str(CUBE_DIR) not in sys.path:
    sys.path.insert(0, str(CUBE_DIR))
