"""Entry point for running frame_detection as a module or directly."""

import sys
from pathlib import Path

# Add parent directory to path when run directly (not as module)
if __package__ is None or __package__ == "":
    parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent))
    from frame_detection.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    main()
