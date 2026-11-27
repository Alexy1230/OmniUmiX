"""Continuous tactile/server check; uses the camera-only test CLI, never moves robots."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test_xense_camera import main

if __name__ == '__main__':
    main()
