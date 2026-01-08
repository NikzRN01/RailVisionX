"""Test script to launch the realtime dashboard page."""
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import and run the realtime page
from src.dashboard.pages.realtime import main

if __name__ == "__main__":
    main()
