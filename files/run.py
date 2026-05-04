import sys
from pathlib import Path

# Add the parent directory to sys.path so 'app.xxx' imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import main

if __name__ == "__main__":
    main()
