"""Repository-local entrypoint for AutoSLM v3 SFT."""
from pathlib import Path
import sys

# Make direct execution work without requiring an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoslm_v3.sft import main

if __name__ == "__main__":
    main()
