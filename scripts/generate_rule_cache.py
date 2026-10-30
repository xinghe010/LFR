
"""Dataset-neutral entry point for corrected LFR rule-cache generation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'atp'))
from gen_lfr_similarity import main

if __name__ == '__main__':
    main()
