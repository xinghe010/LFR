"""Extract data.zip into the repository root."""
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
archive = root / "data.zip"
if not archive.is_file():
    raise FileNotFoundError(f"Missing {archive}")
with zipfile.ZipFile(archive) as zf:
    zf.extractall(root)
print(f"Extracted {archive} to {root}")
