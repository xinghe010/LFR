"""ATP-facing import of the corrected MPTP graph dataset.

ATP D0 contains FOF formulas and the same pair schema. Reusing the corrected
dataset gives ATP training the same cache invalidation, integer node IDs,
formula caching and rule-feature validation as classification experiments.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_DATASET_PATH = Path(__file__).resolve().parents[1] / "mptp" / "dataset.py"
_SPEC = spec_from_file_location("twgnn_corrected_mptp_dataset", _DATASET_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load corrected dataset from {_DATASET_PATH}")
_MODULE = module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, _MODULE)
_SPEC.loader.exec_module(_MODULE)




_PARSER_PATH = Path(__file__).with_name("formula_parser.py")
_PARSER_SPEC = spec_from_file_location("twgnn_atp_formula_parser", _PARSER_PATH)
if _PARSER_SPEC is None or _PARSER_SPEC.loader is None:
    raise ImportError(f"Cannot load ATP formula parser from {_PARSER_PATH}")
_PARSER_MODULE = module_from_spec(_PARSER_SPEC)
_PARSER_SPEC.loader.exec_module(_PARSER_MODULE)
_MODULE.fof_formula_transformer = _PARSER_MODULE.fof_formula_transformer

for _name in dir(_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MODULE, _name)

__all__ = [name for name in dir(_MODULE) if not name.startswith("_")]
