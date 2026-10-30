"""ATP-facing import of the corrected MPTP model implementation.

The ATP dataset uses the same FOF graph representation as MPTP, but its
``node_attr_dict.pkl`` has a different insertion order. Callers therefore
pass ``operator_type_indices`` from that dictionary. Keeping a single model
implementation prevents classification and ATP ranking from drifting.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_MODEL_PATH = Path(__file__).resolve().parents[1] / "mptp" / "model.py"
_SPEC = spec_from_file_location("twgnn_corrected_mptp_model", _MODEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load corrected model from {_MODEL_PATH}")
_MODULE = module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, _MODULE)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MODULE, _name)

__all__ = [name for name in dir(_MODULE) if not name.startswith("_")]
