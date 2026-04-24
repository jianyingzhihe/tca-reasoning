import sys
from pathlib import Path
from typing import TYPE_CHECKING


def _prefer_vendored_transformer_lens() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "third_party" / "TransformerLens"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_prefer_vendored_transformer_lens()

if TYPE_CHECKING:
    from circuit_tracer.attribution.attribute import attribute
    from circuit_tracer.graph import Graph
    from circuit_tracer.replacement_model import ReplacementModel

__all__ = ["ReplacementModel", "Graph", "attribute"]


def __getattr__(name):
    _lazy_imports = {
        "attribute": ("circuit_tracer.attribution.attribute", "attribute"),
        "Graph": ("circuit_tracer.graph", "Graph"),
        "ReplacementModel": ("circuit_tracer.replacement_model", "ReplacementModel"),
    }

    if name in _lazy_imports:
        module_name, attr_name = _lazy_imports[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
