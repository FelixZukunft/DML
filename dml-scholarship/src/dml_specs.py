"""Import shim — exposes 04_dml_specs.py under an importable name.

Lets downstream phases do ``from dml_specs import get_all_models`` even though
the implementation lives in the numeric-prefixed ``04_dml_specs.py``. Importing
this shim runs only the module-level code of the target (no __main__ block),
so it has no side effects.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_src = Path(__file__).with_name("04_dml_specs.py")
_spec = importlib.util.spec_from_file_location("_dml_specs_impl", _src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_all_models = _mod.get_all_models

__all__ = ["get_all_models"]
