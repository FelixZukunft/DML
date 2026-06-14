"""Import shim — exposes 05_hyperparameter.py under an importable name.

Lets downstream phases do ``from hyperparameter import get_tuned_models``
even though the implementation lives in the numeric-prefixed file.
Importing this shim has no side effects (no __main__ block runs).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_src  = Path(__file__).with_name("05_hyperparameter.py")
_spec = importlib.util.spec_from_file_location("_hyperparameter_impl", _src)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_tuned_models = _mod.get_tuned_models

__all__ = ["get_tuned_models"]
