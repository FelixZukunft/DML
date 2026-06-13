"""Import shim — exposes 03_ml_methods.py under an importable name.

Numeric-prefixed module files (``03_ml_methods.py``) cannot be imported with a
normal ``import`` statement, so downstream phases do ``from ml_methods import
get_learners``. This shim loads the numbered file and re-exports its public API.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_src = Path(__file__).with_name("03_ml_methods.py")
_spec = importlib.util.spec_from_file_location("_ml_methods_impl", _src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_learners = _mod.get_learners
get_naive_ols = _mod.get_naive_ols

__all__ = ["get_learners", "get_naive_ols"]
