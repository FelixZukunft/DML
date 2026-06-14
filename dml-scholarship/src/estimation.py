"""Import shim — exposes 06_estimation.py under an importable name.

Lets downstream phases do ``from estimation import load_results``
even though the implementation lives in the numeric-prefixed file.
Importing this shim has no side effects (no __main__ block runs).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_src  = Path(__file__).with_name("06_estimation.py")
_spec = importlib.util.spec_from_file_location("_estimation_impl", _src)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_results = _mod.load_results

__all__ = ["load_results"]
