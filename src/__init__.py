"""
Package init — re-exports all src modules with Python-safe names.
Files starting with digits can't be imported directly via 'import 01_data_loader',
so this init file imports them using importlib and exposes them cleanly.
"""
import importlib.util, os, sys

def _load(filename, alias):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

_01_data_loader         = _load("01_data_loader.py",          "src._01_data_loader")
_02_data_validation     = _load("02_data_validation.py",      "src._02_data_validation")
_03_eda                 = _load("03_eda.py",                   "src._03_eda")
_04_regression          = _load("04_regression.py",           "src._04_regression")
_05_growth_period_analysis = _load("05_growth_period_analysis.py", "src._05_growth_period_analysis")
_06_ml_models           = _load("06_ml_models.py",            "src._06_ml_models")
