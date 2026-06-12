"""Orchestrates all 8 DML pipeline phases in order."""
import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"


def load_phase(filename: str):
    """Load a numbered src file as a module by file path."""
    path = SRC / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TUNE = "--tune" in sys.argv
SPECS = ["plr_rf", "plr_xgb", "plr_lasso"]


def main():
    p1 = load_phase("01_data_pipeline.py")
    p2 = load_phase("02_causal_model.py")
    p4 = load_phase("04_dml_specs.py")
    p5 = load_phase("05_hyperparameter.py")
    p6 = load_phase("06_estimation.py")
    p7 = load_phase("07_inference.py")
    p8 = load_phase("08_sensitivity.py")

    df = p1.run()
    p2.run(df)

    data_backend = p4.make_data_backend(df)
    models = {name: p4.SPECS[name](data_backend) for name in SPECS}

    if TUNE:
        models = p5.run(models)

    fitted = p6.run(models)
    results = p7.run(fitted)
    p8.run(fitted, data_backend)

    print("\nPipeline complete.")
    return results


if __name__ == "__main__":
    main()
