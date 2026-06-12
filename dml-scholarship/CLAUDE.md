# DML Scholarship Project — Claude Brain

## Project Purpose
Apply Double Machine Learning (DML / Partially Linear Regression) to estimate a causal treatment effect using the UCI dataset. Produce a clean, reproducible pipeline from raw data through inference and robustness checks.

## Pipeline Overview
Each numbered source file maps to exactly one phase. `main.py` orchestrates them in order.

| File | Phase | Responsibility |
|------|-------|---------------|
| `src/01_data_pipeline.py` | Data | Load raw UCI data, clean, feature-engineer, save to `data/processed/` |
| `src/02_causal_model.py` | Causal | Define DAG, select treatment D, outcome Y, controls X |
| `src/03_ml_methods.py` | Learners | Configure base learners (RF, XGBoost, Lasso) used in DML nuisance models |
| `src/04_dml_specs.py` | Specs | Assemble DoubleML model objects (PLR, IRM, etc.) |
| `src/05_hyperparameter.py` | Tuning | Hyperparameter search logic for nuisance learners |
| `src/06_estimation.py` | Estimation | Fit DML models, save fitted objects to `outputs/models/` |
| `src/07_inference.py` | Inference | Extract CIs, p-values, coefficient tables; save to `outputs/tables/` |
| `src/08_sensitivity.py` | Sensitivity | Robustness checks: Riesz representer bounds, alternative specs |

## Data
- `data/raw/` — original UCI download, **never modified**
- `data/processed/` — output of `01_data_pipeline.py`; versioned by filename if re-run

## Outputs
- `outputs/figures/` — DAG PNGs, residual plots, coefficient plots
- `outputs/tables/` — LaTeX `.tex` and `.csv` result tables
- `outputs/models/` — pickled fitted DoubleML model objects

## Key Design Decisions
- Use `doubleml` (DoubleML Python package) as the DML framework
- Cross-fitting folds: 5 (default, adjustable in `04_dml_specs.py`)
- Random seed: 42 everywhere for reproducibility
- All scripts are importable modules; `main.py` calls their `run()` functions
- Figures use matplotlib with a consistent style; no interactive plots in scripts

## Running
```bash
# Full pipeline
python main.py

# Single phase (for iteration)
python -m src.01_data_pipeline
```

## Testing
```bash
pytest tests/
```
`tests/test_pipeline.py` contains sanity checks: shape assertions, no-NaN checks, coefficient sign checks.

## Dependencies
See `requirements.txt`. Install with `pip install -r requirements.txt`.

# CLAUDE.md — DML Scholarship Project

## Project Goal
Estimate the Average Treatment Effect (ATE) of scholarship receipt (D)
on 2nd-semester grade (Y) using Double Machine Learning, controlling
for high-dimensional confounders (X).

## Variable Mapping
- Y (Outcome): "Curricular units 2nd sem (grade)" — continuous
- D (Treatment): "Scholarship holder" — binary (0/1)
- X (Confounders): All baseline variables listed below
- EXCLUDED (post-treatment): Any "1st sem" variables — NEVER include in X

## Confounders X — Approved List
Mother's qualification, Father's qualification, Mother's occupation,
Father's occupation, Admission grade, Age at enrollment, Displaced,
Debtor, Tuition fees up to date, Gender, International,
Marital status, Application mode, Application order, Course,
Daytime/evening attendance, Previous qualification (grade),
Nacionality, GDP, Inflation rate, Unemployment rate

## Strictly Excluded Variables (Post-Treatment / Bad Controls)
- Curricular units 1st sem (credited)
- Curricular units 1st sem (enrolled)
- Curricular units 1st sem (evaluations)
- Curricular units 1st sem (approved)
- Curricular units 1st sem (grade)
- Curricular units 1st sem (without evaluations)
- Target (the dropout label — not our outcome)

## Key Library: DoubleML
- Docs: https://docs.doubleml.org/stable/index.html
- GitHub: https://github.com/DoubleML/doubleml-for-py
- Main classes: DoubleMLPLR (Partial Linear Regression) for ATE
- Always use cross-fitting (n_folds=5)
- Always use repeated cross-fitting (n_rep=10)

## Identification Assumption
Selection on observables / Conditional Exogeneity:
  E[U | D, X] = 0  — once we condition on X, D is as good as random

## Running the Project
  python main.py          # full pipeline
  python src/01_data_pipeline.py   # individual phases

## Coding Standards
- All scripts must be runnable standalone
- Save all figures to outputs/figures/
- Print ATE with 95% CI at end of estimation phase
- Use random_state=42 everywhere for reproducibility