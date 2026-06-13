# DML Scholarship Project — Claude Brain

## Project Goal
Estimate the Average Treatment Effect (ATE) of scholarship receipt (D)
on 2nd-semester grade (Y) using Double Machine Learning (Partially Linear Regression),
controlling for high-dimensional baseline confounders (X).

---

## Variable Mapping
| Role | Column | Type |
|------|--------|------|
| Y (Outcome) | `Curricular units 2nd sem (grade)` | continuous |
| D (Treatment) | `Scholarship holder` | binary 0/1 |
| X (Confounders) | see approved list below | mixed |

### Confounders X — Approved List
Mother's qualification, Father's qualification, Mother's occupation,
Father's occupation, Admission grade, Age at enrollment, Displaced,
Debtor, Tuition fees up to date, Gender, International,
Marital Status, Application mode, Application order, Course,
Daytime/evening attendance, Previous qualification (grade),
Nacionality, GDP, Inflation rate, Unemployment rate

### Strictly Excluded (Post-Treatment / Bad Controls)
Never include in X — these are outcomes of D, not pre-treatment variables:
- Curricular units 1st sem (credited / enrolled / evaluations / approved / grade / without evaluations)
- Target (the dropout label — not our outcome variable)

---

## Critical Data Rule — Keep Y = 0 Rows
**DO NOT filter out students with grade = 0.**
Zero-grade students represent a real outcome (withdrawal / failure) that may itself be
causally affected by the scholarship. Removing them would condition on a post-treatment
outcome and introduce collider bias. The dataset has 870 Y=0 rows (19.7%) — all are kept.

---

## Data
| Path | Description |
|------|-------------|
| `data/raw/students_raw.csv` | Original UCI download — never modified |
| `data/raw/categorical_recoding_scheme_1.xlsx` | Semantic recoding scheme (Sheet 1: table, Sheet 2: Python maps) |
| `data/processed/dml_ready_raw.csv` | Raw encoding output: 4,424 × 233 (231 X features) |
| `data/processed/dml_ready_semantic.csv` | Semantic encoding output: 4,424 × 61 (59 X features) |

### Column Name Whitespace — Always Strip
`students_raw.csv` and `dml_ready_semantic.csv` have leading/trailing spaces in column
names for all columns except the first. Always apply after `pd.read_csv`:
```python
df.columns = df.columns.str.strip()
```

---

## Pipeline Overview
Each numbered source file maps to exactly one phase. `main.py` orchestrates them in order.

| File | Phase | Status | Responsibility |
|------|-------|--------|---------------|
| `src/00_load_data.py` | Load | Done | Download UCI dataset, run data audit |
| `src/01_data_pipeline.py` | Data | **Done** | Clean, encode, scale → `dml_ready_*.csv` |
| `src/02_causal_model.py` | Causal | Scaffold | Define DAG, select D / Y / X |
| `src/03_ml_methods.py` | Learners | Scaffold | Configure base learners (RF, XGBoost, Lasso) |
| `src/04_dml_specs.py` | Specs | Scaffold | Assemble DoubleML model objects (PLR, IRM, etc.) |
| `src/05_hyperparameter.py` | Tuning | Scaffold | Hyperparameter search for nuisance learners |
| `src/06_estimation.py` | Estimation | Scaffold | Fit DML models, save to `outputs/models/` |
| `src/07_inference.py` | Inference | Scaffold | CIs, p-values, coefficient tables → `outputs/tables/` |
| `src/08_sensitivity.py` | Sensitivity | Scaffold | Robustness: Riesz bounds, alternative specs |

### Encoding Modes in `01_data_pipeline.py`
Two modes, selectable at runtime:
```bash
python src/01_data_pipeline.py          # runs both
python src/01_data_pipeline.py raw      # pd.get_dummies(drop_first=True) → 231 X features
python src/01_data_pipeline.py semantic # named group dummies from Excel → 59 X features
```
- **raw**: mechanical one-hot, drop_first to avoid dummy trap — use for DML if interpretability not needed
- **semantic**: human-readable group names from recoding scheme — preferred for EDA and partial plots
- Continuous X are z-scored (StandardScaler saved to `outputs/models/scaler.pkl`)
- CONTINUOUS_X = `Previous qualification (grade)`, `Admission grade`, `Age at enrollment`, `GDP`, `Inflation rate`, `Unemployment rate`

---

## Notebooks
| File | Contents |
|------|----------|
| `notebooks/data_viz.ipynb` | EDA: outcome distribution, LOWESS (z-scored + raw), violin/box plots for categoricals, partial regression plots, naive ATE estimate |

`data_viz.ipynb` uses `dml_ready_semantic.csv` for the confounder visualisations.
The partial regression cell uses Ridge (α=1) to residualise both Y and each variable on
all other X — the slope of the red line is the linear coefficient controlling for all
other confounders. This serves as a linear preview of what DML will estimate.

---

## Outputs
| Path | Contents |
|------|----------|
| `outputs/01_data_audit.md` | Shape, dtypes, missingness, treatment/outcome distributions |
| `outputs/figures/` | DAG PNGs, residual plots, coefficient plots |
| `outputs/tables/` | LaTeX `.tex` and `.csv` result tables |
| `outputs/models/scaler.pkl` | Fitted StandardScaler from phase 1 |

---

## Key Library: DoubleML
- Docs: https://docs.doubleml.org/stable/index.html
- Main class: `DoubleMLPLR` (Partially Linear Regression) for ATE
- Cross-fitting: `n_folds=5`, `n_rep=10`, `random_state=42` everywhere

## Identification Assumption
Selection on observables / Conditional Exogeneity:
  `E[U | D, X] = 0` — once we condition on X, D is as good as random

---

## Coding Standards
- All `src/` scripts are importable modules with a `run()` function; `main.py` calls them in order
- All scripts must also be runnable standalone: `python src/01_data_pipeline.py`
- Save all figures to `outputs/figures/`
- Print ATE with 95% CI at end of estimation phase
- `random_state=42` everywhere for reproducibility
- No interactive plots in scripts (matplotlib only, saved to file)

## Running
```bash
python main.py                        # full pipeline
python src/01_data_pipeline.py        # individual phase
pytest tests/                         # sanity checks (shape, no-NaN, sign)
```

## Dependencies
```bash
pip install -r requirements.txt
```
Key packages: `doubleml`, `scikit-learn`, `xgboost`, `pandas`, `seaborn`, `ucimlrepo`
