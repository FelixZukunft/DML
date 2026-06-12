# Data Audit — UCI Student Dataset

**Script:** `src/01_load_data.py`  
**Source:** UCI ML Repository, dataset ID 697  
**Saved to:** `data/raw/students_raw.csv`

---

## Shape

4,424 rows × 37 columns

---

## Missing Values

None. All 37 columns are complete.

---

## Columns (37)

| Column | Dtype |
|--------|-------|
| Marital Status | int64 |
| Application mode | int64 |
| Application order | int64 |
| Course | int64 |
| Daytime/evening attendance | int64 |
| Previous qualification | int64 |
| Previous qualification (grade) | float64 |
| Nacionality | int64 |
| Mother's qualification | int64 |
| Father's qualification | int64 |
| Mother's occupation | int64 |
| Father's occupation | int64 |
| Admission grade | float64 |
| Displaced | int64 |
| Educational special needs | int64 |
| Debtor | int64 |
| Tuition fees up to date | int64 |
| Gender | int64 |
| Scholarship holder | int64 |
| Age at enrollment | int64 |
| International | int64 |
| Curricular units 1st sem (credited) | int64 |
| Curricular units 1st sem (enrolled) | int64 |
| Curricular units 1st sem (evaluations) | int64 |
| Curricular units 1st sem (approved) | int64 |
| Curricular units 1st sem (grade) | float64 |
| Curricular units 1st sem (without evaluations) | int64 |
| Curricular units 2nd sem (credited) | int64 |
| Curricular units 2nd sem (enrolled) | int64 |
| Curricular units 2nd sem (evaluations) | int64 |
| Curricular units 2nd sem (approved) | int64 |
| Curricular units 2nd sem (grade) | float64 |
| Curricular units 2nd sem (without evaluations) | int64 |
| Unemployment rate | float64 |
| Inflation rate | float64 |
| GDP | float64 |
| Target | str |

---

## Treatment — Scholarship holder (D)

| Value | Count | Share |
|-------|-------|-------|
| 0 (no scholarship) | 3,325 | 75.2% |
| 1 (scholarship) | 1,099 | 24.8% |

Treatment is imbalanced ~3:1. No positivity issues expected at this sample size.

---

## Outcome — Curricular units 2nd sem (grade) (Y)

| Stat | Value |
|------|-------|
| count | 4,424 |
| mean | 10.23 |
| std | 5.21 |
| min | 0.00 |
| 25% | 10.75 |
| 50% | 12.20 |
| 75% | 13.33 |
| max | 18.57 |

The outcome is left-skewed with a hard floor at 0 (students who withdrew or failed all units). The gap between the mean (10.23) and median (12.20) reflects this mass at zero.

---

## Target label distribution

| Label | Count | Share |
|-------|-------|-------|
| Graduate | 2,209 | 49.9% |
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |

`Target` is **excluded** from the DML model (post-treatment / dropout label). It is retained in the raw file for reference only.

---

## Notes for next step (`src/01_data_pipeline.py`)

- Drop all `1st sem` columns — post-treatment bad controls per causal graph.
- Drop `Target`, `Educational special needs`, `Curricular units 2nd sem (credited/enrolled/evaluations/approved/without evaluations)` — not in the approved confounder list.
- No imputation needed (zero missings).
- Consider log or winsorised transform of Y given the zero-floor mass.
