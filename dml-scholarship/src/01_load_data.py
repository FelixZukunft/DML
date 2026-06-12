"""Phase 1 — Download UCI student dropout dataset and run a full data audit."""
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"


def download() -> pd.DataFrame:
    dataset = fetch_ucirepo(id=697)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "students_raw.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}\n")
    return df


def audit(df: pd.DataFrame) -> None:
    sep = "-" * 60

    print(sep)
    print("SHAPE")
    print(df.shape)

    print(sep)
    print("COLUMNS")
    print(df.columns.tolist())

    print(sep)
    print("DTYPES")
    print(df.dtypes.to_string())

    print(sep)
    print("MISSING VALUES PER COLUMN")
    print(df.isnull().sum().to_string())

    print(sep)
    print("TREATMENT DISTRIBUTION  —  Scholarship holder")
    print(df["Scholarship holder"].value_counts().to_string())

    print(sep)
    print("OUTCOME STATS  —  Curricular units 2nd sem (grade)")
    print(df["Curricular units 2nd sem (grade)"].describe().to_string())

    print(sep)
    print("TARGET DISTRIBUTION  —  Target")
    print(df["Target"].value_counts().to_string())

    print(sep)


if __name__ == "__main__":
    df = download()
    audit(df)
