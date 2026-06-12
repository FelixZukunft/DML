"""Sanity checks for the DML pipeline."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROCESSED = Path(__file__).parents[1] / "data" / "processed" / "features.parquet"


@pytest.fixture
def processed_df():
    if not PROCESSED.exists():
        pytest.skip("Run phase 1 first to generate processed data")
    return pd.read_parquet(PROCESSED)


def test_no_missing(processed_df):
    assert processed_df.isnull().sum().sum() == 0, "NaNs found in processed data"


def test_min_rows(processed_df):
    assert len(processed_df) >= 100, "Fewer than 100 rows after cleaning"


def test_no_constant_columns(processed_df):
    constant_cols = [c for c in processed_df.columns if processed_df[c].nunique() <= 1]
    assert not constant_cols, f"Constant columns found: {constant_cols}"


def test_numeric_only(processed_df):
    non_numeric = processed_df.select_dtypes(exclude="number").columns.tolist()
    assert not non_numeric, f"Non-numeric columns remain: {non_numeric}"
