"""Smoke tests for the data pipeline. Skip if the master dataset hasn't been built."""
from __future__ import annotations

import pandas as pd
import pytest

from src.utils.config import (
    COUNTRIES, FINAL_DIR, HIGH_INCOME, LOW_INCOME, MIDDLE_INCOME,
    PRIMARY_OUTCOME, PRIMARY_TREATMENT, YEAR_END, YEAR_START,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    path = FINAL_DIR / "master_dataset.csv"
    if not path.exists():
        pytest.skip("master_dataset.csv not built — run `python -m src.data.build_dataset` first")
    return pd.read_csv(path)


def test_country_counts() -> None:
    assert len(HIGH_INCOME) == len(MIDDLE_INCOME) == len(LOW_INCOME) == 10
    assert len(COUNTRIES) == 30


def test_year_window(df: pd.DataFrame) -> None:
    assert df["year"].min() == YEAR_START
    assert df["year"].max() == YEAR_END


def test_primary_columns_present(df: pd.DataFrame) -> None:
    for col in (PRIMARY_OUTCOME, PRIMARY_TREATMENT, "iso3", "year",
                "country", "income_group"):
        assert col in df.columns, f"missing column {col}"


def test_le_plausible_range(df: pd.DataFrame) -> None:
    s = df[PRIMARY_OUTCOME].dropna()
    assert s.min() >= 20 and s.max() <= 95


def test_income_group_ordering(df: pd.DataFrame) -> None:
    means = df.groupby("income_group")[PRIMARY_OUTCOME].mean()
    assert means["high"] > means["middle"] > means["low"]


def test_no_duplicate_country_year(df: pd.DataFrame) -> None:
    assert not df.duplicated(subset=["iso3", "year"]).any()
