"""Feature engineering: lags, log transforms, growth rates, key interactions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.config import PRIMARY_OUTCOME, PRIMARY_TREATMENT
from ..utils.logging_setup import get_logger

logger = get_logger("data.features")

LAG_VARS = [
    "gdp_per_capita_ppp", "gdp_per_capita_usd", "gdp_growth",
    "health_exp_pct_gdp", "health_exp_per_capita", "education_exp_pct_gdp",
    "wgi_gov_effectiveness", "urban_pop_pct",
]
LAG_PERIODS = (1, 2, 3, 5)

LOG_VARS = [
    "gdp_per_capita_usd", "gdp_per_capita_ppp", "gni_per_capita_atlas",
    "health_exp_per_capita", "population_total", "population_density",
]

GROWTH_VARS = [
    "gdp_per_capita_ppp", "gdp_per_capita_usd", "health_exp_per_capita",
    "population_total", "urban_pop_pct",
]

INTERACTIONS: list[tuple[str, str]] = [
    ("gdp_per_capita_ppp", "wgi_gov_effectiveness"),
    ("health_exp_pct_gdp", "physicians_per_1000"),
    ("education_exp_pct_gdp", "urban_pop_pct"),
    ("gdp_per_capita_ppp", "education_exp_pct_gdp"),
]


def add_lags(df: pd.DataFrame, vars_: list[str] = LAG_VARS,
             periods: tuple[int, ...] = LAG_PERIODS) -> pd.DataFrame:
    df = df.sort_values(["iso3", "year"]).copy()
    for v in vars_:
        if v not in df.columns:
            continue
        for k in periods:
            df[f"{v}_lag{k}"] = df.groupby("iso3")[v].shift(k)
    return df


def add_logs(df: pd.DataFrame, vars_: list[str] = LOG_VARS) -> pd.DataFrame:
    for v in vars_:
        if v not in df.columns:
            continue
        # log1p on positive values; clip negatives to NaN so they don't poison the model.
        df[f"log_{v}"] = np.log1p(df[v].where(df[v] > 0))
    return df


def add_growth_rates(df: pd.DataFrame, vars_: list[str] = GROWTH_VARS) -> pd.DataFrame:
    df = df.sort_values(["iso3", "year"]).copy()
    for v in vars_:
        if v not in df.columns:
            continue
        df[f"{v}_growth_pct"] = df.groupby("iso3")[v].pct_change() * 100
    return df


def add_interactions(df: pd.DataFrame,
                     pairs: list[tuple[str, str]] = INTERACTIONS) -> pd.DataFrame:
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}__x__{b}"] = df[a] * df[b]
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    n0 = df.shape[1]
    df = add_logs(df)
    df = add_growth_rates(df)
    df = add_lags(df)
    df = add_interactions(df)
    logger.info(
        "Feature engineering added %d columns (now %d). Outcome=%s, treatment=%s.",
        df.shape[1] - n0, df.shape[1], PRIMARY_OUTCOME, PRIMARY_TREATMENT,
    )
    return df
