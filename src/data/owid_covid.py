"""Our World in Data COVID-19 collection.

We pull the consolidated OWID dataset and aggregate daily series to annual
mean/last values per country. Coverage starts 2020.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.config import ISO3_LIST, OWID_COVID_COLS, RAW_DIR, YEAR_END
from ..utils.logging_setup import get_logger

logger = get_logger("data.owid_covid")

OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
COVID_YEAR_START = 2020
# "Last value of year" makes sense for cumulative series; "mean" for rates.
LAST_VAL_COLS = {
    "total_deaths_per_million",
    "excess_mortality_cumulative_per_million",
    "people_fully_vaccinated_per_hundred",
    "total_tests_per_thousand",
}


def fetch_owid(url: str = OWID_URL) -> pd.DataFrame:
    logger.info("Downloading OWID COVID dataset from %s", url)
    df = pd.read_csv(url, low_memory=False, parse_dates=["date"])
    df = df[df["iso_code"].isin(ISO3_LIST)].copy()
    df["year"] = df["date"].dt.year
    df = df[df["year"].between(COVID_YEAR_START, YEAR_END)]
    return df


def aggregate_annual(df: pd.DataFrame) -> pd.DataFrame:
    src_cols = [c for c in OWID_COVID_COLS if c in df.columns]
    missing = set(OWID_COVID_COLS) - set(src_cols)
    if missing:
        logger.warning("OWID columns missing from upstream: %s", missing)

    rows: list[pd.DataFrame] = []
    for col in src_cols:
        agg = "last" if col in LAST_VAL_COLS else "mean"
        sub = (df.dropna(subset=[col])
                 .sort_values(["iso_code", "date"])
                 .groupby(["iso_code", "year"])[col]
                 .agg(agg)
                 .reset_index()
                 .rename(columns={"iso_code": "iso3", col: OWID_COVID_COLS[col]}))
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=["iso3", "year"])

    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on=["iso3", "year"], how="outer")
    return out


def collect_and_save(out_dir: Path = RAW_DIR) -> pd.DataFrame:
    raw = fetch_owid()
    annual = aggregate_annual(raw)
    out = out_dir / "owid_covid.csv"
    annual.to_csv(out, index=False)
    logger.info("Saved %s (%d rows, %d cols)", out, len(annual), annual.shape[1])
    return annual


if __name__ == "__main__":
    collect_and_save()
