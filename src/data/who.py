"""WHO Global Health Observatory (GHO) collection via the OData API.

Endpoint: https://ghoapi.azureedge.net/api/<INDICATOR_CODE>
Returns JSON; we filter to our 30 countries and 2000-2024.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from ..utils.config import (
    ISO3_LIST, RAW_DIR, WHO_INDICATORS, YEAR_END, YEAR_START,
)
from ..utils.logging_setup import get_logger

logger = get_logger("data.who")

GHO_BASE = "https://ghoapi.azureedge.net/api/{code}"
TIMEOUT = 60


def _fetch_indicator(code: str) -> pd.DataFrame:
    url = GHO_BASE.format(code=code)
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json().get("value", [])
    if not payload:
        return pd.DataFrame(columns=["iso3", "year", "value", "dim1"])
    df = pd.DataFrame(payload)
    keep = ["SpatialDim", "TimeDim", "NumericValue", "Dim1"]
    df = df[[c for c in keep if c in df.columns]].rename(columns={
        "SpatialDim": "iso3", "TimeDim": "year",
        "NumericValue": "value", "Dim1": "dim1",
    })
    return df


def fetch_who(
    indicators: dict[str, str] = WHO_INDICATORS,
    iso3: list[str] = ISO3_LIST,
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for clean_name, code in tqdm(indicators.items(), desc="WHO indicators"):
        try:
            raw = _fetch_indicator(code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("WHO fetch failed for %s (%s): %s", clean_name, code, exc)
            continue
        if raw.empty:
            logger.warning("WHO indicator %s returned no rows", clean_name)
            continue

        raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
        raw = raw[
            raw["iso3"].isin(iso3)
            & raw["year"].between(year_start, year_end)
        ]
        # Some GHO indicators are split by sex/age — keep the "BTSX" (both sexes)
        # row when present, otherwise average across dim1.
        if "dim1" in raw.columns and raw["dim1"].notna().any():
            both = raw[raw["dim1"] == "SEX_BTSX"]
            if not both.empty:
                raw = both
            else:
                raw = raw.groupby(["iso3", "year"], as_index=False)["value"].mean()
        raw["indicator"] = clean_name
        frames.append(raw[["iso3", "year", "indicator", "value"]])

    if not frames:
        logger.warning("No WHO data collected — returning empty frame.")
        return pd.DataFrame(columns=["iso3", "year", "indicator", "value"])

    return pd.concat(frames, ignore_index=True)


def collect_and_save(out_dir: Path = RAW_DIR) -> pd.DataFrame:
    long_df = fetch_who()
    wide = (long_df.pivot_table(index=["iso3", "year"], columns="indicator",
                                values="value", aggfunc="first")
            .reset_index() if not long_df.empty else long_df)
    if not wide.empty:
        wide.columns.name = None
    out = out_dir / "who.csv"
    wide.to_csv(out, index=False)
    logger.info("Saved %s (%d rows)", out, len(wide))
    return wide


if __name__ == "__main__":
    collect_and_save()
