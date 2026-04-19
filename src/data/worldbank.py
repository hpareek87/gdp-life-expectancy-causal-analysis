"""World Bank WDI collection via the wbgapi client.

Output: long DataFrame with columns [iso3, year, indicator, value] saved to
data/raw/worldbank.csv. A wide pivot is also written for convenience.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import wbgapi as wb
from tqdm import tqdm

from ..utils.config import (
    ISO3_LIST, RAW_DIR, WB_INDICATOR_DB, WB_INDICATORS, YEAR_END, YEAR_START,
)
from ..utils.logging_setup import get_logger

logger = get_logger("data.worldbank")


def fetch_worldbank(
    indicators: dict[str, str] = WB_INDICATORS,
    iso3: list[str] = ISO3_LIST,
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
) -> pd.DataFrame:
    """Pull every (indicator, country, year) observation from the WDI API.

    Returned frame is in long form. Indicators that fail are logged and skipped
    rather than aborting the run — partial coverage is preferable to none.
    """
    rows: list[dict] = []
    failures: list[str] = []

    for clean_name, code in tqdm(indicators.items(), desc="WB indicators"):
        db = WB_INDICATOR_DB.get(code)  # None -> wbgapi default (WDI = 2)
        try:
            kwargs = dict(
                series=code, economy=iso3,
                time=range(year_start, year_end + 1),
                labels=False, skipBlanks=False,
            )
            if db is not None:
                kwargs["db"] = db
            df = wb.data.DataFrame(**kwargs)
        except Exception as exc:  # noqa: BLE001 — surface and continue
            logger.warning("WB fetch failed for %s (%s, db=%s): %s",
                           clean_name, code, db, exc)
            failures.append(clean_name)
            continue

        # wbgapi returns wide (countries x time-columns "YR2000"...).
        df = df.reset_index().melt(id_vars="economy", var_name="year_col", value_name="value")
        df["year"] = df["year_col"].str.replace("YR", "", regex=False).astype(int)
        df["indicator"] = clean_name
        df = df.rename(columns={"economy": "iso3"})[["iso3", "year", "indicator", "value"]]
        rows.append(df)

    if not rows:
        raise RuntimeError("World Bank fetch returned no data for any indicator.")

    long_df = pd.concat(rows, ignore_index=True)
    logger.info(
        "Fetched %d indicators (%d failed) → %d rows",
        len(indicators) - len(failures), len(failures), len(long_df),
    )
    if failures:
        logger.warning("Failed indicators: %s", failures)
    return long_df


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long → wide with (iso3, year) index."""
    wide = long_df.pivot_table(
        index=["iso3", "year"], columns="indicator", values="value", aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def collect_and_save(out_dir: Path = RAW_DIR) -> pd.DataFrame:
    long_df = fetch_worldbank()
    wide = to_wide(long_df)
    long_path = out_dir / "worldbank_long.csv"
    wide_path = out_dir / "worldbank_wide.csv"
    long_df.to_csv(long_path, index=False)
    wide.to_csv(wide_path, index=False)
    logger.info("Saved %s (%d rows) and %s (%d rows)",
                long_path, len(long_df), wide_path, len(wide))
    return wide


if __name__ == "__main__":
    collect_and_save()
