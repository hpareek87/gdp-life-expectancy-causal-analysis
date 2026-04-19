"""UNDP Human Development Index collection.

UNDP publishes the full HDI time series as CSV at hdr.undp.org. We pull the
composite HDI value (variable code "hdi") plus its three sub-indices.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from ..utils.config import ISO3_LIST, RAW_DIR, YEAR_END, YEAR_START
from ..utils.logging_setup import get_logger

logger = get_logger("data.undp")

# UNDP's "All composite indices and components time series" CSV.
UNDP_URL = "https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv"
TIMEOUT = 120

KEEP_PREFIXES = ("hdi_", "le_", "eys_", "mys_", "gnipc_")


def fetch_undp(url: str = UNDP_URL) -> pd.DataFrame:
    logger.info("Downloading UNDP composite indices from %s", url)
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), encoding="utf-8")
    df = df[df["iso3"].isin(ISO3_LIST)].copy()
    return df


def reshape_long(df: pd.DataFrame) -> pd.DataFrame:
    """Wide UNDP file -> long [iso3, year, indicator, value] for HDI series."""
    id_cols = [c for c in ("iso3", "country", "hdicode", "region") if c in df.columns]
    val_cols = [c for c in df.columns
                if c.startswith(KEEP_PREFIXES) and c[-4:].isdigit()]
    long_df = df[id_cols + val_cols].melt(
        id_vars=id_cols, var_name="var_year", value_name="value",
    )
    long_df["year"] = long_df["var_year"].str[-4:].astype(int)
    long_df["indicator"] = "undp_" + long_df["var_year"].str[:-5]
    long_df = long_df[long_df["year"].between(YEAR_START, YEAR_END)]
    return long_df[["iso3", "year", "indicator", "value"]]


def collect_and_save(out_dir: Path = RAW_DIR) -> pd.DataFrame:
    try:
        raw = fetch_undp()
    except Exception as exc:  # noqa: BLE001
        logger.warning("UNDP fetch failed: %s — returning empty frame.", exc)
        return pd.DataFrame(columns=["iso3", "year"])

    long_df = reshape_long(raw)
    wide = (long_df.pivot_table(index=["iso3", "year"], columns="indicator",
                                values="value", aggfunc="first")
            .reset_index())
    wide.columns.name = None
    out = out_dir / "undp_hdi.csv"
    wide.to_csv(out, index=False)
    logger.info("Saved %s (%d rows, %d cols)", out, len(wide), wide.shape[1])
    return wide


if __name__ == "__main__":
    collect_and_save()
