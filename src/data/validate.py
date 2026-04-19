"""Data quality validation: range checks, outlier detection, IMF cross-check."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from ..utils.config import GDP_CROSSCHECK_TOLERANCE, ISO3_LIST
from ..utils.logging_setup import get_logger

logger = get_logger("data.validate")

# (lower, upper) — values outside these bounds are flagged as suspect.
RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "life_expectancy":      (20.0, 95.0),
    "infant_mortality":     (0.0, 200.0),
    "under5_mortality":     (0.0, 300.0),
    "gdp_per_capita_usd":   (100.0, 200_000.0),
    "gdp_per_capita_ppp":   (200.0, 200_000.0),
    "gdp_growth":           (-50.0, 50.0),
    "inflation_cpi":        (-30.0, 1_000.0),
    "unemployment":         (0.0, 80.0),
    "fertility_rate":       (0.5, 9.0),
    "urban_pop_pct":        (0.0, 100.0),
    "literacy_adult":       (0.0, 100.0),
    "primary_enroll":       (0.0, 110.0),
    "health_exp_pct_gdp":   (0.0, 30.0),
    "wgi_gov_effectiveness":(-3.0, 3.0),
}

IMF_GDPPC_URL = "https://www.imf.org/external/datamapper/api/v1/NGDPDPC/{iso3}"
IMF_TIMEOUT = 30


@dataclass
class QualityReport:
    range_violations: dict[str, int] = field(default_factory=dict)
    outliers_iqr: dict[str, int] = field(default_factory=dict)
    crosscheck: dict[str, dict] = field(default_factory=dict)


def check_ranges(df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        n = int(((s < lo) | (s > hi)).sum())
        if n:
            out[col] = n
            logger.warning("%d range violations in %s [%s, %s]", n, col, lo, hi)
    return out


def detect_outliers_iqr(df: pd.DataFrame, k: float = 3.0) -> dict[str, int]:
    """Per-country IQR outlier counts for headline indicators."""
    cols = ["gdp_per_capita_ppp", "life_expectancy", "inflation_cpi", "gdp_growth"]
    out: dict[str, int] = {}
    for col in cols:
        if col not in df.columns:
            continue
        n_total = 0
        for _, sub in df.groupby("iso3"):
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(s) < 5:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (s < q1 - k * iqr) | (s > q3 + k * iqr)
            n_total += int(mask.sum())
        if n_total:
            out[col] = n_total
    return out


def crosscheck_gdp_imf(df: pd.DataFrame,
                       tol: float = GDP_CROSSCHECK_TOLERANCE) -> dict[str, dict]:
    """Compare WB GDP-per-capita (USD) against IMF WEO via the DataMapper API."""
    if "gdp_per_capita_usd" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for iso in ISO3_LIST:
        try:
            r = requests.get(IMF_GDPPC_URL.format(iso3=iso), timeout=IMF_TIMEOUT)
            r.raise_for_status()
            payload = r.json().get("values", {}).get("NGDPDPC", {}).get(iso, {})
        except Exception as exc:  # noqa: BLE001
            logger.warning("IMF cross-check failed for %s: %s", iso, exc)
            continue
        if not payload:
            continue
        wb_sub = (df[df["iso3"] == iso][["year", "gdp_per_capita_usd"]]
                    .dropna()
                    .set_index("year")["gdp_per_capita_usd"])
        diffs: list[float] = []
        for yr_str, imf_val in payload.items():
            try:
                yr = int(yr_str)
                if yr in wb_sub.index and imf_val:
                    wb_val = float(wb_sub.loc[yr])
                    if wb_val > 0:
                        diffs.append(abs(wb_val - float(imf_val)) / wb_val)
            except (ValueError, TypeError):
                continue
        if diffs:
            mean_diff = float(np.mean(diffs))
            out[iso] = {
                "mean_relative_diff": round(mean_diff, 4),
                "max_relative_diff": round(float(np.max(diffs)), 4),
                "n_years_compared": len(diffs),
                "exceeds_tolerance": mean_diff > tol,
            }
    n_bad = sum(1 for v in out.values() if v["exceeds_tolerance"])
    logger.info("IMF cross-check: %d/%d countries exceed %.0f%% tolerance",
                n_bad, len(out), tol * 100)
    return out


def validate(df: pd.DataFrame, do_crosscheck: bool = True) -> QualityReport:
    report = QualityReport()
    report.range_violations = check_ranges(df)
    report.outliers_iqr = detect_outliers_iqr(df)
    if do_crosscheck:
        report.crosscheck = crosscheck_gdp_imf(df)
    return report
