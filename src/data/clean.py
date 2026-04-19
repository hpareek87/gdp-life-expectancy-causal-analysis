"""Merge raw sources, deduplicate columns, exclude high-missingness countries,
and impute via IterativeImputer (sklearn's MICE-equivalent)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from ..utils.config import (
    COUNTRIES, INCOME_GROUP, ISO3_LIST, MAX_MISSING_PCT_PER_COUNTRY,
    PROCESSED_DIR, RAW_DIR, YEARS,
)
from ..utils.logging_setup import get_logger
from .validate import RANGE_CHECKS

logger = get_logger("data.clean")

NON_FEATURE = {"iso3", "year", "country", "income_group"}


@dataclass
class CleanReport:
    n_countries_in: int = 0
    n_countries_out: int = 0
    excluded_countries: list[str] = field(default_factory=list)
    missing_before: dict[str, float] = field(default_factory=dict)
    missing_after: dict[str, float] = field(default_factory=dict)
    imputed_cells: int = 0
    total_cells: int = 0
    notes: list[str] = field(default_factory=list)


def _load_panel(raw_dir: Path) -> pd.DataFrame:
    """Build the (iso3 × year) skeleton and merge each raw source onto it."""
    skeleton = pd.MultiIndex.from_product(
        [ISO3_LIST, YEARS], names=["iso3", "year"]
    ).to_frame(index=False)

    panel = skeleton
    for fname in ("worldbank_wide.csv", "who.csv", "owid_covid.csv", "undp_hdi.csv"):
        path = raw_dir / fname
        if not path.exists():
            logger.warning("Skipping missing source file %s", path)
            continue
        df = pd.read_csv(path)
        if "iso3" not in df.columns or "year" not in df.columns or df.empty:
            logger.warning("Source %s has no usable iso3/year columns; skipping.", path)
            continue
        df["year"] = df["year"].astype(int)
        # Avoid duplicate-column collisions on merge.
        overlap = (set(df.columns) - {"iso3", "year"}) & set(panel.columns)
        if overlap:
            df = df.drop(columns=list(overlap))
            logger.info("Dropped overlapping columns from %s: %s", fname, overlap)
        panel = panel.merge(df, on=["iso3", "year"], how="left")
        logger.info("Merged %s → panel shape %s", fname, panel.shape)

    panel["country"] = panel["iso3"].map(COUNTRIES)
    panel["income_group"] = panel["iso3"].map(INCOME_GROUP)
    return panel


def _exclude_high_missing(panel: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [c for c in panel.columns if c not in NON_FEATURE]
    pct = (panel.groupby("iso3")[feature_cols].apply(lambda g: g.isna().mean().mean()) * 100)
    excluded = pct[pct > threshold].index.tolist()
    if excluded:
        logger.warning("Excluding %d country(ies) for missingness > %.0f%%: %s",
                       len(excluded), threshold, excluded)
        panel = panel[~panel["iso3"].isin(excluded)].copy()
    return panel, excluded


def _impute(panel: pd.DataFrame, max_iter: int = 10) -> tuple[pd.DataFrame, int]:
    """Country-aware iterative imputation.

    Strategy: for each country, run IterativeImputer over its 25-year time series.
    Indicators that are entirely missing for a country are filled with the
    income-group median to keep the imputer well-conditioned, then re-imputed.
    """
    feature_cols = [c for c in panel.columns if c not in NON_FEATURE]
    imputed_total = 0

    # 1. Income-group medians as a prior for entirely-missing series.
    grp_medians = panel.groupby("income_group")[feature_cols].median()
    for ig, sub in panel.groupby("income_group"):
        for col in feature_cols:
            if sub[col].isna().all():
                fill = grp_medians.loc[ig, col]
                if pd.notna(fill):
                    mask = (panel["income_group"] == ig) & panel[col].isna()
                    panel.loc[mask, col] = fill
                    imputed_total += int(mask.sum())

    # 2. MICE on the 25-row time series of each country.
    out_frames: list[pd.DataFrame] = []
    for iso, sub in panel.groupby("iso3"):
        sub = sub.sort_values("year").copy()
        X = sub[feature_cols].astype(float).values
        n_missing_before = int(np.isnan(X).sum())
        if n_missing_before > 0 and X.shape[1] > 0:
            usable = ~np.all(np.isnan(X), axis=0)
            if usable.sum() >= 2:
                imputer = IterativeImputer(
                    estimator=BayesianRidge(), max_iter=max_iter,
                    random_state=42, sample_posterior=False,
                )
                X[:, usable] = imputer.fit_transform(X[:, usable])
            sub[feature_cols] = X
        n_missing_after = int(np.isnan(sub[feature_cols].astype(float).values).sum())
        imputed_total += n_missing_before - n_missing_after
        out_frames.append(sub)

    out = pd.concat(out_frames, ignore_index=True)
    # Clip imputed values to plausible bounds so MICE artifacts don't violate
    # reality (e.g., enrollment >100%, negative mortality).
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out, imputed_total


def clean(raw_dir: Path = RAW_DIR,
          processed_dir: Path = PROCESSED_DIR,
          missing_threshold: float = MAX_MISSING_PCT_PER_COUNTRY,
          ) -> tuple[pd.DataFrame, CleanReport]:
    report = CleanReport()
    panel = _load_panel(raw_dir)
    feature_cols = [c for c in panel.columns if c not in NON_FEATURE]
    report.n_countries_in = panel["iso3"].nunique()
    report.missing_before = (panel[feature_cols].isna().mean() * 100).round(2).to_dict()
    report.total_cells = panel[feature_cols].size

    panel, excluded = _exclude_high_missing(panel, missing_threshold)
    report.excluded_countries = excluded

    panel, imputed = _impute(panel)
    report.imputed_cells = imputed
    report.n_countries_out = panel["iso3"].nunique()
    report.missing_after = (panel[feature_cols].isna().mean() * 100).round(2).to_dict()

    out = processed_dir / "panel_clean.csv"
    panel.to_csv(out, index=False)
    logger.info("Saved cleaned panel to %s (%d rows, %d cols)",
                out, len(panel), panel.shape[1])
    return panel, report
