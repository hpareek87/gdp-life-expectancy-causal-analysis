"""End-to-end pipeline orchestrator.

Run as:
    python -m src.data.build_dataset

Steps:
  1. Collect raw data (World Bank, WHO, OWID, UNDP)
  2. Merge + impute -> data/processed/panel_clean.csv
  3. Feature engineering
  4. Validate -> data quality report
  5. Write data/final/master_dataset.csv + data_dictionary.txt + data_quality_report.txt
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..utils.config import (
    COUNTRIES, FINAL_DIR, INCOME_GROUP, OWID_COVID_COLS, PROJECT_ROOT,
    RAW_DIR, VAR_GROUPS, WB_INDICATORS, WHO_INDICATORS,
)
from ..utils.logging_setup import get_logger
from . import clean, features, owid_covid, undp, validate, who, worldbank

logger = get_logger("data.build")


def collect_raw(skip_existing: bool = True) -> None:
    """Run each source collector unless its output is already on disk."""
    targets = [
        ("worldbank_wide.csv", worldbank.collect_and_save),
        ("who.csv",            who.collect_and_save),
        ("owid_covid.csv",     owid_covid.collect_and_save),
        ("undp_hdi.csv",       undp.collect_and_save),
    ]
    for fname, fn in targets:
        path = RAW_DIR / fname
        if skip_existing and path.exists() and path.stat().st_size > 0:
            logger.info("Found existing %s — skipping fetch.", path.name)
            continue
        logger.info("Collecting %s ...", fname)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.error("Collector for %s failed: %s", fname, exc)


def write_data_dictionary(path: Path, df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# DATA DICTIONARY — GDP vs Life Expectancy master dataset")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    lines.append(f"Countries: {df['iso3'].nunique()} | Years: {df['year'].min()}–{df['year'].max()}")
    lines.append("")
    lines.append("## Identifier columns")
    for c in ("iso3", "country", "year", "income_group"):
        if c in df.columns:
            lines.append(f"  - {c}: {('country ISO-3 code' if c=='iso3' else c)}")
    lines.append("")

    def _block(header: str, mapping: dict[str, str]) -> None:
        lines.append(f"## {header}")
        for clean_name, src_code in mapping.items():
            present = "yes" if clean_name in df.columns else "missing"
            lines.append(f"  - {clean_name:<40} src={src_code:<25} present={present}")
        lines.append("")

    _block("World Bank WDI indicators", WB_INDICATORS)
    _block("WHO Global Health Observatory indicators", WHO_INDICATORS)
    _block("OWID COVID indicators", OWID_COVID_COLS)
    lines.append("## UNDP HDI series")
    for c in sorted(c for c in df.columns if c.startswith("undp_")):
        lines.append(f"  - {c}")
    lines.append("")
    lines.append("## Engineered features")
    eng_kinds = ("log_", "_lag", "_growth_pct", "__x__")
    for c in sorted(c for c in df.columns if any(k in c for k in eng_kinds)):
        lines.append(f"  - {c}")
    path.write_text("\n".join(lines))
    logger.info("Wrote data dictionary to %s", path)


def write_quality_report(path: Path, clean_report: clean.CleanReport,
                         qual_report: validate.QualityReport,
                         df_final: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# DATA QUALITY REPORT")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"  Countries in:  {clean_report.n_countries_in}")
    lines.append(f"  Countries out: {clean_report.n_countries_out}")
    if clean_report.excluded_countries:
        names = [f"{i} ({COUNTRIES.get(i, '?')})" for i in clean_report.excluded_countries]
        lines.append(f"  Excluded for >30% missingness: {names}")
    lines.append(f"  Final shape: {df_final.shape}")
    lines.append("")
    lines.append("## Imputation")
    lines.append(f"  Total feature cells: {clean_report.total_cells:,}")
    lines.append(f"  Cells imputed:       {clean_report.imputed_cells:,}")
    if clean_report.total_cells:
        pct = 100 * clean_report.imputed_cells / clean_report.total_cells
        lines.append(f"  % imputed:           {pct:.2f}%")
    lines.append("  Method: sklearn IterativeImputer (BayesianRidge), country-by-country.")
    lines.append("  Indicators entirely missing for a country are pre-filled with the income-group median.")
    lines.append("")

    lines.append("## Missingness — per indicator (BEFORE imputation)")
    for col, miss in sorted(clean_report.missing_before.items(), key=lambda x: -x[1])[:25]:
        lines.append(f"  {col:<45} {miss:>6.2f}%")
    lines.append("")

    lines.append("## Range-check violations")
    if qual_report.range_violations:
        for col, n in sorted(qual_report.range_violations.items(), key=lambda x: -x[1]):
            lines.append(f"  {col:<45} {n} obs out of plausible bounds")
    else:
        lines.append("  None.")
    lines.append("")

    lines.append("## IQR outlier flags (k=3, per country)")
    if qual_report.outliers_iqr:
        for col, n in qual_report.outliers_iqr.items():
            lines.append(f"  {col:<45} {n} outlier obs")
    else:
        lines.append("  None.")
    lines.append("")

    lines.append("## IMF cross-check (GDP per capita, USD)")
    if qual_report.crosscheck:
        n_bad = sum(1 for v in qual_report.crosscheck.values() if v["exceeds_tolerance"])
        lines.append(f"  Countries compared: {len(qual_report.crosscheck)}")
        lines.append(f"  Exceeding 2% tolerance: {n_bad}")
        for iso, info in sorted(qual_report.crosscheck.items(),
                                key=lambda x: -x[1]["mean_relative_diff"])[:15]:
            tag = " (FLAG)" if info["exceeds_tolerance"] else ""
            lines.append(
                f"  {iso} {COUNTRIES.get(iso, '?'):<25} "
                f"mean_diff={info['mean_relative_diff']:.4f} "
                f"max_diff={info['max_relative_diff']:.4f} "
                f"n_years={info['n_years_compared']}{tag}"
            )
    else:
        lines.append("  No cross-check performed (network unreachable or no data).")
    lines.append("")

    lines.append("## Notes")
    lines.append("  - World Bank WDI is the primary source for life expectancy (SP.DYN.LE00.IN).")
    lines.append("  - WHO HALE supplements life expectancy with health-adjusted life years.")
    lines.append("  - OWID COVID series cover 2020–2024 only; missing in earlier years by design.")
    lines.append("  - UNDP HDI series end in their latest published year (typically t-2).")
    path.write_text("\n".join(lines))
    logger.info("Wrote quality report to %s", path)


def write_repro_report(path: Path) -> None:
    import platform
    import sys
    info = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
    }
    try:
        import wbgapi, sklearn, statsmodels, pandas as pd_, numpy as np_
        info["packages"] = {
            "pandas": pd_.__version__, "numpy": np_.__version__,
            "wbgapi": wbgapi.__version__ if hasattr(wbgapi, "__version__") else "n/a",
            "scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
        }
    except Exception:  # noqa: BLE001
        pass
    path.write_text(json.dumps(info, indent=2))


def main(skip_fetch: bool = False, skip_crosscheck: bool = False) -> None:
    if not skip_fetch:
        collect_raw(skip_existing=True)

    panel, clean_report = clean.clean()

    panel = features.engineer(panel)
    panel["income_group"] = panel["iso3"].map(INCOME_GROUP)
    panel["country"] = panel["iso3"].map(COUNTRIES)
    # Bring identifier columns to the left.
    front = [c for c in ("iso3", "country", "year", "income_group") if c in panel.columns]
    panel = panel[front + [c for c in panel.columns if c not in front]]

    out_path = FINAL_DIR / "master_dataset.csv"
    panel.to_csv(out_path, index=False)
    logger.info("Saved master dataset to %s (%d rows × %d cols)",
                out_path, len(panel), panel.shape[1])

    qual_report = validate.validate(panel, do_crosscheck=not skip_crosscheck)

    write_data_dictionary(FINAL_DIR / "data_dictionary.txt", panel)
    write_quality_report(FINAL_DIR / "data_quality_report.txt", clean_report, qual_report, panel)
    write_repro_report(FINAL_DIR / "reproducibility.json")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip raw collection if files already exist")
    parser.add_argument("--skip-crosscheck", action="store_true",
                        help="Skip the IMF GDP cross-check (offline mode)")
    args = parser.parse_args()
    main(skip_fetch=args.skip_fetch, skip_crosscheck=args.skip_crosscheck)
