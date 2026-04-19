"""Generate publication-quality LaTeX tables from causal inference results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.config import TABLES_DIR
from ..utils.logging_setup import get_logger

logger = get_logger("analysis.tables")


def _stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def _fmt(x: float, decimals: int = 3) -> str:
    if np.isnan(x):
        return "--"
    return f"{x:.{decimals}f}"


def _tex_header(title: str, label: str, cols: list[str], caption: str) -> list[str]:
    n = len(cols)
    col_spec = "l" + "c" * (n - 1)
    return [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]


def _tex_footer(note: str) -> list[str]:
    return [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\begin{{tablenotes}}\small\item {note}\end{{tablenotes}}",
        r"\end{table}",
    ]


# ── Granger causality table ───────────────────────────────────────────────────

def table_granger(granger_res: dict,
                  out_dir: Path = TABLES_DIR) -> Path:
    cols = ["Country", "Opt. lag (GDP→LE)", "F-stat", "p-value (raw)",
            "p-value (Bonf.)", "Significant?",
            "Opt. lag (LE→GDP)", "F-stat", "p-value (raw)"]
    rows: list[str] = []
    g2l_map = {r.country: r for r in granger_res["gdp_to_le"]}
    l2g_map = {r.country: r for r in granger_res["le_to_gdp"]}
    countries = sorted(set(g2l_map) | set(l2g_map))
    for cty in countries:
        g = g2l_map.get(cty)
        l = l2g_map.get(cty)
        g_f  = g.fstats.get(g.optimal_lag, np.nan)   if g else np.nan
        g_pr = g.pvalues_raw.get(g.optimal_lag, np.nan) if g else np.nan
        g_pb = list(g.pvalues_corrected.values())[0]  if g and g.pvalues_corrected else np.nan
        l_f  = l.fstats.get(l.optimal_lag, np.nan)   if l else np.nan
        l_pr = l.pvalues_raw.get(l.optimal_lag, np.nan) if l else np.nan
        sig  = r"\checkmark" if (g and g.significant) else "--"
        rows.append(
            f"{cty} & {g.optimal_lag if g else '--'} & "
            f"{_fmt(g_f, 2)} & {_fmt(g_pr)} & {_fmt(g_pb)} & {sig} & "
            f"{l.optimal_lag if l else '--'} & {_fmt(l_f, 2)} & {_fmt(l_pr)} \\\\"
        )

    lines = _tex_header(
        "Granger Causality Tests",
        "tab:granger",
        cols,
        "Granger causality tests: GDP $\\rightarrow$ life expectancy and reverse, maxlag=7, Bonferroni correction",
    )
    lines += rows
    lines += _tex_footer(
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
        "F-statistics from SSR-based Granger test. "
        "Bonferroni correction applied across all 29 countries."
    )
    out = out_dir / "granger_causality_results.tex"
    out.write_text("\n".join(lines))
    logger.info("Wrote %s", out)
    return out


# ── Panel regression table ────────────────────────────────────────────────────

def table_panel_fe(panel_res: dict, subgroup_res: dict,
                   out_dir: Path = TABLES_DIR) -> Path:
    specs = ["baseline", "controls_base", "controls_full"]
    sub_groups = ["high", "middle", "low"]

    col_labels = ([""] + [s.replace("_", " ").title() for s in specs]
                  + [ig.capitalize() + " income" for ig in sub_groups])
    lines = _tex_header(
        "Panel Fixed Effects", "tab:panel_fe", col_labels,
        "Two-way panel fixed effects (country + year). Dependent variable: life expectancy (years). "
        "Estimator: TWFE with cluster-robust standard errors (cluster=country). "
        "log GDP per capita (PPP) is the key regressor.",
    )

    all_res = {**{s: panel_res.get(s) for s in specs},
               **{ig: subgroup_res.get(ig) for ig in sub_groups}}

    vars_show: list[tuple[str, str]] = [
        ("log_gdp_per_capita_ppp", "log GDP per capita (PPP)"),
        ("health_exp_pct_gdp",     "Health exp. (\\% GDP)"),
        ("education_exp_pct_gdp",  "Education exp. (\\% GDP)"),
        ("wgi_gov_effectiveness",  "Gov. effectiveness (WGI)"),
        ("urban_pop_pct",          "Urban pop. (\\%)"),
        ("fertility_rate",          "Fertility rate"),
        ("trade_pct_gdp",          "Trade openness (\\% GDP)"),
        ("age_65_plus_pct",        "Pop. aged 65+ (\\%)"),
        ("inflation_cpi",          "Inflation (CPI, \\%)"),
    ]

    order = specs + sub_groups
    for col_key, label in vars_show:
        row_coefs = [label]
        row_ses   = [""]
        for k in order:
            r = all_res.get(k)
            if r is None or col_key not in r.coefs.index:
                row_coefs.append("--")
                row_ses.append("")
                continue
            c = r.coefs.get(col_key, np.nan)
            p = r.pvalues.get(col_key, np.nan)
            se = r.std_errors.get(col_key, np.nan) if hasattr(r, "std_errors") else np.nan
            row_coefs.append(f"{_fmt(c)}{_stars(p)}")
            row_ses.append(f"({_fmt(se)})" if not np.isnan(se) else "")
        lines.append(" & ".join(row_coefs) + r" \\")
        lines.append(" & ".join(row_ses) + r" \\")

    # Bottom rows
    lines.append(r"\midrule")
    row_n   = ["Obs."]       + [str(all_res[k].nobs)     if all_res.get(k) else "--" for k in order]
    row_r2  = ["R-squared"]  + [_fmt(all_res[k].rsquared) if all_res.get(k) else "--" for k in order]
    row_fe  = ["Country FE"] + [r"\checkmark"] * len(order)
    row_tfe = ["Year FE"]    + [r"\checkmark"] * len(order)
    for r_ in (row_n, row_r2, row_fe, row_tfe):
        lines.append(" & ".join(r_) + r" \\")

    lines += _tex_footer(
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
        "Standard errors in parentheses (clustered by country). "
        "All specifications include country and year fixed effects."
    )
    out = out_dir / "panel_regression_results.tex"
    out.write_text("\n".join(lines))
    logger.info("Wrote %s", out)
    return out


# ── IV table ──────────────────────────────────────────────────────────────────

def table_iv(iv_res: dict, out_dir: Path = TABLES_DIR) -> Path:
    specs = list(iv_res.keys())
    col_labels = [""] + [s.replace("_", "\n") for s in specs]

    lines = _tex_header(
        "IV-2SLS Results", "tab:iv",
        col_labels,
        "Two-stage least squares (2SLS) estimates. Instrument: GDP-weighted mean GDP growth "
        "of all other countries (Bartik-style external demand shock). "
        "Dependent variable: life expectancy. Endogenous: log GDP per capita (PPP).",
    )

    rows_data = [
        ("log GDP per capita (PPP)", "coef_gdp", "se_gdp", "pval_gdp"),
    ]
    for label, c_attr, se_attr, p_attr in rows_data:
        row_c  = [label]
        row_se = [""]
        for s in specs:
            r = iv_res.get(s)
            if r is None:
                row_c.append("--"); row_se.append(""); continue
            c  = getattr(r, c_attr, np.nan)
            se = getattr(r, se_attr, np.nan)
            p  = getattr(r, p_attr, np.nan)
            row_c.append(f"{_fmt(c)}{_stars(p)}")
            row_se.append(f"({_fmt(se)})")
        lines.append(" & ".join(row_c)  + r" \\")
        lines.append(" & ".join(row_se) + r" \\")

    lines.append(r"\midrule")
    diag_rows: list[tuple[str, Any]] = [
        ("Observations", "nobs"),
        ("First-stage F-stat", "first_stage_fstat"),
        ("Sargan over-id p-value", "sargan_pval"),
        ("Wu-Hausman p-value", "wu_hausman_pval"),
    ]
    for label, attr in diag_rows:
        row = [label]
        for s in specs:
            r = iv_res.get(s)
            v = getattr(r, attr, np.nan) if r else np.nan
            row.append(_fmt(v, 2) if (v is not None and not np.isnan(float(v or np.nan)))
                       else "--")
        lines.append(" & ".join(row) + r" \\")

    lines += _tex_footer(
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
        "Robust standard errors in parentheses. "
        "Instrument validity: first-stage $F > 10$ (not weak); Sargan $H_0$: instruments exogenous."
    )
    out = out_dir / "iv_regression_results.tex"
    out.write_text("\n".join(lines))
    logger.info("Wrote %s", out)
    return out


# ── DiD table ─────────────────────────────────────────────────────────────────

def table_did(did_res: dict, out_dir: Path = TABLES_DIR) -> Path:
    events = list(did_res.values())
    col_labels = [""] + [e.label[:25] + "..." if len(e.label) > 25 else e.label
                         for e in events]
    lines = _tex_header(
        "Difference-in-Differences", "tab:did", col_labels,
        "Difference-in-differences estimates (TWFE panel). "
        "Treatment variable: post$\\times$treated interaction. "
        "Dependent variable: life expectancy (years).",
    )

    row_att = ["ATT"]
    row_se  = [""]
    row_pt  = ["Parallel trends p"]
    row_n   = ["Observations"]
    for e in events:
        row_att.append(f"{_fmt(e.att)}{_stars(e.pval_att)}")
        row_se.append(f"({_fmt(e.se_att)})")
        row_pt.append(_fmt(e.parallel_trends_pval)
                      if e.parallel_trends_pval is not None else "--")
        row_n.append(str(e.nobs))

    for row in (row_att, row_se):
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    for row in (row_pt, row_n):
        lines.append(" & ".join(row) + r" \\")

    lines += _tex_footer(
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
        "Cluster-robust SE (by country) in parentheses. "
        "Parallel trends test: joint significance of treated$\\times$year interactions in pre-period."
    )
    out = out_dir / "did_results.tex"
    out.write_text("\n".join(lines))
    logger.info("Wrote %s", out)
    return out


# ── Summary (causal_results.tex) ──────────────────────────────────────────────

def table_synthesis(synthesis: dict, out_dir: Path = TABLES_DIR) -> Path:
    df = synthesis["coef_table"].copy()
    col_labels = ["Method", "Coefficient", "Std. Error", "p-value", "95\\% CI", "N"]
    lines = _tex_header(
        "Causal Estimates — All Methods", "tab:synthesis", col_labels,
        "Causal effect of log GDP per capita (PPP) on life expectancy across all methods. "
        "Coefficients represent years of life expectancy per unit increase in log GDP per capita. "
        "All specifications include country and year fixed effects where applicable.",
    )
    for _, row in df.iterrows():
        ci = (f"[{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}]"
              if not np.isnan(row.get("ci_lo", np.nan)) else "--")
        nobs = str(int(row["nobs"])) if not np.isnan(row.get("nobs", np.nan)) else "--"
        lines.append(
            f"{row['method']} & {_fmt(row['coef'])}{_stars(row['pval'])} & "
            f"{_fmt(row['se'])} & {_fmt(row['pval'])} & {ci} & {nobs} \\\\"
        )
    lines += _tex_footer(
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
        "Panel FE uses two-way fixed effects with cluster-robust SE. "
        "IV uses Bartik external demand instrument. DiD reports ATT. "
        "Synthetic Control uses RMSPE-ratio p-value (placebo distribution)."
    )
    out = out_dir / "causal_results.tex"
    out.write_text("\n".join(lines))
    logger.info("Wrote %s", out)
    return out


def run_all_tables(results: dict) -> list[Path]:
    paths = [
        table_granger(results["granger"]),
        table_panel_fe(results["panel_fe"], results["panel_fe_subgroups"]),
        table_iv(results["iv"]),
        table_did(results["did"]),
        table_synthesis(results["synthesis"]),
    ]
    logger.info("Wrote %d LaTeX tables to %s", len(paths), TABLES_DIR)
    return paths
