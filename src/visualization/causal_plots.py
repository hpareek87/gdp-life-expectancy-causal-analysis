"""Diagnostic and results figures for Phase 2 causal inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.config import EDA_FIG_DIR, OUTPUTS_DIR
from ..utils.logging_setup import get_logger

logger = get_logger("viz.causal")

CAUSAL_FIG_DIR = OUTPUTS_DIR / "figures" / "causal"
CAUSAL_FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"high": "#2c7fb8", "middle": "#fdae61", "low": "#d7191c",
          "positive": "#1a9641", "negative": "#d7191c", "neutral": "#888888"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> Path:
    p = CAUSAL_FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)
    return p


# ── Granger causality ─────────────────────────────────────────────────────────

def fig_granger_heatmap(granger_res: dict) -> Path:
    """Heatmap of log(p-value) for GDP→LE and LE→GDP by country."""
    g2l = {r.country: r.pvalues_raw.get(r.optimal_lag, 1.0)
           for r in granger_res["gdp_to_le"]}
    l2g = {r.country: r.pvalues_raw.get(r.optimal_lag, 1.0)
           for r in granger_res["le_to_gdp"]}
    countries = sorted(set(g2l) | set(l2g))
    data = pd.DataFrame({
        "GDP → LE": [g2l.get(c, 1.0) for c in countries],
        "LE → GDP": [l2g.get(c, 1.0) for c in countries],
    }, index=countries)
    log_data = -np.log10(data.clip(lower=1e-8))

    fig, ax = plt.subplots(figsize=(5, 9))
    sns.heatmap(log_data, annot=data.round(3), fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=3, ax=ax,
                cbar_kws={"label": "-log₁₀(p-value)", "shrink": 0.6},
                annot_kws={"size": 7.5})
    ax.axvline(1, color="white", linewidth=2)
    # Mark significant cells with bold outline
    for i, cty in enumerate(countries):
        sig_g2l = next((r.significant for r in granger_res["gdp_to_le"]
                        if r.country == cty), False)
        sig_l2g = next((r.significant for r in granger_res["le_to_gdp"]
                        if r.country == cty), False)
        for j, sig in enumerate([sig_g2l, sig_l2g]):
            if sig:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor="black", lw=2))
    ax.set_title("Granger causality p-values\n(bold border = Bonferroni-significant)",
                 fontsize=11)
    ax.set_xlabel("")
    return _save(fig, "01_granger_heatmap.png")


def fig_granger_lag_dist(granger_res: dict) -> Path:
    """Distribution of optimal lags for GDP→LE."""
    lags = [r.optimal_lag for r in granger_res["gdp_to_le"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lags, bins=range(1, 9), align="left", rwidth=0.8,
            color="#2c7fb8", edgecolor="white")
    ax.set(title="Distribution of optimal Granger lag (GDP → Life expectancy)",
           xlabel="Lag (years)", ylabel="Number of countries",
           xticks=range(1, 8))
    mode_lag = granger_res["summary"].get("most_common_lag_gdp2le", "?")
    ax.axvline(mode_lag, color="red", linestyle="--",
               label=f"Mode = {mode_lag} yr")
    ax.legend()
    return _save(fig, "02_granger_lag_dist.png")


# ── Panel Fixed Effects ────────────────────────────────────────────────────────

def fig_panel_coef_plot(panel_res: dict, subgroup_res: dict) -> Path:
    """Coefficient plot comparing Panel FE specs and subgroups."""
    rows: list[dict] = []
    for spec_name, r in panel_res.items():
        rows.append({"label": f"Full sample\n({spec_name})", "coef": r.coef_gdp,
                     "lo": r.ci95_gdp[0], "hi": r.ci95_gdp[1], "group": "pooled"})
    for ig, r in subgroup_res.items():
        rows.append({"label": f"{ig.capitalize()} income", "coef": r.coef_gdp,
                     "lo": r.ci95_gdp[0], "hi": r.ci95_gdp[1], "group": ig})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [COLORS.get(r["group"], "#555555") for _, r in df.iterrows()]
    for i, (_, row) in enumerate(df.iterrows()):
        col = COLORS.get(row["group"], "#555555")
        ax.plot([row["lo"], row["hi"]], [i, i], color=col, linewidth=2.5)
        ax.scatter(row["coef"], i, color=col, s=90, zorder=4)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"])
    ax.set(xlabel="Coefficient on log GDP per capita (95% CI)",
           title="Panel fixed effects: effect of GDP on life expectancy\n(two-way FE, cluster-robust SE)")
    return _save(fig, "03_panel_coef_plot.png")


def fig_fe_residuals(df: pd.DataFrame) -> Path:
    """Within-country residuals of life expectancy after removing country mean."""
    sub = df.copy()
    sub["le_within"] = sub.groupby("iso3")["life_expectancy"].transform(
        lambda x: x - x.mean())
    sub["gdp_within"] = sub.groupby("iso3")["log_gdp_per_capita_ppp"].transform(
        lambda x: x - x.mean())
    fig, ax = plt.subplots(figsize=(9, 6))
    for ig in ["high", "middle", "low"]:
        s = sub[sub["income_group"] == ig]
        ax.scatter(s["gdp_within"], s["le_within"], alpha=0.35, s=18,
                   color=COLORS[ig], label=ig.capitalize())
    # OLS fit line
    valid = sub.dropna(subset=["gdp_within", "le_within"])
    b = np.polyfit(valid["gdp_within"], valid["le_within"], 1)
    xl = np.linspace(valid["gdp_within"].min(), valid["gdp_within"].max(), 100)
    ax.plot(xl, np.polyval(b, xl), "k--", linewidth=1.5,
            label=f"OLS slope={b[0]:.2f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set(title="Within-country variation: log GDP vs life expectancy\n(country-mean demeaned)",
           xlabel="Δ log GDP per capita (within country)",
           ylabel="Δ life expectancy (within country)")
    ax.legend()
    return _save(fig, "04_fe_within_variation.png")


# ── IV-2SLS ───────────────────────────────────────────────────────────────────

def fig_iv_first_stage(df: pd.DataFrame) -> Path:
    """First-stage: external demand vs log GDP per capita."""
    from ..analysis.causal import _build_instruments
    df_inst = _build_instruments(df.copy())
    valid = df_inst.dropna(subset=["ext_demand", "log_gdp_per_capita_ppp"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for ig in ["high", "middle", "low"]:
        s = valid[valid["income_group"] == ig]
        ax.scatter(s["ext_demand"], s["log_gdp_per_capita_ppp"], alpha=0.35,
                   s=20, color=COLORS[ig], label=ig.capitalize())
    b = np.polyfit(valid["ext_demand"], valid["log_gdp_per_capita_ppp"], 1)
    xl = np.linspace(valid["ext_demand"].min(), valid["ext_demand"].max(), 100)
    ax.plot(xl, np.polyval(b, xl), "k--", linewidth=1.5)
    ax.set(title="IV first stage: External demand shock → log GDP per capita",
           xlabel="External demand instrument (mean GDP growth of other countries)",
           ylabel="log GDP per capita, PPP")
    ax.legend()
    return _save(fig, "05_iv_first_stage.png")


def fig_iv_comparison(iv_res: dict, panel_res: dict) -> Path:
    """IV vs OLS/FE comparison bar chart."""
    labels, coefs, ses, colors_list = [], [], [], []
    for spec, r in panel_res.items():
        labels.append(f"Panel FE\n({spec})")
        coefs.append(r.coef_gdp)
        ses.append(r.se_gdp)
        colors_list.append("#2c7fb8")
    for spec, r in iv_res.items():
        labels.append(f"IV-2SLS\n({spec.replace('_', ' ')})")
        coefs.append(r.coef_gdp)
        ses.append(r.se_gdp)
        colors_list.append("#f03b20")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, coefs, color=colors_list, alpha=0.8, edgecolor="white", width=0.6)
    ax.errorbar(x, coefs, yerr=1.96 * np.array(ses), fmt="none",
                color="black", capsize=5, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set(title="Comparing OLS/FE vs IV-2SLS estimates\n(coefficient on log GDP per capita)",
           ylabel="Coefficient (years of life expectancy per log-unit GDP)")
    handles = [plt.Rectangle((0,0),1,1, color="#2c7fb8", label="Panel FE"),
               plt.Rectangle((0,0),1,1, color="#f03b20", label="IV-2SLS")]
    ax.legend(handles=handles)
    return _save(fig, "06_iv_vs_ols.png")


# ── DiD ───────────────────────────────────────────────────────────────────────

def fig_did_parallel_trends(did_res: dict, df: pd.DataFrame) -> list[Path]:
    paths = []
    for event_name, r in did_res.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        # Find treated and control countries for this event
        from ..analysis.causal import DID_EVENTS
        cfg = DID_EVENTS.get(event_name, {})
        treated_cties = cfg.get("treated", [])
        control_cties = cfg.get("control", [])
        reform_y = r.event.split("_")[-1] if "_" in r.event else None
        reform_y = cfg.get("reform_year", 2009)

        for iso in treated_cties + control_cties:
            sub = df[df["iso3"] == iso].sort_values("year")
            color = "#d7191c" if iso in treated_cties else "#2c7fb8"
            ls = "-" if iso in treated_cties else "--"
            ax.plot(sub["year"], sub["life_expectancy"],
                    color=color, linewidth=2.0, linestyle=ls,
                    label=f"{iso} ({'treated' if iso in treated_cties else 'control'})")
        ax.axvline(reform_y, color="black", linestyle=":", linewidth=1.5,
                   label=f"Reform year ({reform_y})")
        ax.set(title=f"Parallel trends: {r.label}",
               xlabel="Year", ylabel="Life expectancy (years)")
        ax.legend(fontsize=9)
        tag = event_name.lower().replace("_", "-")
        p = _save(fig, f"07_did_trends_{tag}.png")
        paths.append(p)
    return paths


def fig_event_study(did_res: dict) -> list[Path]:
    paths = []
    for event_name, r in did_res.items():
        if r.event_study.empty:
            continue
        es = r.event_study.sort_values("rel_year")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(es["rel_year"], es["ci_lo"], es["ci_hi"],
                        alpha=0.25, color="#2c7fb8")
        ax.plot(es["rel_year"], es["coef"], "o-", color="#2c7fb8",
                linewidth=2.0, markersize=6)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(-0.5, color="black", linestyle=":", linewidth=1.5,
                   label="Reform year (reference = t-1)")
        pre = es[es["rel_year"] < 0]
        if len(pre) > 0 and not pre["coef"].isna().all():
            ax.annotate("Pre-treatment coefficients\nshould cluster around 0",
                        xy=(pre["rel_year"].iloc[0], pre["coef"].iloc[0]),
                        xytext=(-7, 2),
                        arrowprops=dict(arrowstyle="->", color="gray"),
                        fontsize=9, color="gray")
        ax.set(title=f"Event study: {r.label}",
               xlabel="Years relative to reform", ylabel="ATT estimate (years of LE)")
        ax.legend()
        tag = event_name.lower().replace("_", "-")
        p = _save(fig, f"08_event_study_{tag}.png")
        paths.append(p)
    return paths


# ── Synthetic control ─────────────────────────────────────────────────────────

def fig_synthetic_control(synth_res: Any) -> Path:
    from ..analysis.causal import SYNTH_REFORM_YEAR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: actual vs synthetic
    all_years = synth_res.gaps.index
    gaps      = synth_res.gaps

    # Reconstruct synthetic from actual - gap
    df_plot = pd.read_csv(
        Path(__file__).resolve().parents[2] / "data" / "final" / "master_dataset.csv")
    actual = df_plot[df_plot["iso3"] == synth_res.treated_iso].sort_values("year")
    actual_vals = actual["life_expectancy"].values
    actual_yrs  = actual["year"].values
    synth_vals  = np.array([actual_vals[i] - float(gaps.get(yr, np.nan))
                             for i, yr in enumerate(actual_yrs)])

    ax1.plot(actual_yrs, actual_vals, "k-", linewidth=2.5, label="Actual China")
    ax1.plot(actual_yrs, synth_vals,  "r--", linewidth=2.0, label="Synthetic China")
    ax1.axvline(SYNTH_REFORM_YEAR, color="gray", linestyle=":", linewidth=1.5,
                label=f"Reform {SYNTH_REFORM_YEAR}")
    ax1.set(title="Synthetic Control: China vs Synthetic China",
            xlabel="Year", ylabel="Life expectancy (years)")
    ax1.legend()
    # Annotate top-weighted donors
    top_donors = sorted(synth_res.weights.items(), key=lambda x: -x[1])[:3]
    donor_str = "\n".join(f"{d}: {w:.2f}" for d, w in top_donors)
    ax1.text(0.02, 0.02, f"Top donors:\n{donor_str}",
             transform=ax1.transAxes, fontsize=8.5,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Panel B: gap plot with placebos
    if not synth_res.placebo_gaps.empty:
        for col in synth_res.placebo_gaps.columns:
            pg = synth_res.placebo_gaps[col]
            ax2.plot(pg.index, pg.values, color="gray", alpha=0.3, linewidth=0.8)
    ax2.plot(gaps.index, gaps.values, "k-", linewidth=2.5, label="China (treated)")
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.axvline(SYNTH_REFORM_YEAR, color="gray", linestyle=":", linewidth=1.5)
    ax2.set(title=f"Gap (Actual − Synthetic): China vs donor placebos\n(p = {synth_res.p_value})",
            xlabel="Year", ylabel="Gap in life expectancy (years)")
    ax2.legend()

    plt.tight_layout()
    return _save(fig, "09_synthetic_control.png")


# ── Robustness ────────────────────────────────────────────────────────────────

def fig_robustness(robust_res: dict, main_panel: Any) -> Path:
    """Coefficient plot across robustness checks vs main spec."""
    rows: list[dict] = []
    rows.append({"label": "Main spec\n(controls_full)", "coef": main_panel.coef_gdp,
                 "lo": main_panel.ci95_gdp[0], "hi": main_panel.ci95_gdp[1]})

    for spec_name, spec_res in robust_res.items():
        if spec_name == "pooled_ols":
            c = spec_res["coef_gdp"]
            se = spec_res["se_gdp"]
            rows.append({"label": "Pooled OLS\n(no FE)", "coef": c,
                         "lo": c - 1.96 * se, "hi": c + 1.96 * se})
        elif isinstance(spec_res, dict):
            # dict of PanelFEResult, pick controls_full
            r = spec_res.get("controls_full") or spec_res.get("controls_base")
            if r:
                rows.append({"label": spec_name.replace("_", "\n"), "coef": r.coef_gdp,
                             "lo": r.ci95_gdp[0], "hi": r.ci95_gdp[1]})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        col = "#2c7fb8" if i == 0 else "#888888"
        ax.plot([row["lo"], row["hi"]], [i, i], color=col, linewidth=2.5)
        ax.scatter(row["coef"], i, color=col, s=90, zorder=4)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set(title="Robustness checks: coefficient on log GDP per capita",
           xlabel="Coefficient (years of LE per log-unit GDP, 95% CI)")
    return _save(fig, "10_robustness.png")


# ── Summary / Triangulation ───────────────────────────────────────────────────

def fig_synthesis_forest(synthesis: dict) -> Path:
    """Forest plot of all causal estimates."""
    df = synthesis["coef_table"].copy()
    df = df[df["coef"].notna()].reset_index(drop=True)
    colors_list = []
    for m in df["method"]:
        if "Panel" in m:
            colors_list.append("#2c7fb8")
        elif "IV" in m:
            colors_list.append("#f03b20")
        elif "DiD" in m:
            colors_list.append("#1a9641")
        else:
            colors_list.append("#984ea3")

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, (_, row) in enumerate(df.iterrows()):
        col = colors_list[i]
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                color=col, linewidth=2.5, alpha=0.8)
        ax.scatter(row["coef"], i, color=col, s=100, zorder=5,
                   edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["method"], fontsize=9)
    ax.set(title="Triangulation: all causal estimates (GDP effect on life expectancy)\n"
                 "coefficient = years gained per log-unit of GDP per capita (PPP)",
           xlabel="Effect estimate (years of LE, 95% CI where available)")
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=2.5, label=l)
        for c, l in [("#2c7fb8","Panel FE"),("#f03b20","IV-2SLS"),
                      ("#1a9641","DiD"),("#984ea3","Synthetic Control")]
    ]
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    return _save(fig, "11_synthesis_forest.png")


def fig_mechanism_path(df: pd.DataFrame) -> Path:
    """Mediation intuition: GDP → health spending → life expectancy paths."""
    latest = df[df["year"] >= df["year"].max() - 4].groupby("iso3").agg({
        "gdp_per_capita_ppp": "mean", "health_exp_per_capita": "mean",
        "life_expectancy": "mean", "income_group": "first",
    }).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, (x_col, title, xlabel) in zip(axes, [
        ("log_gdp_per_capita_ppp", "GDP → Health spending",
         "log GDP per capita, PPP"),
        ("health_exp_per_capita",  "Health spending → Life expectancy",
         "Health expenditure per capita (USD)"),
    ]):
        if x_col == "log_gdp_per_capita_ppp":
            latest["log_gdp_per_capita_ppp"] = np.log(latest["gdp_per_capita_ppp"])
            y_col = "health_exp_per_capita"
            y_label = "Health exp. per capita (USD)"
        else:
            y_col = "life_expectancy"
            y_label = "Life expectancy (years)"
        sns.scatterplot(data=latest, x=x_col, y=y_col, hue="income_group",
                        palette={"high": "#2c7fb8", "middle": "#fdae61",
                                 "low": "#d7191c"},
                        s=110, edgecolor="black", ax=ax)
        for _, r in latest.iterrows():
            ax.annotate(r["iso3"], (r[x_col], r[y_col]),
                        fontsize=7.5, alpha=0.7, xytext=(3, 2),
                        textcoords="offset points")
        b = np.polyfit(latest[x_col].dropna(), latest[y_col].dropna(), 1)
        xl = np.linspace(latest[x_col].min(), latest[x_col].max(), 100)
        ax.plot(xl, np.polyval(b, xl), "k--", linewidth=1.2)
        ax.set(title=title, xlabel=xlabel, ylabel=y_label)
    plt.tight_layout()
    return _save(fig, "12_mechanism_paths.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all_causal_plots(results: dict, df: pd.DataFrame) -> list[Path]:
    import warnings; warnings.filterwarnings("ignore")
    paths: list[Path] = []

    paths.append(fig_granger_heatmap(results["granger"]))
    paths.append(fig_granger_lag_dist(results["granger"]))
    paths.append(fig_panel_coef_plot(results["panel_fe"],
                                     results["panel_fe_subgroups"]))
    paths.append(fig_fe_residuals(df))
    paths.append(fig_iv_first_stage(df))
    paths.append(fig_iv_comparison(results["iv"], results["panel_fe"]))
    paths.extend(fig_did_parallel_trends(results["did"], df))
    paths.extend(fig_event_study(results["did"]))
    paths.append(fig_synthetic_control(results["synth"]))
    main_panel = results["panel_fe"].get("controls_full")
    if main_panel:
        paths.append(fig_robustness(results["robustness"], main_panel))
    paths.append(fig_synthesis_forest(results["synthesis"]))
    paths.append(fig_mechanism_path(df))

    logger.info("Generated %d causal figures in %s", len(paths), CAUSAL_FIG_DIR)
    return paths
