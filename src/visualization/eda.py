"""Exploratory data analysis: produce 15+ publication-quality figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.config import (
    COUNTRIES, EDA_FIG_DIR, FINAL_DIR, INCOME_GROUP, PRIMARY_OUTCOME,
    PRIMARY_TREATMENT, VAR_GROUPS,
)
from ..utils.logging_setup import get_logger

logger = get_logger("viz.eda")

INCOME_PALETTE = {"high": "#2c7fb8", "middle": "#fdae61", "low": "#d7191c"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> Path:
    path = EDA_FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def fig_le_trend(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    grp = df.groupby(["year", "income_group"])["life_expectancy"].mean().unstack()
    for ig in ["high", "middle", "low"]:
        if ig in grp.columns:
            ax.plot(grp.index, grp[ig], marker="o", label=ig.capitalize(),
                    color=INCOME_PALETTE[ig], linewidth=2)
    ax.set(title="Life expectancy trend by income group (2000-2024)",
           xlabel="Year", ylabel="Life expectancy (years)")
    ax.legend(title="Income group")
    return _save(fig, "01_life_expectancy_trend.png")


def fig_gdp_trend(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    grp = df.groupby(["year", "income_group"])["gdp_per_capita_ppp"].mean().unstack()
    for ig in ["high", "middle", "low"]:
        if ig in grp.columns:
            ax.plot(grp.index, grp[ig], marker="o", label=ig.capitalize(),
                    color=INCOME_PALETTE[ig], linewidth=2)
    ax.set(title="GDP per capita (PPP) by income group",
           xlabel="Year", ylabel="GDP per capita, PPP (USD)", yscale="log")
    ax.legend(title="Income group")
    return _save(fig, "02_gdp_trend.png")


def fig_preston_curve(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    latest = df[df["year"] == df["year"].max()]
    sns.scatterplot(data=latest, x="gdp_per_capita_ppp", y="life_expectancy",
                    hue="income_group", palette=INCOME_PALETTE, s=110,
                    edgecolor="black", ax=ax)
    for _, r in latest.iterrows():
        ax.annotate(r["iso3"], (r["gdp_per_capita_ppp"], r["life_expectancy"]),
                    fontsize=8, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set(title=f"Preston curve, {int(latest['year'].iloc[0])}",
           xlabel="GDP per capita, PPP (USD, log)", ylabel="Life expectancy (years)",
           xscale="log")
    return _save(fig, "03_preston_curve.png")


def fig_country_trajectories(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    for iso, sub in df.sort_values("year").groupby("iso3"):
        ig = sub["income_group"].iloc[0]
        ax.plot(sub["gdp_per_capita_ppp"], sub["life_expectancy"],
                color=INCOME_PALETTE[ig], alpha=0.55, linewidth=1.4)
        end = sub.iloc[-1]
        ax.scatter(end["gdp_per_capita_ppp"], end["life_expectancy"],
                   color=INCOME_PALETTE[ig], s=22, zorder=3)
    handles = [plt.Line2D([0], [0], color=c, label=ig.capitalize())
               for ig, c in INCOME_PALETTE.items()]
    ax.legend(handles=handles, title="Income group")
    ax.set(title="Country trajectories: GDP vs life expectancy (2000-2024)",
           xlabel="GDP per capita, PPP (USD, log)", ylabel="Life expectancy (years)",
           xscale="log")
    return _save(fig, "04_country_trajectories.png")


def fig_correlation_heatmap(df: pd.DataFrame) -> Path:
    cols = [c for c in [
        "life_expectancy", "gdp_per_capita_ppp", "health_exp_pct_gdp",
        "health_exp_per_capita", "physicians_per_1000", "hospital_beds_per_1000",
        "education_exp_pct_gdp", "literacy_adult", "urban_pop_pct",
        "fertility_rate", "infant_mortality", "under5_mortality",
        "wgi_gov_effectiveness", "water_access", "sanitation_access",
    ] if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix — headline indicators")
    return _save(fig, "05_correlation_heatmap.png")


def fig_missingness_heatmap(df: pd.DataFrame) -> Path:
    feature_cols = [c for c in df.columns
                    if c not in {"iso3", "country", "year", "income_group"}]
    miss = (df.groupby("iso3")[feature_cols].apply(lambda g: g.isna().mean()) * 100)
    miss = miss.loc[:, (miss.mean(0) > 0)]
    if miss.empty:
        miss = pd.DataFrame({"all_complete": [0]}, index=df["iso3"].unique())
    fig, ax = plt.subplots(figsize=(min(22, 0.25 * miss.shape[1] + 5),
                                    0.3 * miss.shape[0] + 2))
    sns.heatmap(miss, cmap="Reds", vmin=0, vmax=100,
                cbar_kws={"label": "% missing (post-imputation)"}, ax=ax)
    ax.set(title="Residual missingness by country × indicator",
           xlabel="Indicator", ylabel="Country (ISO-3)")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=8)
    return _save(fig, "06_missingness_heatmap.png")


def fig_le_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for ig in ["high", "middle", "low"]:
        sub = df[df["income_group"] == ig]["life_expectancy"].dropna()
        ax.hist(sub, bins=30, alpha=0.6, label=ig.capitalize(),
                color=INCOME_PALETTE[ig])
    ax.legend(title="Income group")
    ax.set(title="Distribution of life expectancy by income group",
           xlabel="Life expectancy (years)", ylabel="Count")
    return _save(fig, "07_le_distribution.png")


def fig_gdp_growth_box(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="income_group", y="gdp_growth",
                hue="income_group", order=["low", "middle", "high"],
                palette=INCOME_PALETTE, legend=False, ax=ax)
    ax.set(title="GDP growth distribution by income group",
           xlabel="Income group", ylabel="GDP growth (%)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    return _save(fig, "08_gdp_growth_box.png")


def fig_health_spend_le(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    latest = df[df["year"] >= df["year"].max() - 4]  # last-5-yr average
    avg = (latest.groupby("iso3")[["health_exp_per_capita", "life_expectancy",
                                    "income_group"]]
                  .agg({"health_exp_per_capita": "mean", "life_expectancy": "mean",
                        "income_group": "first"})
                  .reset_index())
    sns.scatterplot(data=avg, x="health_exp_per_capita", y="life_expectancy",
                    hue="income_group", palette=INCOME_PALETTE, s=110,
                    edgecolor="black", ax=ax)
    for _, r in avg.iterrows():
        ax.annotate(r["iso3"], (r["health_exp_per_capita"], r["life_expectancy"]),
                    fontsize=8, alpha=0.75, xytext=(3, 3), textcoords="offset points")
    ax.set(title="Health spending vs life expectancy (last 5-yr mean)",
           xlabel="Health expenditure per capita (USD, log)",
           ylabel="Life expectancy (years)", xscale="log")
    return _save(fig, "09_health_spend_le.png")


def fig_infant_mortality_trend(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    grp = df.groupby(["year", "income_group"])["infant_mortality"].mean().unstack()
    for ig in ["high", "middle", "low"]:
        if ig in grp.columns:
            ax.plot(grp.index, grp[ig], marker="o", label=ig.capitalize(),
                    color=INCOME_PALETTE[ig], linewidth=2)
    ax.set(title="Infant mortality trend by income group",
           xlabel="Year", ylabel="Deaths per 1000 live births")
    ax.legend(title="Income group")
    return _save(fig, "10_infant_mortality.png")


def fig_le_panel_by_country(df: pd.DataFrame) -> Path:
    g = sns.FacetGrid(df, col="iso3", col_wrap=6, height=2.0, sharey=False)
    g.map_dataframe(sns.lineplot, x="year", y="life_expectancy")
    g.set_titles("{col_name}")
    g.fig.suptitle("Life expectancy trajectories — all 29 countries",
                   y=1.02, fontsize=14)
    fig = g.fig
    return _save(fig, "11_le_per_country.png")


def fig_gdp_le_growth(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    g = (df.dropna(subset=["gdp_per_capita_ppp", "life_expectancy"])
            .sort_values(["iso3", "year"])
            .assign(le_change=lambda x: x.groupby("iso3")["life_expectancy"].diff(),
                    gdp_change=lambda x: x.groupby("iso3")["gdp_per_capita_ppp"].pct_change() * 100))
    sns.scatterplot(data=g.dropna(), x="gdp_change", y="le_change",
                    hue="income_group", palette=INCOME_PALETTE, alpha=0.5, s=22, ax=ax)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
    ax.set(title="Year-over-year change: GDP growth vs Δ life expectancy",
           xlabel="GDP per capita YoY change (%)",
           ylabel="Δ life expectancy (years)")
    ax.set_xlim(-25, 25)
    return _save(fig, "12_gdp_vs_le_change.png")


def fig_governance_le(df: pd.DataFrame) -> Path:
    if "wgi_gov_effectiveness" not in df.columns:
        return EDA_FIG_DIR / "skipped"
    fig, ax = plt.subplots(figsize=(10, 6))
    avg = (df.groupby("iso3")[["wgi_gov_effectiveness", "life_expectancy", "income_group"]]
              .agg({"wgi_gov_effectiveness": "mean", "life_expectancy": "mean",
                    "income_group": "first"}).reset_index())
    sns.regplot(data=avg, x="wgi_gov_effectiveness", y="life_expectancy",
                scatter=False, color="gray", ax=ax)
    sns.scatterplot(data=avg, x="wgi_gov_effectiveness", y="life_expectancy",
                    hue="income_group", palette=INCOME_PALETTE, s=110,
                    edgecolor="black", ax=ax)
    for _, r in avg.iterrows():
        ax.annotate(r["iso3"], (r["wgi_gov_effectiveness"], r["life_expectancy"]),
                    fontsize=8, alpha=0.75, xytext=(3, 3), textcoords="offset points")
    ax.set(title="Government effectiveness vs life expectancy (country means)",
           xlabel="WGI Government Effectiveness (z-score)",
           ylabel="Life expectancy (years)")
    return _save(fig, "13_governance_le.png")


def fig_education_le(df: pd.DataFrame) -> Path:
    col = "undp_mys" if "undp_mys" in df.columns else "education_exp_pct_gdp"
    fig, ax = plt.subplots(figsize=(10, 6))
    avg = (df.groupby("iso3")[[col, "life_expectancy", "income_group"]]
              .agg({col: "mean", "life_expectancy": "mean", "income_group": "first"})
              .reset_index())
    sns.scatterplot(data=avg, x=col, y="life_expectancy", hue="income_group",
                    palette=INCOME_PALETTE, s=110, edgecolor="black", ax=ax)
    sns.regplot(data=avg, x=col, y="life_expectancy", scatter=False,
                color="gray", ax=ax)
    for _, r in avg.iterrows():
        ax.annotate(r["iso3"], (r[col], r["life_expectancy"]),
                    fontsize=8, alpha=0.75, xytext=(3, 3), textcoords="offset points")
    label = "Mean years of schooling" if col == "undp_mys" else "Education expenditure (% GDP)"
    ax.set(title=f"{label} vs life expectancy", xlabel=label,
           ylabel="Life expectancy (years)")
    return _save(fig, "14_education_le.png")


def fig_covid_impact(df: pd.DataFrame) -> Path:
    """Δ life expectancy 2019→2021 vs COVID deaths per million."""
    if "covid_deaths_per_million" not in df.columns:
        return EDA_FIG_DIR / "skipped"
    le_2019 = df[df["year"] == 2019].set_index("iso3")["life_expectancy"]
    le_2021 = df[df["year"] == 2021].set_index("iso3")["life_expectancy"]
    deaths = (df[df["year"].isin([2020, 2021])]
                .groupby("iso3")["covid_deaths_per_million"].max())
    impact = pd.DataFrame({
        "le_change": le_2021 - le_2019,
        "covid_deaths_per_million": deaths,
        "income_group": df.drop_duplicates("iso3").set_index("iso3")["income_group"],
    }).dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=impact, x="covid_deaths_per_million", y="le_change",
                    hue="income_group", palette=INCOME_PALETTE, s=110,
                    edgecolor="black", ax=ax)
    for iso, r in impact.iterrows():
        ax.annotate(iso, (r["covid_deaths_per_million"], r["le_change"]),
                    fontsize=8, alpha=0.75, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.set(title="COVID impact: 2019→2021 life-expectancy change vs cumulative deaths",
           xlabel="COVID deaths per million (max 2020-2021)",
           ylabel="Δ Life expectancy 2019→2021 (years)")
    return _save(fig, "15_covid_impact.png")


def fig_summary_stats_table(df: pd.DataFrame) -> Path:
    """Summary statistics table rendered as a figure for the EDA report."""
    cols = ["life_expectancy", "gdp_per_capita_ppp", "health_exp_pct_gdp",
            "infant_mortality", "fertility_rate", "urban_pop_pct"]
    cols = [c for c in cols if c in df.columns]
    desc = df[cols].describe().round(2).T
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(cols) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=desc.values, rowLabels=desc.index, colLabels=desc.columns,
                   loc="center", cellLoc="right")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title("Summary statistics — headline indicators")
    return _save(fig, "16_summary_stats_table.png")


def fig_pairplot(df: pd.DataFrame) -> Path:
    cols = [c for c in ["life_expectancy", "gdp_per_capita_ppp",
                        "health_exp_per_capita", "infant_mortality",
                        "fertility_rate", "urban_pop_pct"] if c in df.columns]
    sub = df[cols + ["income_group"]].dropna().sample(min(500, len(df)),
                                                       random_state=42)
    g = sns.pairplot(sub, hue="income_group", palette=INCOME_PALETTE,
                     corner=True, diag_kind="kde", plot_kws={"alpha": 0.5, "s": 18})
    g.fig.suptitle("Pairwise relationships among headline indicators",
                   y=1.02, fontsize=13)
    return _save(g.fig, "17_pairplot.png")


def fig_le_change_distribution(df: pd.DataFrame) -> Path:
    g = (df.sort_values(["iso3", "year"])
            .assign(le_change=lambda x: x.groupby("iso3")["life_expectancy"].diff()))
    fig, ax = plt.subplots(figsize=(10, 6))
    for ig in ["high", "middle", "low"]:
        sub = g[g["income_group"] == ig]["le_change"].dropna()
        ax.hist(sub, bins=40, alpha=0.55, label=ig.capitalize(),
                color=INCOME_PALETTE[ig])
    ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
    ax.set(title="Year-over-year change in life expectancy (distribution)",
           xlabel="Δ Life expectancy (years)", ylabel="Count")
    ax.legend(title="Income group")
    return _save(fig, "18_le_change_distribution.png")


def fig_top_correlates(df: pd.DataFrame) -> Path:
    feature_cols = [c for c in df.columns
                    if c not in {"iso3", "country", "year", "income_group"}
                    and c != PRIMARY_OUTCOME
                    and not any(s in c for s in ("_lag", "__x__"))]
    corrs = df[feature_cols].apply(
        lambda c: c.corr(df[PRIMARY_OUTCOME]) if c.dtype.kind in "fi" else np.nan)
    corrs = corrs.dropna().sort_values()
    top = pd.concat([corrs.head(10), corrs.tail(10)])
    fig, ax = plt.subplots(figsize=(10, 8))
    top.plot(kind="barh", ax=ax,
             color=["#d7191c" if v < 0 else "#2c7fb8" for v in top.values])
    ax.set(title="Top 20 correlates of life expectancy (Pearson r)",
           xlabel="Correlation with life expectancy")
    return _save(fig, "19_top_correlates.png")


def run_all(master_path: Path | None = None) -> list[Path]:
    path = master_path or (FINAL_DIR / "master_dataset.csv")
    df = pd.read_csv(path)
    figs = [
        fig_le_trend(df), fig_gdp_trend(df), fig_preston_curve(df),
        fig_country_trajectories(df), fig_correlation_heatmap(df),
        fig_missingness_heatmap(df), fig_le_distribution(df),
        fig_gdp_growth_box(df), fig_health_spend_le(df),
        fig_infant_mortality_trend(df), fig_le_panel_by_country(df),
        fig_gdp_le_growth(df), fig_governance_le(df), fig_education_le(df),
        fig_covid_impact(df), fig_summary_stats_table(df), fig_pairplot(df),
        fig_le_change_distribution(df), fig_top_correlates(df),
    ]
    logger.info("Generated %d EDA figures in %s", len(figs), EDA_FIG_DIR)
    return figs


if __name__ == "__main__":
    run_all()
