"""Reusable Plotly chart builders for the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INCOME_COLORS = {
    "Low income":          "#E53935",
    "Lower-middle income": "#FB8C00",
    "Upper-middle income": "#43A047",
    "High income":         "#1E88E5",
}

TEMPLATE = "plotly_white"


def world_map(df: pd.DataFrame, year: int,
              metric: str = "life_expectancy",
              title: str = "Life Expectancy") -> go.Figure:
    sub = df[df["year"] == year].dropna(subset=[metric, "iso3"])
    fig = px.choropleth(
        sub, locations="iso3",
        color=metric,
        hover_name="country",
        hover_data={metric: ":.1f", "gdp_per_capita_ppp": ":,.0f"},
        color_continuous_scale="RdYlGn",
        range_color=[sub[metric].quantile(0.05), sub[metric].quantile(0.95)],
        title=f"{title} ({year})",
        template=TEMPLATE,
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True,
                 projection_type="natural earth"),
        coloraxis_colorbar=dict(title=title, len=0.7),
        margin=dict(l=0, r=0, t=40, b=0),
        height=430,
    )
    return fig


def country_trajectory(df: pd.DataFrame, iso3: str,
                        metrics: list[str], labels: list[str]) -> go.Figure:
    sub = df[df["iso3"] == iso3].sort_values("year")
    country_name = sub["country"].iloc[0] if "country" in sub.columns else iso3

    n = len(metrics)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=labels, vertical_spacing=0.08)
    colors = ["#1E88E5", "#43A047", "#FB8C00", "#E53935"]
    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors), 1):
        if m not in sub.columns:
            continue
        ys = sub[m].dropna()
        xs = sub.loc[ys.index, "year"]
        fig.add_trace(go.Scatter(x=xs, y=ys, name=lbl,
                                  line=dict(color=col, width=2.5),
                                  mode="lines+markers",
                                  marker=dict(size=4)), row=i, col=1)
        fig.update_yaxes(title_text=lbl, row=i, col=1)

    fig.update_layout(
        title=f"{country_name} — 2000–2024 Trajectory",
        height=120 * n + 80,
        showlegend=False,
        template=TEMPLATE,
        hovermode="x unified",
    )
    return fig


def income_group_trajectories(df: pd.DataFrame, metric: str,
                               ylabel: str) -> go.Figure:
    grp = (df.groupby(["year", "income_group"])[metric]
             .mean().reset_index().dropna())
    fig = px.line(grp, x="year", y=metric, color="income_group",
                  color_discrete_map=INCOME_COLORS,
                  labels={"year": "Year", metric: ylabel,
                           "income_group": "Income Group"},
                  template=TEMPLATE)
    fig.update_traces(line_width=2.5)
    fig.update_layout(height=360, legend_title="Income Group",
                      hovermode="x unified")
    return fig


def scatter_gdp_le(df: pd.DataFrame, year: int | None = None) -> go.Figure:
    sub = df if year is None else df[df["year"] == year]
    sub = sub.dropna(subset=["gdp_per_capita_ppp", "life_expectancy"])
    fig = px.scatter(
        sub,
        x="gdp_per_capita_ppp", y="life_expectancy",
        color="income_group",
        color_discrete_map=INCOME_COLORS,
        hover_name="country" if "country" in sub.columns else "iso3",
        hover_data={"year": True, "gdp_per_capita_ppp": ":,.0f",
                    "life_expectancy": ":.1f"},
        log_x=True,
        labels={
            "gdp_per_capita_ppp": "GDP per Capita PPP (log scale)",
            "life_expectancy": "Life Expectancy (years)",
            "income_group": "Income Group",
        },
        template=TEMPLATE,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=420, hovermode="closest",
                      legend_title="Income Group")
    return fig


def feature_importance_bar(fi_df: pd.DataFrame, col: str = "xgb_shap",
                            top_n: int = 15, title: str = "") -> go.Figure:
    sub = fi_df[col].dropna().nlargest(top_n).sort_values()
    clean_names = [n.replace("_", " ").replace("  ", " ")
                   .replace("x x", "×") for n in sub.index]
    fig = go.Figure(go.Bar(
        x=sub.values, y=clean_names, orientation="h",
        marker_color="#1E88E5",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title or f"Top {top_n} Features — {col.upper()}",
        xaxis_title="Importance Score",
        height=max(350, top_n * 28),
        template=TEMPLATE,
        margin=dict(l=220, r=20),
    )
    return fig


def threshold_plot(df: pd.DataFrame, thresholds: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["log_gdp_per_capita_ppp", "life_expectancy"])
    X = sub["log_gdp_per_capita_ppp"].values
    y = sub["life_expectancy"].values
    gdp = np.exp(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gdp, y=y, mode="markers",
        marker=dict(size=5, color="steelblue", opacity=0.35),
        name="Observed", hovertemplate="GDP: $%{x:,.0f}<br>LE: %{y:.1f} yrs<extra></extra>",
    ))

    splits = [-np.inf] + sorted(thresholds["log_gdp_threshold"].tolist()) + [np.inf]
    seg_colors = ["#E91E63", "#FF9800", "#4CAF50", "#9C27B0"]
    for i, (lo, hi) in enumerate(zip(splits[:-1], splits[1:])):
        mask = (X >= lo) & (X < hi)
        if mask.sum() < 5:
            continue
        xs = np.linspace(X[mask].min(), X[mask].max(), 80)
        coef = np.polyfit(X[mask], y[mask], 1)
        ys = np.polyval(coef, xs)
        fig.add_trace(go.Scatter(
            x=np.exp(xs), y=ys, mode="lines",
            line=dict(color=seg_colors[i % len(seg_colors)], width=3),
            name=f"Segment {i+1}: β={coef[0]:.2f}",
        ))

    for _, row in thresholds.iterrows():
        gdp_val = np.exp(row["log_gdp_threshold"])
        stars = "***" if row["chow_p_value"] < 0.001 else ("**" if row["chow_p_value"] < 0.01 else "*")
        fig.add_vline(x=gdp_val, line_dash="dash", line_color="gray",
                      annotation_text=f"${gdp_val:,.0f} {stars}",
                      annotation_position="top right")

    fig.update_xaxes(type="log", title="GDP per Capita PPP (USD, log scale)")
    fig.update_yaxes(title="Life Expectancy (years)")
    fig.update_layout(
        title="GDP–Life Expectancy: Structural Break Analysis",
        height=460, template=TEMPLATE, hovermode="closest",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )
    return fig


def causal_bar(coefs: dict[str, float], errors: dict[str, float],
               title: str = "Causal Estimates") -> go.Figure:
    names = list(coefs.keys())
    vals  = [coefs[n] for n in names]
    errs  = [errors.get(n, 0) for n in names]
    colors = ["#43A047" if v > 0 else "#E53935" for v in vals]

    fig = go.Figure(go.Bar(
        x=names, y=vals,
        error_y=dict(type="data", array=errs, visible=True),
        marker_color=colors,
        hovertemplate="%{x}: β=%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=title, yaxis_title="Coefficient (years per log-GDP)",
        template=TEMPLATE, height=350,
    )
    return fig


def covid_recovery(df: pd.DataFrame, countries: list[str]) -> go.Figure:
    sub = df[df["iso3"].isin(countries) & (df["year"] >= 2018)].copy()
    fig = px.line(
        sub, x="year", y="life_expectancy", color="country",
        markers=True,
        labels={"life_expectancy": "Life Expectancy (years)", "year": "Year"},
        template=TEMPLATE,
        title="Life Expectancy Trajectories 2018–2024 (COVID period)",
    )
    fig.add_vline(x=2020, line_dash="dot", line_color="red",
                  annotation_text="COVID-19", annotation_position="top right")
    fig.update_layout(height=380, hovermode="x unified")
    return fig


def policy_simulator_gauge(predicted: float, baseline: float) -> go.Figure:
    delta = predicted - baseline
    sign = "+" if delta >= 0 else ""
    fig = go.Figure(go.Indicator(
        mode="number+delta+gauge",
        value=predicted,
        delta=dict(reference=baseline, valueformat=".2f",
                   increasing=dict(color="#43A047"),
                   decreasing=dict(color="#E53935")),
        number=dict(suffix=" yrs", valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[40, 90], ticksuffix=" yrs"),
            bar=dict(color="#1E88E5"),
            steps=[
                dict(range=[40, 60], color="#FFCDD2"),
                dict(range=[60, 75], color="#FFF9C4"),
                dict(range=[75, 90], color="#C8E6C9"),
            ],
            threshold=dict(line=dict(color="red", width=3),
                           thickness=0.75, value=baseline),
        ),
        title=dict(text=f"Predicted Life Expectancy<br><sup>Δ vs baseline: {sign}{delta:.2f} yrs</sup>"),
    ))
    fig.update_layout(height=280, template=TEMPLATE)
    return fig
