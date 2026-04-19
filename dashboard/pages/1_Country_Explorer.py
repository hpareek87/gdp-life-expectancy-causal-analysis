"""Country Explorer page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from components.data_loader import load_master, load_country_list, get_country_iso
from components.charts import country_trajectory, income_group_trajectories, scatter_gdp_le

st.set_page_config(page_title="Country Explorer", page_icon="🔍", layout="wide")
st.title("🔍 Country Explorer")
st.markdown("Explore 24-year trajectories for any country and compare with income-group peers.")

df = load_master()
countries = load_country_list(df)
iso_map   = get_country_iso(df)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    selected_country = st.selectbox("Select country", countries,
                                    index=countries.index("China") if "China" in countries else 0)
    compare_mode = st.checkbox("Compare with income-group peers", value=True)
    metrics_to_show = st.multiselect(
        "Metrics to display",
        ["life_expectancy", "gdp_per_capita_ppp", "health_exp_pct_gdp",
         "education_exp_pct_gdp", "fertility_rate"],
        default=["life_expectancy", "gdp_per_capita_ppp", "health_exp_pct_gdp"],
        format_func=lambda x: {
            "life_expectancy": "Life Expectancy",
            "gdp_per_capita_ppp": "GDP per Capita PPP",
            "health_exp_pct_gdp": "Health Spending % GDP",
            "education_exp_pct_gdp": "Education Spending % GDP",
            "fertility_rate": "Fertility Rate",
        }.get(x, x),
    )

iso = iso_map.get(selected_country, selected_country)
country_df = df[df["iso3"] == iso].sort_values("year")

if country_df.empty:
    st.error(f"No data found for {selected_country}")
    st.stop()

# ── Country header ─────────────────────────────────────────────────────────────
latest = country_df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Country", selected_country)
col2.metric("Latest LE",
            f"{latest.get('life_expectancy', 'N/A'):.1f} yrs"
            if pd.notna(latest.get("life_expectancy")) else "N/A",
            delta=f"{country_df['life_expectancy'].iloc[-1] - country_df['life_expectancy'].iloc[0]:.1f} vs 2000"
            if len(country_df) > 1 else None)
col3.metric("GDP per Capita",
            f"${latest.get('gdp_per_capita_ppp', 0):,.0f}"
            if pd.notna(latest.get("gdp_per_capita_ppp")) else "N/A")
col4.metric("Income Group",
            str(latest.get("income_group", "N/A"))
            if pd.notna(latest.get("income_group")) else "N/A")

st.divider()

# ── Trajectory plot ───────────────────────────────────────────────────────────
if metrics_to_show:
    label_map = {
        "life_expectancy":      "Life Expectancy (yrs)",
        "gdp_per_capita_ppp":   "GDP per Capita PPP (USD)",
        "health_exp_pct_gdp":   "Health Spending % GDP",
        "education_exp_pct_gdp":"Education Spending % GDP",
        "fertility_rate":       "Fertility Rate",
    }
    st.plotly_chart(
        country_trajectory(df, iso, metrics_to_show,
                           [label_map.get(m, m) for m in metrics_to_show]),
        use_container_width=True,
    )

# ── Income-group comparison ────────────────────────────────────────────────────
if compare_mode:
    st.markdown("### Income-Group Peer Comparison")
    income_grp = latest.get("income_group")
    if pd.notna(income_grp):
        peers = df[df["income_group"] == income_grp]

        # Avg trajectory of peers vs selected country
        peer_avg = (peers[peers["iso3"] != iso]
                    .groupby("year")["life_expectancy"]
                    .mean().reset_index())
        own = country_df[["year", "life_expectancy"]].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=peer_avg["year"], y=peer_avg["life_expectancy"],
            name=f"{income_grp} average",
            line=dict(dash="dash", color="#90A4AE", width=2),
            hovertemplate="Peers avg: %{y:.1f} yrs<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=own["year"], y=own["life_expectancy"],
            name=selected_country,
            line=dict(color="#1E88E5", width=3),
            hovertemplate=f"{selected_country}: %{{y:.1f}} yrs<extra></extra>",
        ))
        fig.add_vline(x=2019, line_dash="dot", line_color="gray",
                      annotation_text="Test period start")
        fig.update_layout(
            title=f"{selected_country} vs {income_grp} Peers — Life Expectancy",
            yaxis_title="Life Expectancy (years)",
            xaxis_title="Year",
            template="plotly_white",
            height=380,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

# ── GDP vs LE scatter highlight ────────────────────────────────────────────────
st.markdown("### GDP–Life Expectancy Position")
scatter_yr = st.slider("Year for scatter", 2000, 2024, 2023)
base_fig = scatter_gdp_le(df, scatter_yr)

# Highlight selected country
hl = df[(df["iso3"] == iso) & (df["year"] == scatter_yr)].dropna(
    subset=["gdp_per_capita_ppp", "life_expectancy"])
if not hl.empty:
    base_fig.add_trace(go.Scatter(
        x=hl["gdp_per_capita_ppp"], y=hl["life_expectancy"],
        mode="markers+text",
        marker=dict(size=18, color="#FF5722", symbol="star", line=dict(width=2, color="white")),
        text=[selected_country], textposition="top center",
        name=f"▶ {selected_country}",
        hovertemplate=f"{selected_country}<br>GDP: $%{{x:,.0f}}<br>LE: %{{y:.1f}} yrs<extra></extra>",
    ))
st.plotly_chart(base_fig, use_container_width=True)

# ── Data table ────────────────────────────────────────────────────────────────
with st.expander("📋 Raw data for this country"):
    show_cols = ["year", "life_expectancy", "gdp_per_capita_ppp",
                 "health_exp_pct_gdp", "education_exp_pct_gdp",
                 "fertility_rate", "urban_pop_pct"]
    show_cols = [c for c in show_cols if c in country_df.columns]
    st.dataframe(
        country_df[show_cols].set_index("year").round(2),
        use_container_width=True,
    )
