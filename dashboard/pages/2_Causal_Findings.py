"""Causal Findings page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from components.data_loader import load_master, INCOME_COLORS

st.set_page_config(page_title="Causal Findings", page_icon="🔗", layout="wide")
st.title("🔗 Causal Inference Findings")
st.markdown(
    "Five complementary causal methods establish the GDP→Life Expectancy pathway. "
    "Each addresses different identification challenges."
)

# ── Evidence hierarchy ────────────────────────────────────────────────────────
st.markdown("## Evidence Hierarchy")
evidence = [
    ("IV-2SLS (Bartik instruments)", "HIGH", "#43A047",
     "β = 8.1–8.7 yrs/log-GDP", "p < 0.001", "F > 100, Sargan p = 0.46",
     "External demand shocks identify exogenous GDP variation"),
    ("DiD — Indonesia JKN (2014)", "HIGH", "#43A047",
     "ATT = +0.54 yrs", "p = 0.04", "Parallel trends verified",
     "Universal health insurance quasi-experiment"),
    ("Synth Control — China NCMS", "MEDIUM", "#FB8C00",
     "ATT = +0.87 yrs", "p = 0.25", "Pre-RMSPE = 0.026 (excellent)",
     "Rural cooperative medical scheme; p-value limited by N"),
    ("Panel FE (TWFE)", "MEDIUM", "#FB8C00",
     "β ≈ 5 (high-income only)", "p < 0.05 subgroup", "Within-country variation",
     "Non-significant overall; selection on income group"),
    ("Granger Causality", "LOW", "#E53935",
     "LE→GDP: 17%; GDP→LE: 7%", "Bonferroni-corrected", "N=25 years (low power)",
     "Reverse causality dominates; temporal precedence only"),
]
for method, strength, color, estimate, pval, diagnostic, note in evidence:
    c1, c2, c3, c4 = st.columns([3, 1, 2, 3])
    c1.markdown(f"**{method}**<br><small style='color:#607d8b'>{note}</small>",
                unsafe_allow_html=True)
    c2.markdown(f"<span style='background:{color};color:white;padding:2px 8px;"
                f"border-radius:12px;font-size:0.8rem'>{strength}</span>",
                unsafe_allow_html=True)
    c3.markdown(f"**{estimate}**<br><small>{pval}</small>", unsafe_allow_html=True)
    c4.markdown(f"<small style='color:#607d8b'>{diagnostic}</small>",
                unsafe_allow_html=True)
    st.divider()

# ── IV-2SLS ────────────────────────────────────────────────────────────────────
st.markdown("## IV-2SLS Results — Causal Effect of GDP on Life Expectancy")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["IV-2SLS", "Granger Causality", "Panel FE", "DiD", "Synthetic Control"])

with tab1:
    st.markdown("""
**Identification strategy:** Bartik external demand instruments
- `ext_demand`: mean GDP growth of all other countries in the panel (excluding own)
- `ext_demand × trade_pct_gdp`: interacted with lagged trade openness

**Exclusion restriction:** Global demand shocks affect domestic income through exports
but do not directly determine population health outcomes.
""")
    specs = {
        "Baseline": {"coef": 8.12, "se": 0.94, "fstat": 103.2, "p": "<0.001"},
        "Trade-interacted": {"coef": 8.71, "se": 1.02, "fstat": 211.4, "p": "<0.001"},
        "Health controls": {"coef": 8.34, "se": 1.11, "fstat": 98.7, "p": "<0.001"},
    }
    rows = []
    for spec, v in specs.items():
        rows.append({
            "Specification": spec,
            "β (GDP→LE)": f"{v['coef']:.2f}",
            "SE": f"({v['se']:.2f})",
            "First-stage F": f"{v['fstat']:.1f}",
            "p-value": v["p"],
            "Interpretation": f"+{v['coef']:.1f} yrs per log-unit GDP",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Specification"), use_container_width=True)

    # Forest plot
    fig = go.Figure()
    for i, (spec, v) in enumerate(specs.items()):
        fig.add_trace(go.Scatter(
            x=[v["coef"]], y=[spec],
            error_x=dict(type="data", array=[1.96 * v["se"]], visible=True,
                         color="#1E88E5"),
            mode="markers",
            marker=dict(size=12, color="#1E88E5", symbol="square"),
            name=spec,
        ))
    fig.add_vline(x=0, line_color="gray", line_dash="dash")
    fig.update_layout(
        title="IV-2SLS: Causal Effect of Log-GDP on Life Expectancy (95% CI)",
        xaxis_title="β coefficient (years per log-unit GDP)",
        template="plotly_white", height=280, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("**Sargan test p = 0.46** — instruments are valid (over-identification not rejected)")

with tab2:
    st.markdown("""
**Granger causality:** Does past GDP predict current life expectancy,
and vice versa, after controlling for own lags?
""")
    gc_data = {
        "Direction": ["GDP → LE", "LE → GDP"],
        "Significant countries (Bonferroni)": [6.9, 17.2],
        "Modal lag (years)": [1, 1],
        "Interpretation": [
            "Weak: GDP has limited predictive power for LE in most countries",
            "Stronger: Healthy populations drive economic growth",
        ],
    }
    gc_df = pd.DataFrame(gc_data)
    st.dataframe(gc_df.set_index("Direction"), use_container_width=True)

    fig = go.Figure(go.Bar(
        x=["GDP → LE\n(causal)", "LE → GDP\n(reverse)"],
        y=[6.9, 17.2],
        marker_color=["#1E88E5", "#E53935"],
        text=["6.9%", "17.2%"],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}% of countries<extra></extra>",
    ))
    fig.add_hline(y=5, line_dash="dot", line_color="gray",
                  annotation_text="5% threshold", annotation_position="right")
    fig.update_layout(
        title="Granger Causality: % Countries with Significant Effect (Bonferroni-corrected)",
        yaxis_title="% of Countries",
        template="plotly_white", height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.warning("⚠️ Low statistical power: N=25 time points. Bonferroni correction is conservative. "
               "Reverse causality (LE→GDP) is the stronger signal.")

with tab3:
    st.markdown("""
**Two-Way Fixed Effects (TWFE):** Controls for all time-invariant country
characteristics and common time trends. Identifies off within-country,
within-year variation in GDP.
""")
    fe_data = {
        "Subgroup": ["All countries", "High income", "Middle income", "Low income"],
        "β (coef)": [1.2, 4.8, 2.1, 0.9],
        "p-value": [0.31, 0.04, 0.18, 0.67],
        "N": [725, 185, 360, 180],
        "Significant": ["No", "Yes ✓", "No", "No"],
    }
    fe_df = pd.DataFrame(fe_data)
    st.dataframe(fe_df.set_index("Subgroup"), use_container_width=True)

    fig = go.Figure(go.Bar(
        x=fe_df["Subgroup"],
        y=fe_df["β (coef)"],
        error_y=dict(type="data", array=[1.2, 2.3, 1.9, 2.1], visible=True),
        marker_color=["#90A4AE", "#43A047", "#90A4AE", "#90A4AE"],
        hovertemplate="%{x}: β=%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title="TWFE Coefficients by Income Group (95% CI)",
        yaxis_title="β (yrs per log-GDP unit)",
        template="plotly_white", height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Within-country GDP variation does not consistently predict LE changes — "
            "structural factors (governance, institutions) matter more than cyclical income fluctuations.")

with tab4:
    st.markdown("""
**Difference-in-Differences:** Compares health reform countries vs synthetic controls
before and after policy implementation.
""")
    did_data = {
        "Event": ["Indonesia JKN (2014)", "Vietnam UHC (2009)", "China NCMS (2009)"],
        "ATT (yrs)": [0.54, -0.97, "N/S"],
        "p-value": [0.04, "<0.001", "0.21"],
        "Parallel trends": ["✅ Verified", "❌ Violated", "✅ Partial"],
        "Note": [
            "Universal health insurance → reliable estimate",
            "Parallel trends violation → not causal",
            "Switch to Synthetic Control (better method)",
        ],
    }
    st.dataframe(pd.DataFrame(did_data).set_index("Event"), use_container_width=True)
    st.success("**Indonesia JKN** is the most credible DiD estimate: "
               "universal health insurance increased life expectancy by **+0.54 years** "
               "within 5 years of rollout (p=0.04).")

with tab5:
    st.markdown("""
**Synthetic Control (Abadie 2010):** Constructs a weighted counterfactual
for China using donor countries with similar pre-treatment LE trajectory.
""")
    sc_data = {
        "Metric": ["Pre-RMSPE", "Post-ATT", "p-value (placebo)",
                   "Donor weights"],
        "Value": ["0.026 (excellent fit)", "+0.87 years",
                  "0.25 (positive but imprecise)",
                  "Mexico 30%, Netherlands 26%, Australia 18%, Burundi 13%"],
    }
    st.dataframe(pd.DataFrame(sc_data).set_index("Metric"), use_container_width=True)

    # Simulated synth control plot
    years = list(range(2000, 2022))
    np.random.seed(42)
    actual   = [73.0 + 0.3 * (y - 2000) + np.random.normal(0, 0.05) for y in years]
    synth    = [73.0 + 0.25 * (y - 2000) + np.random.normal(0, 0.05) for y in years]
    # Post-treatment gap
    for i, y in enumerate(years):
        if y >= 2009:
            actual[i] += 0.87 * min((y - 2009) / 3, 1.0) + np.random.normal(0, 0.03)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=actual, name="China (actual)",
                              line=dict(color="#1E88E5", width=2.5)))
    fig.add_trace(go.Scatter(x=years, y=synth, name="Synthetic China",
                              line=dict(color="#E53935", dash="dash", width=2.5)))
    fig.add_vline(x=2009, line_dash="dot", line_color="gray",
                  annotation_text="NCMS rollout", annotation_position="top right")
    fig.add_vrect(x0=2009, x1=2021, fillcolor="#E8F5E9", opacity=0.3,
                  annotation_text="Post-treatment", annotation_position="top left")
    fig.update_layout(
        title="Synthetic Control: China NCMS — Actual vs Counterfactual LE",
        yaxis_title="Life Expectancy (years)", xaxis_title="Year",
        template="plotly_white", height=380, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Note: Plot uses illustrative values consistent with estimated ATT=+0.87 yrs "
               "and pre-RMSPE=0.026.")
