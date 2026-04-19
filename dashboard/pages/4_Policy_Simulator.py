"""Policy Simulator page — real-time predictions from trained XGBoost model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from components.data_loader import load_master, load_xgb_model, load_scaler, INCOME_COLORS
from components.charts import policy_simulator_gauge

st.set_page_config(page_title="Policy Simulator", page_icon="🎛️", layout="wide")
st.title("🎛️ Policy Simulator")
st.markdown(
    "Adjust economic and social determinants to predict life expectancy in real time. "
    "Powered by the trained XGBoost model (R² = 0.906 on 2019–2024 test set)."
)

df = load_master()
model = load_xgb_model()
scaler, feat_cols = load_scaler()

if model is None or scaler is None:
    st.error("⚠️ Trained model not found. Please run `run_all_ml()` first.")
    st.stop()

# ── Baseline selector ──────────────────────────────────────────────────────────
st.markdown("## 1. Choose a Baseline Country")
col_sel, col_yr = st.columns([3, 1])
with col_sel:
    country_options = sorted(df["country"].dropna().unique().tolist())
    baseline_country = st.selectbox("Baseline country", country_options,
                                     index=country_options.index("India")
                                     if "India" in country_options else 0)
with col_yr:
    baseline_year = st.selectbox("Baseline year", [2023, 2022, 2021, 2020, 2019, 2015, 2010])

iso_map = dict(zip(df["country"], df["iso3"]))
iso = iso_map.get(baseline_country, "")
baseline_row = df[(df["iso3"] == iso) & (df["year"] == baseline_year)]

if baseline_row.empty:
    st.warning(f"No data found for {baseline_country} in {baseline_year}. Using synthetic baseline.")
    baseline_vals: dict = {}
    baseline_le = 70.0
else:
    baseline_vals = baseline_row.iloc[0].to_dict()
    baseline_le = float(baseline_vals.get("life_expectancy", 70.0))

# Show baseline metrics
st.markdown(f"**{baseline_country} ({baseline_year}) baseline:**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Life Expectancy", f"{baseline_le:.1f} yrs")
c2.metric("GDP per Capita", f"${baseline_vals.get('gdp_per_capita_ppp', 0):,.0f}")
c3.metric("Health Spending", f"{baseline_vals.get('health_exp_pct_gdp', 0):.1f}% GDP")
c4.metric("Income Group", str(baseline_vals.get("income_group", "N/A")))

st.divider()

# ── Policy sliders ─────────────────────────────────────────────────────────────
st.markdown("## 2. Adjust Policy Variables")

col_sliders, col_results = st.columns([2, 1])

with col_sliders:
    st.markdown("**Economic factors**")
    gdp_mult = st.slider(
        "GDP growth multiplier (×current)",
        0.5, 3.0, 1.0, step=0.05,
        help="1.0 = no change; 2.0 = double GDP",
    )
    gdp_current = float(baseline_vals.get("gdp_per_capita_ppp", 10000))
    gdp_new     = gdp_current * gdp_mult

    st.markdown(f"*GDP: ${gdp_current:,.0f} → ${gdp_new:,.0f}*")

    st.markdown("**Health sector**")
    health_new = st.slider(
        "Health spending (% GDP)",
        1.0, 15.0,
        float(baseline_vals.get("health_exp_pct_gdp", 5.0)),
        step=0.1,
    )
    physicians_new = st.slider(
        "Physicians (per 1,000 pop.)",
        0.0, 8.0,
        min(float(baseline_vals.get("physicians_per_1000", 1.5)), 8.0),
        step=0.1,
    )
    sanitation_new = st.slider(
        "Sanitation access (%)",
        10.0, 100.0,
        min(float(baseline_vals.get("sanitation_access", 60.0)), 100.0),
        step=1.0,
    )

    st.markdown("**Education & governance**")
    edu_new = st.slider(
        "Education spending (% GDP)",
        1.0, 12.0,
        min(float(baseline_vals.get("education_exp_pct_gdp", 4.0)), 12.0),
        step=0.1,
    )
    gov_new = st.slider(
        "Governance effectiveness (WGI score, −2.5 to 2.5)",
        -2.5, 2.5,
        float(baseline_vals.get("wgi_gov_effectiveness", 0.0)),
        step=0.05,
    )


# ── Prediction logic ───────────────────────────────────────────────────────────
def build_feature_vector(baseline: dict, overrides: dict) -> np.ndarray:
    """Build feature array from baseline + overrides."""
    from src.analysis.ml_models import prepare_features, ALL_CANDIDATE_FEATURES
    # Fill from baseline
    feat_vals = {}
    for fc in feat_cols:
        feat_vals[fc] = float(baseline.get(fc, 0.0) or 0.0)

    # Override key features
    if "log_gdp_per_capita_ppp" in feat_cols:
        feat_vals["log_gdp_per_capita_ppp"] = np.log(max(overrides["gdp"], 100))
    if "gdp_per_capita_ppp" in feat_cols:
        feat_vals["gdp_per_capita_ppp"] = overrides["gdp"]
    if "health_exp_pct_gdp" in feat_cols:
        feat_vals["health_exp_pct_gdp"] = overrides["health"]
    if "physicians_per_1000" in feat_cols:
        feat_vals["physicians_per_1000"] = overrides["physicians"]
    if "sanitation_access" in feat_cols:
        feat_vals["sanitation_access"] = overrides["sanitation"]
    if "education_exp_pct_gdp" in feat_cols:
        feat_vals["education_exp_pct_gdp"] = overrides["edu"]
    if "wgi_gov_effectiveness" in feat_cols:
        feat_vals["wgi_gov_effectiveness"] = overrides["gov"]
    # Interaction features
    if "gdp_per_capita_ppp__x__education_exp_pct_gdp" in feat_cols:
        feat_vals["gdp_per_capita_ppp__x__education_exp_pct_gdp"] = (
            overrides["gdp"] * overrides["edu"])
    if "gdp_per_capita_ppp__x__wgi_gov_effectiveness" in feat_cols:
        feat_vals["gdp_per_capita_ppp__x__wgi_gov_effectiveness"] = (
            overrides["gdp"] * overrides["gov"])
    if "health_exp_pct_gdp__x__physicians_per_1000" in feat_cols:
        feat_vals["health_exp_pct_gdp__x__physicians_per_1000"] = (
            overrides["health"] * overrides["physicians"])
    if "gdp_x_health" in feat_cols:
        feat_vals["gdp_x_health"] = overrides["gdp"] * overrides["health"]
    if "log_gdp_sq" in feat_cols:
        log_g = np.log(max(overrides["gdp"], 100))
        feat_vals["log_gdp_sq"] = log_g ** 2
    if "log_gdp_cu" in feat_cols:
        log_g = np.log(max(overrides["gdp"], 100))
        feat_vals["log_gdp_cu"] = log_g ** 3

    row = np.array([[feat_vals.get(fc, 0.0) for fc in feat_cols]])
    row_scaled = scaler.transform(row)
    return row_scaled


overrides_policy = {
    "gdp": gdp_new,
    "health": health_new,
    "physicians": physicians_new,
    "sanitation": sanitation_new,
    "edu": edu_new,
    "gov": gov_new,
}

try:
    row_policy = build_feature_vector(baseline_vals, overrides_policy)
    pred_le = float(model.predict(row_policy)[0])

    # Baseline prediction (from baseline values without changes)
    overrides_base = {k: v for k, v in overrides_policy.items()}
    overrides_base["gdp"] = gdp_current
    overrides_base["health"] = float(baseline_vals.get("health_exp_pct_gdp", health_new))
    overrides_base["physicians"] = float(baseline_vals.get("physicians_per_1000", physicians_new))
    overrides_base["sanitation"] = float(baseline_vals.get("sanitation_access", sanitation_new))
    overrides_base["edu"] = float(baseline_vals.get("education_exp_pct_gdp", edu_new))
    overrides_base["gov"] = float(baseline_vals.get("wgi_gov_effectiveness", gov_new))
    row_base = build_feature_vector(baseline_vals, overrides_base)
    pred_baseline_le = float(model.predict(row_base)[0])
    prediction_ok = True
except Exception as e:
    pred_le = baseline_le
    pred_baseline_le = baseline_le
    prediction_ok = False
    col_results.error(f"Prediction error: {e}")

with col_results:
    st.markdown("**Predicted Life Expectancy**")
    st.plotly_chart(
        policy_simulator_gauge(pred_le, pred_baseline_le),
        use_container_width=True,
    )
    delta = pred_le - pred_baseline_le
    actual_delta = pred_le - baseline_le
    sign = "+" if delta >= 0 else ""
    st.metric("vs model baseline", f"{sign}{delta:.2f} yrs")
    st.metric("vs actual baseline", f"{'+' if actual_delta >= 0 else ''}{actual_delta:.2f} yrs")
    if not prediction_ok:
        st.warning("Using fallback estimate.")

st.divider()

# ── Scenario comparison ────────────────────────────────────────────────────────
st.markdown("## 3. Scenario Comparison")
st.markdown("Compare multiple policy scenarios side by side.")

scenarios = {
    "Current (baseline)": {
        "gdp": gdp_current,
        "health": float(baseline_vals.get("health_exp_pct_gdp", 5)),
        "physicians": float(baseline_vals.get("physicians_per_1000", 1.5)),
        "sanitation": float(baseline_vals.get("sanitation_access", 60)),
        "edu": float(baseline_vals.get("education_exp_pct_gdp", 4)),
        "gov": float(baseline_vals.get("wgi_gov_effectiveness", 0)),
    },
    "Double GDP only": {
        "gdp": gdp_current * 2,
        "health": float(baseline_vals.get("health_exp_pct_gdp", 5)),
        "physicians": float(baseline_vals.get("physicians_per_1000", 1.5)),
        "sanitation": float(baseline_vals.get("sanitation_access", 60)),
        "edu": float(baseline_vals.get("education_exp_pct_gdp", 4)),
        "gov": float(baseline_vals.get("wgi_gov_effectiveness", 0)),
    },
    "Double health spending": {
        "gdp": gdp_current,
        "health": min(float(baseline_vals.get("health_exp_pct_gdp", 5)) * 2, 15),
        "physicians": float(baseline_vals.get("physicians_per_1000", 1.5)),
        "sanitation": float(baseline_vals.get("sanitation_access", 60)),
        "edu": float(baseline_vals.get("education_exp_pct_gdp", 4)),
        "gov": float(baseline_vals.get("wgi_gov_effectiveness", 0)),
    },
    "Universal sanitation": {
        "gdp": gdp_current,
        "health": float(baseline_vals.get("health_exp_pct_gdp", 5)),
        "physicians": float(baseline_vals.get("physicians_per_1000", 1.5)),
        "sanitation": 95.0,
        "edu": float(baseline_vals.get("education_exp_pct_gdp", 4)),
        "gov": float(baseline_vals.get("wgi_gov_effectiveness", 0)),
    },
    "Package: GDP+Health+Education": {
        "gdp": gdp_current * 1.5,
        "health": min(float(baseline_vals.get("health_exp_pct_gdp", 5)) * 1.5, 15),
        "physicians": float(baseline_vals.get("physicians_per_1000", 1.5)) * 1.3,
        "sanitation": min(float(baseline_vals.get("sanitation_access", 60)) + 20, 100),
        "edu": min(float(baseline_vals.get("education_exp_pct_gdp", 4)) * 1.3, 12),
        "gov": float(baseline_vals.get("wgi_gov_effectiveness", 0)) + 0.3,
    },
    "Your scenario": overrides_policy,
}

scenario_preds: list[dict] = []
for scen_name, scen_vals in scenarios.items():
    try:
        rv = build_feature_vector(baseline_vals, scen_vals)
        le_pred = float(model.predict(rv)[0])
    except Exception:
        le_pred = baseline_le
    scenario_preds.append({
        "Scenario": scen_name,
        "Predicted LE": round(le_pred, 2),
        "Δ vs actual baseline": round(le_pred - baseline_le, 2),
        "Δ vs model baseline": round(le_pred - pred_baseline_le, 2),
    })

scen_df = pd.DataFrame(scenario_preds).set_index("Scenario")

col_scen_chart, col_scen_table = st.columns([3, 2])
with col_scen_chart:
    fig_scen = go.Figure(go.Bar(
        x=scen_df.index,
        y=scen_df["Predicted LE"],
        marker_color=[
            "#90A4AE", "#1E88E5", "#43A047", "#FB8C00", "#9C27B0", "#FF5722"
        ][:len(scen_df)],
        text=scen_df["Predicted LE"].round(1).astype(str) + " yrs",
        textposition="outside",
        hovertemplate="%{x}<br>Predicted LE: %{y:.1f} yrs<extra></extra>",
    ))
    fig_scen.add_hline(y=baseline_le, line_dash="dot", line_color="red",
                       annotation_text=f"Actual baseline: {baseline_le:.1f} yrs")
    fig_scen.update_layout(
        title=f"Scenario Comparison — {baseline_country} ({baseline_year})",
        yaxis_title="Predicted Life Expectancy (years)",
        template="plotly_white", height=380,
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig_scen, use_container_width=True)

with col_scen_table:
    st.dataframe(scen_df.style.background_gradient(subset=["Δ vs actual baseline"],
                                                    cmap="RdYlGn"),
                 use_container_width=True)

st.divider()

# ── Policy ROI ─────────────────────────────────────────────────────────────────
st.markdown("## 4. Policy ROI Estimates (Based on IV-2SLS Causal Effect)")
st.markdown("""
The IV-2SLS estimate (β ≈ 8.1 years per log-unit GDP) provides the most credible
**causal** multiplier. The table below translates interventions into life-year gains.
""")

roi_data = {
    "Policy intervention": [
        "Double GDP (100% income growth)",
        "+50% GDP",
        "+20% GDP",
        "Increase health spending 5%→10% GDP",
        "Universal sanitation (reach 100%)",
        "Universal health insurance (DiD estimate)",
        "China-style rural health scheme (Synth estimate)",
    ],
    "Estimated LE gain (yrs)": [5.6, 3.3, 1.5, "1–3 (indirect)", "+1–2", "+0.54", "+0.87"],
    "Evidence source": [
        "IV-2SLS (β=8.1 × ln(2))",
        "IV-2SLS (β=8.1 × ln(1.5))",
        "IV-2SLS (β=8.1 × ln(1.2))",
        "SHAP partial dependence",
        "SHAP partial dependence",
        "DiD — Indonesia JKN (2014)",
        "Synthetic Control — China NCMS (2009)",
    ],
    "Time horizon": ["5–15 yrs", "5–15 yrs", "3–8 yrs", "3–10 yrs",
                     "2–5 yrs", "2–5 yrs", "3–8 yrs"],
    "Confidence": ["High (IV)", "High (IV)", "High (IV)",
                   "Medium (ML)", "Medium (ML)", "High (DiD)", "Medium (Synth)"],
}
st.dataframe(pd.DataFrame(roi_data).set_index("Policy intervention"),
             use_container_width=True)

st.caption(
    "📌 **Caveat:** IV-2SLS estimates average effects across 30 countries. "
    "Country-specific elasticities vary by income level and institutional context. "
    "DiD and Synthetic Control estimates are specific to the policy context described."
)
