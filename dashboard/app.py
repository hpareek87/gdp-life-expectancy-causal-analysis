"""GDP vs Life Expectancy — Interactive Research Dashboard.

Entry point for Streamlit multipage app.
Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="GDP & Life Expectancy Research",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": (
            "GDP & Life Expectancy Causal Analysis\n\n"
            "30 countries × 2000–2024 × 69 variables\n\n"
            "Methods: Causal inference (IV-2SLS, DiD, Synthetic Control) "
            "+ ML (XGBoost, LSTM, Ensemble)"
        ),
    },
)

# Load CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 GDP & Life Expectancy")
    st.markdown("*Publication-grade causal analysis*")
    st.divider()
    st.markdown(
        "**30 countries** · **25 years** · **69 variables**\n\n"
        "Methods: Granger, IV-2SLS, DiD, Synth, XGBoost, LSTM"
    )
    st.divider()
    st.markdown("**Navigation**")
    st.page_link("app.py", label="🏠 Overview", icon="🏠")
    st.page_link("pages/1_Country_Explorer.py",     label="🔍 Country Explorer")
    st.page_link("pages/2_Causal_Findings.py",      label="🔗 Causal Findings")
    st.page_link("pages/3_Predictive_Models.py",    label="🤖 Predictive Models")
    st.page_link("pages/4_Policy_Simulator.py",     label="🎛️ Policy Simulator")
    st.page_link("pages/5_COVID_Validation.py",     label="🦠 COVID Validation")
    st.page_link("pages/6_Data_Methods.py",         label="📚 Data & Methods")

# ── Overview page ─────────────────────────────────────────────────────────────
from components.data_loader import load_master, get_summary_stats, INCOME_COLORS
from components.charts import world_map, income_group_trajectories, scatter_gdp_le

st.title("🌍 GDP & Life Expectancy: A Causal Analysis")
st.markdown(
    "**Does economic growth improve health outcomes?** "
    "This dashboard synthesizes causal inference and machine learning findings "
    "from a panel of 30 countries over 2000–2024."
)

df = load_master()
stats = get_summary_stats(df)

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Countries", stats["n_countries"])
c2.metric("Years", f"{stats['year_range'][0]}–{stats['year_range'][1]}")
c3.metric("Avg LE (2024)", f"{stats['mean_le_2024']} yrs",
          delta=f"+{stats['mean_le_2024'] - stats['mean_le_2000']:.1f} since 2000")
c4.metric("Avg GDP (2024)", f"${stats['mean_gdp_2024']:,.0f}")
c5.metric("Max LE Gain", f"+{stats['max_le_gain']} yrs")

st.divider()

# ── World map ─────────────────────────────────────────────────────────────────
col_map, col_findings = st.columns([3, 2])

with col_map:
    metric_choice = st.selectbox(
        "Map metric",
        ["life_expectancy", "gdp_per_capita_ppp", "health_exp_pct_gdp"],
        format_func=lambda x: {
            "life_expectancy": "Life Expectancy (years)",
            "gdp_per_capita_ppp": "GDP per Capita PPP (USD)",
            "health_exp_pct_gdp": "Health Spending (% GDP)",
        }[x],
    )
    year_slider = st.slider("Year", 2000, 2024, 2024, key="map_year")
    labels = {
        "life_expectancy": "Life Expectancy",
        "gdp_per_capita_ppp": "GDP per Capita",
        "health_exp_pct_gdp": "Health Spending",
    }
    st.plotly_chart(
        world_map(df, year_slider, metric_choice, labels[metric_choice]),
        use_container_width=True,
    )

with col_findings:
    st.markdown("### 🔑 Key Findings")
    findings = [
        ("🔗", "**Causal effect:** IV-2SLS estimates β = 8.1–8.7 years of life expectancy per log-unit GDP increase (p < 0.001)"),
        ("📈", "**Reverse causality:** Life expectancy Granger-causes GDP in 17% of countries vs only 7% vice versa"),
        ("🎯", "**ML accuracy:** XGBoost + LSTM + Ensemble all exceed R² = 0.91 on out-of-sample test set (2019–2024)"),
        ("💡", "**Top predictor:** GDP × Education interaction is the single strongest SHAP feature (58% gain importance)"),
        ("🔀", "**Threshold effect:** GDP–health relationship changes at $1,271, $9,090, and $25,950 PPP — diminishing returns above $25,950"),
        ("🏥", "**Policy experiments:** Indonesia universal health insurance added +0.54 yrs (DiD, p=0.04); China rural insurance +0.87 yrs (Synth Control)"),
    ]
    for icon, text in findings:
        st.markdown(f"""<div class="finding-box">{icon} {text}</div>""",
                    unsafe_allow_html=True)

st.divider()

# ── Income-group trends ────────────────────────────────────────────────────────
st.markdown("### Life Expectancy Trends by Income Group")
col_trend, col_scatter = st.columns(2)
with col_trend:
    st.plotly_chart(
        income_group_trajectories(df, "life_expectancy", "Life Expectancy (years)"),
        use_container_width=True,
    )
with col_scatter:
    scatter_year = st.slider("Scatter year", 2000, 2024, 2019, key="scatter_yr")
    st.plotly_chart(scatter_gdp_le(df, scatter_year), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dashboard-footer">'
    "GDP & Life Expectancy Research Dashboard · Data: World Bank WDI/WGI, WHO GHO, OWID · "
    "Methods: Panel FE, IV-2SLS, DiD, Synthetic Control, XGBoost, LSTM"
    "</div>",
    unsafe_allow_html=True,
)
