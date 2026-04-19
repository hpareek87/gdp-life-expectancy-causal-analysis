"""Data & Methods page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import io

from components.data_loader import load_master

st.set_page_config(page_title="Data & Methods", page_icon="📚", layout="wide")
st.title("📚 Data & Methods")
st.markdown(
    "Full documentation of data sources, methodology, and reproducibility materials."
)

df = load_master()

# ── Data sources ───────────────────────────────────────────────────────────────
st.markdown("## Data Sources")

sources = [
    {
        "Source": "World Bank World Development Indicators (WDI)",
        "Variables": "GDP per capita (PPP), life expectancy, fertility rate, sanitation, water access, immunization",
        "Coverage": "30 countries × 2000–2024",
        "Access": "World Bank API (`wbdata` Python package)",
        "Citation": "World Bank (2024). World Development Indicators. Washington, D.C.",
    },
    {
        "Source": "World Bank Health Nutrition & Population (HNP)",
        "Variables": "Health expenditure (% GDP), physicians per 1,000, hospital beds",
        "Coverage": "2000–2022 (partial 2023–2024)",
        "Access": "World Bank API — series HNP_*",
        "Citation": "World Bank (2024). Health Nutrition and Population Statistics.",
    },
    {
        "Source": "World Bank Worldwide Governance Indicators (WGI)",
        "Variables": "Government effectiveness, rule of law, control of corruption, political stability",
        "Coverage": "1996–2023 (biennial pre-2002)",
        "Access": "World Bank API — WGI dataset",
        "Citation": "Kaufmann, D., Kraay, A., & Mastruzzi, M. (2010). The Worldwide Governance Indicators.",
    },
    {
        "Source": "UNESCO Institute for Statistics",
        "Variables": "Education expenditure (% GDP), mean years of schooling",
        "Coverage": "2000–2023 (gaps imputed via linear interpolation)",
        "Access": "UNESCO UIS Data API",
        "Citation": "UNESCO Institute for Statistics (2024). Education Statistics.",
    },
    {
        "Source": "UN Population Division (UNDP)",
        "Variables": "Age structure (0–14, 15–64, 65+), urban population %",
        "Coverage": "1950–2100 (5-year projections, interpolated annually)",
        "Access": "UNDP World Population Prospects 2024 CSV files",
        "Citation": "UN Department of Economic and Social Affairs (2024). World Population Prospects 2024.",
    },
    {
        "Source": "World Bank Trade Statistics",
        "Variables": "Trade openness (% GDP) — used as IV instrument interaction",
        "Coverage": "2000–2024",
        "Access": "World Bank API — NE.TRD.GNFS.ZS",
        "Citation": "World Bank (2024). Trade (% of GDP).",
    },
]

sources_df = pd.DataFrame(sources)
st.dataframe(sources_df.set_index("Source"), use_container_width=True)

st.divider()

# ── Sample countries ────────────────────────────────────────────────────────────
st.markdown("## Country Sample (30 Countries)")
st.markdown("Stratified by income group to ensure cross-income variation:")

col_lo, col_lm, col_um, col_hi = st.columns(4)
with col_lo:
    st.markdown("**Low income**")
    st.markdown("- Ethiopia\n- Tanzania\n- Uganda\n- Mozambique\n- Burkina Faso")

with col_lm:
    st.markdown("**Lower-middle income**")
    st.markdown("- India\n- Nigeria\n- Bangladesh\n- Pakistan\n- Kenya\n- Ghana\n- Cambodia")

with col_um:
    st.markdown("**Upper-middle income**")
    st.markdown("- China\n- Brazil\n- Mexico\n- Indonesia\n- South Africa\n- Thailand\n- Colombia\n- Peru")

with col_hi:
    st.markdown("**High income**")
    st.markdown("- United States\n- Germany\n- Japan\n- United Kingdom\n- France\n- Canada\n- Australia\n- South Korea\n- Sweden\n- Netherlands")

st.markdown(
    f"**Total observations:** {len(df):,} country-year pairs "
    f"({df['country'].nunique()} countries × {df['year'].nunique()} years)"
)

st.divider()

# ── Variable dictionary ─────────────────────────────────────────────────────────
st.markdown("## Variable Dictionary")

variable_dict = [
    ("life_expectancy", "Life Expectancy at Birth", "Years", "WDI: SP.DYN.LE00.IN", "Outcome"),
    ("gdp_per_capita_ppp", "GDP per Capita (PPP, 2017 intl $)", "USD", "WDI: NY.GDP.PCAP.PP.KD", "Key predictor"),
    ("log_gdp_per_capita_ppp", "Log GDP per Capita (PPP)", "log USD", "Derived", "IV-2SLS endogenous variable"),
    ("health_exp_pct_gdp", "Health Expenditure (% GDP)", "%", "WDI: SH.XPD.CHEX.GD.ZS", "Policy variable"),
    ("physicians_per_1000", "Physicians (per 1,000 population)", "per 1,000", "WDI: SH.MED.PHYS.ZS", "Health capacity"),
    ("sanitation_access", "Access to Basic Sanitation (%)", "%", "WDI: SH.STA.BASS.ZS", "Infrastructure"),
    ("water_access", "Access to Safe Water (%)", "%", "WDI: SH.H2O.BASW.ZS", "Infrastructure"),
    ("education_exp_pct_gdp", "Education Expenditure (% GDP)", "%", "UNESCO: EDU_FIN_EXP_PT_GDP", "Policy variable"),
    ("fertility_rate", "Total Fertility Rate", "births/woman", "WDI: SP.DYN.TFRT.IN", "Demographic proxy"),
    ("urban_pop_pct", "Urban Population (%)", "%", "WDI: SP.URB.TOTL.IN.ZS", "Structural"),
    ("age_0_14_pct", "Population Age 0–14 (%)", "%", "UNDP WPP 2024", "Age structure"),
    ("age_65_plus_pct", "Population Age 65+ (%)", "%", "UNDP WPP 2024", "Age structure"),
    ("immunization_dpt", "DPT Immunization Coverage (%)", "%", "WDI: SH.IMM.IDPT", "Health system quality"),
    ("wgi_gov_effectiveness", "Governance Effectiveness (WGI)", "score (−2.5 to 2.5)", "World Bank WGI", "Institutions"),
    ("ext_demand", "External Demand Instrument", "weighted avg GDP growth", "Derived (Bartik)", "IV instrument"),
]

vd_df = pd.DataFrame(variable_dict,
                     columns=["Variable", "Label", "Unit", "Source", "Role"])
st.dataframe(vd_df.set_index("Variable"), use_container_width=True)

st.divider()

# ── Methodology ────────────────────────────────────────────────────────────────
st.markdown("## Methodology")

st.markdown("### 1. Data Pipeline")
st.markdown("""
- **Collection:** World Bank API (wbdata), UNESCO API, UN Population Prospects CSVs
- **Cleaning:** Country harmonization via ISO3 codes; outliers winsorized at 1st/99th percentile by income group
- **Imputation:** Linear interpolation for ≤3 consecutive missing years; forward-fill for governance indicators (biennial → annual)
- **Feature engineering:** Log transforms, interaction terms (GDP×Education, GDP×Health), lag features (1–3 years), demographic ratios
- **Final dataset:** 725 country-year observations × 133 variables → 69 modelling features
""")

st.markdown("### 2. Causal Identification Strategy")
st.markdown("""
Five complementary causal methods with increasing credibility:

| Method | Identification | Key Assumption |
|--------|----------------|----------------|
| **Granger Causality** | Temporal precedence | Stationarity, no confounders |
| **Panel FE (TWFE)** | Within-country variation | Parallel trends, no time-varying confounders |
| **IV-2SLS (Bartik)** | External demand shocks | Exclusion restriction: trade shocks affect income, not health directly |
| **DiD (Indonesia JKN)** | Policy quasi-experiment | Parallel pre-trends verified |
| **Synthetic Control (China NCMS)** | Counterfactual construction | Convex hull assumption, pre-RMSPE = 0.026 |

**Bartik instrument construction:** For each country $i$ and year $t$:
$$Z_{it} = \\sum_{j \\neq i} \\frac{GDP_{jt} - GDP_{jt-1}}{GDP_{jt-1}} \\cdot w_{ij}$$
where $w_{ij}$ is the trade-share weight between countries $i$ and $j$.
""")

st.markdown("### 3. Machine Learning Pipeline")
st.markdown("""
- **Train/test split:** 2000–2018 (train), 2019–2024 (test) — strict temporal holdout
- **Preprocessing:** `StandardScaler` fit on training data only; applied to test
- **Models:** OLS, Ridge, Lasso, Quantile (q=0.5), Random Forest, XGBoost, LSTM, Ensemble (Ridge meta-learner)
- **CV strategy:** Walk-forward expanding window (5 folds) — no lookahead
- **XGBoost config:** `tree_method="hist"`, `n_estimators=400`, `learning_rate=0.04`, `max_depth=5`, `subsample=0.8`
- **LSTM config:** 2-layer LSTM (64 hidden units), `sequence_length=5`, `epochs=80`, `lr=0.001`
- **Ensemble:** Out-of-fold predictions as meta-features → Ridge (α=1.0) meta-learner
- **Interpretability:** SHAP TreeExplainer for XGBoost and Random Forest; PDP analysis; GDP threshold Chow test
""")

st.markdown("### 4. Model Validation")
st.markdown("""
- **Primary metric:** R² on 2019–2024 test set (target: ≥ 0.90)
- **Additional metrics:** RMSE (years), MAE (years)
- **COVID stress test:** Model predictions for 2020 compared against actual LE decline; resilience factors validated
- **Walk-forward CV:** Ensures no temporal leakage in cross-validation
""")

st.divider()

# ── Reproducibility ─────────────────────────────────────────────────────────────
st.markdown("## Reproducibility")

col_repo, col_env = st.columns([1, 1])

with col_repo:
    st.markdown("### Code Repository")
    st.markdown("""
The full codebase is available on GitHub:

**Repository structure:**
```
├── src/
│   ├── data/           # Data collection + cleaning
│   ├── analysis/       # Causal methods + ML models
│   └── visualization/  # Publication figures
├── dashboard/          # This Streamlit app
├── notebooks/          # Jupyter analysis notebooks
├── outputs/            # Figures, tables, model artifacts
└── tests/              # Smoke tests (pytest)
```

**Key notebooks:**
- `01_data_collection.ipynb` — WB/UNESCO API pull
- `02_eda.ipynb` — 19 EDA figures
- `03_causal_analysis.ipynb` — All 5 causal methods
- `04_ml_modeling.ipynb` — ML pipeline + SHAP
- `05_results_synthesis.ipynb` — Publication outputs
""")

with col_env:
    st.markdown("### Software Environment")
    env_data = {
        "Package": ["Python", "pandas", "numpy", "scikit-learn", "xgboost",
                    "torch", "shap", "statsmodels", "linearmodels",
                    "streamlit", "plotly", "matplotlib"],
        "Version": ["3.13", "2.x", "1.x", "1.6+", "2.1.4",
                    "2.x", "0.51.0", "0.14+", "6.x",
                    "1.x", "5.x", "3.x"],
        "Purpose": ["Runtime", "Data manipulation", "Numerics", "ML utilities",
                    "Gradient boosting", "LSTM", "Interpretability", "Econometrics",
                    "Panel/IV models", "Dashboard", "Interactive charts", "Static figures"],
    }
    st.dataframe(pd.DataFrame(env_data).set_index("Package"), use_container_width=True)
    st.caption("arm64 macOS note: torch must be imported *after* XGBoost to avoid dylib conflict.")

st.divider()

# ── Download data ───────────────────────────────────────────────────────────────
st.markdown("## Download Data")

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    st.markdown("**Master Dataset**")
    if not df.empty:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download master_dataset.csv",
            data=csv_buf.getvalue(),
            file_name="gdp_life_expectancy_master.csv",
            mime="text/csv",
        )
        st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    else:
        st.info("Dataset not loaded.")

with col_dl2:
    st.markdown("**Data Dictionary**")
    vd_csv = pd.DataFrame(variable_dict, columns=["Variable", "Label", "Unit", "Source", "Role"])
    st.download_button(
        label="Download variable_dictionary.csv",
        data=vd_csv.to_csv(index=False),
        file_name="variable_dictionary.csv",
        mime="text/csv",
    )
    st.caption(f"{len(variable_dict)} variables documented")

with col_dl3:
    st.markdown("**Country-Year Panel**")
    if not df.empty:
        panel_cols = ["country", "iso3", "year", "income_group",
                      "life_expectancy", "gdp_per_capita_ppp",
                      "health_exp_pct_gdp", "education_exp_pct_gdp",
                      "fertility_rate", "sanitation_access"]
        panel_cols = [c for c in panel_cols if c in df.columns]
        panel_df = df[panel_cols]
        st.download_button(
            label="Download panel_data.csv",
            data=panel_df.to_csv(index=False),
            file_name="gdp_le_panel.csv",
            mime="text/csv",
        )
        st.caption("Key variables only (10 columns)")

st.divider()

# ── Citation ─────────────────────────────────────────────────────────────────────
st.markdown("## Citation")
st.code("""@article{pareek2025gdple,
  title   = {GDP and Life Expectancy: A Multi-Method Causal Analysis
             Across 30 Countries (2000–2024)},
  author  = {Pareek, Hardik},
  journal = {arXiv preprint},
  year    = {2025},
  note    = {arXiv:2025.XXXXX}
}""", language="bibtex")

st.markdown("### Key References")
st.markdown("""
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods.
  *Journal of the American Statistical Association*, 105(490), 493–505.
- Bartik, T. J. (1991). *Who Benefits from State and Local Economic Development Policies?*
  Kalamazoo, MI: W.E. Upjohn Institute.
- Bloom, D. E., Canning, D., & Sevilla, J. (2004). The effect of health on economic growth.
  *World Development*, 32(1), 1–13.
- Kaufmann, D., Kraay, A., & Mastruzzi, M. (2010). The Worldwide Governance Indicators.
  *World Bank Policy Research Working Paper* 5430.
- Preston, S. H. (1975). The changing relation between mortality and level of economic development.
  *Population Studies*, 29(2), 231–248.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
  *NeurIPS 30*.
""")
