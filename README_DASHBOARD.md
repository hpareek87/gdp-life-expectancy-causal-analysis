# GDP vs Life Expectancy — Interactive Dashboard

A publication-grade Streamlit dashboard for the causal analysis of GDP and life expectancy across 30 countries (2000–2024).

## Pages

| # | Page | Description |
|---|------|-------------|
| 0 | **Overview** | World map, KPIs, income-group trends, key findings |
| 1 | **Country Explorer** | 24-year trajectories, peer comparison, GDP scatter |
| 2 | **Causal Findings** | IV-2SLS, Granger, Panel FE, DiD, Synthetic Control |
| 3 | **Predictive Models** | Model performance, SHAP importance, GDP thresholds |
| 4 | **Policy Simulator** | Real-time XGBoost predictions with slider controls |
| 5 | **COVID Validation** | Pandemic stress test, resilience factors, recovery trajectories |
| 6 | **Data & Methods** | Sources, methodology, variable dictionary, data download |

## Local Setup

```bash
# 1. Clone / navigate to project
cd "GDP vs Life Expectancy"

# 2. Install dependencies (Python 3.11–3.13)
pip install -r dashboard/requirements.txt

# 3. Run the pipeline first (if model artifacts don't exist)
python -c "from src.analysis.ml_models import run_all_ml; run_all_ml()"
python -c "from src.analysis.interpretability import run_interpretability; run_interpretability()"

# 4. Launch dashboard
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Deployment: Streamlit Community Cloud

1. Push the repository to GitHub (public or private with access granted)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path:** `dashboard/app.py`
5. Set **Python version:** 3.11 (recommended for arm64 compatibility)
6. Click **Deploy**

> **Note:** Streamlit Cloud runs on Linux x86-64. The arm64-specific XGBoost/torch dylib conflict does not apply there. You may upgrade xgboost to the latest stable version in `requirements.txt` for cloud deployments.

## Model Artifacts

The Policy Simulator (page 4) requires trained model artifacts:

```
outputs/
├── models/
│   ├── xgboost_model.pkl
│   └── scaler.pkl
└── ml/
    ├── feature_importance.csv
    └── threshold_analysis.csv
```

If these are missing, run `run_all_ml()` from `src/analysis/ml_models.py`. The dashboard degrades gracefully — all other pages work without the model artifacts.

## Architecture

```
dashboard/
├── app.py                    # Entry point (Overview page)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme + server settings
├── components/
│   ├── data_loader.py        # Cached data loading, model loading
│   └── charts.py             # Reusable Plotly chart functions
├── assets/
│   └── style.css             # Custom CSS (metric cards, sidebar)
└── pages/
    ├── 1_Country_Explorer.py
    ├── 2_Causal_Findings.py
    ├── 3_Predictive_Models.py
    ├── 4_Policy_Simulator.py
    ├── 5_COVID_Validation.py
    └── 6_Data_Methods.py
```

## Key Findings

- **Causal effect (IV-2SLS):** β = 8.1–8.7 years per log-unit GDP (p < 0.001)
- **ML accuracy:** XGBoost R² = 0.906, Ensemble R² = 0.913 on 2019–2024 holdout
- **Top predictors (SHAP):** GDP × Education interaction, fertility rate, sanitation access
- **Policy ROI:** Doubling GDP → +5.6 years LE; Universal sanitation → +1–2 years
- **COVID validation:** Pre-pandemic health spending >7% GDP reduced 2020 LE decline by ~1.5 years
