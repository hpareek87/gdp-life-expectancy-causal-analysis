# Methodology

Detailed documentation of all analytical methods used in the GDP & Life Expectancy causal analysis.

---

## 1. Data Pipeline

### 1.1 Data Collection

Data collected programmatically from six sources via official APIs and curated CSV files:

- **World Bank API** (`wbgapi`): WDI + HNP + WGI series
- **UNESCO UIS API**: Education expenditure
- **UN Population Division**: World Population Prospects 2024 (CSV)
- **WHO GHO API**: Health system indicators
- **OWID COVID-19 dataset**: Pandemic mortality data

### 1.2 Country Selection

30 countries selected via stratified sampling by income group:

| Income Group | N | Selection Criteria |
|-------------|---|-------------------|
| Low income | 5 | SSA region, >20M population, complete data >60% |
| Lower-middle income | 7 | Asia + Africa, diverse governance contexts |
| Upper-middle income | 8 | BRICS + LAC, sufficient GDP variation |
| High income | 10 | OECD members, developed health systems |

### 1.3 Cleaning & Imputation

- **Outlier treatment:** Winsorize at 1st/99th percentile by income group
- **Missing data:**
  - ≤3 consecutive missing years → linear interpolation
  - Governance indicators (biennial pre-2002) → linear interpolation → annual
  - Structural break in series (e.g., methodological change) → chain-linked
- **Country harmonization:** ISO 3166-1 alpha-3 codes as canonical identifiers

### 1.4 Feature Engineering

69 modelling features from raw variables:

| Feature Type | Examples | Count |
|-------------|---------|-------|
| Raw economic | `gdp_per_capita_ppp`, `health_exp_pct_gdp` | 18 |
| Log transforms | `log_gdp_per_capita_ppp` | 4 |
| Polynomial | `log_gdp_sq`, `log_gdp_cu` | 2 |
| Interactions | `gdp × education`, `gdp × health`, `health × physicians` | 8 |
| Lag features | `gdp_per_capita_ppp_lag1/2/3` | 9 |
| Demographic | `fertility_rate`, `age_0_14_pct`, `age_65_plus_pct` | 8 |
| Infrastructure | `sanitation_access`, `water_access`, `urban_pop_pct` | 6 |
| Governance | `wgi_gov_effectiveness`, `wgi_rule_of_law` | 5 |
| IV instruments | `ext_demand`, `ext_demand × trade_pct_gdp` | 2 |
| Growth rates | `gdp_growth`, `le_change_1yr` | 7 |

---

## 2. Causal Inference

### 2.1 Granger Causality

**Purpose:** Test whether past values of GDP help predict current life expectancy (and vice versa) beyond own lags.

**Specification:**
```
LE_t = α + Σ_k β_k LE_{t-k} + Σ_k γ_k GDP_{t-k} + ε_t
```

**Implementation:**
- `statsmodels.tsa.stattools.grangercausalitytests`
- Tested at lags 1–4; selected lag 1 (modal AIC)
- Bonferroni correction for multiple country comparisons
- Run country-by-country (N=30)

**Limitations:**
- Only establishes temporal precedence, not causation
- N=25 time points → low statistical power
- Cannot control for time-varying confounders

**Results:**
- GDP → LE: Significant in 6.9% of countries (Bonferroni-corrected)
- LE → GDP: Significant in 17.2% (stronger reverse causality signal)

---

### 2.2 Panel Fixed Effects (TWFE)

**Specification:**
```
LE_it = α_i + λ_t + β log(GDP_it) + X_it'γ + ε_it
```

Where α_i = country FE, λ_t = time FE, X_it = health/governance controls.

**Implementation:**
- `linearmodels.panel.PanelOLS` with `entity_effects=True, time_effects=True`
- Standard errors clustered at country level
- Estimated separately by income subgroup

**Identification:** Within-country, within-year variation in log GDP.

**Results:**
- Overall: β = 1.2 (p = 0.31) — not significant
- High income only: β = 4.8 (p = 0.04) — significant
- Interpretation: Cyclical GDP variation within countries explains less LE variation than structural level differences

---

### 2.3 IV-2SLS (Bartik Instruments)

**This is our primary causal estimate.**

**Instruments:**
```
Z1_it = Σ_{j≠i} w_ij × ΔlogGDP_jt    (external demand)
Z2_it = Z1_it × trade_pct_gdp_{i,t-1}  (trade-interaction)
```

Where w_ij = bilateral trade share between countries i and j.

**Exclusion restriction:** Global demand shocks affect domestic income through export channels. They do not directly determine population-level health outcomes conditional on income.

**Implementation (`linearmodels.iv.IV2SLS`):**
```python
IV2SLS(
    dependent=LE,
    exog=[const, controls],
    endog=log_GDP,
    instruments=[ext_demand, ext_demand_x_trade]
).fit(cov_type='clustered', clusters=country_id)
```

**Diagnostics:**
- First-stage F-statistic: 103–211 (>> 10 threshold; strong instrument)
- Sargan over-identification test: p = 0.46 (cannot reject exclusion restriction)
- Kleibergen-Paap robust F: 98.7 (weak instrument robust)

**Results:**
| Specification | β | SE | F-stat |
|---|---|---|---|
| Baseline | 8.12 | (0.94) | 103.2 |
| Trade-interacted | 8.71 | (1.02) | 211.4 |
| + Health controls | 8.34 | (1.11) | 98.7 |

**Interpretation:** A 1 log-unit increase in GDP per capita (PPP) causes 8.1–8.7 additional years of life expectancy in the long run.

**Policy translation (β = 8.1):**
- Doubling GDP (ln 2 ≈ 0.693): +5.6 years
- +50% GDP (ln 1.5 ≈ 0.405): +3.3 years
- +20% GDP (ln 1.2 ≈ 0.182): +1.5 years

---

### 2.4 Difference-in-Differences (DiD)

**Event 1: Indonesia JKN (Jaminan Kesehatan Nasional), 2014**

Universal health insurance rollout; treated = Indonesia, control = synthetic control group.

```
ΔLE_it = α + β × Post_t × Treated_i + γ_i + δ_t + ε_it
```

- Parallel pre-trends: verified (p = 0.71 for pre-trend test)
- ATT = +0.54 years (p = 0.04, 5-year window post-rollout)
- Placebo test: insignificant for 2010–2013 "false treatments"

**Event 2: Vietnam UHC (2009)**
- Parallel trends test FAILED (p < 0.001) → estimate discarded

**Event 3: China NCMS (2009)**
- Parallel trends partial → upgraded to Synthetic Control (see §2.5)

---

### 2.5 Synthetic Control (Abadie 2010)

**Target:** China's 2009 NCMS rural cooperative medical scheme.

**Method:** Construct synthetic China as weighted combination of donor countries:
```
ŷ_China,t = Σ_j w_j* × y_j,t    for t < 2009
```

Optimize weights to minimize pre-treatment RMSPE.

**Donor pool weights:**
- Mexico: 30%
- Netherlands: 26%
- Australia: 18%
- Burundi: 13%
- Others: 13%

**Diagnostics:**
- Pre-RMSPE = 0.026 (excellent pre-treatment fit; threshold < 0.10)
- ATT = +0.87 years (2009–2021 average)
- Placebo p-value = 0.25 (positive but imprecise; limited by donor N)

---

## 3. Machine Learning

### 3.1 Train/Test Split

**Strict temporal holdout — no leakage:**
- **Train:** 2000–2018 (all countries)
- **Test:** 2019–2024 (all countries)
- `StandardScaler` fit on training data only; transform applied to test

### 3.2 Models

| Model | Key Hyperparameters |
|-------|-------------------|
| OLS | Standard linear regression |
| Ridge | α tuned via 5-fold CV |
| Lasso | α tuned via 5-fold CV |
| Quantile (q=0.5) | Median regression |
| Random Forest | n_estimators=200, max_depth=None, n_jobs=1 |
| XGBoost | n_estimators=400, lr=0.04, max_depth=5, subsample=0.8, tree_method='hist' |
| LSTM | 2-layer, hidden=64, seq_len=5, epochs=80, lr=0.001 |
| Ensemble | Ridge meta-learner (α=1.0) on OOF predictions |

### 3.3 Cross-Validation

Walk-forward expanding window (5 folds):
```
Fold 1: train [2000–2003] → val [2004–2005]
Fold 2: train [2000–2005] → val [2006–2007]
Fold 3: train [2000–2007] → val [2008–2010]
Fold 4: train [2000–2010] → val [2011–2013]
Fold 5: train [2000–2013] → val [2014–2018]
```
No future data leaks into any validation fold.

### 3.4 SHAP Interpretability

- **TreeExplainer** for XGBoost and Random Forest (exact Shapley values)
- **Global importance:** Mean |SHAP| across test set
- **Local explanation:** Waterfall plots for individual predictions
- **Dependence plots:** SHAP value vs feature value (with color-coded interaction term)
- **API note:** SHAP 0.51.0 requires `.shap_values()` method; manual `Explanation` object construction for beeswarm/waterfall

### 3.5 GDP Threshold Analysis

Chow structural break test at candidate breakpoints (percentiles of log GDP distribution):

```
LE = α + β_low × logGDP × I(GDP < threshold) + β_high × logGDP × I(GDP ≥ threshold) + ε
```

Three significant break points identified:
- **$1,271 PPP** (F = 23.1, p = 0.023): Steep returns in low-income range
- **$9,090 PPP** (F = 41.7, p < 0.001): Middle-income health dividend
- **$25,950 PPP** (F = 38.2, p < 0.001): High-income diminishing returns

---

## 4. COVID-19 Validation

**Out-of-sample stress test:**
- XGBoost trained on 2000–2018 predicts 2020 LE (without seeing COVID data)
- "COVID mortality shock" = actual 2020 LE − model prediction
- RMSE on 2020 holdout ≈ 1 year (demonstrating structural predictors hold even in crisis)

**Resilience factors analysis:**
- Countries with pre-2020 health spending > 7% GDP saw ~1.5 years less LE decline
- Governance quality (WGI > 0.5) associated with 1.7 years less decline
- High-income countries recovered to pre-pandemic LE by 2022–2023

**IV transmission of GDP shock:**
Using IV estimate β = 8.1: a 5% GDP drop → log(0.95) ≈ −0.051 → predicted LE loss ≈ −0.41 years (consistent with observed 2020 LE changes in most income groups).

---

## 5. Reproducibility

All analysis is fully reproducible from raw API pulls:

```bash
# Full pipeline (takes ~20 min on first run due to API calls)
python -m src.data.build_dataset
python -m src.analysis.causal
python -c "from src.analysis.ml_models import run_all_ml; run_all_ml()"
python -c "from src.analysis.interpretability import run_interpretability; run_interpretability()"
pytest -q                        # 12 smoke tests
```

Pre-computed outputs are included in `outputs/` for immediate dashboard use.
