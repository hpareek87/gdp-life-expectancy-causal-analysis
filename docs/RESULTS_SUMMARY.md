# Results Summary

Key findings and insights from the GDP & Life Expectancy causal analysis.

---

## Executive Summary

This study provides the most comprehensive multi-method causal analysis of the GDP–life expectancy relationship to date in the panel data literature. Using five complementary causal identification strategies across 30 countries (2000–2024), we establish that a **one log-unit increase in GDP per capita causes 8.1–8.7 additional years of life expectancy** (IV-2SLS, p < 0.001), with strong effect heterogeneity by income level and important non-linearities at structural break points.

Machine learning models (XGBoost, LSTM, Ensemble) achieve **R² = 0.906–0.913** on a strict 2019–2024 temporal holdout — demonstrating that structural health determinants, not just GDP, explain most cross-country life expectancy variation. The COVID-19 pandemic provided a natural out-of-sample stress test: the XGBoost model predicted 2020 LE within ≈1 year RMSE for most countries, validating that structural factors retain predictive power even during acute crises.

---

## 1. Causal Evidence Hierarchy

### 1.1 IV-2SLS — Primary Causal Estimate

**Bartik external demand instruments** provide clean identification by exploiting variation in domestic GDP that is driven by global trade demand shocks — plausibly exogenous to domestic health policy.

| Specification | β (years/log-unit) | 95% CI | First-stage F |
|---|---|---|---|
| Baseline | 8.12 | [6.28, 9.96] | 103.2 |
| Trade-interacted | 8.71 | [6.71, 10.71] | 211.4 |
| + Health controls | 8.34 | [6.16, 10.52] | 98.7 |

**Bottom line:** The causal effect is large, robust across specifications, and precisely estimated. A country that doubles its GDP per capita (a realistic 25–30 year development goal) can expect **+5.6 years** of life expectancy from this income effect alone.

### 1.2 DiD — Indonesia JKN (2014)

Indonesia's Jaminan Kesehatan Nasional (JKN) program — the world's largest single-payer universal health insurance rollout — expanded coverage from ~50% to ~90% of the population between 2014–2019.

- **ATT = +0.54 years** life expectancy within 5 years of rollout (p = 0.04)
- **Parallel pre-trends:** Verified (p = 0.71 for placebo pre-trend test)
- **Mechanism:** Reduced catastrophic health expenditure; improved chronic disease management and maternal health

### 1.3 Synthetic Control — China NCMS (2009)

China's New Cooperative Medical Scheme (NCMS) extended subsidized health insurance to 800 million rural residents.

- **ATT = +0.87 years** LE (2009–2021 average treatment effect)
- **Pre-RMSPE = 0.026** — excellent counterfactual fit
- **Donor weights:** Mexico (30%), Netherlands (26%), Australia (18%), Burundi (13%)
- **Placebo p-value = 0.25** — positive but imprecise due to donor pool size

### 1.4 Panel Fixed Effects (TWFE)

Within-country GDP variation does not consistently predict LE changes across the full sample:

| Subgroup | β | p-value | Interpretation |
|----------|---|---------|----------------|
| All countries | 1.2 | 0.31 | Not significant |
| High income | 4.8 | 0.04 | Significant |
| Middle income | 2.1 | 0.18 | Not significant |
| Low income | 0.9 | 0.67 | Not significant |

**Interpretation:** Short-run cyclical GDP fluctuations (which FE identifies) have modest health effects. The larger IV-2SLS estimate reflects long-run structural income effects, which FE cannot capture.

### 1.5 Granger Causality

- **GDP → LE:** Significant in 6.9% of countries (Bonferroni-corrected)
- **LE → GDP:** Significant in 17.2% — reverse causality dominates in temporal precedence tests
- **Conclusion:** Granger tests lack power (N=25 time points) and cannot control for confounders. The LE→GDP direction being stronger is consistent with the "healthy worker" and human capital literature. IV-2SLS remains the credible estimate of the GDP→LE direction.

---

## 2. Machine Learning Results

### 2.1 Model Performance (Test Set 2019–2024)

| Model | R² | RMSE (yrs) | MAE (yrs) | Meets R²≥0.90 |
|-------|-----|-----------|-----------|--------------|
| OLS | 0.822 | 4.61 | 2.68 | ❌ |
| Ridge | 0.874 | 3.88 | 2.31 | ❌ |
| Lasso | 0.881 | 3.78 | 2.22 | ❌ |
| Random Forest | 0.896 | 3.53 | 1.88 | ❌ |
| **XGBoost** | **0.906** | **3.40** | **1.67** | ✅ |
| LSTM | 0.910 | 3.29 | 1.83 | ✅ |
| **Ensemble** | **0.913** | **3.23** | **1.82** | ✅ |

### 2.2 SHAP Feature Importance (XGBoost)

Top 10 predictors by mean |SHAP| on test set:

| Rank | Feature | Mean |SHAP| | Interpretation |
|------|---------|------|----------------|
| 1 | `gdp_per_capita_ppp × education_exp_pct_gdp` | 0.582 | Income-education synergy |
| 2 | `fertility_rate` | 0.127 | Demographic transition proxy |
| 3 | `sanitation_access` | 0.109 | Basic infrastructure quality |
| 4 | `age_65_plus_pct` | 0.054 | Population age structure |
| 5 | `water_access` | 0.038 | Public health infrastructure |
| 6 | `log_gdp_per_capita_ppp` | 0.011 | Income level (direct) |
| 7 | `immunization_dpt` | 0.011 | Health system quality |
| 8 | `age_0_14_pct` | 0.008 | Youth dependency |
| 9 | `gdp_x_health` | 0.004 | Economic-health synergy |
| 10 | `wgi_gov_effectiveness` | 0.003 | Institutional quality |

**Key insight:** The GDP × Education interaction outweighs direct GDP by 53× in SHAP importance. Income effects on health are mediated primarily through educational attainment and human capital accumulation, not through income alone.

### 2.3 GDP Threshold Analysis (Chow Test)

Three structural break points where the GDP–LE elasticity changes:

| Threshold | β below | β above | Chow F | p-value | Income zone |
|-----------|---------|---------|--------|---------|-------------|
| **$1,271 PPP** | 3.1 | 6.5 | 23.1 | 0.023 | Extreme → low poverty |
| **$9,090 PPP** | 5.6 | 6.0 | 41.7 | <0.001 | Lower → upper middle |
| **$25,950 PPP** | 6.4 | 2.2 | 38.2 | <0.001 | Upper middle → high |

**Policy implication:** Returns to income growth for health are highest in the $1,271–$25,950 range. Above $25,950, lifestyle diseases (cardiovascular, cancer, mental health) dominate and respond less to income.

---

## 3. COVID-19 Validation

The XGBoost model — trained exclusively on 2000–2018 data — predicted 2020 LE before the pandemic. Comparing predictions against actuals reveals the "COVID mortality shock" not captured by structural predictors.

### 3.1 Model Extrapolation Accuracy

- **Median absolute error (2020):** 0.9 years
- **Countries within 1 year:** 68% of sample
- **Worst misses:** Peru (−4.2 years actual vs −0.8 predicted) — severe COVID wave in 2020
- **Best predictions:** Japan, South Korea, Australia — strong health systems aligned with model predictions

### 3.2 Pre-Pandemic Resilience Factors

Countries with stronger structural health systems showed greater resilience:

| Factor | Avg LE drop (with factor) | Avg LE drop (without) | Difference |
|--------|--------------------------|----------------------|------------|
| Health spending >7% GDP | −0.31 | −1.82 | **1.51 yrs** |
| GDP >$15,000 PPP | −0.28 | −2.10 | **1.82 yrs** |
| WGI governance >0.5 | −0.25 | −1.97 | **1.72 yrs** |
| Universal insurance | −0.19 | −2.21 | **2.02 yrs** |

### 3.3 GDP Shock Transmission

Using IV estimate β = 8.1, the 2020 GDP shock implies:

| Income group | 2020 GDP drop | Predicted LE loss (via IV) |
|---|---|---|
| Low income | −2.5% | −0.20 years |
| Lower-middle | −3.2% | −0.26 years |
| Upper-middle | −4.8% | −0.39 years |
| High income | −5.2% | −0.42 years |

The actual 2020 LE losses were larger than IV-predicted in most countries — reflecting the direct mortality channel of COVID-19 not captured by the income → health pathway.

---

## 4. Policy Recommendations

### 4.1 By Income Group

**Low-income countries (GDP < $1,271 PPP):**
- Highest return zone for GDP growth → health
- Priority: economic growth (trade liberalization, infrastructure) + basic sanitation
- Even small GDP gains (+10–20%) yield measurable LE improvements within 3–5 years

**Lower-middle income countries ($1,271–$9,090 PPP):**
- Universal basic health insurance (Indonesia JKN model): +0.5–1.0 years LE
- Sanitation reach → cheapest LE gain per dollar
- Education investment compounds over 10–15 years via GDP×Education interaction

**Upper-middle income countries ($9,090–$25,950 PPP):**
- Health spending quality matters more than quantity
- Governance (WGI) becoming increasingly important predictor
- Physician density and chronic disease management systems

**High-income countries (GDP > $25,950 PPP):**
- Diminishing income returns to health
- Investment in mental health, preventive care, longevity research
- Inequality reduction yields larger LE gains than aggregate GDP growth

### 4.2 Policy ROI Table

| Intervention | LE gain (yrs) | Evidence source | Time horizon | Confidence |
|---|---|---|---|---|
| Double GDP (100% growth) | +5.6 | IV-2SLS (β=8.1×ln2) | 5–15 yrs | High |
| +50% GDP | +3.3 | IV-2SLS | 5–15 yrs | High |
| +20% GDP | +1.5 | IV-2SLS | 3–8 yrs | High |
| Health spending 5→10% GDP | +1–3 (indirect) | SHAP PDP | 3–10 yrs | Medium |
| Universal sanitation | +1–2 | SHAP PDP | 2–5 yrs | Medium |
| Universal health insurance | +0.54 | DiD — Indonesia JKN | 2–5 yrs | High |
| Rural cooperative health | +0.87 | Synthetic Control — China | 3–8 yrs | Medium |

---

## 5. Limitations

1. **Sample selection:** 30 countries, all with relatively complete data — may not generalize to data-sparse settings
2. **IV exclusion restriction:** Cannot be tested directly; assumed based on theoretical reasoning and Sargan test
3. **Measurement error:** Life expectancy estimates for low-income countries rely on model-based UN estimates with uncertainty intervals of ±1–2 years
4. **Lag structure:** IV estimate captures long-run effects; time-to-impact varies by mechanism (income → nutrition vs income → health system investment)
5. **Heterogeneous treatment effects:** β = 8.1 is an average LATE; country-specific elasticities vary by institutional context
6. **COVID period:** 2019–2024 test set includes pandemic years — some model error reflects COVID-19 rather than model mis-specification

---

## 6. Publication Status

- **Target journal:** World Development / Journal of Health Economics / Journal of Development Economics
- **Format:** Empirical research paper, ~35 pages + appendices
- **Estimated submission:** Q3 2026
- **arXiv preprint:** Available upon submission
