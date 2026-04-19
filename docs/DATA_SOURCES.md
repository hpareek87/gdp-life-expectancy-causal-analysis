# Data Sources

Complete documentation of all data sources used in the GDP & Life Expectancy analysis.

---

## Overview

| Source | Type | Access Method | Years | Countries |
|--------|------|--------------|-------|-----------|
| World Bank WDI | API | `wbgapi` Python package | 2000–2024 | 30 |
| World Bank WGI | API | `wbgapi` Python package | 1996–2023 | 30 |
| World Bank HNP | API | `wbgapi` Python package | 2000–2022 | 30 |
| UNESCO UIS | API | REST API + CSV | 2000–2023 | 30 |
| UN Population Division | CSV | Manual download | 2000–2024 | 30 |
| WHO GHO | API | REST API | 2000–2022 | 30 |
| OWID COVID-19 | CSV | GitHub raw | 2020–2023 | 30 |

---

## 1. World Bank World Development Indicators (WDI)

**URL:** https://data.worldbank.org/

**Access:**
```python
import wbgapi as wb
df = wb.data.DataFrame(series_codes, economy=country_codes, time=range(2000, 2025))
```

**Series collected:**

| WDI Code | Description | Unit |
|----------|-------------|------|
| `SP.DYN.LE00.IN` | Life expectancy at birth (total) | Years |
| `NY.GDP.PCAP.PP.KD` | GDP per capita, PPP (2017 intl $) | USD |
| `NY.GDP.MKTP.KD.ZG` | GDP growth rate | % |
| `SP.DYN.TFRT.IN` | Total fertility rate | Births/woman |
| `SP.POP.TOTL` | Total population | Persons |
| `SP.URB.TOTL.IN.ZS` | Urban population (% of total) | % |
| `SH.STA.BASS.ZS` | Access to basic sanitation services | % |
| `SH.H2O.BASW.ZS` | Access to basic drinking water | % |
| `SH.IMM.IDPT` | DPT immunization (% children 12–23 months) | % |
| `NE.TRD.GNFS.ZS` | Trade openness (% of GDP) | % |
| `SL.UEM.TOTL.ZS` | Unemployment (% of labor force) | % |

**Known issues:**
- `SP.DYN.LE00.IN` has a 2-year reporting lag; 2023–2024 values are provisional
- PPP GDP series revised in 2023 base year update — use `KD` (constant) series

---

## 2. World Bank Worldwide Governance Indicators (WGI)

**URL:** https://info.worldbank.org/governance/wgi/

**Citation:** Kaufmann, D., Kraay, A., & Mastruzzi, M. (2010). The Worldwide Governance Indicators: A Summary of Methodology, Data and Analytical Issues. *World Bank Policy Research Working Paper* No. 5430.

**Access:**
```python
wb.data.DataFrame(['GE.EST', 'RL.EST', 'CC.EST', 'PV.EST'], economy=codes, time=range(2000, 2024))
```

| WGI Code | Description | Scale |
|----------|-------------|-------|
| `GE.EST` | Government effectiveness | −2.5 to +2.5 |
| `RL.EST` | Rule of law | −2.5 to +2.5 |
| `CC.EST` | Control of corruption | −2.5 to +2.5 |
| `PV.EST` | Political stability and absence of violence | −2.5 to +2.5 |

**Known issues:**
- Published biennially before 2002; interpolated to annual using linear method
- Some country-years show no change (indicator published same value in adjacent years)

---

## 3. World Bank Health Nutrition & Population (HNP)

**URL:** https://databank.worldbank.org/source/health-nutrition-and-population-statistics

| HNP Code | Description | Unit |
|----------|-------------|------|
| `SH.XPD.CHEX.GD.ZS` | Current health expenditure (% GDP) | % |
| `SH.MED.PHYS.ZS` | Physicians (per 1,000 population) | Per 1,000 |
| `SH.MED.BEDS.ZS` | Hospital beds (per 1,000 population) | Per 1,000 |
| `SH.DYN.MORT` | Under-5 mortality rate | Per 1,000 live births |

---

## 4. UNESCO Institute for Statistics (UIS)

**URL:** https://uis.unesco.org/

**Citation:** UNESCO Institute for Statistics (2024). *Education Statistics*. Montreal: UIS.

**Series:**

| UIS Code | Description | Unit |
|----------|-------------|------|
| `EDU_FIN_EXP_PT_GDP` | Government expenditure on education (% GDP) | % |
| `NERA_1_CP` | Net enrollment rate, primary | % |
| `XUNIT_PPPCONST_1_FSGOV` | Gov expenditure per pupil, primary (PPP $) | USD |

**Access:**
```python
# REST API: https://api.uis.unesco.org/sdmx/v2/data/{dataset}/{indicator}
import requests
resp = requests.get(f"https://api.uis.unesco.org/sdmx/v2/data/UNESCO,EDU_FINANCE/...")
```

**Known issues:**
- Significant missingness for low-income countries (>30% missing in some series)
- Multiple-imputation approach: linear interpolation for gaps ≤5 years; median by income group for longer gaps

---

## 5. UN Population Division — World Population Prospects 2024

**URL:** https://population.un.org/wpp/

**Citation:** United Nations, Department of Economic and Social Affairs, Population Division (2024). *World Population Prospects 2024*. UN DESA/POP/2024/TR/NO. 1.

**Files downloaded:**
- `WPP2024_PopulationBySingleAgeSex_Medium_1950-2100.csv`
- Aggregated to age groups: 0–14, 15–64, 65+

**Variables constructed:**

| Variable | Description |
|----------|-------------|
| `age_0_14_pct` | Population aged 0–14 as % of total |
| `age_15_64_pct` | Working-age population (15–64) as % |
| `age_65_plus_pct` | Population aged 65+ as % |
| `old_age_dependency` | (65+) / (15–64) ratio |

**Access note:** 5-year projection intervals interpolated to annual using cubic spline.

---

## 6. WHO Global Health Observatory (GHO)

**URL:** https://www.who.int/data/gho

**Citation:** World Health Organization (2024). *Global Health Observatory data repository*. Geneva: WHO.

**API:**
```python
url = "https://ghoapi.azureedge.net/api/{indicator_code}"
```

| GHO Code | Description |
|----------|-------------|
| `WHOSIS_000001` | Life expectancy at birth (WHO estimate, used for validation) |
| `GHED_CHEGDP_SHA2011` | Current health expenditure as % GDP (cross-check vs WDI) |

---

## 7. Our World in Data — COVID-19 Dataset

**URL:** https://ourworldindata.org/covid-deaths

**GitHub:** https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv

**Variables used:**

| Variable | Description |
|----------|-------------|
| `excess_mortality_cumulative_per_million` | Cumulative excess deaths per million (2020–2023) |
| `new_deaths_smoothed_per_million` | 7-day smoothed COVID-19 deaths per million |
| `total_vaccinations_per_hundred` | Cumulative vaccine doses per 100 people |

**Usage:** COVID-19 Validation page only. Used to contextualize 2020–2021 LE drops and test model resilience.

---

## Country List

| ISO3 | Country | Income Group |
|------|---------|-------------|
| USA | United States | High income |
| DEU | Germany | High income |
| JPN | Japan | High income |
| GBR | United Kingdom | High income |
| FRA | France | High income |
| CAN | Canada | High income |
| AUS | Australia | High income |
| KOR | South Korea | High income |
| SWE | Sweden | High income |
| NLD | Netherlands | High income |
| CHN | China | Upper-middle income |
| BRA | Brazil | Upper-middle income |
| MEX | Mexico | Upper-middle income |
| IDN | Indonesia | Upper-middle income |
| ZAF | South Africa | Upper-middle income |
| THA | Thailand | Upper-middle income |
| COL | Colombia | Upper-middle income |
| PER | Peru | Upper-middle income |
| IND | India | Lower-middle income |
| NGA | Nigeria | Lower-middle income |
| BGD | Bangladesh | Lower-middle income |
| PAK | Pakistan | Lower-middle income |
| KEN | Kenya | Lower-middle income |
| GHA | Ghana | Lower-middle income |
| KHM | Cambodia | Lower-middle income |
| ETH | Ethiopia | Low income |
| TZA | Tanzania | Low income |
| UGA | Uganda | Low income |
| MOZ | Mozambique | Low income |
| BFA | Burkina Faso | Low income |

---

## Data Quality Notes

- **Overall completeness:** 725 country-year observations; mean missingness 8.3% across all variables
- **Worst missingness:** `education_exp_pct_gdp` (23% missing, especially low-income countries pre-2005)
- **Best coverage:** `life_expectancy`, `gdp_per_capita_ppp` (<2% missing)
- **Temporal coverage:** All 30 countries have data for 2000–2022; 2023–2024 are provisional estimates

Full data quality report available in `data/final/data_quality_report.txt`.
