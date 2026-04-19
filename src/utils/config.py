"""Project-wide configuration: paths, countries, indicators, time window."""
from __future__ import annotations

from pathlib import Path
from typing import Final

# ---- Paths ------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
FINAL_DIR: Final[Path] = DATA_DIR / "final"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
FIG_DIR: Final[Path] = OUTPUTS_DIR / "figures"
EDA_FIG_DIR: Final[Path] = FIG_DIR / "eda"
TABLES_DIR: Final[Path] = OUTPUTS_DIR / "tables"
MODELS_DIR: Final[Path] = OUTPUTS_DIR / "models"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

for _d in (RAW_DIR, PROCESSED_DIR, FINAL_DIR, EDA_FIG_DIR, TABLES_DIR, MODELS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Time window ------------------------------------------------------------
YEAR_START: Final[int] = 2000
YEAR_END: Final[int] = 2024
YEARS: Final[list[int]] = list(range(YEAR_START, YEAR_END + 1))

# ---- Countries (ISO-3) ------------------------------------------------------
HIGH_INCOME: Final[dict[str, str]] = {
    "USA": "United States", "DEU": "Germany", "GBR": "United Kingdom",
    "FRA": "France", "JPN": "Japan", "CAN": "Canada", "KOR": "South Korea",
    "ITA": "Italy", "AUS": "Australia", "NLD": "Netherlands",
}
MIDDLE_INCOME: Final[dict[str, str]] = {
    "CHN": "China", "MEX": "Mexico", "BRA": "Brazil", "TUR": "Turkey",
    "IND": "India", "IDN": "Indonesia", "VNM": "Vietnam", "PHL": "Philippines",
    "EGY": "Egypt", "NGA": "Nigeria",
}
LOW_INCOME: Final[dict[str, str]] = {
    "BDI": "Burundi", "SSD": "South Sudan", "MWI": "Malawi", "MOZ": "Mozambique",
    "NER": "Niger", "MDG": "Madagascar", "TCD": "Chad",
    "CAF": "Central African Republic", "COD": "DR Congo", "LBR": "Liberia",
}
COUNTRIES: Final[dict[str, str]] = {**HIGH_INCOME, **MIDDLE_INCOME, **LOW_INCOME}
INCOME_GROUP: Final[dict[str, str]] = (
    {c: "high" for c in HIGH_INCOME}
    | {c: "middle" for c in MIDDLE_INCOME}
    | {c: "low" for c in LOW_INCOME}
)
ISO3_LIST: Final[list[str]] = list(COUNTRIES.keys())

# ---- World Bank indicators --------------------------------------------------
# Map: clean_name -> WDI code. Indicator codes verified against api.worldbank.org.
WB_INDICATORS: Final[dict[str, str]] = {
    # Economic (15)
    "gdp_per_capita_usd":              "NY.GDP.PCAP.CD",
    "gdp_per_capita_ppp":              "NY.GDP.PCAP.PP.CD",
    "gdp_growth":                      "NY.GDP.MKTP.KD.ZG",
    "gni_per_capita_atlas":            "NY.GNP.PCAP.CD",
    "gini":                            "SI.POV.GINI",
    "poverty_rate_215":                "SI.POV.DDAY",
    "unemployment":                    "SL.UEM.TOTL.ZS",
    "inflation_cpi":                   "FP.CPI.TOTL.ZG",
    "gov_expenditure_pct_gdp":         "GC.XPN.TOTL.GD.ZS",
    "fdi_inflows_pct_gdp":             "BX.KLT.DINV.WD.GD.ZS",
    "remittances_pct_gdp":             "BX.TRF.PWKR.DT.GD.ZS",
    "trade_pct_gdp":                   "NE.TRD.GNFS.ZS",
    "agriculture_pct_gdp":             "NV.AGR.TOTL.ZS",
    "industry_pct_gdp":                "NV.IND.TOTL.ZS",
    "services_pct_gdp":                "NV.SRV.TOTL.ZS",

    # Health (20)
    "life_expectancy":                 "SP.DYN.LE00.IN",
    "life_expectancy_male":            "SP.DYN.LE00.MA.IN",
    "life_expectancy_female":          "SP.DYN.LE00.FE.IN",
    "infant_mortality":                "SP.DYN.IMRT.IN",
    "under5_mortality":                "SH.DYN.MORT",
    "maternal_mortality":              "SH.STA.MMRT",
    "adult_mortality_male":            "SP.DYN.AMRT.MA",
    "adult_mortality_female":          "SP.DYN.AMRT.FE",
    "health_exp_pct_gdp":              "SH.XPD.CHEX.GD.ZS",
    "health_exp_per_capita":           "SH.XPD.CHEX.PC.CD",
    "oop_health_exp_pct":              "SH.XPD.OOPC.CH.ZS",
    "hospital_beds_per_1000":          "SH.MED.BEDS.ZS",
    "physicians_per_1000":             "SH.MED.PHYS.ZS",
    "nurses_per_1000":                 "SH.MED.NUMW.P3",
    "immunization_dpt":                "SH.IMM.IDPT",
    "immunization_measles":            "SH.IMM.MEAS",
    "uhc_index":                       "SH_UHC_SCI",
    "water_access":                    "SH.H2O.BASW.ZS",
    "sanitation_access":               "SH.STA.BASS.ZS",
    "tb_incidence":                    "SH.TBS.INCD",

    # Education (8)
    "education_exp_pct_gdp":           "SE.XPD.TOTL.GD.ZS",
    "literacy_adult":                  "SE.ADT.LITR.ZS",
    "primary_enroll":                  "SE.PRM.NENR",
    "secondary_enroll":                "SE.SEC.NENR",
    "tertiary_enroll":                 "SE.TER.ENRR",
    "gender_parity_secondary":         "SE.ENR.SECO.FM.ZS",
    # NOTE: mean_years_schooling and expected_years_schooling are sourced from
    # UNDP HDR (undp_mys, undp_eys), which has full 2000-2024 coverage. The WB
    # codes BAR.SCHL.15UP / HD.HCI.EYRS are sparse or archived.

    # Governance (6 from WGI; 4 supplementary from WDI)
    "wgi_political_stability":         "GOV_WGI_PV.EST",
    "wgi_gov_effectiveness":           "GOV_WGI_GE.EST",
    "wgi_regulatory_quality":          "GOV_WGI_RQ.EST",
    "wgi_rule_of_law":                 "GOV_WGI_RL.EST",
    "wgi_corruption_control":          "GOV_WGI_CC.EST",
    "wgi_voice_accountability":        "GOV_WGI_VA.EST",
    "military_exp_pct_gdp":            "MS.MIL.XPND.GD.ZS",
    "tax_revenue_pct_gdp":             "GC.TAX.TOTL.GD.ZS",
    "internet_users_pct":              "IT.NET.USER.ZS",
    "mobile_subs_per_100":             "IT.CEL.SETS.P2",

    # Demographics (8)
    "population_total":                "SP.POP.TOTL",
    "population_density":              "EN.POP.DNST",
    "urban_pop_pct":                   "SP.URB.TOTL.IN.ZS",
    "fertility_rate":                  "SP.DYN.TFRT.IN",
    "age_dependency_ratio":            "SP.POP.DPND",
    "age_0_14_pct":                    "SP.POP.0014.TO.ZS",
    "age_15_64_pct":                   "SP.POP.1564.TO.ZS",
    "age_65_plus_pct":                 "SP.POP.65UP.TO.ZS",
}

# ---- WHO indicators (GHO codes) --------------------------------------------
# HALE supplements the World Bank life-expectancy series.
WHO_INDICATORS: Final[dict[str, str]] = {
    "hale_total":                      "WHOSIS_000002",
}

# ---- OWID COVID columns we keep --------------------------------------------
OWID_COVID_COLS: Final[dict[str, str]] = {
    "total_deaths_per_million":        "covid_deaths_per_million",
    "excess_mortality_cumulative_per_million": "covid_excess_mortality_per_million",
    "people_fully_vaccinated_per_hundred":     "covid_fully_vaccinated_pct",
    "stringency_index":                "covid_stringency_index",
    "total_tests_per_thousand":        "covid_tests_per_thousand",
    "new_cases_per_million":           "covid_new_cases_per_million",
    "icu_patients_per_million":        "covid_icu_per_million",
    "reproduction_rate":               "covid_reproduction_rate",
}

# ---- Variable groups for analysis & reporting -------------------------------
VAR_GROUPS: Final[dict[str, list[str]]] = {
    "economic":   [k for k in WB_INDICATORS if any(s in k for s in (
        "gdp", "gni", "gini", "poverty", "unemployment", "inflation",
        "gov_expenditure", "fdi", "remittances", "trade",
        "agriculture", "industry", "services"))],
    "health":     [k for k in WB_INDICATORS if any(s in k for s in (
        "life_expectancy", "mortality", "health_exp", "oop", "hospital", "physicians",
        "nurses", "immunization", "uhc", "water", "sanitation", "tb_"))] + list(WHO_INDICATORS),
    "education":  [k for k in WB_INDICATORS if any(s in k for s in (
        "education_exp", "literacy", "enroll", "schooling", "gender_parity"))],
    "governance": [k for k in WB_INDICATORS if k.startswith("wgi_")
                  or k in {"military_exp_pct_gdp", "tax_revenue_pct_gdp",
                           "internet_users_pct", "mobile_subs_per_100"}],
    "demographic":[k for k in WB_INDICATORS if any(s in k for s in (
        "population", "urban_pop", "fertility", "age_"))],
    "covid":      list(OWID_COVID_COLS.values()),
}

PRIMARY_OUTCOME: Final[str] = "life_expectancy"
PRIMARY_TREATMENT: Final[str] = "gdp_per_capita_ppp"

# ---- World Bank database overrides ------------------------------------------
# Most WDI indicators live in source 2 (default). WGI lives in source 3.
WB_INDICATOR_DB: Final[dict[str, int]] = {
    code: 3 for code in WB_INDICATORS.values() if code.startswith("GOV_WGI_")
}

# ---- Quality thresholds -----------------------------------------------------
MAX_MISSING_PCT_PER_COUNTRY: Final[float] = 30.0  # exclude country if exceeded
GDP_CROSSCHECK_TOLERANCE: Final[float] = 0.02     # World Bank vs IMF within 2%
