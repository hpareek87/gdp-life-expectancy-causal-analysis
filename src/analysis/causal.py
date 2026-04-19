"""Phase 2 — Causal Inference.

Five complementary methods quantifying the causal effect of GDP on life
expectancy: Granger causality, Panel Fixed Effects, IV-2SLS, Difference-in-
Differences (event study), and Synthetic Control.

Usage::

    from src.analysis.causal import run_all_causal
    results = run_all_causal()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize as opt
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from ..utils.config import FINAL_DIR, INCOME_GROUP
from ..utils.logging_setup import get_logger

logger = get_logger("analysis.causal")

# ── Constants ─────────────────────────────────────────────────────────────────
OUTCOME = "life_expectancy"
TREATMENT = "gdp_per_capita_ppp"
LOG_TREATMENT = "log_gdp_per_capita_ppp"

CONTROLS_BASE = [
    "health_exp_pct_gdp", "education_exp_pct_gdp",
    "wgi_gov_effectiveness", "urban_pop_pct",
]
CONTROLS_FULL = CONTROLS_BASE + [
    "fertility_rate", "trade_pct_gdp", "age_65_plus_pct",
    "inflation_cpi",
]

# DiD events: (treated_iso3, control_iso3_list, reform_year, window)
DID_EVENTS: dict[str, dict] = {
    "IDN_JKN_2014": {
        "treated": ["IDN"],
        "control": ["PHL", "VNM", "EGY"],
        "reform_year": 2014,
        "window": (2005, 2022),
        "label": "Indonesia JKN universal insurance (2014)",
    },
    "VNM_UHC_2009": {
        "treated": ["VNM"],
        "control": ["IDN", "PHL", "EGY"],
        "reform_year": 2009,
        "window": (2000, 2018),
        "label": "Vietnam health insurance expansion (2009)",
    },
    "CHN_NCMS_2009": {
        "treated": ["CHN"],
        "control": ["IND", "IDN", "BRA", "MEX"],
        "reform_year": 2009,
        "window": (2000, 2021),
        "label": "China New Rural Cooperative Medical Scheme (2009)",
    },
}

SYNTH_TREATED = "CHN"
SYNTH_REFORM_YEAR = 2009
SYNTH_PREDICTORS = [
    "gdp_per_capita_ppp", "health_exp_pct_gdp", "fertility_rate",
    "education_exp_pct_gdp", "urban_pop_pct", "wgi_gov_effectiveness",
]

# ── Data loading ──────────────────────────────────────────────────────────────

def load_df() -> pd.DataFrame:
    path = FINAL_DIR / "master_dataset.csv"
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    return df


def _panel_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set (iso3, year) MultiIndex required by linearmodels."""
    return df.set_index(["iso3", "year"])


def _balanced_subset(df: pd.DataFrame, cols: list[str],
                     min_obs: int = 15) -> pd.DataFrame:
    """Drop country-years with missing required columns; exclude thin panels."""
    sub = df.dropna(subset=cols).copy()
    counts = sub.groupby("iso3").size()
    keep = counts[counts >= min_obs].index
    return sub[sub["iso3"].isin(keep)]


# ── Method 1: Granger Causality ───────────────────────────────────────────────

@dataclass
class GrangerResult:
    direction: str               # "GDP→LE" or "LE→GDP"
    country: str
    optimal_lag: int
    fstats: dict[int, float]     # lag -> F-stat
    pvalues_raw: dict[int, float]
    pvalues_corrected: dict[int, float]
    significant: bool            # after Bonferroni


def _adf_test(series: pd.Series) -> bool:
    """True if series is stationary (ADF rejects unit root at 5%)."""
    if len(series.dropna()) < 10:
        return True
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1] < 0.05


def _granger_one(y: np.ndarray, x: np.ndarray,
                 maxlag: int = 7) -> tuple[dict, dict]:
    """Return {lag: F-stat} and {lag: p-value} for y caused by x."""
    data = np.column_stack([y, x])
    try:
        res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    except Exception:
        return {}, {}
    fstats = {lag: float(res[lag][0]["ssr_ftest"][0]) for lag in res}
    pvals  = {lag: float(res[lag][0]["ssr_ftest"][1]) for lag in res}
    return fstats, pvals


def run_granger(
    df: pd.DataFrame | None = None,
    maxlag: int = 7,
    alpha: float = 0.05,
) -> dict[str, Any]:
    df = df if df is not None else load_df()
    required = [OUTCOME, TREATMENT]
    sub = _balanced_subset(df, required, min_obs=maxlag + 5)

    results_gdp2le: list[GrangerResult] = []
    results_le2gdp: list[GrangerResult] = []
    countries = sorted(sub["iso3"].unique())

    all_pvals_gdp2le: list[float] = []
    all_pvals_le2gdp: list[float] = []

    for iso in countries:
        cs = sub[sub["iso3"] == iso].sort_values("year")
        le = cs[OUTCOME].values
        gdp = cs[TREATMENT].values

        # Stationarity — use first differences if non-stationary
        if not _adf_test(pd.Series(le)):
            le = np.diff(le)
            gdp = np.diff(gdp)

        f_g2l, p_g2l = _granger_one(le, gdp, maxlag)
        f_l2g, p_l2g = _granger_one(gdp, le, maxlag)

        if not f_g2l:
            continue

        # Collect best-lag p-values for Bonferroni correction
        best_lag_g2l = min(p_g2l, key=p_g2l.get) if p_g2l else 1
        best_lag_l2g = min(p_l2g, key=p_l2g.get) if p_l2g else 1
        all_pvals_gdp2le.append(p_g2l.get(best_lag_g2l, 1.0))
        all_pvals_le2gdp.append(p_l2g.get(best_lag_l2g, 1.0))

        results_gdp2le.append(GrangerResult(
            direction="GDP→LE", country=iso,
            optimal_lag=best_lag_g2l,
            fstats=f_g2l, pvalues_raw=p_g2l,
            pvalues_corrected={}, significant=False,
        ))
        results_le2gdp.append(GrangerResult(
            direction="LE→GDP", country=iso,
            optimal_lag=best_lag_l2g,
            fstats=f_l2g, pvalues_raw=p_l2g,
            pvalues_corrected={}, significant=False,
        ))

    # Bonferroni correction
    def _apply_bonf(results: list[GrangerResult],
                    raw_pvals: list[float]) -> None:
        if not raw_pvals:
            return
        rej, p_corr, _, _ = multipletests(raw_pvals, method="bonferroni")
        for i, r in enumerate(results):
            r.significant = bool(rej[i])
            r.pvalues_corrected = {r.optimal_lag: p_corr[i]}

    _apply_bonf(results_gdp2le, all_pvals_gdp2le)
    _apply_bonf(results_le2gdp, all_pvals_le2gdp)

    # Pooled panel Granger (all countries stacked — demeaned within country)
    le_dm = sub.groupby("iso3")[OUTCOME].transform(lambda x: x - x.mean())
    gdp_dm = sub.groupby("iso3")[TREATMENT].transform(lambda x: x - x.mean())
    pooled_f, pooled_p = _granger_one(le_dm.values, gdp_dm.values, maxlag)

    n_sig_g2l = sum(r.significant for r in results_gdp2le)
    n_sig_l2g = sum(r.significant for r in results_le2gdp)
    logger.info(
        "Granger: GDP→LE significant in %d/%d countries; LE→GDP in %d/%d",
        n_sig_g2l, len(results_gdp2le), n_sig_l2g, len(results_le2gdp),
    )

    return {
        "gdp_to_le": results_gdp2le,
        "le_to_gdp": results_le2gdp,
        "pooled_fstats_gdp2le": pooled_f,
        "pooled_pvals_gdp2le": pooled_p,
        "n_countries": len(countries),
        "n_sig_gdp2le": n_sig_g2l,
        "n_sig_le2gdp": n_sig_l2g,
        "summary": {
            "pct_sig_gdp2le": round(n_sig_g2l / max(len(results_gdp2le), 1) * 100, 1),
            "pct_sig_le2gdp": round(n_sig_l2g / max(len(results_le2gdp), 1) * 100, 1),
            "most_common_lag_gdp2le": (
                pd.Series([r.optimal_lag for r in results_gdp2le]).mode().iloc[0]
                if results_gdp2le else None),
        },
    }


# ── Method 2: Panel Fixed Effects ────────────────────────────────────────────

@dataclass
class PanelFEResult:
    spec: str
    nobs: int
    rsquared: float
    coef_gdp: float
    se_gdp: float
    pval_gdp: float
    ci95_gdp: tuple[float, float]
    coefs: pd.Series
    pvalues: pd.Series
    std_errors: pd.Series = field(default_factory=pd.Series)
    fstat: float | None = None


def _make_panel_data(df: pd.DataFrame, controls: list[str],
                     log_gdp: bool = True) -> pd.DataFrame:
    gdp_col = LOG_TREATMENT if log_gdp else TREATMENT
    cols = [OUTCOME, gdp_col] + controls + ["income_group"]
    sub = _balanced_subset(df, cols, min_obs=10)
    return sub


def run_panel_fe(df: pd.DataFrame | None = None) -> dict[str, PanelFEResult]:
    df = df if df is not None else load_df()
    out: dict[str, PanelFEResult] = {}

    specs: dict[str, list[str]] = {
        "baseline":    [],
        "controls_base": CONTROLS_BASE,
        "controls_full": CONTROLS_FULL,
    }

    for spec_name, controls in specs.items():
        sub = _make_panel_data(df, controls)
        pdata = _panel_index(sub)
        dep = pdata[OUTCOME]
        # linearmodels requires exog to exclude the dependent variable
        exog_cols = [LOG_TREATMENT] + controls
        exog = pdata[exog_cols].copy()

        try:
            model = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                             drop_absorbed=True)
            res = model.fit(cov_type="clustered", cluster_entity=True)
        except Exception as exc:
            logger.warning("Panel FE (%s) failed: %s", spec_name, exc)
            continue

        params = res.params
        pvals  = res.pvalues
        ci     = res.conf_int()
        gdp_ci = (
            float(ci.loc[LOG_TREATMENT, "lower"]),
            float(ci.loc[LOG_TREATMENT, "upper"]),
        ) if LOG_TREATMENT in ci.index else (np.nan, np.nan)

        out[spec_name] = PanelFEResult(
            spec=spec_name,
            nobs=int(res.nobs),
            rsquared=float(res.rsquared),
            coef_gdp=float(params.get(LOG_TREATMENT, np.nan)),
            se_gdp=float(res.std_errors.get(LOG_TREATMENT, np.nan)),
            pval_gdp=float(pvals.get(LOG_TREATMENT, np.nan)),
            ci95_gdp=gdp_ci,
            coefs=params,
            pvalues=pvals,
            std_errors=res.std_errors,
        )
        logger.info(
            "Panel FE [%s]: β_GDP=%.3f (se=%.3f, p=%.4f), N=%d",
            spec_name,
            out[spec_name].coef_gdp,
            out[spec_name].se_gdp,
            out[spec_name].pval_gdp,
            out[spec_name].nobs,
        )

    return out


def run_panel_fe_subgroups(
    df: pd.DataFrame | None = None,
) -> dict[str, PanelFEResult]:
    df = df if df is not None else load_df()
    df["income_group"] = df["iso3"].map(INCOME_GROUP)
    out: dict[str, PanelFEResult] = {}

    for ig in ["high", "middle", "low"]:
        sub_ig = df[df["income_group"] == ig]
        sub = _make_panel_data(sub_ig, CONTROLS_BASE)
        if sub["iso3"].nunique() < 3:
            continue
        pdata = _panel_index(sub)
        dep = pdata[OUTCOME]
        exog = pdata[[LOG_TREATMENT] + CONTROLS_BASE].copy()
        try:
            model = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                             drop_absorbed=True)
            res = model.fit(cov_type="clustered", cluster_entity=True)
        except Exception as exc:
            logger.warning("Panel FE subgroup %s failed: %s", ig, exc)
            continue

        params = res.params
        out[ig] = PanelFEResult(
            spec=ig,
            nobs=int(res.nobs),
            rsquared=float(res.rsquared),
            coef_gdp=float(params.get(LOG_TREATMENT, np.nan)),
            se_gdp=float(res.std_errors.get(LOG_TREATMENT, np.nan)),
            pval_gdp=float(res.pvalues.get(LOG_TREATMENT, np.nan)),
            ci95_gdp=(
                float(res.conf_int().loc[LOG_TREATMENT, "lower"]),
                float(res.conf_int().loc[LOG_TREATMENT, "upper"]),
            ) if LOG_TREATMENT in res.conf_int().index else (np.nan, np.nan),
            coefs=params, pvalues=res.pvalues,
            std_errors=res.std_errors,
        )
        logger.info("Panel FE [%s]: β_GDP=%.3f (p=%.4f)", ig,
                    out[ig].coef_gdp, out[ig].pval_gdp)

    return out


# ── Method 3: IV-2SLS ────────────────────────────────────────────────────────

def _build_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """Construct two Bartik-style instruments for log GDP per capita.

    Instrument 1 — External demand shock:
      For country i, year t: mean GDP growth of all OTHER countries (global
      demand proxy that moves income through exports but not health directly).

    Instrument 2 — Trade-weighted external demand:
      External-demand × own trade openness (lagged). Countries more open to
      trade feel global shocks more strongly.
    """
    df = df.copy()
    df["year"] = df["year"].astype(int)

    # Instrument 1: mean gdp_growth of all OTHER countries
    world_stats = df.groupby("year")["gdp_growth"].agg(
        world_mean="mean", world_sum="sum", world_n="count"
    ).reset_index()
    df = df.merge(world_stats, on="year", how="left")
    # Exclude own country: (sum_all - own) / (N-1)
    df["ext_demand"] = (df["world_sum"] - df["gdp_growth"]) / (df["world_n"] - 1)

    # Instrument 2: ext_demand × lagged trade openness
    df = df.sort_values(["iso3", "year"])
    df["trade_lag1"] = df.groupby("iso3")["trade_pct_gdp"].shift(1)
    df["ext_demand_x_trade"] = df["ext_demand"] * df["trade_lag1"]

    return df


@dataclass
class IVResult:
    spec: str
    nobs: int
    coef_gdp: float
    se_gdp: float
    pval_gdp: float
    ci95_gdp: tuple[float, float]
    first_stage_fstat: float
    sargan_pval: float | None   # overidentification (None if just-identified)
    wu_hausman_pval: float | None  # endogeneity test
    weak_instrument: bool


def run_iv(df: pd.DataFrame | None = None) -> dict[str, IVResult]:
    df = df if df is not None else load_df()
    df = _build_instruments(df)
    out: dict[str, IVResult] = {}

    inst_cols = ["ext_demand", "ext_demand_x_trade"]
    required = [OUTCOME, LOG_TREATMENT, "ext_demand", "trade_lag1"] + CONTROLS_BASE
    sub = _balanced_subset(df, required, min_obs=10)
    pdata = _panel_index(sub)

    dep  = pdata[OUTCOME]
    endog = pdata[[LOG_TREATMENT]]

    specs_iv: dict[str, dict] = {
        "iv_baseline": {
            "exog":        CONTROLS_BASE[:2],   # partial controls to avoid overparameterization
            "instruments": ["ext_demand"],
            "note":        "Just-identified: 1 endogenous, 1 instrument",
        },
        "iv_overidentified": {
            "exog":        CONTROLS_BASE,
            "instruments": inst_cols,
            "note":        "Overidentified: 2 instruments; Sargan test valid",
        },
    }

    for spec_name, spec in specs_iv.items():
        inst_use = [c for c in spec["instruments"] if c in pdata.columns]
        exog_cols = [c for c in spec["exog"] if c in pdata.columns]
        if not inst_use:
            continue
        exog = pdata[exog_cols].copy() if exog_cols else None
        instruments = pdata[inst_use].copy()

        try:
            model = IV2SLS(dep, exog, endog, instruments)
            res = model.fit(cov_type="robust")
        except Exception as exc:
            logger.warning("IV (%s) failed: %s", spec_name, exc)
            continue

        params = res.params
        ci    = res.conf_int()
        gdp_ci = (
            float(ci.loc[LOG_TREATMENT, "lower"]),
            float(ci.loc[LOG_TREATMENT, "upper"]),
        ) if LOG_TREATMENT in ci.index else (np.nan, np.nan)

        # First-stage F-stat (linearmodels 7 column is "f.stat")
        try:
            fs_diag = res.first_stage.diagnostics
            fs_fstat = float(fs_diag["f.stat"].iloc[0])
        except Exception:
            fs_fstat = np.nan

        # Sargan overidentification test
        sargan_p: float | None = None
        if len(inst_use) > 1:
            try:
                oi = res.sargan
                sargan_p = float(oi.pval)
            except Exception:
                pass

        # Wu-Hausman endogeneity test
        wh_p: float | None = None
        try:
            wh = res.wu_hausman()
            wh_p = float(wh.pval)
        except Exception:
            pass

        out[spec_name] = IVResult(
            spec=spec_name,
            nobs=int(res.nobs),
            coef_gdp=float(params.get(LOG_TREATMENT, np.nan)),
            se_gdp=float(res.std_errors.get(LOG_TREATMENT, np.nan)),
            pval_gdp=float(res.pvalues.get(LOG_TREATMENT, np.nan)),
            ci95_gdp=gdp_ci,
            first_stage_fstat=fs_fstat,
            sargan_pval=sargan_p,
            wu_hausman_pval=wh_p,
            weak_instrument=fs_fstat < 10,
        )
        logger.info(
            "IV [%s]: β_GDP=%.3f (se=%.3f, p=%.4f), 1st-stage F=%.1f, "
            "Sargan p=%s, WH p=%s",
            spec_name, out[spec_name].coef_gdp, out[spec_name].se_gdp,
            out[spec_name].pval_gdp, fs_fstat,
            f"{sargan_p:.3f}" if sargan_p else "N/A",
            f"{wh_p:.3f}" if wh_p else "N/A",
        )

    return out


# ── Method 4: Difference-in-Differences ──────────────────────────────────────

@dataclass
class DIDResult:
    event: str
    label: str
    nobs: int
    att: float             # average treatment effect on treated
    se_att: float
    pval_att: float
    ci95_att: tuple[float, float]
    parallel_trends_pval: float | None
    event_study: pd.DataFrame   # relative_year × (coef, se, ci_lo, ci_hi)


def _parallel_trends_test(df_event: pd.DataFrame, pre_years: list[int],
                          reform_year: int) -> float | None:
    """Regress outcome on treat×year_fe in pre-period; test joint significance."""
    pre = df_event[df_event["year"] < reform_year].copy()
    if pre.empty or pre["treat"].nunique() < 2:
        return None
    pre["year_fe"] = pre["year"].astype(str)
    try:
        import statsmodels.formula.api as smf
        res = smf.ols("life_expectancy ~ treat * C(year)", data=pre).fit()
        interact_terms = [c for c in res.params.index if "treat:C(year)" in c]
        if not interact_terms:
            return None
        R = np.zeros((len(interact_terms), len(res.params)))
        for i, t in enumerate(interact_terms):
            j = list(res.params.index).index(t)
            R[i, j] = 1
        from statsmodels.stats.anova import anova_lm
        ftest = res.f_test(R)
        return float(ftest.pvalue)
    except Exception:
        return None


def run_did(df: pd.DataFrame | None = None) -> dict[str, DIDResult]:
    df = df if df is not None else load_df()
    out: dict[str, DIDResult] = {}

    for event_name, cfg in DID_EVENTS.items():
        treated  = cfg["treated"]
        control  = cfg["control"]
        reform_y = cfg["reform_year"]
        t0, t1   = cfg["window"]

        sub = df[df["iso3"].isin(treated + control) &
                 df["year"].between(t0, t1)].copy()
        if sub.empty or sub["iso3"].nunique() < 2:
            logger.warning("DiD [%s]: insufficient countries", event_name)
            continue

        sub["treat"] = sub["iso3"].isin(treated).astype(int)
        sub["post"]  = (sub["year"] >= reform_y).astype(int)
        sub["post_treat"] = sub["treat"] * sub["post"]
        sub["rel_year"] = sub["year"] - reform_y

        # ── Standard DiD ──────────────────────────────────
        controls_did = [c for c in CONTROLS_BASE if c in sub.columns]
        did_cols = [OUTCOME, "treat", "post", "post_treat"] + controls_did
        sub_did = sub.dropna(subset=did_cols)

        try:
            pdata = sub_did.set_index(["iso3", "year"])
            dep = pdata[OUTCOME]
            exog = pdata[["post_treat"] + controls_did].copy()
            model = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                             drop_absorbed=True)
            res = model.fit(cov_type="clustered", cluster_entity=True)
            params  = res.params
            pvals   = res.pvalues
            se_vals = res.std_errors
            ci      = res.conf_int()
            att     = float(params.get("post_treat", np.nan))
            se_att  = float(se_vals.get("post_treat", np.nan))
            pval_at = float(pvals.get("post_treat", np.nan))
            ci95_at = (
                float(ci.loc["post_treat", "lower"]),
                float(ci.loc["post_treat", "upper"]),
            ) if "post_treat" in ci.index else (np.nan, np.nan)
        except Exception as exc:
            logger.warning("DiD [%s] panel regression failed: %s", event_name, exc)
            continue

        # ── Event study (relative-year coefficients) ──────
        event_rows: list[dict] = []
        rel_years = sorted(sub["rel_year"].unique())
        # Omit rel_year == -1 as reference
        rel_years_use = [r for r in rel_years if r != -1 and abs(r) <= 8]
        for ryr in rel_years_use:
            sub[f"d_{ryr}"] = ((sub["rel_year"] == ryr) & (sub["treat"] == 1)).astype(int)
        d_cols = [f"d_{ryr}" for ryr in rel_years_use if f"d_{ryr}" in sub.columns]
        if d_cols:
            try:
                pdata2 = sub.dropna(subset=[OUTCOME] + d_cols).set_index(["iso3", "year"])
                dep2   = pdata2[OUTCOME]
                exog2  = pdata2[d_cols + controls_did].copy()
                model2 = PanelOLS(dep2, exog2, entity_effects=True,
                                  time_effects=True, drop_absorbed=True)
                res2   = model2.fit(cov_type="clustered", cluster_entity=True)
                ci2    = res2.conf_int()
                for ryr, dcol in zip(rel_years_use, d_cols):
                    coef = float(res2.params.get(dcol, np.nan))
                    se   = float(res2.std_errors.get(dcol, np.nan))
                    lo   = float(ci2.loc[dcol, "lower"]) if dcol in ci2.index else np.nan
                    hi   = float(ci2.loc[dcol, "upper"]) if dcol in ci2.index else np.nan
                    event_rows.append({"rel_year": ryr, "coef": coef, "se": se,
                                       "ci_lo": lo, "ci_hi": hi})
                # Also add the reference year (rel_year=-1) with zeros
                event_rows.append({"rel_year": -1, "coef": 0.0, "se": 0.0,
                                   "ci_lo": 0.0, "ci_hi": 0.0})
            except Exception:
                pass

        event_df = (pd.DataFrame(event_rows).sort_values("rel_year")
                    if event_rows else pd.DataFrame())

        pt_pval = _parallel_trends_test(sub, [], reform_y)

        out[event_name] = DIDResult(
            event=event_name, label=cfg["label"],
            nobs=int(res.nobs), att=att, se_att=se_att,
            pval_att=pval_at, ci95_att=ci95_at,
            parallel_trends_pval=pt_pval,
            event_study=event_df,
        )
        logger.info(
            "DiD [%s]: ATT=%.3f (se=%.3f, p=%.4f)",
            event_name, att, se_att, pval_at,
        )

    return out


# ── Method 5: Synthetic Control ───────────────────────────────────────────────

@dataclass
class SynthResult:
    treated_iso: str
    reform_year: int
    donors: list[str]
    weights: dict[str, float]
    pre_rmspe: float        # root mean squared prediction error in pre-period
    post_att: float         # average gap post-reform
    gaps: pd.Series         # year -> (actual - synthetic) life expectancy
    placebo_gaps: pd.DataFrame  # donor × year
    p_value: float | None   # fraction of placebos with |post/pre RMSPE| >= treated


def _synth_weights(Y_treat: np.ndarray, Y_donors: np.ndarray) -> np.ndarray:
    """Abadie et al. (2010) synthetic control: convex weights W* minimising
    ||Y_treat - Y_donors @ W||^2 over the pre-treatment period in raw space.
    To avoid scale domination we column-center Y_donors by the donor means.
    """
    n_donors = Y_donors.shape[1]
    # Minimise raw pre-period RMSPE — no normalisation so the convex combo
    # can match China's LEVEL from a weighted average of donors.
    def objective(w: np.ndarray) -> float:
        return float(np.sum((Y_treat - Y_donors @ w) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1}]
    bounds = [(0, 1)] * n_donors

    best_w, best_val = np.ones(n_donors) / n_donors, np.inf
    rng = np.random.default_rng(42)
    starts = [np.ones(n_donors) / n_donors]
    starts += [rng.dirichlet(np.ones(n_donors)) for _ in range(9)]
    for w0 in starts:
        try:
            r = opt.minimize(objective, w0, method="SLSQP",
                             constraints=constraints, bounds=bounds,
                             options={"ftol": 1e-14, "maxiter": 3000})
            if r.success and r.fun < best_val:
                best_val, best_w = r.fun, np.clip(r.x, 0, 1)
        except Exception:
            continue
    w_out = best_w / max(best_w.sum(), 1e-12)
    return w_out


def _rmspe(actual: np.ndarray, synth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - synth) ** 2)))


def run_synthetic_control(
    df: pd.DataFrame | None = None,
    treated: str = SYNTH_TREATED,
    reform_year: int = SYNTH_REFORM_YEAR,
    predictors: list[str] = SYNTH_PREDICTORS,
) -> SynthResult:
    df = df if df is not None else load_df()

    # Pre-period: all years before reform
    pre  = df[df["year"] < reform_year]
    post = df[df["year"] >= reform_year]

    required = [OUTCOME] + predictors
    available = df.dropna(subset=required)
    donors = [c for c in available["iso3"].unique()
              if c != treated
              and available[available["iso3"] == c]["year"].nunique() >= 10]

    logger.info("Synth control: treated=%s, %d donors, reform=%d",
                treated, len(donors), reform_year)

    # Pre-period outcome matrix (T_pre × N_donors)
    pre_years = sorted(pre["year"].unique())
    Y_treat = (pre[pre["iso3"] == treated].sort_values("year")[OUTCOME].values)
    # Use .values directly (already sorted); .reindex(range(...)) was buggy
    # because it matched on RangeIndex positions, not on year labels.
    Y_donors = np.column_stack([
        pre[pre["iso3"] == d].sort_values("year")[OUTCOME].values
        if len(pre[pre["iso3"] == d]) == len(pre_years)
        else np.full(len(pre_years), np.nan)
        for d in donors
    ])

    # Predictor matrix: column-normalized pre-period means
    X_treat = np.array([
        float(pre[pre["iso3"] == treated][p].mean())
        for p in predictors if p in pre.columns
    ])
    X_donors = np.column_stack([
        np.array([float(pre[pre["iso3"] == d][p].mean())
                  for p in predictors if p in pre.columns])
        for d in donors
    ])

    # Remove donor columns with NaNs
    valid_cols = ~np.any(np.isnan(Y_donors), axis=0) & ~np.any(np.isnan(X_donors), axis=0)
    Y_donors  = Y_donors[:, valid_cols]
    X_donors  = X_donors[:, valid_cols]
    donors_ok = [d for d, v in zip(donors, valid_cols) if v]

    if Y_donors.shape[1] == 0:
        logger.warning("No valid donors for synthetic control")
        return SynthResult(treated, reform_year, [], {}, np.nan, np.nan,
                           pd.Series(dtype=float), pd.DataFrame(), None)

    w = _synth_weights(Y_treat, Y_donors)

    # Compute synthetic outcome across all years
    all_years = sorted(df["year"].unique())
    synth_vals: list[float] = []
    actual_vals: list[float] = []
    for yr in all_years:
        yr_data = df[df["year"] == yr]
        actual_vals.append(
            float(yr_data[yr_data["iso3"] == treated][OUTCOME].values[0])
            if len(yr_data[yr_data["iso3"] == treated]) > 0 else np.nan)
        donor_vals = np.array([
            float(yr_data[yr_data["iso3"] == d][OUTCOME].values[0])
            if len(yr_data[yr_data["iso3"] == d]) > 0 else np.nan
            for d in donors_ok
        ])
        synth_vals.append(float(w @ donor_vals) if not np.any(np.isnan(donor_vals))
                          else np.nan)

    actual_s = pd.Series(actual_vals, index=all_years)
    synth_s  = pd.Series(synth_vals,  index=all_years)
    gaps     = actual_s - synth_s

    pre_rmspe  = _rmspe(
        actual_s[actual_s.index < reform_year].dropna().values,
        synth_s[synth_s.index < reform_year].dropna().values,
    )
    post_att = float(gaps[gaps.index >= reform_year].mean())

    # ── Placebo tests: apply synthetic control to each donor ──
    placebo_rows: dict[str, pd.Series] = {}
    for d in donors_ok:
        donors_for_d = [x for x in donors_ok if x != d]
        if len(donors_for_d) < 2:
            continue
        Yd = np.column_stack([
            pre[pre["iso3"] == dd].sort_values("year")[OUTCOME].values
            for dd in donors_for_d
            if len(pre[pre["iso3"] == dd]) == len(pre_years)
        ])
        Xd = np.column_stack([
            np.array([float(pre[pre["iso3"] == dd][p].mean())
                      for p in predictors if p in pre.columns])
            for dd in donors_for_d
            if len(pre[pre["iso3"] == dd]) == len(pre_years)
        ])
        Y_d_treat = pre[pre["iso3"] == d].sort_values("year")[OUTCOME].values
        X_d_treat = np.array([float(pre[pre["iso3"] == d][p].mean())
                               for p in predictors if p in pre.columns])
        if Yd.shape[0] != len(Y_d_treat) or Yd.shape[1] == 0:
            continue
        try:
            wd = _synth_weights(Y_d_treat, Yd)
            d_synth_vals = []
            for yr in all_years:
                yr_data = df[df["year"] == yr]
                dv = np.array([
                    float(yr_data[yr_data["iso3"] == dd][OUTCOME].values[0])
                    if len(yr_data[yr_data["iso3"] == dd]) > 0 else np.nan
                    for dd in donors_for_d
                    if len(pre[pre["iso3"] == dd]) == len(pre_years)
                ])
                d_synth_vals.append(
                    float(wd @ dv) if not np.any(np.isnan(dv)) else np.nan)
            d_actual = pd.Series([
                float(df[(df["iso3"] == d) & (df["year"] == yr)][OUTCOME].values[0])
                if len(df[(df["iso3"] == d) & (df["year"] == yr)]) > 0 else np.nan
                for yr in all_years
            ], index=all_years)
            d_gap = d_actual - pd.Series(d_synth_vals, index=all_years)
            placebo_rows[d] = d_gap
        except Exception:
            continue

    placebo_df = pd.DataFrame(placebo_rows)

    # RMSPE-ratio p-value
    p_value: float | None = None
    if not placebo_df.empty:
        def rmspe_ratio(gap_s: pd.Series) -> float:
            pre_gap = gap_s[gap_s.index < reform_year].dropna()
            post_gap = gap_s[gap_s.index >= reform_year].dropna()
            if len(pre_gap) == 0:
                return np.nan
            pre_r = _rmspe(pre_gap.values, np.zeros(len(pre_gap)))
            if pre_r < 1e-6:
                return np.nan
            return _rmspe(post_gap.values, np.zeros(len(post_gap))) / pre_r

        treated_ratio = rmspe_ratio(gaps)
        placebo_ratios = [rmspe_ratio(placebo_df[c]) for c in placebo_df.columns]
        valid_ratios = [r for r in placebo_ratios if not np.isnan(r)]
        if valid_ratios:
            p_value = float(sum(1 for r in valid_ratios if r >= treated_ratio)
                            / len(valid_ratios))

    logger.info(
        "Synth ctrl [%s]: pre-RMSPE=%.2f, post_ATT=%.2f yrs, p=%.3f",
        treated, pre_rmspe, post_att, p_value or np.nan,
    )
    return SynthResult(
        treated_iso=treated,
        reform_year=reform_year,
        donors=donors_ok,
        weights={d: round(float(wi), 4) for d, wi in zip(donors_ok, w)},
        pre_rmspe=pre_rmspe,
        post_att=post_att,
        gaps=gaps,
        placebo_gaps=placebo_df,
        p_value=p_value,
    )


# ── Robustness checks ─────────────────────────────────────────────────────────

def run_robustness(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Panel FE across alternative samples and specifications."""
    df = df if df is not None else load_df()
    out: dict[str, Any] = {}

    # R1: Exclude COVID years
    r1 = run_panel_fe(df[df["year"] < 2020].copy())
    out["exclude_covid"] = r1

    # R2: Pre-2010 only
    r2 = run_panel_fe(df[df["year"] <= 2010].copy())
    out["pre_2010"] = r2

    # R3: High-income excluded (avoid structural break)
    r3 = run_panel_fe(df[df["income_group"] != "high"].copy())
    out["excl_high_income"] = r3

    # R4: OLS (no fixed effects) — compare magnitude
    sub = _make_panel_data(df, CONTROLS_BASE)
    sub_ols = sub.copy()
    try:
        import statsmodels.formula.api as smf
        formula = f"{OUTCOME} ~ {LOG_TREATMENT} + " + " + ".join(CONTROLS_BASE)
        res_ols = smf.ols(formula, data=sub_ols).fit(cov_type="HC3")
        out["pooled_ols"] = {
            "coef_gdp": float(res_ols.params.get(LOG_TREATMENT, np.nan)),
            "se_gdp":   float(res_ols.bse.get(LOG_TREATMENT, np.nan)),
            "pval_gdp": float(res_ols.pvalues.get(LOG_TREATMENT, np.nan)),
            "rsquared": float(res_ols.rsquared),
            "nobs":     int(res_ols.nobs),
        }
    except Exception as exc:
        logger.warning("Pooled OLS failed: %s", exc)

    logger.info("Robustness checks complete: %s", list(out.keys()))
    return out


# ── Synthesis ─────────────────────────────────────────────────────────────────

def synthesise_findings(
    granger: dict, panel: dict, iv: dict,
    did: dict, synth: SynthResult,
) -> dict[str, Any]:
    """Collect point estimates across all methods into one summary frame."""
    rows: list[dict] = []

    # Panel FE — main specs
    for spec_name, r in panel.items():
        rows.append({"method": f"Panel FE ({spec_name})", "coef": r.coef_gdp,
                     "se": r.se_gdp, "pval": r.pval_gdp,
                     "ci_lo": r.ci95_gdp[0], "ci_hi": r.ci95_gdp[1],
                     "nobs": r.nobs})

    # IV
    for spec_name, r in iv.items():
        rows.append({"method": f"IV-2SLS ({spec_name})", "coef": r.coef_gdp,
                     "se": r.se_gdp, "pval": r.pval_gdp,
                     "ci_lo": r.ci95_gdp[0], "ci_hi": r.ci95_gdp[1],
                     "nobs": r.nobs})

    # DiD
    for event_name, r in did.items():
        rows.append({"method": f"DiD ({r.label[:30]})", "coef": r.att,
                     "se": r.se_att, "pval": r.pval_att,
                     "ci_lo": r.ci95_att[0], "ci_hi": r.ci95_att[1],
                     "nobs": r.nobs})

    # Synthetic control (post-period ATT)
    rows.append({"method": "Synthetic Control (China NCMS)", "coef": synth.post_att,
                 "se": np.nan, "pval": synth.p_value or np.nan,
                 "ci_lo": np.nan, "ci_hi": np.nan, "nobs": np.nan})

    summary_df = pd.DataFrame(rows)

    return {
        "coef_table": summary_df,
        "granger_summary": granger.get("summary", {}),
        "panel_fe_main": panel.get("controls_full"),
        "iv_main": iv.get("iv_overidentified"),
        "synth_pval": synth.p_value,
    }


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all_causal(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = df if df is not None else load_df()
    logger.info("═" * 60)
    logger.info("Phase 2 — Causal Inference")
    logger.info("═" * 60)

    logger.info("1/6  Granger causality ...")
    granger = run_granger(df)

    logger.info("2/6  Panel fixed effects ...")
    panel   = run_panel_fe(df)
    panel_sub = run_panel_fe_subgroups(df)

    logger.info("3/6  IV-2SLS ...")
    iv_res  = run_iv(df)

    logger.info("4/6  Difference-in-Differences ...")
    did_res = run_did(df)

    logger.info("5/6  Synthetic control ...")
    synth_res = run_synthetic_control(df)

    logger.info("6/6  Robustness checks ...")
    robust  = run_robustness(df)

    synthesis = synthesise_findings(granger, panel, iv_res, did_res, synth_res)

    return {
        "granger": granger,
        "panel_fe": panel,
        "panel_fe_subgroups": panel_sub,
        "iv": iv_res,
        "did": did_res,
        "synth": synth_res,
        "robustness": robust,
        "synthesis": synthesis,
    }
