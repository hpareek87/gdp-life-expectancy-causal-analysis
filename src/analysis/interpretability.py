"""Phase 3 — SHAP interpretability and PDP analysis.

Generates SHAP values, partial dependence plots, and feature importance
summaries for RF, XGBoost, and ensemble models.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..utils.config import FINAL_DIR, OUTPUTS_DIR
from ..utils.logging_setup import get_logger
from .ml_models import (
    DataSplit, ModelMetrics, make_split, run_tree_models, OUTCOME,
)

warnings.filterwarnings("ignore")
logger = get_logger("ml.interpretability")

FIG_DIR = OUTPUTS_DIR / "figures" / "ml"
TAB_DIR = OUTPUTS_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


# ── SHAP computation ────────────────────────────────────────────────────────────

def compute_shap_tree(model: Any, ds: DataSplit,
                      model_name: str = "XGBoost") -> shap.Explanation:
    """Compute SHAP values via TreeExplainer for RF or XGBoost."""
    logger.info("Computing SHAP values for %s ...", model_name)
    explainer = shap.TreeExplainer(model)
    # Pass numpy array to avoid XGBoost/SHAP DataFrame API mismatch.
    # Wrap in a DataFrame so SHAP carries feature names through.
    import pandas as _pd
    X_np = _pd.DataFrame(ds.X_train.values, columns=ds.feature_cols)
    shap_vals = explainer.shap_values(ds.X_train.values)
    # Build Explanation manually so downstream plots get feature names
    exp = shap.Explanation(
        values=shap_vals if isinstance(shap_vals, np.ndarray)
               else np.array(shap_vals),
        base_values=explainer.expected_value
                    if np.isscalar(explainer.expected_value)
                    else float(np.mean(explainer.expected_value)),
        data=ds.X_train.values,
        feature_names=ds.feature_cols,
    )
    logger.info("SHAP computed: shape=%s", exp.values.shape)
    return exp


def compute_shap_test(model: Any, ds: DataSplit) -> shap.Explanation:
    """SHAP on the test set."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(ds.X_test.values)
    return shap.Explanation(
        values=shap_vals if isinstance(shap_vals, np.ndarray)
               else np.array(shap_vals),
        base_values=explainer.expected_value
                    if np.isscalar(explainer.expected_value)
                    else float(np.mean(explainer.expected_value)),
        data=ds.X_test.values,
        feature_names=ds.feature_cols,
    )


# ── Global feature importance ───────────────────────────────────────────────────

def shap_global_importance(shap_vals: shap.Explanation,
                           feature_cols: list[str],
                           top_n: int = 20) -> pd.Series:
    """Mean |SHAP| per feature, ranked descending."""
    mean_abs = np.abs(np.array(shap_vals.values)).mean(axis=0)
    imp = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False)
    return imp.head(top_n)


def save_feature_importance_csv(xgb_shap: pd.Series, rf_shap: pd.Series,
                                 xgb_gain: pd.Series, rf_mdi: pd.Series) -> Path:
    """Merge four importance measures into one table."""
    all_feats = list(dict.fromkeys(
        list(xgb_gain.index) + list(rf_mdi.index)))  # union, deduped
    df = pd.DataFrame(index=all_feats)
    df["xgb_shap"] = xgb_shap.reindex(all_feats)
    df["rf_shap"]  = rf_shap.reindex(all_feats)
    df["xgb_gain"] = xgb_gain.reindex(all_feats)
    df["rf_mdi"]   = rf_mdi.reindex(all_feats)
    df = df.fillna(0)
    df["mean_rank"] = df.rank(ascending=False).mean(axis=1)
    df = df.sort_values("mean_rank")
    path = TAB_DIR / "feature_importance.csv"
    df.to_csv(path)
    logger.info("Saved feature importance table to %s", path)
    return path


# ── Model performance LaTeX table ───────────────────────────────────────────────

def _stars(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_model_performance_table(metrics: dict[str, ModelMetrics]) -> Path:
    models_order = ["OLS", "Ridge", "Lasso", "Q50",
                    "RandomForest", "XGBoost", "LSTM", "Ensemble"]
    rows = []
    for name in models_order:
        if name not in metrics:
            continue
        m = metrics[name]
        rows.append(
            f"    {name:<16} & {m.r2_train:.4f} & {m.r2_test:.4f}"
            f" & {m.rmse_train:.3f} & {m.rmse_test:.3f}"
            f" & {m.mae_test:.3f} & {m.n_train} & {m.n_test} \\\\"
        )

    tex = r"""\begin{table}[htbp]
  \centering
  \caption{ML Model Performance: Life Expectancy Prediction (2000--2024)}
  \label{tab:model_performance}
  \begin{tabular}{lcccccrr}
    \toprule
    Model & $R^2_{\text{train}}$ & $R^2_{\text{test}}$ &
    RMSE$_{\text{train}}$ & RMSE$_{\text{test}}$ &
    MAE$_{\text{test}}$ & $N_{\text{train}}$ & $N_{\text{test}}$ \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}
    \small
    \item Train: 2000--2018; Test: 2019--2024. RMSE and MAE in years.
    \item LSTM uses 5-year sliding windows (PyTorch, 2-layer 64 units).
    \item Ensemble = Ridge meta-learner over RF + XGBoost + LSTM OOF predictions.
  \end{tablenotes}
\end{table}
"""
    path = TAB_DIR / "model_performance.tex"
    path.write_text(tex)
    logger.info("Saved model performance table to %s", path)
    return path


# ── PDP computation ─────────────────────────────────────────────────────────────

def compute_pdp(model: Any, ds: DataSplit, feature: str,
                n_grid: int = 50) -> pd.DataFrame:
    """Manual partial dependence: vary one feature, average predictions."""
    X = ds.X_train.copy()
    idx = ds.feature_cols.index(feature)
    grid = np.linspace(X.iloc[:, idx].quantile(0.02),
                       X.iloc[:, idx].quantile(0.98), n_grid)
    ys = []
    for val in grid:
        Xmod = X.values.copy()
        Xmod[:, idx] = val
        ys.append(model.predict(Xmod).mean())
    return pd.DataFrame({"feature_val_scaled": grid, "pdp_y": ys})


def compute_pdp_original_scale(model: Any, ds: DataSplit, feature: str,
                                n_grid: int = 50) -> pd.DataFrame:
    """PDP in original (unscaled) feature units by inverting the StandardScaler."""
    pdp = compute_pdp(model, ds, feature, n_grid)
    feat_idx = ds.feature_cols.index(feature)
    mu  = ds.scaler.mean_[feat_idx]
    sig = ds.scaler.scale_[feat_idx]
    pdp["feature_val"] = pdp["feature_val_scaled"] * sig + mu
    return pdp


# ── Master runner ───────────────────────────────────────────────────────────────

def run_interpretability(ml_results: dict | None = None) -> dict:
    """Compute and return SHAP + PDP results, and save tables."""
    if ml_results is None:
        from .ml_models import run_all_ml
        ml_results = run_all_ml()

    ds: DataSplit = ml_results["ds"]
    xgb_model = ml_results["trees"]["XGBoost"][0]
    rf_model  = ml_results["trees"]["RandomForest"][0]
    metrics   = ml_results["metrics"]

    logger.info("Computing SHAP values ...")
    xgb_shap_train = compute_shap_tree(xgb_model, ds, "XGBoost")
    rf_shap_train  = compute_shap_tree(rf_model,  ds, "RandomForest")
    xgb_shap_test  = compute_shap_test(xgb_model, ds)
    rf_shap_test   = compute_shap_test(rf_model,  ds)

    xgb_global = shap_global_importance(xgb_shap_train, ds.feature_cols)
    rf_global  = shap_global_importance(rf_shap_train,  ds.feature_cols)
    xgb_gain   = ml_results["xgb_importance"]
    rf_mdi     = ml_results["rf_importance"]

    save_feature_importance_csv(xgb_global, rf_global, xgb_gain, rf_mdi)
    save_model_performance_table(metrics)

    # PDPs for key features
    pdp_results: dict[str, pd.DataFrame] = {}
    for feat in ["log_gdp_per_capita_ppp", "health_exp_pct_gdp",
                 "sanitation_access", "fertility_rate"]:
        if feat in ds.feature_cols:
            pdp_results[feat] = compute_pdp_original_scale(xgb_model, ds, feat)

    logger.info("Interpretability complete.")
    return {
        "xgb_shap_train": xgb_shap_train,
        "rf_shap_train":  rf_shap_train,
        "xgb_shap_test":  xgb_shap_test,
        "rf_shap_test":   rf_shap_test,
        "xgb_global_importance": xgb_global,
        "rf_global_importance":  rf_global,
        "pdp": pdp_results,
    }
