"""Phase 3 — ML visualization module.

All figures for the ML analysis: performance comparisons, SHAP plots,
PDP plots, threshold plots, prediction scatter, and residual diagnostics.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap

from ..utils.config import OUTPUTS_DIR
from ..utils.logging_setup import get_logger
from ..analysis.ml_models import DataSplit, OUTCOME, TRAIN_END, TEST_START

warnings.filterwarnings("ignore")
logger = get_logger("ml.plots")

FIG_DIR = OUTPUTS_DIR / "figures" / "ml"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)

PALETTE = {
    "OLS": "#aaaaaa", "Ridge": "#888888", "Lasso": "#666666",
    "Q50": "#444444",
    "RandomForest": "#2196F3", "XGBoost": "#FF5722",
    "LSTM": "#9C27B0", "Ensemble": "#4CAF50",
}
R2_TARGET = 0.90


# ── 1. Model performance bar chart ─────────────────────────────────────────────

def fig_model_performance(metrics: dict) -> Path:
    names = [n for n in ["OLS", "Ridge", "Lasso", "RandomForest",
                          "XGBoost", "LSTM", "Ensemble"] if n in metrics]
    r2_test  = [metrics[n].r2_test  for n in names]
    rmse_test = [metrics[n].rmse_test for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [PALETTE.get(n, "#999999") for n in names]

    ax = axes[0]
    bars = ax.barh(names, r2_test, color=colors, edgecolor="white", height=0.6)
    ax.axvline(R2_TARGET, color="red", linestyle="--", lw=1.5, label=f"Target R²={R2_TARGET}")
    ax.set_xlabel("R² (test set, 2019–2024)")
    ax.set_title("Predictive Performance — R²")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.02)
    for bar, val in zip(bars, r2_test):
        ax.text(max(val + 0.01, 0.02), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax = axes[1]
    ax.barh(names, rmse_test, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("RMSE (years)")
    ax.set_title("Predictive Performance — RMSE")
    for i, val in enumerate(rmse_test):
        ax.text(val + 0.05, i, f"{val:.2f}", va="center", fontsize=8)

    fig.suptitle("ML Model Comparison: Life Expectancy Prediction", fontsize=12, y=1.01)
    fig.tight_layout()
    path = FIG_DIR / "model_performance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── 2. Prediction vs actual scatter ────────────────────────────────────────────

def fig_pred_vs_actual(ml_results: dict) -> Path:
    ds: DataSplit = ml_results["ds"]
    tree_res = ml_results["trees"]
    xgb = tree_res["XGBoost"][0]
    rf  = tree_res["RandomForest"][0]

    Xte = ds.X_test.values
    yte = ds.y_test.values
    yhat_xgb = xgb.predict(Xte)
    yhat_rf  = rf.predict(Xte)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, yhat, color) in zip(axes, [
        ("XGBoost", yhat_xgb, PALETTE["XGBoost"]),
        ("Random Forest", yhat_rf,  PALETTE["RandomForest"]),
    ]):
        lo = min(yte.min(), yhat.min()) - 1
        hi = max(yte.max(), yhat.max()) + 1
        ax.scatter(yte, yhat, alpha=0.5, s=20, color=color, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect prediction")
        r2 = float(1 - np.sum((yte - yhat)**2) / np.sum((yte - np.mean(yte))**2))
        ax.set_xlabel("Actual Life Expectancy (years)")
        ax.set_ylabel("Predicted Life Expectancy (years)")
        ax.set_title(f"{name}  |  R²={r2:.3f} (test)")
        ax.legend(fontsize=9)

    fig.suptitle("Predicted vs Actual — Test Set (2019–2024)", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "pred_vs_actual.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── 3. Residual plot ────────────────────────────────────────────────────────────

def fig_residuals(ml_results: dict) -> Path:
    ds: DataSplit = ml_results["ds"]
    xgb = ml_results["trees"]["XGBoost"][0]
    Xte = ds.X_test.values
    yte = ds.y_test.values
    resid = yte - xgb.predict(Xte)
    years = ds.df_test["year"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(xgb.predict(Xte), resid, alpha=0.5, s=18, color=PALETTE["XGBoost"])
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Predicted Life Expectancy")
    axes[0].set_ylabel("Residual (years)")
    axes[0].set_title("Residuals vs Fitted — XGBoost (test set)")

    axes[1].scatter(years, resid, alpha=0.5, s=18, color=PALETTE["XGBoost"])
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Residual (years)")
    axes[1].set_title("Residuals Over Time — XGBoost (test set)")

    fig.tight_layout()
    path = FIG_DIR / "residuals.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── 4. SHAP beeswarm ────────────────────────────────────────────────────────────

def fig_shap_beeswarm(shap_vals: shap.Explanation, model_name: str,
                       max_display: int = 15) -> Path:
    # SHAP beeswarm does not accept an ax argument; use show=False + gcf()
    shap.plots.beeswarm(shap_vals, max_display=max_display, show=False,
                         plot_size=(10, 7))
    fig = plt.gcf()
    fig.suptitle(f"SHAP Beeswarm — {model_name} (train set)", fontsize=11)
    path = FIG_DIR / f"shap_beeswarm_{model_name.lower()}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    logger.info("Saved %s", path)
    return path


# ── 5. SHAP global bar ──────────────────────────────────────────────────────────

def fig_shap_bar(xgb_imp: pd.Series, rf_imp: pd.Series, top_n: int = 15) -> Path:
    top_feats = xgb_imp.head(top_n).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, imp, color) in zip(axes, [
        ("XGBoost", xgb_imp, PALETTE["XGBoost"]),
        ("Random Forest", rf_imp, PALETTE["RandomForest"]),
    ]):
        vals = imp.reindex(top_feats).fillna(0)
        ax.barh(range(len(vals)), vals.values[::-1], color=color, edgecolor="white")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([f.replace("_", " ") for f in vals.index[::-1]], fontsize=8)
        ax.set_xlabel("Mean |SHAP| value")
        ax.set_title(f"SHAP Global Importance — {name}")

    fig.suptitle("Top Feature Importances (SHAP)", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "shap_global_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── 6. SHAP dependence plot ─────────────────────────────────────────────────────

def fig_shap_dependence(shap_vals: shap.Explanation, ds: DataSplit,
                         feature: str, model_name: str = "XGBoost") -> Path:
    if feature not in ds.feature_cols:
        logger.warning("Feature %s not in feature_cols — skipping dependence plot", feature)
        return FIG_DIR / "skip"

    feat_idx = ds.feature_cols.index(feature)
    shap.plots.scatter(shap_vals[:, feat_idx], show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Dependence — {feature} ({model_name})", fontsize=10)
    fname = f"shap_dependence_{feature}_{model_name.lower()}.png"
    path = FIG_DIR / fname
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    logger.info("Saved %s", path)
    return path


# ── 7. PDP plots ────────────────────────────────────────────────────────────────

def fig_pdp_grid(pdp_results: dict[str, pd.DataFrame]) -> Path:
    n = len(pdp_results)
    if n == 0:
        return FIG_DIR / "skip"
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax, (feat, pdp) in zip(axes, pdp_results.items()):
        ax.plot(pdp["feature_val"], pdp["pdp_y"],
                color=PALETTE["XGBoost"], lw=2)
        ax.fill_between(pdp["feature_val"], pdp["pdp_y"],
                         alpha=0.15, color=PALETTE["XGBoost"])
        ax.set_xlabel(feat.replace("_", " "), fontsize=9)
        ax.set_ylabel("Avg. Predicted LE (years)", fontsize=9)
        ax.set_title(f"PDP: {feat.replace('_', ' ')}", fontsize=10)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Partial Dependence Plots — XGBoost (train set)", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "pdp_grid.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── 8. GDP threshold visualization ─────────────────────────────────────────────

def fig_threshold(df: pd.DataFrame, thresholds: pd.DataFrame) -> Path:
    sub = df.dropna(subset=["log_gdp_per_capita_ppp", OUTCOME]).copy()
    X = sub["log_gdp_per_capita_ppp"].values
    y = sub[OUTCOME].values
    gdp = np.exp(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(gdp / 1000, y, alpha=0.25, s=12, color="steelblue")

    # Draw piecewise linear fit at each threshold split
    colors_seg = ["#E91E63", "#FF9800", "#4CAF50"]
    splits = [-np.inf] + sorted(thresholds["log_gdp_threshold"].tolist()) + [np.inf]
    for i, (lo, hi) in enumerate(zip(splits[:-1], splits[1:])):
        mask = (X >= lo) & (X < hi)
        if mask.sum() < 5:
            continue
        xs = np.linspace(X[mask].min(), X[mask].max(), 100)
        coef = np.polyfit(X[mask], y[mask], 1)
        ys = np.polyval(coef, xs)
        ax.plot(np.exp(xs) / 1000, ys,
                color=colors_seg[i % len(colors_seg)], lw=2.5,
                label=f"Segment {i+1}: β={coef[0]:.2f}")

    for _, row in thresholds.iterrows():
        thresh_k = np.exp(row["log_gdp_threshold"]) / 1000
        ax.axvline(thresh_k, color="gray", linestyle="--", lw=1.2,
                   label=f"${thresh_k:,.0f}k (Chow p={row['chow_p_value']:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("GDP per Capita PPP (thousands USD, log scale)")
    ax.set_ylabel("Life Expectancy (years)")
    ax.set_title("GDP–Life Expectancy Threshold Analysis\n(Piecewise linear fit with Chow test breakpoints)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = FIG_DIR / "gdp_threshold.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── 9. CV fold performance ─────────────────────────────────────────────────────

def fig_cv_results(cv_results: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["r2", "rmse"]):
        for model_name, grp in cv_results.groupby("model"):
            color = PALETTE.get(model_name, "#888888")
            ax.plot(grp["fold"], grp[metric], marker="o", label=model_name,
                    color=color, lw=1.8)
        ax.set_xlabel("CV Fold")
        ax.set_ylabel("R²" if metric == "r2" else "RMSE (years)")
        ax.set_title(f"Walk-Forward CV — {metric.upper()}")
        ax.legend(fontsize=9)
    fig.suptitle("Time-Series Cross-Validation (expanding window)", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "cv_results.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ── 10. SHAP waterfall (single observation) ────────────────────────────────────

def fig_shap_waterfall(shap_vals: shap.Explanation,
                        model_name: str, idx: int = 0) -> Path:
    shap.plots.waterfall(shap_vals[idx], show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP Waterfall — {model_name} (observation {idx})", fontsize=11)
    path = FIG_DIR / f"shap_waterfall_{model_name.lower()}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    logger.info("Saved %s", path)
    return path


# ── Master runner ───────────────────────────────────────────────────────────────

def run_all_ml_plots(ml_results: dict, interp_results: dict,
                      df: pd.DataFrame | None = None) -> list[Path]:
    ds: DataSplit = ml_results["ds"]
    metrics  = ml_results["metrics"]
    cv_res   = ml_results["cv_results"]
    thresholds = ml_results["thresholds"]

    if df is None:
        from ..utils.config import FINAL_DIR
        df = pd.read_csv(FINAL_DIR / "master_dataset.csv")

    paths: list[Path] = []
    paths.append(fig_model_performance(metrics))
    paths.append(fig_pred_vs_actual(ml_results))
    paths.append(fig_residuals(ml_results))
    paths.append(fig_cv_results(cv_res))
    paths.append(fig_threshold(df, thresholds))
    paths.append(fig_pdp_grid(interp_results["pdp"]))

    xgb_imp = interp_results["xgb_global_importance"]
    rf_imp  = interp_results["rf_global_importance"]
    paths.append(fig_shap_bar(xgb_imp, rf_imp))

    xgb_shap_train = interp_results["xgb_shap_train"]
    rf_shap_train  = interp_results["rf_shap_train"]
    paths.append(fig_shap_beeswarm(xgb_shap_train, "XGBoost"))
    paths.append(fig_shap_beeswarm(rf_shap_train, "RandomForest"))

    for feat in ["log_gdp_per_capita_ppp", "sanitation_access",
                 "health_exp_pct_gdp"]:
        p = fig_shap_dependence(xgb_shap_train, ds, feat, "XGBoost")
        paths.append(p)

    paths.append(fig_shap_waterfall(xgb_shap_train, "XGBoost", idx=0))

    logger.info("Generated %d ML figures in %s", len(paths), FIG_DIR)
    return [p for p in paths if p.exists()]
