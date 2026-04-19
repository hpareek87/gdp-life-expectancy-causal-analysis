"""Smoke tests for Phase 3 ML pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.config import FINAL_DIR


@pytest.fixture(scope="module")
def ml_results():
    if not (FINAL_DIR / "master_dataset.csv").exists():
        pytest.skip("master_dataset.csv not built")
    from src.analysis.ml_models import run_all_ml
    import pandas as pd
    df = pd.read_csv(FINAL_DIR / "master_dataset.csv")
    return run_all_ml(df)


def test_feature_count(ml_results):
    ds = ml_results["ds"]
    assert len(ds.feature_cols) >= 40, f"Only {len(ds.feature_cols)} features"
    assert ds.X_train.shape[1] == len(ds.feature_cols)


def test_no_data_leakage(ml_results):
    ds = ml_results["ds"]
    train_years = ml_results["ds"].df_train["year"].unique()
    test_years  = ml_results["ds"].df_test["year"].unique()
    assert max(train_years) < min(test_years), "Train/test years overlap!"


def test_linear_models_run(ml_results):
    for name in ["OLS", "Ridge", "Lasso"]:
        assert name in ml_results["metrics"]
        m = ml_results["metrics"][name]
        assert np.isfinite(m.r2_test)
        assert m.r2_test > 0.60, f"{name} R²_test={m.r2_test:.3f} too low"


def test_tree_models_exceed_target(ml_results):
    TARGET = 0.90
    for name in ["XGBoost"]:
        m = ml_results["metrics"][name]
        assert m.r2_test >= TARGET, (
            f"{name} R²_test={m.r2_test:.4f} < target {TARGET}")


def test_lstm_runs(ml_results):
    m = ml_results["metrics"].get("LSTM")
    if m is None:
        pytest.skip("LSTM did not run")
    assert np.isfinite(m.r2_test)
    assert m.r2_test > 0.70, f"LSTM R²_test={m.r2_test:.3f} too low"


def test_ensemble_runs(ml_results):
    m = ml_results["metrics"].get("Ensemble")
    assert m is not None
    assert np.isfinite(m.r2_test)


def test_thresholds_detected(ml_results):
    thresh = ml_results["thresholds"]
    assert len(thresh) >= 1, "No GDP thresholds detected"
    assert "chow_p_value" in thresh.columns
    # At least one threshold should be statistically significant
    assert (thresh["chow_p_value"] < 0.05).any()


def test_feature_importances(ml_results):
    xgb_imp = ml_results["xgb_importance"]
    rf_imp  = ml_results["rf_importance"]
    assert len(xgb_imp) > 0
    assert len(rf_imp) > 0
    # GDP-related features should be in top 20
    top20 = set(xgb_imp.head(20).index) | set(rf_imp.head(20).index)
    gdp_feats = [f for f in top20 if "gdp" in f.lower()]
    assert len(gdp_feats) >= 1, f"No GDP features in top 20: {top20}"


def test_models_saved(ml_results):
    from src.utils.config import MODELS_DIR
    assert (MODELS_DIR / "xgboost_model.pkl").exists()
    assert (MODELS_DIR / "randomforest_model.pkl").exists()
    assert (MODELS_DIR / "lstm_model.pth").exists()
