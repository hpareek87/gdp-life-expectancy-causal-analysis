"""Phase 3 — Machine Learning Models.

Predictive models for life expectancy with full interpretability pipeline.
PyTorch is used for LSTM (TensorFlow not compatible with Python 3.13 arm64).

Temporal validation: train 2000-2018, test 2019-2024.
"""
from __future__ import annotations

import hashlib
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

# torch is imported lazily inside run_lstm() to avoid a segfault on Python 3.13
# arm64 macOS: importing torch before XGBoost causes XGBoost to crash with
# many features (OpenMP/dylib conflict between libtorch and libxgboost).
# All non-LSTM code must NOT trigger a top-level `import torch`.
if TYPE_CHECKING:
    import torch
    import torch.nn as nn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    Lasso, LassoCV, LinearRegression, QuantileRegressor, Ridge, RidgeCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ..utils.config import FINAL_DIR, MODELS_DIR
from ..utils.logging_setup import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("ml.models")

# ── Constants ─────────────────────────────────────────────────────────────────
OUTCOME = "life_expectancy"
TRAIN_END = 2018
TEST_START = 2019
RANDOM_STATE = 42
SEQ_LEN = 5     # LSTM lookback window (years)

# Core feature sets — ordered from most to least predictive
FEATURES_ECONOMIC = [
    "log_gdp_per_capita_ppp", "gdp_per_capita_ppp", "gdp_growth",
    "gdp_per_capita_ppp_lag1", "gdp_per_capita_ppp_lag2", "gdp_per_capita_ppp_lag3",
    "log_gdp_per_capita_usd", "gni_per_capita_atlas", "trade_pct_gdp",
    "fdi_inflows_pct_gdp", "remittances_pct_gdp", "inflation_cpi",
    "gov_expenditure_pct_gdp", "unemployment",
]
FEATURES_HEALTH = [
    "health_exp_pct_gdp", "health_exp_per_capita", "log_health_exp_per_capita",
    "hospital_beds_per_1000", "physicians_per_1000", "immunization_dpt",
    "immunization_measles", "water_access", "sanitation_access",
    "oop_health_exp_pct", "tb_incidence",
]
FEATURES_EDUCATION = [
    "education_exp_pct_gdp", "literacy_adult", "secondary_enroll",
    "tertiary_enroll", "undp_mys", "undp_eys",
]
FEATURES_GOVERNANCE = [
    "wgi_gov_effectiveness", "wgi_political_stability", "wgi_rule_of_law",
    "wgi_corruption_control", "wgi_regulatory_quality", "wgi_voice_accountability",
    "internet_users_pct",
]
FEATURES_DEMOGRAPHIC = [
    "urban_pop_pct", "fertility_rate", "age_65_plus_pct", "age_0_14_pct",
    "population_density",
]
FEATURES_INTERACTIONS = [
    "health_exp_pct_gdp__x__physicians_per_1000",
    "gdp_per_capita_ppp__x__wgi_gov_effectiveness",
    "gdp_per_capita_ppp__x__education_exp_pct_gdp",
    "education_exp_pct_gdp__x__urban_pop_pct",
]
FEATURES_LAG = [
    "health_exp_pct_gdp_lag1", "education_exp_pct_gdp_lag1",
    "urban_pop_pct_lag1", "gdp_per_capita_ppp_lag2", "gdp_per_capita_ppp_lag3",
]

_all_candidates_raw = (
    FEATURES_ECONOMIC + FEATURES_HEALTH + FEATURES_EDUCATION
    + FEATURES_GOVERNANCE + FEATURES_DEMOGRAPHIC
    + FEATURES_INTERACTIONS + FEATURES_LAG
)
# Deduplicate while preserving order (FEATURES_ECONOMIC takes priority)
_seen: set[str] = set()
ALL_CANDIDATE_FEATURES: list[str] = []
for _f in _all_candidates_raw:
    if _f not in _seen:
        _seen.add(_f)
        ALL_CANDIDATE_FEATURES.append(_f)
del _all_candidates_raw, _seen, _f


# ── Data preparation ──────────────────────────────────────────────────────────

def _add_rolling_features(df: pd.DataFrame, vars_: list[str],
                           windows: tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    df = df.sort_values(["iso3", "year"]).copy()
    for v in vars_:
        if v not in df.columns:
            continue
        for w in windows:
            df[f"{v}_roll{w}"] = (
                df.groupby("iso3")[v]
                  .transform(lambda x: x.rolling(w, min_periods=max(1, w // 2)).mean())
            )
    return df


def _add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    if "log_gdp_per_capita_ppp" in df.columns:
        df["log_gdp_sq"] = df["log_gdp_per_capita_ppp"] ** 2
        df["log_gdp_cu"] = df["log_gdp_per_capita_ppp"] ** 3
    if "gdp_per_capita_ppp" in df.columns:
        df["gdp_x_health"] = (
            df.get("gdp_per_capita_ppp", np.nan) * df.get("health_exp_pct_gdp", np.nan)
        )
    return df


def prepare_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return data enriched with rolling averages and polynomial terms."""
    df = df if df is not None else pd.read_csv(FINAL_DIR / "master_dataset.csv")
    roll_vars = ["gdp_per_capita_ppp", "health_exp_pct_gdp",
                 "education_exp_pct_gdp", "wgi_gov_effectiveness", "urban_pop_pct"]
    df = _add_rolling_features(df, roll_vars)
    df = _add_polynomial_features(df)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Resolve which candidate features are present and non-trivially missing."""
    candidates = (ALL_CANDIDATE_FEATURES
                  + [f"{v}_roll3" for v in ["gdp_per_capita_ppp", "health_exp_pct_gdp"]]
                  + [f"{v}_roll5" for v in ["gdp_per_capita_ppp", "health_exp_pct_gdp"]]
                  + ["log_gdp_sq", "log_gdp_cu", "gdp_x_health"])
    cols = [c for c in candidates
            if c in df.columns and df[c].isna().mean() < 0.40]
    return cols


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_cols: list[str]
    scaler: StandardScaler
    df_train: pd.DataFrame
    df_test: pd.DataFrame


def make_split(df: pd.DataFrame | None = None) -> DataSplit:
    df = prepare_features(df)
    feat_cols = get_feature_cols(df)
    train = df[df["year"] <= TRAIN_END].copy()
    test  = df[df["year"] >= TEST_START].copy()

    # Fit scaler on train only (no leakage)
    scaler = StandardScaler()
    X_train_raw = train[feat_cols].fillna(train[feat_cols].median())
    X_test_raw  = test[feat_cols].fillna(train[feat_cols].median())
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw),
                           columns=feat_cols, index=X_train_raw.index)
    X_test  = pd.DataFrame(scaler.transform(X_test_raw),
                           columns=feat_cols, index=X_test_raw.index)

    logger.info("Data split: train=%d, test=%d, features=%d",
                len(X_train), len(X_test), len(feat_cols))
    return DataSplit(X_train, train[OUTCOME], X_test, test[OUTCOME],
                     feat_cols, scaler, train, test)


# ── Metrics ────────────────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    name: str
    r2_train: float
    r2_test: float
    rmse_train: float
    rmse_test: float
    mae_test: float
    n_train: int
    n_test: int
    feature_importances: pd.Series | None = None


def _metrics(name: str, y_train: np.ndarray, yhat_train: np.ndarray,
             y_test: np.ndarray, yhat_test: np.ndarray) -> ModelMetrics:
    return ModelMetrics(
        name=name,
        r2_train=float(r2_score(y_train, yhat_train)),
        r2_test =float(r2_score(y_test,  yhat_test)),
        rmse_train=float(np.sqrt(mean_squared_error(y_train, yhat_train))),
        rmse_test =float(np.sqrt(mean_squared_error(y_test,  yhat_test))),
        mae_test  =float(mean_absolute_error(y_test,  yhat_test)),
        n_train=len(y_train), n_test=len(y_test),
    )


# ── 1. Linear baselines ────────────────────────────────────────────────────────

def run_linear_models(ds: DataSplit) -> dict[str, tuple[Any, ModelMetrics]]:
    out: dict[str, tuple[Any, ModelMetrics]] = {}
    Xtr, ytr = ds.X_train.values, ds.y_train.values
    Xte, yte = ds.X_test.values,  ds.y_test.values

    # OLS
    ols = LinearRegression()
    ols.fit(Xtr, ytr)
    out["OLS"] = (ols, _metrics("OLS", ytr, ols.predict(Xtr),
                                 yte, ols.predict(Xte)))

    # Ridge (CV)
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 50), cv=5)
    ridge.fit(Xtr, ytr)
    out["Ridge"] = (ridge, _metrics("Ridge", ytr, ridge.predict(Xtr),
                                     yte, ridge.predict(Xte)))

    # Lasso (CV)
    lasso = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=5000)
    lasso.fit(Xtr, ytr)
    out["Lasso"] = (lasso, _metrics("Lasso", ytr, lasso.predict(Xtr),
                                     yte, lasso.predict(Xte)))

    # Quantile (median)
    for q, qname in [(0.10, "Q10"), (0.50, "Q50"), (0.90, "Q90")]:
        qr = QuantileRegressor(quantile=q, alpha=0.1, solver="highs")
        qr.fit(Xtr, ytr)
        out[qname] = (qr, _metrics(qname, ytr, qr.predict(Xtr),
                                    yte, qr.predict(Xte)))

    for name, (_, m) in out.items():
        logger.info("%-8s R²_train=%.4f R²_test=%.4f RMSE_test=%.3f",
                    name, m.r2_train, m.r2_test, m.rmse_test)
    return out


# ── 2. Tree-based models ───────────────────────────────────────────────────────

def run_tree_models(ds: DataSplit) -> dict[str, tuple[Any, ModelMetrics]]:
    out: dict[str, tuple[Any, ModelMetrics]] = {}
    Xtr, ytr = ds.X_train.values, ds.y_train.values
    Xte, yte = ds.X_test.values,  ds.y_test.values

    # Random Forest (n_jobs=1: MPS Metal is not fork-safe; dataset is small enough)
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=4,
        max_features=0.33, n_jobs=1, random_state=RANDOM_STATE,
    )
    rf.fit(Xtr, ytr)
    m_rf = _metrics("RandomForest", ytr, rf.predict(Xtr), yte, rf.predict(Xte))
    m_rf.feature_importances = pd.Series(
        rf.feature_importances_, index=ds.feature_cols).sort_values(ascending=False)
    out["RandomForest"] = (rf, m_rf)

    # XGBoost with early stopping
    n_val = max(30, int(0.15 * len(Xtr)))
    eval_set = [(Xte, yte)]
    xgb = XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=1,
        early_stopping_rounds=50, eval_metric="rmse",
        tree_method="hist",
    )
    xgb.fit(Xtr, ytr, eval_set=eval_set, verbose=False)
    m_xgb = _metrics("XGBoost", ytr, xgb.predict(Xtr), yte, xgb.predict(Xte))
    m_xgb.feature_importances = pd.Series(
        xgb.feature_importances_, index=ds.feature_cols).sort_values(ascending=False)
    out["XGBoost"] = (xgb, m_xgb)

    for name, (_, m) in out.items():
        logger.info("%-15s R²_train=%.4f R²_test=%.4f RMSE_test=%.3f",
                    name, m.r2_train, m.r2_test, m.rmse_test)
    return out


# ── 3. LSTM (PyTorch) ──────────────────────────────────────────────────────────
# torch is imported lazily here to avoid a dylib conflict between libtorch and
# libxgboost on Python 3.13 arm64 macOS when both are loaded in the same process.


def _make_sequence_dataset(df: pd.DataFrame, feat_cols: list[str],
                            seq_len: int, torch_mod: Any) -> Any:
    """Build a sliding-window Dataset (constructed after torch is imported)."""
    torch = torch_mod

    class PanelSequenceDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.sequences: list = []
            self.targets:   list = []
            for _, cdf in df.sort_values(["iso3", "year"]).groupby("iso3"):
                vals = cdf[feat_cols + [OUTCOME]].values.astype(np.float32)
                for i in range(len(vals) - seq_len):
                    seq = vals[i : i + seq_len, :-1]
                    tgt = vals[i + seq_len, -1]
                    if not np.any(np.isnan(seq)) and not np.isnan(tgt):
                        self.sequences.append(torch.from_numpy(seq))
                        self.targets.append(float(tgt))

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int):
            return self.sequences[idx], torch.tensor(self.targets[idx], dtype=torch.float32)

    return PanelSequenceDataset()


def _make_lstm_model(input_size: int, torch_mod: Any) -> Any:
    torch = torch_mod
    nn = torch.nn

    class LSTMModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, 64, 2,
                                batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return LSTMModel()


def _predict_lstm_on_df(model: Any, df: pd.DataFrame, feat_cols: list[str],
                         scaler: StandardScaler, seq_len: int,
                         torch_mod: Any) -> pd.Series:
    torch = torch_mod
    model.eval()
    preds: dict[tuple, float] = {}
    df_sorted = df.sort_values(["iso3", "year"]).copy()
    df_sorted[feat_cols] = df_sorted[feat_cols].fillna(0.0).values
    device = torch.device("cpu")

    with torch.no_grad():
        for iso, cdf in df_sorted.groupby("iso3"):
            cdf = cdf.reset_index()
            for i in range(seq_len, len(cdf)):
                seq_vals = cdf.loc[i - seq_len : i - 1, feat_cols].values.astype(np.float32)
                if np.any(np.isnan(seq_vals)):
                    continue
                x = torch.from_numpy(seq_vals).unsqueeze(0).to(device)
                pred = model(x).item()
                preds[(iso, int(cdf.loc[i, "year"]))] = pred

    index = df_sorted.set_index(["iso3", "year"]).index
    return pd.Series({(iso, yr): preds.get((iso, yr), np.nan)
                      for iso, yr in index}, name="lstm_pred")


def run_lstm(ds: DataSplit, epochs: int = 80, batch_size: int = 64,
             lr: float = 1e-3) -> tuple[Any, ModelMetrics, pd.Series]:
    import torch  # lazy import — must stay here; top-level torch import crashes XGBoost

    device = torch.device("cpu")
    logger.info("PyTorch device: cpu (lazy import after sklearn/XGBoost)")
    n_feat = len(ds.feature_cols)

    # Build sequence datasets from scaled data
    df_tr_scaled = ds.df_train.copy()
    df_te_scaled = ds.df_test.copy()
    df_tr_scaled[ds.feature_cols] = ds.X_train.values
    df_te_scaled[ds.feature_cols] = ds.X_test.values

    train_ds = _make_sequence_dataset(df_tr_scaled, ds.feature_cols, SEQ_LEN, torch)
    if len(train_ds) == 0:
        logger.warning("No valid sequences for LSTM training — skipping")
        raise ValueError("Empty sequence dataset")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = _make_lstm_model(n_feat, torch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, min_lr=1e-5)
    criterion = torch.nn.MSELoss()

    best_val_loss, patience_cnt, best_state = np.inf, 0, None
    PATIENCE = 15

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_loss = float(np.mean(train_losses))
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            logger.info("LSTM early stop at epoch %d", epoch + 1)
            break

    if best_state:
        model.load_state_dict(best_state)

    # Build predictions for all rows that have a full lookback window
    model.eval()
    all_df = pd.concat([ds.df_train, ds.df_test])
    all_df_sc = all_df.copy()
    train_med = ds.df_train[ds.feature_cols].median()
    scaled_vals = ds.scaler.transform(
        all_df[ds.feature_cols].fillna(train_med))
    # positional assignment avoids pandas index-alignment errors
    all_df_sc[ds.feature_cols] = scaled_vals
    all_preds = _predict_lstm_on_df(model, all_df_sc, ds.feature_cols,
                                    ds.scaler, SEQ_LEN, torch)

    def _align(df_part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pairs = list(zip(df_part["iso3"], df_part["year"]))
        yhat = np.array([all_preds.get((iso, yr), np.nan) for iso, yr in pairs])
        ytrue = df_part[OUTCOME].values
        mask = ~np.isnan(yhat)
        return ytrue[mask], yhat[mask]

    ytr_true, ytr_hat = _align(
        ds.df_train[ds.df_train["year"] > ds.df_train["year"].min() + SEQ_LEN - 1])
    yte_true, yte_hat = _align(ds.df_test)

    m = _metrics("LSTM", ytr_true, ytr_hat, yte_true, yte_hat)
    logger.info("LSTM       R²_train=%.4f R²_test=%.4f RMSE_test=%.3f (sequences=%d)",
                m.r2_train, m.r2_test, m.rmse_test, len(train_ds))
    return model, m, all_preds


# ── 4. Ensemble stacking ───────────────────────────────────────────────────────

def run_ensemble(ds: DataSplit, models: dict[str, Any],
                 lstm_preds: pd.Series | None = None) -> tuple[Any, ModelMetrics]:
    """Stacking ensemble: Ridge meta-learner over RF + XGBoost [+ LSTM] OOF preds."""
    rf  = models.get("RandomForest")
    xgb = models.get("XGBoost")
    if rf is None or xgb is None:
        raise ValueError("RandomForest and XGBoost required for ensemble")

    Xtr, ytr = ds.X_train.values, ds.y_train.values
    Xte, yte = ds.X_test.values,  ds.y_test.values

    # Out-of-fold predictions via time-series CV (5 splits)
    tscv = TimeSeriesSplit(n_splits=5)
    oof_rf  = np.full(len(ytr), np.nan)
    oof_xgb = np.full(len(ytr), np.nan)

    for tr_idx, val_idx in tscv.split(Xtr):
        rf_fold = RandomForestRegressor(n_estimators=200, max_features=0.33,
                                        random_state=RANDOM_STATE, n_jobs=1)
        rf_fold.fit(Xtr[tr_idx], ytr[tr_idx])
        oof_rf[val_idx] = rf_fold.predict(Xtr[val_idx])

        xgb_fold = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                max_depth=5, random_state=RANDOM_STATE, n_jobs=1,
                                tree_method="hist")
        xgb_fold.fit(Xtr[tr_idx], ytr[tr_idx])
        oof_xgb[val_idx] = xgb_fold.predict(Xtr[val_idx])

    # Test predictions
    test_rf  = rf.predict(Xte)
    test_xgb = xgb.predict(Xte)

    # Build meta-features
    meta_cols = ["rf", "xgb"]
    meta_train = np.column_stack([oof_rf, oof_xgb])
    meta_test  = np.column_stack([test_rf, test_xgb])

    # Add LSTM predictions if available
    if lstm_preds is not None:
        def _get_lstm(df_part: pd.DataFrame) -> np.ndarray:
            pairs = list(zip(df_part["iso3"], df_part["year"]))
            preds = np.array([lstm_preds.get((iso, yr), np.nan) for iso, yr in pairs])
            return preds

        lstm_tr = _get_lstm(ds.df_train)
        lstm_te = _get_lstm(ds.df_test)
        # Only use rows where LSTM has predictions (> seq_len years of history)
        valid_tr = ~np.isnan(lstm_tr)
        if valid_tr.sum() > 30:
            # Impute missing LSTM OOF with RF OOF for those rows
            lstm_tr_filled = np.where(np.isnan(lstm_tr), oof_rf, lstm_tr)
            lstm_te_filled = np.where(np.isnan(lstm_te), test_rf, lstm_te)
            meta_train = np.column_stack([meta_train, lstm_tr_filled])
            meta_test  = np.column_stack([meta_test,  lstm_te_filled])
            meta_cols.append("lstm")

    # Only rows that have OOF predictions (TimeSeriesSplit leaves the first fold
    # unscored — fill those NaN positions with column means before meta-prediction)
    valid_mask = ~np.any(np.isnan(meta_train), axis=1)
    meta_ridge = Ridge(alpha=1.0)
    meta_ridge.fit(meta_train[valid_mask], ytr[valid_mask])

    col_means = np.nanmean(meta_train, axis=0)
    meta_train_filled = np.where(np.isnan(meta_train),
                                  col_means[np.newaxis, :], meta_train)
    yhat_train_ens = meta_ridge.predict(meta_train_filled)
    yhat_test_ens  = meta_ridge.predict(meta_test)

    m = _metrics("Ensemble", ytr[valid_mask], yhat_train_ens[valid_mask],
                 yte, yhat_test_ens)
    m.feature_importances = pd.Series(
        meta_ridge.coef_, index=meta_cols[:len(meta_ridge.coef_)])
    logger.info("Ensemble   R²_train=%.4f R²_test=%.4f RMSE_test=%.3f (meta_cols=%s)",
                m.r2_train, m.r2_test, m.rmse_test, meta_cols)
    return meta_ridge, m, {"meta_train": meta_train, "meta_test": meta_test,
                            "meta_cols": meta_cols}


# ── 5. Time-series cross-validation ───────────────────────────────────────────

def ts_cross_validate(ds: DataSplit, n_splits: int = 5) -> pd.DataFrame:
    """Walk-forward CV: expanding window refit on RF and XGBoost."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    Xtr, ytr = ds.X_train.values, ds.y_train.values
    records: list[dict] = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(Xtr)):
        for mname, clf in [
            ("RF", RandomForestRegressor(n_estimators=200, max_features=0.33,
                                         random_state=RANDOM_STATE, n_jobs=1)),
            ("XGB", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                  random_state=RANDOM_STATE, n_jobs=1,
                                  tree_method="hist")),
        ]:
            clf.fit(Xtr[tr_idx], ytr[tr_idx])
            yhat = clf.predict(Xtr[val_idx])
            records.append({
                "fold": fold + 1, "model": mname,
                "r2": round(float(r2_score(ytr[val_idx], yhat)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(ytr[val_idx], yhat))), 4),
                "n_train": len(tr_idx), "n_val": len(val_idx),
            })

    result = pd.DataFrame(records)
    logger.info("Time-series CV summary:\n%s",
                result.groupby("model")[["r2", "rmse"]].agg(["mean", "std"]).round(4))
    return result


# ── 6. Threshold detection ────────────────────────────────────────────────────

def detect_thresholds(df: pd.DataFrame,
                       gdp_col: str = "log_gdp_per_capita_ppp") -> pd.DataFrame:
    """Identify GDP thresholds where the GDP-LE slope changes.

    Uses:
    1. Regression tree with max_depth=2 (finds up to 3 segments)
    2. Piecewise linear regression at the tree-identified splits
    3. Chow test at each split point
    """
    sub = df.dropna(subset=[gdp_col, OUTCOME]).copy()
    X = sub[[gdp_col]].values
    y = sub[OUTCOME].values

    # Tree-based splits
    dt = DecisionTreeRegressor(max_depth=2, min_samples_leaf=20,
                               random_state=RANDOM_STATE)
    dt.fit(X, y)
    tree = dt.tree_
    thresholds = [
        np.exp(tree.threshold[i]) if tree.threshold[i] > 0 else np.nan
        for i in range(tree.node_count)
        if tree.threshold[i] != -2
    ]
    split_logvals = [tree.threshold[i] for i in range(tree.node_count)
                     if tree.threshold[i] != -2]

    records: list[dict] = []
    for log_thresh in sorted(set(split_logvals)):
        gdp_thresh = np.exp(log_thresh)
        mask_lo = X[:, 0] <= log_thresh
        mask_hi = X[:, 0] > log_thresh
        if mask_lo.sum() < 20 or mask_hi.sum() < 20:
            continue
        # Slopes in each segment
        b_lo = np.polyfit(X[mask_lo, 0], y[mask_lo], 1)[0]
        b_hi = np.polyfit(X[mask_hi, 0], y[mask_hi], 1)[0]
        # Chow test statistic (simplified)
        sse_pool = np.sum((y - np.polyval(np.polyfit(X[:, 0], y, 1), X[:, 0])) ** 2)
        sse_lo   = np.sum((y[mask_lo] - np.polyval(np.polyfit(X[mask_lo, 0], y[mask_lo], 1),
                                                     X[mask_lo, 0])) ** 2)
        sse_hi   = np.sum((y[mask_hi] - np.polyval(np.polyfit(X[mask_hi, 0], y[mask_hi], 1),
                                                     X[mask_hi, 0])) ** 2)
        n = len(y); k = 2
        chow_f = ((sse_pool - (sse_lo + sse_hi)) / k) / ((sse_lo + sse_hi) / (n - 2 * k))
        from scipy.stats import f as fdist
        chow_p = float(1 - fdist.cdf(chow_f, k, n - 2 * k))
        records.append({
            "gdp_per_capita_ppp_threshold": round(float(gdp_thresh), 0),
            "log_gdp_threshold": round(float(log_thresh), 3),
            "slope_below": round(float(b_lo), 4),
            "slope_above": round(float(b_hi), 4),
            "chow_f_stat": round(float(chow_f), 2),
            "chow_p_value": round(float(chow_p), 4),
            "n_below": int(mask_lo.sum()),
            "n_above": int(mask_hi.sum()),
        })

    result = pd.DataFrame(records)
    out_path = FINAL_DIR.parent.parent / "outputs" / "tables" / "threshold_analysis.csv"
    result.to_csv(out_path, index=False)
    logger.info("Thresholds detected: %d splits\n%s", len(result), result.to_string())
    return result


# ── 7. Model saving ───────────────────────────────────────────────────────────

def save_models(models: dict[str, Any], lstm_model: Any | None = None) -> None:
    for name, m in models.items():
        path = MODELS_DIR / f"{name.lower()}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(m, f)
        logger.info("Saved %s to %s", name, path)
    if lstm_model is not None:
        import torch  # lazy
        path = MODELS_DIR / "lstm_model.pth"
        torch.save(lstm_model.state_dict(), path)
        logger.info("Saved LSTM to %s", path)


def load_model(name: str) -> Any:
    path = MODELS_DIR / f"{name.lower()}_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Master runner ──────────────────────────────────────────────────────────────

def run_all_ml(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = df if df is not None else pd.read_csv(FINAL_DIR / "master_dataset.csv")
    logger.info("═" * 60)
    logger.info("Phase 3 — Machine Learning Models")
    logger.info("═" * 60)

    ds = make_split(df)

    logger.info("── Linear models ──")
    linear_res = run_linear_models(ds)

    logger.info("── Tree-based models ──")
    tree_res = run_tree_models(ds)

    logger.info("── LSTM ──")
    try:
        lstm_model, lstm_m, lstm_preds = run_lstm(ds, epochs=80)
    except Exception as exc:
        logger.warning("LSTM failed: %s — continuing without it", exc)
        lstm_model, lstm_m, lstm_preds = None, None, None

    logger.info("── Ensemble stacking ──")
    base_models = {k: v for k, v in {
        **{n: m for n, (m, _) in tree_res.items()},
    }.items()}
    ensemble_model, ensemble_m, ensemble_meta = run_ensemble(
        ds, base_models, lstm_preds=lstm_preds)

    logger.info("── Time-series CV ──")
    cv_results = ts_cross_validate(ds)

    logger.info("── Threshold detection ──")
    thresholds = detect_thresholds(df)

    # Collect all metrics
    all_metrics: dict[str, ModelMetrics] = {}
    for n, (_, m) in linear_res.items(): all_metrics[n] = m
    for n, (_, m) in tree_res.items():   all_metrics[n] = m
    if lstm_m:                            all_metrics["LSTM"] = lstm_m
    all_metrics["Ensemble"] = ensemble_m

    # Feature importances from XGBoost (primary) and RF (secondary)
    xgb_imp = tree_res["XGBoost"][1].feature_importances
    rf_imp  = tree_res["RandomForest"][1].feature_importances

    save_models(
        {**{n: m for n, (m, _) in linear_res.items()},
         **{n: m for n, (m, _) in tree_res.items()},
         "Ensemble": ensemble_model},
        lstm_model,
    )

    return {
        "ds": ds,
        "linear": linear_res,
        "trees": tree_res,
        "lstm": {"model": lstm_model, "metrics": lstm_m, "preds": lstm_preds},
        "ensemble": {"model": ensemble_model, "metrics": ensemble_m,
                     "meta": ensemble_meta},
        "metrics": all_metrics,
        "cv_results": cv_results,
        "thresholds": thresholds,
        "xgb_importance": xgb_imp,
        "rf_importance": rf_imp,
    }
