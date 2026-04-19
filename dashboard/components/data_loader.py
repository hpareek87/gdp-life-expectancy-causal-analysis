"""Cached data loading for the dashboard."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH   = ROOT / "data" / "final" / "master_dataset.csv"
MODELS_DIR  = ROOT / "outputs" / "models"
TABLES_DIR  = ROOT / "outputs" / "tables"
CACHE_DIR   = ROOT / "dashboard" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Income-group boundaries (World Bank Atlas, 2024)
INCOME_THRESHOLDS = {
    "Low income":         (0, 1135),
    "Lower-middle income":(1135, 4465),
    "Upper-middle income":(4465, 13845),
    "High income":        (13845, 1e9),
}

INCOME_COLORS = {
    "Low income":          "#E53935",
    "Lower-middle income": "#FB8C00",
    "Upper-middle income": "#43A047",
    "High income":         "#1E88E5",
}


@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["iso3", "year"])
    df["income_group"] = pd.cut(
        df["gdp_per_capita_ppp"],
        bins=[0, 1135, 4465, 13845, 1e9],
        labels=["Low income", "Lower-middle income",
                "Upper-middle income", "High income"],
        right=True,
    )
    return df


@st.cache_data(ttl=3600)
def load_country_list(df: pd.DataFrame) -> list[str]:
    mapping = df.dropna(subset=["country"])[["iso3", "country"]].drop_duplicates()
    return sorted(mapping["country"].tolist())


@st.cache_data(ttl=3600)
def get_country_iso(df: pd.DataFrame) -> dict[str, str]:
    mapping = df.dropna(subset=["country"])[["iso3", "country"]].drop_duplicates()
    return dict(zip(mapping["country"], mapping["iso3"]))


@st.cache_data(ttl=3600)
def load_threshold_analysis() -> pd.DataFrame:
    path = TABLES_DIR / "threshold_analysis.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_feature_importance() -> pd.DataFrame:
    path = TABLES_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


@st.cache_resource
def load_xgb_model():
    path = MODELS_DIR / "xgboost_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_rf_model():
    path = MODELS_DIR / "randomforest_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    # Rebuild scaler from training data to match dashboard predictions
    sys.path.insert(0, str(ROOT))
    try:
        from src.analysis.ml_models import make_split
        df = load_master()
        ds = make_split(df)
        return ds.scaler, ds.feature_cols
    except Exception:
        return None, []


def predict_life_expectancy(feature_dict: dict) -> float | None:
    """Predict LE from a dict of {feature_name: value}."""
    model = load_xgb_model()
    scaler, feat_cols = load_scaler()
    if model is None or scaler is None or not feat_cols:
        return None
    row = np.array([[feature_dict.get(c, 0.0) for c in feat_cols]])
    row_scaled = scaler.transform(row)
    return float(model.predict(row_scaled)[0])


@st.cache_data(ttl=3600)
def get_summary_stats(df: pd.DataFrame) -> dict:
    latest = df.groupby("iso3").last().reset_index()
    return {
        "n_countries": df["iso3"].nunique(),
        "year_range": (int(df["year"].min()), int(df["year"].max())),
        "mean_le_2024": round(float(latest["life_expectancy"].mean()), 1),
        "mean_le_2000": round(float(df[df["year"] == 2000]["life_expectancy"].mean()), 1),
        "mean_gdp_2024": round(float(latest["gdp_per_capita_ppp"].mean()), 0),
        "max_le_gain": round(float(
            df.groupby("iso3").apply(
                lambda x: x["life_expectancy"].max() - x["life_expectancy"].min()
            ).max()
        ), 1),
    }
