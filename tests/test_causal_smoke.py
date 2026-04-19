"""Smoke tests for Phase 2 causal inference module."""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.causal import (
    run_granger, run_iv, run_panel_fe, run_panel_fe_subgroups,
    run_did, run_synthetic_control, load_df,
)


@pytest.fixture(scope="module")
def df():
    from src.utils.config import FINAL_DIR
    if not (FINAL_DIR / "master_dataset.csv").exists():
        pytest.skip("master_dataset.csv not built")
    return load_df()


def test_granger_runs(df):
    res = run_granger(df)
    assert "gdp_to_le" in res
    assert "le_to_gdp" in res
    assert res["n_countries"] > 0


def test_panel_fe_runs(df):
    res = run_panel_fe(df)
    assert "baseline" in res
    # Coefficient should be a finite float
    assert np.isfinite(res["baseline"].coef_gdp)


def test_panel_fe_subgroups(df):
    res = run_panel_fe_subgroups(df)
    assert set(res.keys()) >= {"high", "middle", "low"}


def test_iv_runs(df):
    res = run_iv(df)
    assert len(res) >= 1
    for spec, r in res.items():
        assert np.isfinite(r.coef_gdp)
        # Strong instruments: F should exceed 10
        if not np.isnan(r.first_stage_fstat):
            assert r.first_stage_fstat > 10, f"Weak instrument in {spec}: F={r.first_stage_fstat}"


def test_did_runs(df):
    res = run_did(df)
    assert len(res) >= 1


def test_synthetic_control_good_prefit(df):
    res = run_synthetic_control(df)
    assert res.pre_rmspe < 1.0, f"Poor pre-period fit: RMSPE={res.pre_rmspe:.3f}"
    assert not np.isnan(res.post_att)
    assert len(res.donors) > 3
