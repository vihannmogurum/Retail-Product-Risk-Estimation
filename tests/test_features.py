"""
Test Feature Engineering Module
Verifies:
- Required exact feature names exist
- No future data leakage across temporal shifts
- Proper chronological sorting by product_id and date
"""

import numpy as np
import pandas as pd
import pytest
from pipeline.features import (
    engineer_features,
    REQUIRED_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

def create_mock_sales_df():
    dates = pd.date_range("2024-01-01", periods=45, freq="D")
    records = []
    for pid in ["PRD-01", "PRD-02"]:
        for idx, dt in enumerate(dates):
            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "product_id": pid,
                "price": 100.0,
                "competitor_price": 102.0,
                "promotion": 1 if idx % 7 == 0 else 0,
                "holiday": 0,
                "units_sold": 20 + idx, # increasing demand
            })
    return pd.DataFrame(records)

def test_required_feature_names_present():
    df = create_mock_sales_df()
    feat_df = engineer_features(df, fill_lags=True)

    for req_feat in REQUIRED_FEATURES:
        assert req_feat in feat_df.columns, f"Missing required feature: {req_feat}"

    assert "day_of_week" in feat_df.columns
    assert "month" in feat_df.columns
    assert "is_weekend" in feat_df.columns
    assert "lag_1" in feat_df.columns
    assert "lag_7" in feat_df.columns
    assert "lag_14" in feat_df.columns
    assert "rolling_mean_7" in feat_df.columns
    assert "rolling_std_7" in feat_df.columns
    assert "rolling_mean_30" in feat_df.columns

def test_leakage_safety_shift():
    """
    Changing the target value of row t must NEVER affect the lag_1, lag_7,
    or rolling_mean_7 of row t itself.
    """
    df = create_mock_sales_df()
    feat_original = engineer_features(df, fill_lags=False)

    # Make a modified copy where day 20's sales is multiplied by 100
    df_modified = df.copy()
    idx_target = (df_modified["product_id"] == "PRD-01") & (df_modified["date"] == "2024-01-20")
    df_modified.loc[idx_target, "units_sold"] = 99999

    feat_modified = engineer_features(df_modified, fill_lags=False)

    # For day 20 itself, lag_1 and rolling_mean_7 must remain completely identical!
    orig_row = feat_original.loc[idx_target].iloc[0]
    mod_row = feat_modified.loc[idx_target].iloc[0]

    assert orig_row["lag_1"] == mod_row["lag_1"], "Leakage detected! lag_1 changed when current day target changed"
    assert orig_row["rolling_mean_7"] == mod_row["rolling_mean_7"], "Leakage detected! rolling_mean_7 changed"

    # But for day 21, lag_1 should reflect the day 20 change
    idx_next = (df_modified["product_id"] == "PRD-01") & (df_modified["date"] == "2024-01-21")
    mod_next = feat_modified.loc[idx_next].iloc[0]
    assert mod_next["lag_1"] == 99999, "Day 21 must capture Day 20's target in lag_1"

def test_sku_id_alias_support():
    df = create_mock_sales_df()
    df["sku_id"] = "SKU-E101"
    df.drop(columns=["product_id"], inplace=True)

    feat_df = engineer_features(df, fill_lags=True)
    assert "product_id" in feat_df.columns
    assert feat_df["product_id"].iloc[0] == "PRD-01"
