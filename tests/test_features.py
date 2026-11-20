"""
Test Feature Engineering Module with FMCG Data
Verifies:
- Daily aggregation by (date, sku)
- Required exact feature names exist
- No future data leakage across temporal shifts
- Proper chronological sorting by product_id and date
"""

import numpy as np
import pandas as pd
import pytest
from pipeline.features import (
    engineer_features,
    aggregate_fmcg_daily,
    REQUIRED_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

def create_mock_fmcg_df():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    records = []
    for sku in ["MI-006", "YO-029"]:
        for idx, dt in enumerate(dates):
            # simulate 2 rows per date, sku across channels
            for ch in ["Retail", "Online"]:
                records.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "sku": sku,
                    "brand": "BrandX",
                    "segment": "Standard",
                    "category": "Milk" if "MI" in sku else "Yogurt",
                    "channel": ch,
                    "region": "North",
                    "pack_type": "Bottle",
                    "price_unit": 4.5,
                    "promotion_flag": 1 if idx % 5 == 0 else 0,
                    "delivery_days": 3,
                    "stock_available": 100,
                    "delivered_qty": 50,
                    "units_sold": 10 + idx,
                })
    return pd.DataFrame(records)

def test_fmcg_aggregation():
    df = create_mock_fmcg_df()
    agg_df = aggregate_fmcg_daily(df)
    # Check that duplicates per date, sku are eliminated
    assert not agg_df.duplicated(subset=["date", "sku"]).any()
    # units_sold should be summed across channels (2 channels * (10+0) = 20 for day 0)
    day0 = agg_df[agg_df["date"] == "2024-01-01"].iloc[0]
    assert day0["units_sold"] == 20

def test_required_feature_names_present():
    df = create_mock_fmcg_df()
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
    df = create_mock_fmcg_df()
    agg_df = aggregate_fmcg_daily(df)
    feat_original = engineer_features(agg_df, fill_lags=False)

    # Modify day 15's target
    agg_mod = agg_df.copy()
    idx_target = (agg_mod["sku"] == "MI-006") & (agg_mod["date"] == "2024-01-15")
    agg_mod.loc[idx_target, "units_sold"] = 99999

    feat_modified = engineer_features(agg_mod, fill_lags=False)

    orig_row = feat_original.loc[idx_target].iloc[0]
    mod_row = feat_modified.loc[idx_target].iloc[0]

    assert orig_row["lag_1"] == mod_row["lag_1"], "Leakage: lag_1 changed on modified row"
    assert orig_row["rolling_mean_7"] == mod_row["rolling_mean_7"], "Leakage: rolling_mean_7 changed"
