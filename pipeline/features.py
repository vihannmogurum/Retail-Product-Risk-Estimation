"""
Leakage-safe Chronological Feature Engineering Module
Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III

Aggregates FMCG daily multi-channel data to daily SKU totals, then constructs
chronological features strictly using past observations (t-1, t-7, t-14, etc.)
grouped by SKU/product_id without future leakage.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Required feature names
REQUIRED_FEATURES = [
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_30",
]

ADDITIONAL_FEATURES = [
    "price_unit",
    "promotion_flag",
    "product_encoded",
]

FEATURE_COLUMNS = REQUIRED_FEATURES + ADDITIONAL_FEATURES
TARGET_COLUMN = "units_sold"

# 30 FMCG SKUs
FMCG_SKUS = [
    'JU-021', 'MI-002', 'MI-006', 'MI-008', 'MI-011', 'MI-022', 'MI-023', 'MI-026',
    'RE-004', 'RE-007', 'RE-015', 'RE-017', 'RE-025', 'SN-010', 'SN-013', 'SN-019',
    'SN-027', 'SN-028', 'SN-030', 'YO-001', 'YO-003', 'YO-005', 'YO-009', 'YO-012',
    'YO-014', 'YO-016', 'YO-018', 'YO-020', 'YO-024', 'YO-029'
]

PRODUCT_MAP = {sku: idx for idx, sku in enumerate(FMCG_SKUS)}

def aggregate_fmcg_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Critical Aggregation Step:
    Because each (date, sku) has multiple rows across channels/regions/pack types,
    aggregate to daily SKU totals BEFORE sorting and computing lag/rolling features:
    - units_sold: sum
    - stock_available: sum
    - delivered_qty: sum
    - price_unit: mean (round 2)
    - promotion_flag: max
    - delivery_days: mean (round to integer)
    - brand, segment, category: first
    """
    data = df.copy()
    # Normalize column names if needed
    if "sku" not in data.columns and "product_id" in data.columns:
        data["sku"] = data["product_id"]
    if "product_id" not in data.columns and "sku" in data.columns:
        data["product_id"] = data["sku"]

    # Check if raw FMCG multi-channel format
    group_cols = ["date", "sku"]
    if all(col in data.columns for col in ["date", "sku"]):
        # Aggregate
        agg_dict = {}
        if "units_sold" in data.columns:
            agg_dict["units_sold"] = "sum"
        if "stock_available" in data.columns:
            agg_dict["stock_available"] = "sum"
        elif "inventory_level" in data.columns:
            agg_dict["inventory_level"] = "sum"
        elif "current_stock" in data.columns:
            agg_dict["current_stock"] = "sum"

        if "delivered_qty" in data.columns:
            agg_dict["delivered_qty"] = "sum"

        if "price_unit" in data.columns:
            agg_dict["price_unit"] = "mean"
        elif "price" in data.columns:
            agg_dict["price"] = "mean"

        if "promotion_flag" in data.columns:
            agg_dict["promotion_flag"] = "max"
        elif "promotion" in data.columns:
            agg_dict["promotion"] = "max"

        if "delivery_days" in data.columns:
            agg_dict["delivery_days"] = "mean"
        elif "lead_time_days" in data.columns:
            agg_dict["lead_time_days"] = "mean"

        for meta_col in ["brand", "segment", "category"]:
            if meta_col in data.columns:
                agg_dict[meta_col] = "first"

        if agg_dict:
            # Check if grouping is necessary (if multiple rows per date, sku)
            if data.duplicated(subset=["date", "sku"]).any():
                data = data.groupby(["date", "sku"], as_index=False).agg(agg_dict)

        if "price_unit" in data.columns:
            data["price_unit"] = data["price_unit"].round(2)
        elif "price" in data.columns:
            data["price_unit"] = data["price"].round(2)

        if "delivery_days" in data.columns:
            data["delivery_days"] = data["delivery_days"].round().astype(int)
        elif "lead_time_days" in data.columns:
            data["delivery_days"] = data["lead_time_days"].round().astype(int)

        if "stock_available" in data.columns:
            data["stock_available"] = data["stock_available"].round().astype(int)
            data["current_stock"] = data["stock_available"]
            data["inventory_level"] = data["stock_available"]

        data["product_id"] = data["sku"]
        data["sku_id"] = data["sku"]

    return data

def engineer_features(df: pd.DataFrame, fill_lags: bool = True) -> pd.DataFrame:
    """
    Computes calendar, pricing, and strictly shifted lagged & rolling features
    for the aggregated daily SKU series without leakage.
    """
    data = aggregate_fmcg_daily(df)
    data["date"] = pd.to_datetime(data["date"])

    # Ensure sku/product_id
    if "product_id" not in data.columns and "sku" in data.columns:
        data["product_id"] = data["sku"]
    elif "sku" not in data.columns and "product_id" in data.columns:
        data["sku"] = data["product_id"]
    if "sku_id" not in data.columns:
        data["sku_id"] = data["product_id"]

    # Align price and promotion column names
    if "price_unit" not in data.columns:
        data["price_unit"] = data["price"] if "price" in data.columns else 10.0
    if "promotion_flag" not in data.columns:
        data["promotion_flag"] = data["promotion"] if "promotion" in data.columns else 0

    # Ensure integer/float types
    data["price_unit"] = data["price_unit"].astype(float)
    data["promotion_flag"] = data["promotion_flag"].astype(int)

    # Sort strictly chronologically by product_id and date
    data = data.sort_values(by=["product_id", "date"]).reset_index(drop=True)

    # 1. Calendar Features
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    # 2. Product encoding
    data["product_encoded"] = data["product_id"].map(PRODUCT_MAP).fillna(0).astype(int)

    # 3. Leakage-safe Lags and Rolling Windows
    if TARGET_COLUMN in data.columns:
        grouped = data.groupby("product_id")[TARGET_COLUMN]

        # Shifted lags (t - 1, t - 7, t - 14)
        data["lag_1"] = grouped.shift(1)
        data["lag_7"] = grouped.shift(7)
        data["lag_14"] = grouped.shift(14)

        # Shifted series for rolling calculations so day t cannot see day t's sales
        data["_target_t_minus_1"] = grouped.shift(1)
        grouped_prev = data.groupby("product_id")["_target_t_minus_1"]

        # rolling_mean_7, rolling_std_7, rolling_mean_30
        data["rolling_mean_7"] = grouped_prev.transform(
            lambda s: s.rolling(window=7, min_periods=1).mean()
        )
        data["rolling_std_7"] = grouped_prev.transform(
            lambda s: s.rolling(window=7, min_periods=1).std().fillna(0.0)
        )
        data["rolling_mean_30"] = grouped_prev.transform(
            lambda s: s.rolling(window=30, min_periods=1).mean()
        )

        data.drop(columns=["_target_t_minus_1"], inplace=True)
    else:
        for col in ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7", "rolling_mean_30"]:
            if col not in data.columns:
                data[col] = np.nan

    if fill_lags:
        # Impute missing warm-up values with product-level medians
        for pid in FMCG_SKUS:
            p_mask = data["product_id"] == pid
            for col in ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30"]:
                med = data.loc[p_mask, col].median()
                if pd.isna(med):
                    med = data[col].median() if not data[col].dropna().empty else 20.0
                data.loc[p_mask, col] = data.loc[p_mask, col].fillna(med)

            data.loc[p_mask, "rolling_std_7"] = data.loc[p_mask, "rolling_std_7"].fillna(0.0)

        # Catch-all
        data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].fillna(0.0)

    return data
