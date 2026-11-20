"""
Leakage-safe Chronological Feature Engineering Module
Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III

Constructs chronological features strictly using past observations (t-1, t-7, etc.)
grouped by product_id/sku_id.
Required features:
- day_of_week
- month
- is_weekend
- lag_1
- lag_7
- lag_14
- rolling_mean_7
- rolling_std_7
- rolling_mean_30
Plus additional contextual features (price, competitor_price, price_ratio, promotion, holiday, product_encoded).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Primary features used for training & inference
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
    "price",
    "competitor_price",
    "price_ratio",
    "promotion",
    "holiday",
    "product_encoded",
]

FEATURE_COLUMNS = REQUIRED_FEATURES + ADDITIONAL_FEATURES
TARGET_COLUMN = "units_sold"

# 16 products lookup
PRODUCT_LIST = [f"PRD-{i:02d}" for i in range(1, 17)]
PRODUCT_MAP = {pid: idx for idx, pid in enumerate(PRODUCT_LIST)}

SKU_TO_PRODUCT = {
    f"SKU-{'E' if i<=4 else 'G' if i<=8 else 'A' if i<=12 else 'H'}{100*(((i-1)//4)+1) + ((i-1)%4)+1}": f"PRD-{i:02d}"
    for i in range(1, 17)
}
# Direct manual aliases for standard SKU patterns
SKU_TO_PRODUCT.update({
    "SKU-E101": "PRD-01", "SKU-E102": "PRD-02", "SKU-E103": "PRD-03", "SKU-E104": "PRD-04",
    "SKU-G201": "PRD-05", "SKU-G202": "PRD-06", "SKU-G203": "PRD-07", "SKU-G204": "PRD-08",
    "SKU-A301": "PRD-09", "SKU-A302": "PRD-10", "SKU-A303": "PRD-11", "SKU-A304": "PRD-12",
    "SKU-H401": "PRD-13", "SKU-H402": "PRD-14", "SKU-H403": "PRD-15", "SKU-H404": "PRD-16",
})
PRODUCT_TO_SKU = {v: k for k, v in SKU_TO_PRODUCT.items()}

def engineer_features(df: pd.DataFrame, fill_lags: bool = True) -> pd.DataFrame:
    """
    Computes calendar, pricing, and strictly shifted lagged & rolling features.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    # Ensure product_id exists (support sku_id alias)
    if "product_id" not in data.columns and "sku_id" in data.columns:
        data["product_id"] = data["sku_id"].map(SKU_TO_PRODUCT).fillna("PRD-01")
    elif "product_id" in data.columns and "sku_id" not in data.columns:
        data["sku_id"] = data["product_id"].map(PRODUCT_TO_SKU).fillna("SKU-E101")

    # Sort strictly chronologically by product and date
    data = data.sort_values(by=["product_id", "date"]).reset_index(drop=True)

    # 1. Calendar Features
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    # 2. Pricing & Marketing
    if "competitor_price" not in data.columns:
        data["competitor_price"] = data["price"]
    data["competitor_price"] = data["competitor_price"].fillna(data["price"])
    data["price_ratio"] = data["price"] / np.maximum(data["competitor_price"], 0.01)

    if "promotion" not in data.columns:
        data["promotion"] = 0
    if "holiday" not in data.columns:
        data["holiday"] = 0

    # Categorical encoding
    data["product_encoded"] = data["product_id"].map(PRODUCT_MAP).fillna(0).astype(int)

    # 3. Leakage-safe Lags and Rolling Windows
    # grouped by product_id
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
        for pid in PRODUCT_MAP.keys():
            p_mask = data["product_id"] == pid
            for col in ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30"]:
                med = data.loc[p_mask, col].median()
                if pd.isna(med):
                    med = data[col].median() if not data[col].dropna().empty else 35.0
                data.loc[p_mask, col] = data.loc[p_mask, col].fillna(med)

            data.loc[p_mask, "rolling_std_7"] = data.loc[p_mask, "rolling_std_7"].fillna(0.0)

        # Catch-all
        data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].fillna(0.0)

    return data
