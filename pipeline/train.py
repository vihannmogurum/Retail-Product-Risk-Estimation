"""
Model Training and 3-Model Comparison Pipeline
Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III

Trains and compares:
1. Ridge Regression (linear baseline)
2. Random Forest Regressor (tree ensemble)
3. HistGradientBoostingRegressor (gradient boosting)

Using Kaggle FMCG Dataset (FMCG_2022_2024.csv, aggregated to daily SKU totals).
Chronological 80/20 train/test holdout split.
Hyperparameter tuning via TimeSeriesSplit RandomizedSearchCV.
Outputs:
- artifacts/model_comparison.json
- artifacts/demand_model.joblib
- artifacts/best_model.joblib
- artifacts/metrics.json
- artifacts/sku_metadata.json
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.getcwd())

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from pipeline.features import (
        engineer_features,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        FMCG_SKUS,
        PRODUCT_MAP,
    )
except ImportError:
    from features import (
        engineer_features,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        FMCG_SKUS,
        PRODUCT_MAP,
    )

def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(y_true)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

def run_train_pipeline(
    data_path: str = "data/FMCG_2022_2024.csv",
    artifacts_dir: str = "artifacts"
) -> Dict[str, Any]:
    os.makedirs(artifacts_dir, exist_ok=True)

    if not os.path.exists(data_path):
        data_path = "data/retail_sales.csv"

    print(f"Reading dataset from {data_path}...", flush=True)
    raw_df = pd.read_csv(data_path)
    df = engineer_features(raw_df, fill_lags=True)

    # Chronological sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "product_id"]).reset_index(drop=True)

    unique_dates = sorted(df["date"].unique())
    split_idx = int(len(unique_dates) * 0.80)
    split_date = unique_dates[split_idx]

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    print(f"Aggregated daily dataset shape: {df.shape}", flush=True)
    print(f"Train split: {len(train_df)} rows ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')})", flush=True)
    print(f"Holdout split: {len(test_df)} rows ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})", flush=True)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # TimeSeries cross-validation on train split
    tscv = TimeSeriesSplit(n_splits=3)

    models_config = {
        "Ridge Regression": {
            "model": Ridge(random_state=42),
            "params": {"alpha": [0.1, 1.0, 10.0, 50.0]},
            "scale": True,
            "n_iter": 4,
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=1),
            "params": {
                "n_estimators": [30, 50],
                "max_depth": [6, 10],
                "min_samples_split": [2, 5],
            },
            "scale": False,
            "n_iter": 3,
        },
        "Gradient Boosting": {
            "model": HistGradientBoostingRegressor(random_state=42),
            "params": {
                "learning_rate": [0.05, 0.1],
                "max_iter": [40, 80],
                "max_leaf_nodes": [15, 31],
            },
            "scale": False,
            "n_iter": 3,
        },
    }

    comparison_results = {}
    model_predictions = {}
    best_wape = float("inf")
    best_name = None
    best_model_obj = None

    for name, conf in models_config.items():
        print(f"\n--- Training {name} ---", flush=True)
        X_tr = X_train_scaled if conf["scale"] else X_train
        X_te = X_test_scaled if conf["scale"] else X_test

        search = RandomizedSearchCV(
            estimator=conf["model"],
            param_distributions=conf["params"],
            n_iter=conf["n_iter"],
            cv=tscv,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=1,
        )
        search.fit(X_tr, y_train)

        best_estimator = search.best_estimator_
        y_pred = np.maximum(0.0, best_estimator.predict(X_te))
        model_predictions[name] = y_pred

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        wape = float(compute_wape(y_test, y_pred))

        comparison_results[name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "wape": round(wape, 4),
            "best_params": search.best_params_,
            "cv_best_score_neg_mae": round(float(search.best_score_), 4),
        }
        print(f"Results {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, WAPE={wape*100:.2f}%", flush=True)

        if wape < best_wape:
            best_wape = wape
            best_name = name
            best_model_obj = best_estimator

    print(f"\nWinning Model: {best_name} with Holdout WAPE: {best_wape*100:.2f}%", flush=True)

    # SKU Metadata and Residual Standard Deviations for Risk Engine
    test_df["predicted_demand"] = model_predictions[best_name]
    test_df["residual"] = test_df[TARGET_COLUMN] - test_df["predicted_demand"]

    sku_metadata = {}
    for pid in sorted(df["product_id"].unique()):
        sub_df = test_df[test_df["product_id"] == pid]
        if sub_df.empty:
            sub_df = df[df["product_id"] == pid]

        brand = sub_df["brand"].iloc[0] if "brand" in sub_df.columns else "Brand"
        segment = sub_df["segment"].iloc[0] if "segment" in sub_df.columns else "Standard"
        category = sub_df["category"].iloc[0] if "category" in sub_df.columns else "FMCG"
        product_name = f"{brand} {segment} ({pid})"

        base_price = float(sub_df["price_unit"].iloc[-1]) if "price_unit" in sub_df.columns else 10.0
        lead_time = int(sub_df["delivery_days"].iloc[-1]) if "delivery_days" in sub_df.columns else 3
        if lead_time <= 0:
            lead_time = 3

        res_std = float(sub_df["residual"].std()) if "residual" in sub_df.columns and not sub_df["residual"].empty else 5.0
        if np.isnan(res_std) or res_std < 1.0:
            res_std = max(1.5, float(sub_df[TARGET_COLUMN].std() * 0.4))

        mean_demand = float(sub_df[TARGET_COLUMN].mean())
        mean_pred = float(sub_df["predicted_demand"].mean()) if "predicted_demand" in sub_df.columns else mean_demand
        
        # Recent stock available
        stock_col = "stock_available" if "stock_available" in sub_df.columns else ("current_stock" if "current_stock" in sub_df.columns else "inventory_level")
        recent_inv = int(sub_df[stock_col].iloc[-1]) if stock_col in sub_df.columns else int(mean_demand * lead_time * 2)

        # Recent lag/rolling values for quick prefilling
        last_row = sub_df.iloc[-1]
        sku_metadata[pid] = {
            "product_id": pid,
            "sku_id": pid,
            "sku": pid,
            "product_name": product_name,
            "brand": brand,
            "segment": segment,
            "category": category,
            "base_price": round(base_price, 2),
            "price_unit": round(base_price, 2),
            "lead_time_days": lead_time,
            "delivery_days": lead_time,
            "demand_std": round(res_std, 2),
            "rolling_std_7": round(float(last_row.get("rolling_std_7", res_std)), 2),
            "historical_mean_demand": round(mean_demand, 2),
            "predicted_mean_demand": round(mean_pred, 2),
            "recent_inventory": recent_inv,
            "current_stock": recent_inv,
            "lag_1": round(float(last_row.get("lag_1", mean_demand)), 1),
            "lag_7": round(float(last_row.get("lag_7", mean_demand)), 1),
            "lag_14": round(float(last_row.get("lag_14", mean_demand)), 1),
            "rolling_mean_7": round(float(last_row.get("rolling_mean_7", mean_demand)), 1),
            "rolling_mean_30": round(float(last_row.get("rolling_mean_30", mean_demand)), 1),
        }

    # Extract feature importance
    feature_importance_list = []
    if hasattr(best_model_obj, "feature_importances_"):
        raw_imp = best_model_obj.feature_importances_
        feature_importance_list = [
            {"feature": col, "importance": round(float(imp), 5)}
            for col, imp in sorted(zip(FEATURE_COLUMNS, raw_imp), key=lambda x: x[1], reverse=True)
        ]
    elif hasattr(best_model_obj, "coef_"):
        raw_imp = np.abs(best_model_obj.coef_)
        s = np.sum(raw_imp) or 1.0
        feature_importance_list = [
            {"feature": col, "importance": round(float(imp / s), 5)}
            for col, imp in sorted(zip(FEATURE_COLUMNS, raw_imp), key=lambda x: x[1], reverse=True)
        ]
    else:
        from sklearn.inspection import permutation_importance
        X_eval = X_test_scaled if models_config[best_name]["scale"] else X_test
        perm = permutation_importance(best_model_obj, X_eval, y_test, n_repeats=2, random_state=42)
        raw_imp = np.maximum(0, perm.importances_mean)
        s = np.sum(raw_imp) or 1.0
        feature_importance_list = [
            {"feature": col, "importance": round(float(imp / s), 5)}
            for col, imp in sorted(zip(FEATURE_COLUMNS, raw_imp), key=lambda x: x[1], reverse=True)
        ]

    # Model comparison payload
    model_comparison_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_name,
        "holdout_split_date": split_date.strftime("%Y-%m-%d"),
        "train_rows": len(train_df),
        "holdout_rows": len(test_df),
        "total_rows": len(df),
        "dataset_name": "Kaggle FMCG 2022-2024",
        "models": {
            k: {
                "mae": v["mae"],
                "rmse": v["rmse"],
                "r2": v["r2"],
                "wape": v["wape"],
                "best_params": v["best_params"],
            }
            for k, v in comparison_results.items()
        },
        "feature_importances": feature_importance_list,
    }

    # Save artifacts
    model_comparison_file = os.path.join(artifacts_dir, "model_comparison.json")
    with open(model_comparison_file, "w", encoding="utf-8") as f:
        json.dump(model_comparison_payload, f, indent=2)

    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(model_comparison_payload, f, indent=2)

    sku_metadata_file = os.path.join(artifacts_dir, "sku_metadata.json")
    with open(sku_metadata_file, "w", encoding="utf-8") as f:
        json.dump(sku_metadata, f, indent=2)

    # Demand model joblib
    demand_model_bundle = {
        "model": best_model_obj,
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "scale_features": models_config[best_name]["scale"],
        "scaler": scaler,
        "product_map": PRODUCT_MAP,
        "sku_metadata": sku_metadata,
        "train_date_range": [train_df["date"].min().strftime("%Y-%m-%d"), train_df["date"].max().strftime("%Y-%m-%d")],
        "test_date_range": [test_df["date"].min().strftime("%Y-%m-%d"), test_df["date"].max().strftime("%Y-%m-%d")],
    }
    demand_model_file = os.path.join(artifacts_dir, "demand_model.joblib")
    joblib.dump(demand_model_bundle, demand_model_file)
    joblib.dump(demand_model_bundle, os.path.join(artifacts_dir, "best_model.joblib"))

    print("\nFMCG Training completed successfully!", flush=True)
    print(f"Emitted: {model_comparison_file}", flush=True)
    print(f"Emitted: {demand_model_file}", flush=True)
    print(f"Emitted: {sku_metadata_file}", flush=True)

    return model_comparison_payload

if __name__ == "__main__":
    run_train_pipeline()
