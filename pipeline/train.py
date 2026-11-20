"""
Model Training and 3-Model Comparison Pipeline
Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III

Trains and compares:
1. Ridge Regression (linear baseline)
2. Random Forest Regressor (tree ensemble)
3. HistGradientBoostingRegressor (gradient boosting)

Chronological 80/20 train/test holdout split.
Hyperparameter tuning via TimeSeriesSplit RandomizedSearchCV.
Outputs:
- artifacts/model_comparison.json
- artifacts/demand_model.joblib
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

# Ensure project root and script directory are in sys.path
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
        PRODUCT_MAP,
        SKU_TO_PRODUCT,
        PRODUCT_TO_SKU,
    )
except ImportError:
    from features import (
        engineer_features,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        PRODUCT_MAP,
        SKU_TO_PRODUCT,
        PRODUCT_TO_SKU,
    )

def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(y_true)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)

def run_train_pipeline(
    data_path: str = "data/retail_sales.csv",
    artifacts_dir: str = "artifacts"
) -> Dict[str, Any]:
    os.makedirs(artifacts_dir, exist_ok=True)

    if not os.path.exists(data_path):
        from data.generate_data import generate_retail_sales_dataset
        print(f"Dataset {data_path} not found. Generating fresh dataset...")
        generate_retail_sales_dataset(output_path=data_path)

    print(f"Reading dataset from {data_path}...")
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

    print(f"Dataset shape: {df.shape}")
    print(f"Train split: {len(train_df)} rows ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"Holdout split: {len(test_df)} rows ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    # Preprocessing scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # TimeSeries cross-validation on train split
    tscv = TimeSeriesSplit(n_splits=3)

    models_config = {
        "Ridge Regression": {
            "model": Ridge(random_state=42),
            "params": {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0]},
            "scale": True,
            "n_iter": 5,
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=1),
            "params": {
                "n_estimators": [30, 60],
                "max_depth": [6, 10],
                "min_samples_split": [2, 5],
            },
            "scale": False,
            "n_iter": 4,
        },
        "Gradient Boosting": {
            "model": HistGradientBoostingRegressor(random_state=42),
            "params": {
                "learning_rate": [0.05, 0.1],
                "max_iter": [40, 80],
                "max_leaf_nodes": [15, 31],
            },
            "scale": False,
            "n_iter": 4,
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

    print(f"\nWinning Model: {best_name} with Holdout WAPE: {best_wape*100:.2f}%")

    # SKU / Product Metadata and Residual Standard Deviations for Risk Engine
    test_df["predicted_demand"] = model_predictions[best_name]
    test_df["residual"] = test_df[TARGET_COLUMN] - test_df["predicted_demand"]

    product_metadata = {}
    for pid in PRODUCT_MAP.keys():
        sub_df = test_df[test_df["product_id"] == pid]
        if sub_df.empty:
            sub_df = df[df["product_id"] == pid]

        p_name = sub_df["product_name"].iloc[0] if "product_name" in sub_df.columns else pid
        category = sub_df["category"].iloc[0] if "category" in sub_df.columns else "General"
        sku_id = sub_df["sku_id"].iloc[0] if "sku_id" in sub_df.columns else PRODUCT_TO_SKU.get(pid, "SKU-001")
        base_price = float(sub_df["base_price"].iloc[0]) if "base_price" in sub_df.columns else float(sub_df["price"].mean())
        lead_time = int(sub_df["lead_time_days"].iloc[0]) if "lead_time_days" in sub_df.columns else 7
        
        res_std = float(sub_df["residual"].std()) if "residual" in sub_df.columns and not sub_df["residual"].empty else 5.0
        if np.isnan(res_std) or res_std < 1.0:
            res_std = max(1.5, float(sub_df[TARGET_COLUMN].std() * 0.4))

        mean_demand = float(sub_df[TARGET_COLUMN].mean())
        mean_pred = float(sub_df["predicted_demand"].mean()) if "predicted_demand" in sub_df.columns else mean_demand
        recent_inv = int(sub_df["inventory_level"].iloc[-1]) if "inventory_level" in sub_df.columns else int(mean_demand * lead_time)

        product_metadata[pid] = {
            "product_id": pid,
            "sku_id": sku_id,
            "product_name": p_name,
            "category": category,
            "base_price": round(base_price, 2),
            "lead_time_days": lead_time,
            "demand_std": round(res_std, 2),
            "historical_mean_demand": round(mean_demand, 2),
            "predicted_mean_demand": round(mean_pred, 2),
            "recent_inventory": recent_inv,
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
        perm = permutation_importance(best_model_obj, X_eval, y_test, n_repeats=3, random_state=42)
        raw_imp = np.maximum(0, perm.importances_mean)
        s = np.sum(raw_imp) or 1.0
        feature_importance_list = [
            {"feature": col, "importance": round(float(imp / s), 5)}
            for col, imp in sorted(zip(FEATURE_COLUMNS, raw_imp), key=lambda x: x[1], reverse=True)
        ]

    # Required output 1: artifacts/model_comparison.json
    model_comparison_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_name,
        "holdout_split_date": split_date.strftime("%Y-%m-%d"),
        "train_rows": len(train_df),
        "holdout_rows": len(test_df),
        "total_rows": len(df),
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
    model_comparison_file = os.path.join(artifacts_dir, "model_comparison.json")
    with open(model_comparison_file, "w", encoding="utf-8") as f:
        json.dump(model_comparison_payload, f, indent=2)

    # Also save standard metrics.json
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(model_comparison_payload, f, indent=2)

    # Save sku_metadata.json
    sku_metadata_file = os.path.join(artifacts_dir, "sku_metadata.json")
    with open(sku_metadata_file, "w", encoding="utf-8") as f:
        json.dump(product_metadata, f, indent=2)

    # Required output 2: artifacts/demand_model.joblib
    # Stores winning model + feature metadata package
    demand_model_bundle = {
        "model": best_model_obj,
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "scale_features": models_config[best_name]["scale"],
        "scaler": scaler,
        "product_map": PRODUCT_MAP,
        "sku_to_product": SKU_TO_PRODUCT,
        "product_to_sku": PRODUCT_TO_SKU,
        "product_metadata": product_metadata,
        "train_date_range": [train_df["date"].min().strftime("%Y-%m-%d"), train_df["date"].max().strftime("%Y-%m-%d")],
        "test_date_range": [test_df["date"].min().strftime("%Y-%m-%d"), test_df["date"].max().strftime("%Y-%m-%d")],
    }
    demand_model_file = os.path.join(artifacts_dir, "demand_model.joblib")
    joblib.dump(demand_model_bundle, demand_model_file)

    # Also mirror to best_model.joblib for redundancy
    joblib.dump(demand_model_bundle, os.path.join(artifacts_dir, "best_model.joblib"))

    print("\nTraining completed successfully!")
    print(f"Emitted: {model_comparison_file}")
    print(f"Emitted: {demand_model_file}")
    print(f"Emitted: {sku_metadata_file}")

    return model_comparison_payload

if __name__ == "__main__":
    run_train_pipeline()
