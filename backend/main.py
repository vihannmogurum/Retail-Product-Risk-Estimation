"""
FastAPI Backend Application
Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III

Exact Routes:
- GET /health and GET /api/health
- GET /api/metrics
- GET /api/products and GET /api/skus
- POST /api/predict
- POST /api/batch-predict and POST /api/predict/batch (multipart CSV)
- GET /api/export/batch-template
- GET /api/history
- Single-port static mount of frontend at /
"""

import os
import sys
import io
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.getcwd())

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from backend.risk_engine import StockoutRiskEngine
except ImportError:
    from risk_engine import StockoutRiskEngine

try:
    from pipeline.features import (
        FEATURE_COLUMNS,
        PRODUCT_MAP,
        SKU_TO_PRODUCT,
        PRODUCT_TO_SKU,
    )
except ImportError:
    from features import (
        FEATURE_COLUMNS,
        PRODUCT_MAP,
        SKU_TO_PRODUCT,
        PRODUCT_TO_SKU,
    )

START_TIME = time.time()

app = FastAPI(
    title="Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III",
    description="Enterprise API for demand forecasting and inventory stockout risk quantification.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime cache
MODEL_BUNDLE = None
MODEL_COMPARISON = None
PRODUCT_METADATA = None
RISK_ENGINE = StockoutRiskEngine()

def load_system_artifacts():
    global MODEL_BUNDLE, MODEL_COMPARISON, PRODUCT_METADATA
    artifacts_dir = "artifacts"
    model_path = os.path.join(artifacts_dir, "demand_model.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(artifacts_dir, "best_model.joblib")

    comp_path = os.path.join(artifacts_dir, "model_comparison.json")
    if not os.path.exists(comp_path):
        comp_path = os.path.join(artifacts_dir, "metrics.json")

    sku_path = os.path.join(artifacts_dir, "sku_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(comp_path):
        print("Artifacts missing. Running pipeline/train.py...")
        from pipeline.train import run_train_pipeline
        run_train_pipeline()

    if os.path.exists(model_path):
        MODEL_BUNDLE = joblib.load(model_path)
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            MODEL_COMPARISON = json.load(f)
    if os.path.exists(sku_path):
        with open(sku_path, "r", encoding="utf-8") as f:
            PRODUCT_METADATA = json.load(f)

# Auto-load on import
load_system_artifacts()

class PredictionRequest(BaseModel):
    product_id: Optional[str] = Field(None, example="PRD-01")
    sku_id: Optional[str] = Field(None, example="SKU-E101")
    date: Optional[str] = Field(None, example="2025-01-15")
    current_stock: Optional[int] = Field(None, example=85)
    inventory_level: Optional[int] = Field(None, example=85) # backward-compatible alias
    lead_time_days: Optional[int] = Field(None, example=7)
    demand_multiplier: Optional[float] = Field(1.0, example=1.0)
    price: Optional[float] = Field(None, example=289.99)
    competitor_price: Optional[float] = Field(None, example=295.00)
    promotion: Optional[int] = Field(0, example=1)
    holiday: Optional[int] = Field(0, example=0)
    # Optional lag overrides
    lag_1: Optional[float] = None
    lag_7: Optional[float] = None
    lag_14: Optional[float] = None
    rolling_mean_7: Optional[float] = None
    rolling_std_7: Optional[float] = None
    rolling_mean_30: Optional[float] = None

def resolve_product(pid: Optional[str], sku: Optional[str]) -> str:
    if pid and pid in PRODUCT_MAP:
        return pid
    if sku and sku in SKU_TO_PRODUCT:
        return SKU_TO_PRODUCT[sku]
    if pid and pid in SKU_TO_PRODUCT:
        return SKU_TO_PRODUCT[pid]
    return "PRD-01"

def predict_single_row(payload: PredictionRequest) -> Dict[str, Any]:
    global MODEL_BUNDLE, PRODUCT_METADATA
    if MODEL_BUNDLE is None:
        load_system_artifacts()

    pid = resolve_product(payload.product_id, payload.sku_id)
    meta = (PRODUCT_METADATA or {}).get(pid, {})
    base_price = meta.get("base_price", 50.0)
    price = float(payload.price if payload.price is not None else base_price)
    comp_price = float(payload.competitor_price if payload.competitor_price is not None else price)
    lead_time = int(payload.lead_time_days or meta.get("lead_time_days", 7))

    # Resolve current stock
    if payload.current_stock is not None:
        stock = int(payload.current_stock)
    elif payload.inventory_level is not None:
        stock = int(payload.inventory_level)
    else:
        stock = int(meta.get("recent_inventory", 100))

    rolling_std_7 = float(payload.rolling_std_7 if payload.rolling_std_7 is not None else meta.get("demand_std", 6.0))
    mean_demand = float(meta.get("historical_mean_demand", 35.0))
    multiplier = float(payload.demand_multiplier or 1.0)
    multiplier = max(0.01, min(multiplier, 10.0))

    # Date resolution
    req_date = payload.date or datetime.now().strftime("%Y-%m-%d")
    try:
        dt = pd.to_datetime(req_date)
    except Exception:
        dt = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))

    day_of_week = dt.dayofweek
    month = dt.month
    is_weekend = 1 if day_of_week in [5, 6] else 0
    price_ratio = price / max(comp_price, 0.01)

    lag_1 = float(payload.lag_1 if payload.lag_1 is not None else mean_demand)
    lag_7 = float(payload.lag_7 if payload.lag_7 is not None else mean_demand)
    lag_14 = float(payload.lag_14 if payload.lag_14 is not None else mean_demand)
    roll_m7 = float(payload.rolling_mean_7 if payload.rolling_mean_7 is not None else mean_demand)
    roll_s7 = rolling_std_7
    roll_m30 = float(payload.rolling_mean_30 if payload.rolling_mean_30 is not None else mean_demand)

    features = {
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "rolling_mean_7": roll_m7,
        "rolling_std_7": roll_s7,
        "rolling_mean_30": roll_m30,
        "price": price,
        "competitor_price": comp_price,
        "price_ratio": price_ratio,
        "promotion": int(payload.promotion or 0),
        "holiday": int(payload.holiday or 0),
        "product_encoded": PRODUCT_MAP.get(pid, 0),
    }

    feature_df = pd.DataFrame([features])[FEATURE_COLUMNS]

    model = MODEL_BUNDLE["model"]
    if MODEL_BUNDLE.get("scale_features", False) and MODEL_BUNDLE.get("scaler"):
        X_eval = MODEL_BUNDLE["scaler"].transform(feature_df)
    else:
        X_eval = feature_df

    raw_pred = float(model.predict(X_eval)[0])
    predicted_daily_demand = max(0.0, round(raw_pred * multiplier, 2))

    # Evaluate deterministic risk
    risk_output = RISK_ENGINE.evaluate_risk(
        predicted_daily_demand=predicted_daily_demand,
        current_stock=stock,
        lead_time_days=lead_time,
        rolling_std_7=roll_s7,
        unit_price=price,
    )

    return {
        "product_id": pid,
        "sku_id": meta.get("sku_id", PRODUCT_TO_SKU.get(pid, "SKU-001")),
        "product_name": meta.get("product_name", pid),
        "category": meta.get("category", "General"),
        "date": req_date,
        "demand_multiplier": multiplier,
        "price": price,
        "competitor_price": comp_price,
        "promotion": features["promotion"],
        "holiday": features["holiday"],
        "model_used": MODEL_BUNDLE.get("best_model_name", "Winning Regressor"),
        **risk_output,
    }

# Required endpoint 1: GET /health and alias GET /api/health
@app.get("/health")
@app.get("/api/health")
def get_health():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        load_system_artifacts()

    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_loaded": MODEL_BUNDLE is not None,
        "best_model_name": MODEL_BUNDLE.get("best_model_name", "N/A") if MODEL_BUNDLE else "None",
        "product_count": len(PRODUCT_MAP),
        "version": "3.0.0",
    }

@app.get("/api/metrics")
def get_metrics():
    global MODEL_COMPARISON
    if MODEL_COMPARISON is None:
        load_system_artifacts()
    if MODEL_COMPARISON is None:
        raise HTTPException(status_code=500, detail="Metrics not available")
    return MODEL_COMPARISON

@app.get("/api/products")
@app.get("/api/skus")
def get_products():
    global PRODUCT_METADATA
    if PRODUCT_METADATA is None:
        load_system_artifacts()
    return list(PRODUCT_METADATA.values()) if PRODUCT_METADATA else []

@app.post("/api/predict")
def predict_scenario(payload: PredictionRequest):
    return predict_single_row(payload)

# Required endpoint 2: POST /api/batch-predict and alias POST /api/predict/batch
@app.post("/api/batch-predict")
@app.post("/api/predict/batch")
async def batch_predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    items = []
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if uploaded_file and hasattr(uploaded_file, "read"):
            content = await uploaded_file.read()
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

            for _, row in df.iterrows():
                item = PredictionRequest(
                    product_id=str(row.get("product_id")) if pd.notna(row.get("product_id")) else None,
                    sku_id=str(row.get("sku_id")) if pd.notna(row.get("sku_id")) else None,
                    date=str(row.get("date")) if pd.notna(row.get("date")) else None,
                    current_stock=int(row.get("current_stock")) if pd.notna(row.get("current_stock")) else (
                        int(row.get("inventory_level")) if pd.notna(row.get("inventory_level")) else 100
                    ),
                    lead_time_days=int(row.get("lead_time_days")) if pd.notna(row.get("lead_time_days")) else None,
                    demand_multiplier=float(row.get("demand_multiplier")) if pd.notna(row.get("demand_multiplier")) else 1.0,
                    price=float(row.get("price")) if pd.notna(row.get("price")) else None,
                    competitor_price=float(row.get("competitor_price")) if pd.notna(row.get("competitor_price")) else None,
                    promotion=int(row.get("promotion")) if pd.notna(row.get("promotion")) else 0,
                    holiday=int(row.get("holiday")) if pd.notna(row.get("holiday")) else 0,
                )
                items.append(item)
    elif "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, list):
                for obj in body:
                    items.append(PredictionRequest(**obj))
            elif isinstance(body, dict):
                items.append(PredictionRequest(**body))
        except Exception:
            pass

    if not items:
        for pid in PRODUCT_MAP.keys():
            items.append(PredictionRequest(product_id=pid, current_stock=60, promotion=0, holiday=0))

    results = []
    critical_count = 0
    warning_count = 0
    healthy_count = 0
    total_reorder_qty = 0
    total_reorder_cost = 0.0

    for itm in items:
        res = predict_single_row(itm)
        results.append(res)
        tier = res["risk_tier"]
        if tier == "CRITICAL":
            critical_count += 1
        elif "WARNING" in tier:
            warning_count += 1
        else:
            healthy_count += 1

        qty = res["reorder_qty"]
        total_reorder_qty += qty
        total_reorder_cost += qty * res["price"]

    summary = {
        "total_items": len(results),
        "critical_count": critical_count,
        "warning_count": warning_count,
        "healthy_count": healthy_count,
        "total_recommended_reorder_units": total_reorder_qty,
        "total_estimated_reorder_cost": round(total_reorder_cost, 2),
    }

    return {"summary": summary, "results": results}

@app.get("/api/export/batch-template")
def export_batch_template():
    csv_content = (
        "product_id,date,current_stock,lead_time_days,demand_multiplier,price,competitor_price,promotion,holiday\n"
        "PRD-01,2025-02-01,45,7,1.0,289.99,295.00,0,0\n"
        "PRD-02,2025-02-01,25,5,1.2,139.99,145.00,1,0\n"
        "PRD-05,2025-02-01,110,3,1.0,17.50,18.00,0,0\n"
        "PRD-09,2025-02-01,15,10,1.0,85.00,88.00,0,0\n"
        "PRD-13,2025-02-01,20,8,1.0,124.99,129.99,0,0\n"
    )
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=retail_risk_batch_template.csv"}
    )

@app.get("/api/history")
def get_sample_history(product_id: Optional[str] = "PRD-01"):
    data_path = "data/retail_sales.csv"
    if not os.path.exists(data_path):
        return []

    df = pd.read_csv(data_path)
    pid = resolve_product(product_id, None)
    sub = df[df["product_id"] == pid].tail(30)

    records = []
    for _, r in sub.iterrows():
        records.append({
            "date": str(r["date"]),
            "units_sold": int(r["units_sold"]),
            "inventory_level": int(r["inventory_level"]),
            "price": float(r["price"]),
            "stockout_flag": int(r["stockout_flag"]),
        })
    return records

# Mount frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if not os.path.exists(frontend_dir):
    frontend_dir = "frontend"

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
