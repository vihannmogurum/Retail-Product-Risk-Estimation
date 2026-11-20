"""
Test FastAPI Endpoints
Tests:
- GET /health and GET /api/health
- GET /api/metrics
- GET /api/products and /api/skus
- POST /api/predict
- POST /api/batch-predict
- GET /api/export/batch-template
"""

import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoints():
    res1 = client.get("/health")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "healthy"
    assert data1["model_loaded"] is True

    res2 = client.get("/api/health")
    assert res2.status_code == 200
    assert res2.json()["status"] == "healthy"

def test_metrics_endpoint():
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "best_model" in data
    models = data["models"]
    assert "Ridge Regression" in models
    assert "Random Forest" in models
    assert "Gradient Boosting" in models

def test_products_endpoint():
    res = client.get("/api/products")
    assert res.status_code == 200
    products = res.json()
    assert len(products) >= 15
    assert any(p["product_id"] == "PRD-01" for p in products)

def test_predict_endpoint():
    payload = {
        "product_id": "PRD-01",
        "date": "2025-02-15",
        "current_stock": 30,
        "lead_time_days": 7,
        "demand_multiplier": 1.0,
        "price": 289.99,
        "competitor_price": 295.00,
        "promotion": 0,
        "holiday": 0
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_daily_demand" in data
    assert "predicted_7day_demand" in data
    assert "safety_stock" in data
    assert "reorder_point" in data
    assert "reorder_qty" in data
    assert data["risk_tier"] in ["CRITICAL", "WARNING / REORDER SOON", "HEALTHY"]

def test_batch_predict_endpoint_json():
    payload = [
        {"product_id": "PRD-01", "current_stock": 20, "lead_time_days": 7},
        {"product_id": "PRD-05", "current_stock": 250, "lead_time_days": 3},
    ]
    res = client.post("/api/batch-predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "results" in data
    assert data["summary"]["total_items"] == 2
    assert "critical_count" in data["summary"]
    assert "warning_count" in data["summary"]
    assert "healthy_count" in data["summary"]

def test_batch_predict_multipart_csv():
    csv_data = (
        "product_id,date,current_stock,lead_time_days,demand_multiplier,price,competitor_price,promotion,holiday\n"
        "PRD-01,2025-02-01,15,7,1.0,289.99,295.00,0,0\n"
        "PRD-02,2025-02-01,200,5,1.0,139.99,145.00,0,0\n"
    )
    files = {"file": ("test_batch.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    res = client.post("/api/batch-predict", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["total_items"] == 2

def test_batch_template_export():
    res = client.get("/api/export/batch-template")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "product_id,date,current_stock" in res.text
