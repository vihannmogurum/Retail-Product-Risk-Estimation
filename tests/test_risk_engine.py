"""
Test Stockout Risk Engine
Verifies:
- CRITICAL condition when current_stock < predicted_daily_demand * lead_time_days
- WARNING / REORDER SOON when current_stock < predicted_7day_demand + safety_stock
- HEALTHY condition when current_stock >= reorder_point
- Exact safety stock and reorder quantity formulas
"""

import pytest
import math
from backend.risk_engine import StockoutRiskEngine

@pytest.fixture
def engine():
    return StockoutRiskEngine()

def test_critical_risk_tier(engine):
    # Daily demand = 10, lead time = 5 days -> Lead time demand = 50
    # Stock = 40 (< 50) -> Must be CRITICAL
    res = engine.evaluate_risk(
        predicted_daily_demand=10.0,
        current_stock=40,
        lead_time_days=5,
        rolling_std_7=3.0,
    )
    assert res["risk_tier"] == "CRITICAL"
    assert res["lead_time_demand"] == 50.0
    assert res["reorder_qty"] > 0

def test_warning_risk_tier(engine):
    # Daily demand = 10, lead time = 5 days -> Lead time demand = 50
    # 7-day demand = 70
    # rolling_std_7 = 4.0, sqrt(5) ~ 2.236 -> safety_stock = ceil(1.65 * 4 * 2.236) = 15
    # ROP = 70 + 15 = 85
    # Current stock = 60 (>= 50 but < 85) -> Must be WARNING / REORDER SOON
    res = engine.evaluate_risk(
        predicted_daily_demand=10.0,
        current_stock=60,
        lead_time_days=5,
        rolling_std_7=4.0,
    )
    assert res["risk_tier"] == "WARNING / REORDER SOON"
    assert res["reorder_point"] == 85
    assert res["safety_stock"] == 15
    assert res["reorder_qty"] == 25 # 85 - 60 = 25

def test_healthy_risk_tier(engine):
    # Daily demand = 10, lead time = 5, rolling_std_7 = 4.0 -> ROP = 85
    # Current stock = 100 (>= 85) -> Must be HEALTHY
    res = engine.evaluate_risk(
        predicted_daily_demand=10.0,
        current_stock=100,
        lead_time_days=5,
        rolling_std_7=4.0,
    )
    assert res["risk_tier"] == "HEALTHY"
    assert res["reorder_qty"] == 0 # No reorder needed

def test_safety_stock_calculation(engine):
    daily_d = 20.0
    lt = 9
    sigma_7 = 5.0
    # 1.65 * 5.0 * sqrt(9) = 1.65 * 15 = 24.75 -> ceil = 25
    res = engine.evaluate_risk(
        predicted_daily_demand=daily_d,
        current_stock=200,
        lead_time_days=lt,
        rolling_std_7=sigma_7,
    )
    assert res["safety_stock"] == 25
    assert res["predicted_7day_demand"] == 140.0
    assert res["reorder_point"] == 165
