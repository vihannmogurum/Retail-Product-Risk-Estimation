"""
Deterministic Risk Engine for Inventory & Stockout Analysis
Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III

Exact deterministic specification:
1. CRITICAL:
   if current_stock < predicted_daily_demand * lead_time_days
2. WARNING / REORDER SOON:
   elif current_stock < predicted_7day_demand + safety_stock
   where safety_stock = 1.65 * rolling_std_7 * sqrt(lead_time_days)
3. HEALTHY:
   else HEALTHY

Outputs:
- predicted_daily_demand
- predicted_7day_demand (predicted_daily_demand * 7)
- safety_stock (1.65 * rolling_std_7 * sqrt(lead_time_days))
- reorder_point (predicted_7day_demand + safety_stock)
- reorder_qty (max(0, ceil(reorder_point - current_stock)))
- risk_tier (CRITICAL, WARNING / REORDER SOON, HEALTHY)
"""

import math
from typing import Dict, Any, Optional

class StockoutRiskEngine:
    def __init__(self, z_score: float = 1.65):
        self.z_score = z_score

    def evaluate_risk(
        self,
        predicted_daily_demand: float,
        current_stock: int,
        lead_time_days: int,
        rolling_std_7: float,
        unit_price: float = 50.0,
    ) -> Dict[str, Any]:
        d_daily = max(0.0, float(predicted_daily_demand))
        stock = max(0, int(current_stock))
        lt = max(1, int(lead_time_days))
        sigma_7 = max(0.1, float(rolling_std_7))

        # 1. Demand projections
        predicted_7day_demand = round(d_daily * 7.0, 2)
        lead_time_demand = round(d_daily * lt, 2)

        # 2. Safety stock = 1.65 * rolling_std_7 * sqrt(lead_time_days)
        raw_ss = 1.65 * sigma_7 * math.sqrt(lt)
        safety_stock = int(math.ceil(max(0.0, raw_ss)))

        # 3. Reorder point = predicted_7day_demand + safety_stock
        reorder_point = int(math.ceil(predicted_7day_demand + safety_stock))

        # 4. Reorder quantity = max(0, ceil(reorder_point - current_stock))
        reorder_qty = max(0, int(math.ceil(reorder_point - stock)))

        # 5. Deterministic Tiers in exact order:
        # Step 1: CRITICAL if current_stock < predicted_daily_demand * lead_time_days
        if stock < lead_time_demand:
            risk_tier = "CRITICAL"
            action_title = "Critical Stockout Imminent"
            rationale = (
                f"Current stock ({stock} units) is less than lead time demand ({lead_time_demand:.1f} units). "
                f"Stockout will occur before replenishment arrives in {lt} days."
            )
        # Step 2: WARNING / REORDER SOON if current_stock < predicted_7day_demand + safety_stock
        elif stock < (predicted_7day_demand + safety_stock):
            risk_tier = "WARNING / REORDER SOON"
            action_title = "Reorder Soon"
            rationale = (
                f"Current stock ({stock} units) has fallen below Reorder Point ({reorder_point} units = "
                f"{predicted_7day_demand:.1f} 7-day demand + {safety_stock} safety stock). "
                f"Place order for {reorder_qty} units now."
            )
        # Step 3: HEALTHY
        else:
            risk_tier = "HEALTHY"
            action_title = "Healthy Stock Buffer"
            rationale = (
                f"Current stock ({stock} units) safely exceeds Reorder Point ({reorder_point} units). "
                f"Buffer is adequate for projected demand."
            )

        runway_days = round(stock / max(d_daily, 0.01), 1)

        return {
            "predicted_daily_demand": round(d_daily, 2),
            "predicted_7day_demand": predicted_7day_demand,
            "lead_time_demand": lead_time_demand,
            "current_stock": stock,
            "current_inventory": stock, # alias for backward compatibility
            "lead_time_days": lt,
            "rolling_std_7": round(sigma_7, 2),
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "reorder_qty": reorder_qty,
            "recommended_reorder_qty": reorder_qty, # alias
            "runway_days": runway_days,
            "risk_tier": risk_tier,
            "action_title": action_title,
            "rationale": rationale,
        }
