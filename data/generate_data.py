"""
Realistic Multi-Product Retail Sales Dataset Generator
Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III

Generates 5,000+ daily observations across 16 distinct product_id values (with sku_id alias),
incorporating realistic price elasticity, competitor pricing, calendar seasonality,
promotional campaigns, holidays, and inventory depletion.
Saves to data/retail_sales.csv.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PRODUCTS = [
    {
        "product_id": "PRD-01",
        "sku_id": "SKU-E101",
        "product_name": "4K Ultra-HD Smart Monitor 27-inch",
        "category": "Electronics",
        "base_price": 299.99,
        "base_daily_demand": 24.0,
        "lead_time_days": 7,
        "price_elasticity": -1.6,
        "weekend_factor": 1.25,
        "promo_lift": 0.40,
    },
    {
        "product_id": "PRD-02",
        "sku_id": "SKU-E102",
        "product_name": "Noise-Cancelling Wireless Headphones",
        "category": "Electronics",
        "base_price": 149.99,
        "base_daily_demand": 42.0,
        "lead_time_days": 5,
        "price_elasticity": -1.8,
        "weekend_factor": 1.30,
        "promo_lift": 0.45,
    },
    {
        "product_id": "PRD-03",
        "sku_id": "SKU-E103",
        "product_name": "Mechanical RGB Gaming Keyboard",
        "category": "Electronics",
        "base_price": 89.99,
        "base_daily_demand": 35.0,
        "lead_time_days": 6,
        "price_elasticity": -1.5,
        "weekend_factor": 1.35,
        "promo_lift": 0.40,
    },
    {
        "product_id": "PRD-04",
        "sku_id": "SKU-E104",
        "product_name": "Smart Home Mesh Wi-Fi 6 Router",
        "category": "Electronics",
        "base_price": 179.99,
        "base_daily_demand": 28.0,
        "lead_time_days": 8,
        "price_elasticity": -1.4,
        "weekend_factor": 1.15,
        "promo_lift": 0.35,
    },
    {
        "product_id": "PRD-05",
        "sku_id": "SKU-G201",
        "product_name": "Organic Whole Bean Coffee 1kg",
        "category": "Groceries",
        "base_price": 18.50,
        "base_daily_demand": 110.0,
        "lead_time_days": 3,
        "price_elasticity": -1.1,
        "weekend_factor": 1.15,
        "promo_lift": 0.30,
    },
    {
        "product_id": "PRD-06",
        "sku_id": "SKU-G202",
        "product_name": "Extra Virgin Olive Oil 750ml",
        "category": "Groceries",
        "base_price": 14.25,
        "base_daily_demand": 85.0,
        "lead_time_days": 4,
        "price_elasticity": -1.2,
        "weekend_factor": 1.20,
        "promo_lift": 0.35,
    },
    {
        "product_id": "PRD-07",
        "sku_id": "SKU-G203",
        "product_name": "Artisanal Sourdough Bread Loaf",
        "category": "Groceries",
        "base_price": 6.50,
        "base_daily_demand": 140.0,
        "lead_time_days": 2,
        "price_elasticity": -1.0,
        "weekend_factor": 1.35,
        "promo_lift": 0.25,
    },
    {
        "product_id": "PRD-08",
        "sku_id": "SKU-G204",
        "product_name": "Pure Clover Raw Honey 500g",
        "category": "Groceries",
        "base_price": 9.75,
        "base_daily_demand": 65.0,
        "lead_time_days": 4,
        "price_elasticity": -1.2,
        "weekend_factor": 1.10,
        "promo_lift": 0.30,
    },
    {
        "product_id": "PRD-09",
        "sku_id": "SKU-A301",
        "product_name": "Breathable Performance Running Shoes",
        "category": "Apparel",
        "base_price": 89.00,
        "base_daily_demand": 38.0,
        "lead_time_days": 10,
        "price_elasticity": -1.7,
        "weekend_factor": 1.40,
        "promo_lift": 0.50,
    },
    {
        "product_id": "PRD-10",
        "sku_id": "SKU-A302",
        "product_name": "Weatherproof All-Season Outdoor Jacket",
        "category": "Apparel",
        "base_price": 119.50,
        "base_daily_demand": 22.0,
        "lead_time_days": 12,
        "price_elasticity": -1.5,
        "weekend_factor": 1.35,
        "promo_lift": 0.45,
    },
    {
        "product_id": "PRD-11",
        "sku_id": "SKU-A303",
        "product_name": "Merino Wool Thermal Crew Socks (3-Pack)",
        "category": "Apparel",
        "base_price": 24.99,
        "base_daily_demand": 50.0,
        "lead_time_days": 7,
        "price_elasticity": -1.3,
        "weekend_factor": 1.25,
        "promo_lift": 0.35,
    },
    {
        "product_id": "PRD-12",
        "sku_id": "SKU-A304",
        "product_name": "Quick-Dry Moisture-Wicking Training Tee",
        "category": "Apparel",
        "base_price": 29.50,
        "base_daily_demand": 58.0,
        "lead_time_days": 8,
        "price_elasticity": -1.6,
        "weekend_factor": 1.30,
        "promo_lift": 0.40,
    },
    {
        "product_id": "PRD-13",
        "sku_id": "SKU-H401",
        "product_name": "Cordless Brushless Drill Kit 20V",
        "category": "Home & Garden",
        "base_price": 129.99,
        "base_daily_demand": 30.0,
        "lead_time_days": 8,
        "price_elasticity": -1.4,
        "weekend_factor": 1.45,
        "promo_lift": 0.38,
    },
    {
        "product_id": "PRD-14",
        "sku_id": "SKU-H402",
        "product_name": "True HEPA Air Purifier Filter Replacement",
        "category": "Home & Garden",
        "base_price": 34.95,
        "base_daily_demand": 55.0,
        "lead_time_days": 6,
        "price_elasticity": -1.0,
        "weekend_factor": 1.10,
        "promo_lift": 0.25,
    },
    {
        "product_id": "PRD-15",
        "sku_id": "SKU-H403",
        "product_name": "Smart Programmable Thermostat",
        "category": "Home & Garden",
        "base_price": 159.00,
        "base_daily_demand": 25.0,
        "lead_time_days": 9,
        "price_elasticity": -1.3,
        "weekend_factor": 1.20,
        "promo_lift": 0.35,
    },
    {
        "product_id": "PRD-16",
        "sku_id": "SKU-H404",
        "product_name": "Heavy-Duty Steel Garden Pruner Shears",
        "category": "Home & Garden",
        "base_price": 27.50,
        "base_daily_demand": 40.0,
        "lead_time_days": 5,
        "price_elasticity": -1.4,
        "weekend_factor": 1.50,
        "promo_lift": 0.30,
    },
]

def generate_retail_sales_dataset(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    output_path: str = "data/retail_sales.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates 365 days * 16 products = 5,840 records of retail sales data.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    holidays_set = {
        "2024-01-01", "2024-02-14", "2024-05-27", "2024-07-04",
        "2024-09-02", "2024-10-31", "2024-11-28", "2024-11-29",
        "2024-12-02", "2024-12-24", "2024-12-25", "2024-12-31"
    }

    all_rows = []

    for prod in PRODUCTS:
        lead_time = prod["lead_time_days"]
        base_demand = prod["base_daily_demand"]
        reorder_point = int(base_demand * lead_time * 1.5)
        order_batch_size = int(base_demand * lead_time * 2.2)

        current_inventory = order_batch_size * 2
        pending_orders = []

        for day_idx, dt in enumerate(dates):
            date_str = dt.strftime("%Y-%m-%d")
            day_of_week = dt.dayofweek
            day_of_year = dt.dayofyear
            month = dt.month

            # Deliver pending replenishments
            new_pending = []
            for arr_day, qty in pending_orders:
                if arr_day <= day_idx:
                    current_inventory += qty
                else:
                    new_pending.append((arr_day, qty))
            pending_orders = new_pending

            is_holiday = 1 if date_str in holidays_set else 0
            is_weekend = 1 if day_of_week in [5, 6] else 0

            # Promotions schedule
            promo_prob = 0.25 if month in [11, 12] else (0.18 if is_weekend else 0.10)
            is_promo = 1 if np.random.rand() < promo_prob else 0

            # Price calculations
            base_price = prod["base_price"]
            if is_promo:
                discount = np.random.uniform(0.10, 0.22)
                price = round(base_price * (1.0 - discount), 2)
            else:
                price = round(base_price * np.random.uniform(0.98, 1.02), 2)

            competitor_price = round(base_price * np.random.uniform(0.93, 1.07), 2)

            # Demand factors
            annual_season = 1.0 + 0.22 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
            if month in [11, 12]:
                annual_season += 0.35

            dow_multiplier = prod["weekend_factor"] if is_weekend else 0.95
            price_ratio = price / base_price
            price_effect = price_ratio ** prod["price_elasticity"]
            comp_ratio = competitor_price / price
            comp_effect = 1.0 + 0.18 * (comp_ratio - 1.0)
            promo_effect = (1.0 + prod["promo_lift"]) if is_promo else 1.0
            holiday_effect = 1.45 if is_holiday else 1.0

            expected_demand = (
                base_demand
                * annual_season
                * dow_multiplier
                * price_effect
                * comp_effect
                * promo_effect
                * holiday_effect
            )

            noise = np.random.normal(loc=0.0, scale=0.12 * expected_demand)
            true_demand = max(1, int(round(expected_demand + noise)))

            stockout_flag = 1 if current_inventory == 0 else 0
            units_sold = min(true_demand, current_inventory)
            current_inventory = max(0, current_inventory - units_sold)

            # Reorder trigger
            if current_inventory <= reorder_point and len(pending_orders) == 0:
                arrival_day = day_idx + lead_time + np.random.randint(-1, 2)
                arrival_day = max(day_idx + 1, arrival_day)
                pending_orders.append((arrival_day, order_batch_size))

            all_rows.append({
                "date": date_str,
                "product_id": prod["product_id"],
                "sku_id": prod["sku_id"],
                "product_name": prod["product_name"],
                "category": prod["category"],
                "base_price": base_price,
                "price": price,
                "competitor_price": competitor_price,
                "promotion": is_promo,
                "holiday": is_holiday,
                "inventory_level": current_inventory,
                "lead_time_days": lead_time,
                "stockout_flag": stockout_flag,
                "units_sold": units_sold,
            })

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "product_id"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows across {len(PRODUCTS)} products at '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_retail_sales_dataset()
