"""
Exploratory Data Analysis and Visual Artifact Generator
Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III
Generates:
- artifacts/eda_summary.json
- artifacts/eda_sales_trend.png
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_eda_artifacts(
    data_path: str = "data/retail_sales.csv",
    artifacts_dir: str = "artifacts"
):
    os.makedirs(artifacts_dir, exist_ok=True)
    if not os.path.exists(data_path):
        from data.generate_data import generate_retail_sales_dataset
        generate_retail_sales_dataset(output_path=data_path)

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # 1. Compute summary statistics
    category_summary = df.groupby("category").agg(
        total_units_sold=("units_sold", "sum"),
        mean_daily_units=("units_sold", "mean"),
        mean_price=("price", "mean"),
        stockout_occurrences=("stockout_flag", "sum"),
    ).round(2).to_dict(orient="index")

    product_summary = df.groupby("product_id").agg(
        product_name=("product_name", "first"),
        category=("category", "first"),
        mean_demand=("units_sold", "mean"),
        std_demand=("units_sold", "std"),
        total_sold=("units_sold", "sum"),
    ).round(2).to_dict(orient="index")

    overall_summary = {
        "dataset_rows": len(df),
        "product_count": int(df["product_id"].nunique()),
        "date_start": df["date"].min().strftime("%Y-%m-%d"),
        "date_end": df["date"].max().strftime("%Y-%m-%d"),
        "total_units_sold": int(df["units_sold"].sum()),
        "average_daily_demand": round(float(df["units_sold"].mean()), 2),
        "overall_stockout_rate_pct": round(float(df["stockout_flag"].mean() * 100), 2),
        "category_metrics": category_summary,
        "product_metrics": product_summary,
    }

    eda_json_path = os.path.join(artifacts_dir, "eda_summary.json")
    with open(eda_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    # 2. Render plot: Daily sales trend and stockout rate
    daily_agg = df.groupby("date").agg(
        total_sales=("units_sold", "sum"),
        stockouts=("stockout_flag", "sum")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    ax1.plot(daily_agg["date"], daily_agg["total_sales"], color="#1D4ED8", linewidth=1.2, label="Daily Total Sales (Units)")
    ax1.set_ylabel("Total Units Sold Across Catalog", color="#0F172A", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#0F172A")
    ax1.set_title("Project 41 // Retail Multi-SKU Demand & Depletion Dynamics", fontsize=12, fontweight="bold", pad=12)

    ax2 = ax1.twinx()
    ax2.fill_between(daily_agg["date"], 0, daily_agg["stockouts"], color="#B91C1C", alpha=0.3, label="Stockout Depletions")
    ax2.set_ylabel("Stockout Events", color="#B91C1C", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#B91C1C")
    ax2.grid(False)

    fig.tight_layout()
    chart_path = os.path.join(artifacts_dir, "eda_sales_trend.png")
    fig.savefig(chart_path)
    plt.close(fig)

    print(f"Generated EDA artifacts:")
    print(f"- {eda_json_path}")
    print(f"- {chart_path}")
    return overall_summary

if __name__ == "__main__":
    generate_eda_artifacts()
