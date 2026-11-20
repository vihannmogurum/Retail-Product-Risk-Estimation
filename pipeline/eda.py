"""
Exploratory Data Analysis and Visual Artifact Generator
Project 41: Retail Product Demand Estimation and Stockout Risk Analysis-III
Generates:
- artifacts/eda_summary.json
- artifacts/eda_sales_trend.png
Using Kaggle FMCG Dataset (FMCG_2022_2024.csv).
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np

try:
    from pipeline.features import aggregate_fmcg_daily
except ImportError:
    from features import aggregate_fmcg_daily

def generate_eda_artifacts(
    data_path: str = "data/FMCG_2022_2024.csv",
    artifacts_dir: str = "artifacts"
):
    os.makedirs(artifacts_dir, exist_ok=True)
    if not os.path.exists(data_path):
        data_path = "data/retail_sales.csv"

    print(f"Reading dataset from {data_path} for EDA...")
    raw_df = pd.read_csv(data_path)
    df = aggregate_fmcg_daily(raw_df)
    df["date"] = pd.to_datetime(df["date"])

    # Category summary
    cat_col = "category" if "category" in df.columns else None
    if cat_col:
        category_summary = df.groupby(cat_col).agg(
            total_units_sold=("units_sold", "sum"),
            mean_daily_units=("units_sold", "mean"),
            mean_price=("price_unit" if "price_unit" in df.columns else "price", "mean"),
        ).round(2).to_dict(orient="index")
    else:
        category_summary = {}

    # Product/SKU summary
    prod_summary = df.groupby("sku").agg(
        category=("category", "first") if "category" in df.columns else ("sku", "first"),
        mean_demand=("units_sold", "mean"),
        std_demand=("units_sold", "std"),
        total_sold=("units_sold", "sum"),
        avg_price=("price_unit" if "price_unit" in df.columns else "price", "mean"),
        lead_time_days=("delivery_days" if "delivery_days" in df.columns else "lead_time_days", "mean"),
    ).round(2).to_dict(orient="index")

    overall_summary = {
        "dataset_rows_raw": len(raw_df),
        "aggregated_daily_sku_rows": len(df),
        "product_count": int(df["sku"].nunique()),
        "date_start": df["date"].min().strftime("%Y-%m-%d"),
        "date_end": df["date"].max().strftime("%Y-%m-%d"),
        "total_units_sold": int(df["units_sold"].sum()),
        "average_daily_demand": round(float(df["units_sold"].mean()), 2),
        "category_metrics": category_summary,
        "product_metrics": prod_summary,
    }

    eda_json_path = os.path.join(artifacts_dir, "eda_summary.json")
    with open(eda_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    # Render plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        daily_agg = df.groupby("date").agg(
            total_sales=("units_sold", "sum")
        ).reset_index()

        fig, ax1 = plt.subplots(figsize=(11, 4.5), dpi=150)
        ax1.plot(daily_agg["date"], daily_agg["total_sales"], color="#1D4ED8", linewidth=1.2, label="Daily Total Units Sold")
        ax1.set_ylabel("Daily Total Sales (All SKUs)", color="#0F172A", fontsize=10, fontweight="bold")
        ax1.set_title("Project 41 // FMCG Daily Demand (2022-2024, 30 SKUs)", fontsize=12, fontweight="bold", pad=12)
        ax1.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        chart_path = os.path.join(artifacts_dir, "eda_sales_trend.png")
        fig.savefig(chart_path)
        plt.close(fig)
        print(f"Generated plot: {chart_path}")
    except Exception as e:
        print(f"Matplotlib plot generation skipped or failed: {e}")

    print(f"Generated EDA summary: {eda_json_path}")
    return overall_summary

if __name__ == "__main__":
    generate_eda_artifacts()
