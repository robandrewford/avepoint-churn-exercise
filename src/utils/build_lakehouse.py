#!/usr/bin/env python3
"""
Build Lakehouse from Synthetic Data

This script:
1. Loads synthetic data from Parquet files
2. Initializes DuckDB lakehouse with medallion schema
3. Loads data into Bronze layer
4. Transforms to Silver layer
5. Builds Gold layer (Customer 360)

Usage:
    uv run python scripts/build_lakehouse.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.duckdb_lakehouse import create_lakehouse


def main():
    print("=" * 60)
    print("Building Churn Prediction Lakehouse")
    print("=" * 60)
    
    # Paths
    data_dir = Path("outputs/synthetic_data")
    db_path = "outputs/churn_lakehouse.duckdb"
    
    # Check if synthetic data exists
    if not data_dir.exists():
        print(f"\nERROR: Synthetic data not found at {data_dir}")
        print("Run: uv run python -m src.data.generate_synthetic")
        return 1
    
    # Load synthetic data
    print("\n[1/5] Loading synthetic data...")
    
    customers = pd.read_parquet(data_dir / "customers.parquet")
    daily_engagement = pd.read_parquet(data_dir / "daily_engagement.parquet")
    support_tickets = pd.read_parquet(data_dir / "support_tickets.parquet")
    
    print(f"  Customers: {len(customers):,}")
    print(f"  Daily engagement: {len(daily_engagement):,}")
    print(f"  Support tickets: {len(support_tickets):,}")
    
    # Initialize lakehouse
    print("\n[2/5] Initializing lakehouse...")
    lakehouse = create_lakehouse(db_path)
    
    # Prepare customer data for bronze layer
    print("\n[3/5] Loading Bronze layer...")
    
    # Map customer columns to bronze schema
    customers_bronze = customers.rename(columns={
        "partner": "partner",
        "dependents": "dependents", 
        "phone_service": "phone_service",
        "paperless_billing": "paperless_billing",
    })
    
    # Ensure boolean columns are correct type
    bool_cols = ["senior_citizen", "partner", "dependents", "phone_service", "paperless_billing"]
    for col in bool_cols:
        if col in customers_bronze.columns:
            customers_bronze[col] = customers_bronze[col].fillna(False).astype(bool)
    
    lakehouse.load_bronze_customers(customers_bronze)
    lakehouse.load_bronze_tickets(support_tickets)
    
    # Transform to Silver
    print("\n[4/5] Transforming to Silver layer...")
    lakehouse.transform_to_silver_customers()
    lakehouse.load_silver_daily_engagement(daily_engagement)
    lakehouse.transform_to_silver_tickets()
    
    # Build Gold layer (Customer 360)
    print("\n[5/5] Building Gold layer (Customer 360)...")
    
    # Get date range from data
    min_date = daily_engagement["activity_date"].min()
    max_date = daily_engagement["activity_date"].max()
    
    # Build Customer 360 for multiple prediction dates
    # Use weekly snapshots for the last 3 months
    prediction_dates = []
    current_date = max_date - timedelta(days=90)
    
    while current_date <= max_date - timedelta(days=30):  # Need 30 days for label
        prediction_dates.append(current_date)
        current_date += timedelta(days=7)
    
    print(f"  Building {len(prediction_dates)} prediction snapshots...")
    
    total_records = 0
    for pred_date in prediction_dates:
        count = lakehouse.build_customer_360(pred_date)
        total_records += count
    
    print(f"  Total Customer 360 records: {total_records:,}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Lakehouse Build Complete!")
    print("=" * 60)
    
    counts = lakehouse.get_table_counts()
    print("\nTable Summary:")
    for table, count in counts.items():
        print(f"  {table}: {count:,}")
    
    # Validate Gold layer
    sample = lakehouse.query("""
        SELECT 
            cohort,
            COUNT(*) as n,
            AVG(CASE WHEN churned_in_window THEN 1.0 ELSE 0.0 END) as churn_rate
        FROM gold.customer_360
        GROUP BY cohort
    """)
    
    print("\nGold Layer Validation (by cohort):")
    print(sample.to_string(index=False))
    
    lakehouse.close()
    
    print(f"\nLakehouse saved to: {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
