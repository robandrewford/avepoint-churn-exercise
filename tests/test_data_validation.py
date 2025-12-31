import pytest
import pandas as pd
from pydantic import BaseModel, Field, validator
from datetime import date
from typing import Optional
import duckdb
from pathlib import Path

# Pydantic Models for Data Validation
class CustomerBronze(BaseModel):
    customer_id: str
    signup_date: date
    monthly_charges: float = Field(ge=0)
    contract_type: str
    churn_label: bool

class CustomerSilver(BaseModel):
    customer_id: str
    cohort: str
    ltv_tier: str
    tenure_days: int = Field(ge=0)
    estimated_ltv: float = Field(ge=0)

class Customer360Gold(BaseModel):
    customer_id: str
    prediction_date: date
    logins_30d: float = Field(ge=0)
    churned_in_window: bool

def test_pydantic_validation_success():
    # Bronze validation
    data = {
        "customer_id": "C1",
        "signup_date": date(2023, 1, 1),
        "monthly_charges": 100.0,
        "contract_type": "Month-to-month",
        "churn_label": False
    }
    CustomerBronze(**data)

def test_pydantic_validation_failure():
    # Bronze validation with invalid data
    data = {
        "customer_id": "C1",
        "signup_date": date(2023, 1, 1),
        "monthly_charges": -50.0, # Invalid: negative charge
        "contract_type": "Month-to-month",
        "churn_label": False
    }
    with pytest.raises(Exception):
        CustomerBronze(**data)

def test_duckdb_schema_validation():
    # Check if we can validate a sample from the database
    project_root = Path(__file__).parent.parent
    db_path = project_root / "outputs" / "churn_lakehouse.duckdb"
    
    if not db_path.exists():
        pytest.skip("DuckDB database not found. Skipping integration test.")
        
    conn = duckdb.connect(str(db_path))
    
    # Validate a sample from silver.customers
    try:
        sample_df = conn.execute("SELECT customer_id, cohort, ltv_tier, tenure_days, estimated_ltv FROM silver.customers LIMIT 10").df()
        for _, row in sample_df.iterrows():
            CustomerSilver(**row.to_dict())
    finally:
        conn.close()

def test_feature_table_validation():
    # Validate a sample from gold.customer_360
    project_root = Path(__file__).parent.parent
    db_path = project_root / "outputs" / "churn_lakehouse.duckdb"
    
    if not db_path.exists():
        pytest.skip("DuckDB database not found. Skipping integration test.")
        
    conn = duckdb.connect(str(db_path))
    
    try:
        # Check if table exists
        exists = conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'gold' AND table_name = 'customer_360'").fetchone()[0]
        if exists == 0:
            pytest.skip("gold.customer_360 table not found.")
            
        sample_df = conn.execute("SELECT customer_id, prediction_date, logins_30d, churned_in_window FROM gold.customer_360 LIMIT 10").df()
        for _, row in sample_df.iterrows():
            Customer360Gold(**row.to_dict())
    finally:
        conn.close()
