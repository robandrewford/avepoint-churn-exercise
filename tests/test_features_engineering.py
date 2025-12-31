import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.features.engineering import (
    assign_cohort,
    calculate_velocity,
    calculate_trend_slope,
    FeatureEngineer
)

def test_assign_cohort():
    assert assign_cohort(15) == "new_user"
    assert assign_cohort(30) == "new_user"
    assert assign_cohort(31) == "established"
    assert assign_cohort(180) == "established"
    assert assign_cohort(181) == "mature"

def test_calculate_velocity():
    # Regular growth
    assert calculate_velocity(120, 100) == 0.2
    # Decline
    assert calculate_velocity(80, 100) == -0.2
    # No change
    assert calculate_velocity(100, 100) == 0.0
    # Zero previous (growth from zero)
    assert calculate_velocity(50, 0) == 1.0
    # Both zero
    assert calculate_velocity(0, 0) == 0.0

def test_calculate_trend_slope():
    # Increasing trend
    assert calculate_trend_slope([1, 2, 3, 4]) > 0
    # Decreasing trend
    assert calculate_trend_slope([4, 3, 2, 1]) < 0
    # Flat trend
    assert calculate_trend_slope([5, 5, 5, 5]) == 0.0
    # Single value
    assert calculate_trend_slope([10]) == 0.0
    # Empty list
    assert calculate_trend_slope([]) == 0.0

@pytest.fixture
def sample_data():
    prediction_date = date(2024, 1, 1)
    
    customers = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "signup_date": [
            date(2023, 12, 15), # New User (17 days)
            date(2023, 10, 1),  # Established (92 days)
            date(2023, 1, 1)    # Mature (365 days)
        ],
        "ltv_tier": ["smb", "mid_market", "enterprise"],
        "monthly_charges": [100.0, 500.0, 2000.0],
        "contract_type": ["Month-to-month", "One year", "Two year"],
        "churn_label": [False, False, False],
        "churn_date": [pd.NaT, pd.NaT, pd.NaT]
    })
    
    # Simple engagement: 1 login per day
    engagement = pd.DataFrame({
        "customer_id": ["C1"] * 10 + ["C2"] * 10 + ["C3"] * 10,
        "activity_date": [prediction_date - timedelta(days=i) for i in range(1, 11)] * 3,
        "login_count": [1] * 30,
        "features_used": [2] * 30,
        "session_duration_minutes": [10.0] * 30,
        "unique_features": [3] * 30
    })
    
    tickets = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "created_date": [prediction_date - timedelta(days=5), prediction_date - timedelta(days=20)],
        "sentiment_score": [0.5, -0.2],
        "escalated": [False, True]
    })
    
    return prediction_date, customers, engagement, tickets

def test_feature_engineer_build(sample_data):
    prediction_date, customers, engagement, tickets = sample_data
    engineer = FeatureEngineer(prediction_date)
    
    features = engineer.build_features(customers, engagement, tickets)
    
    assert len(features) == 3
    assert set(features["customer_id"]) == {"C1", "C2", "C3"}
    
    # Check cohorts
    assert features.loc[features["customer_id"] == "C1", "cohort"].iloc[0] == "new_user"
    assert features.loc[features["customer_id"] == "C2", "cohort"].iloc[0] == "established"
    assert features.loc[features["customer_id"] == "C3", "cohort"].iloc[0] == "mature"
    
    # Check sample weights
    assert features.loc[features["customer_id"] == "C1", "sample_weight"].iloc[0] == 1.0
    assert features.loc[features["customer_id"] == "C2", "sample_weight"].iloc[0] == 3.0
    assert features.loc[features["customer_id"] == "C3", "sample_weight"].iloc[0] == 10.0
    
    # Check some engagement features
    # C1 (new_user) 7d window
    assert features.loc[features["customer_id"] == "C1", "logins_7d"].iloc[0] == 7
    # C3 (mature) 14d window
    assert features.loc[features["customer_id"] == "C3", "logins_14d"].iloc[0] == 10 # only 10 days of data
