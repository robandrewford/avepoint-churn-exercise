import pytest
import pandas as pd
import numpy as np
from datetime import date
from src.features.leakage_audit import LeakageAuditor, LeakageCheckResult

def test_leakage_auditor_clean():
    prediction_date = date(2024, 1, 1)
    auditor = LeakageAuditor(prediction_date)
    
    # Clean data
    df = pd.DataFrame({
        "logins_7d": [5, 10, 15],
        "churned_in_window": [0, 1, 0]
    })
    
    results = auditor.audit_features(df, ["logins_7d"])
    summary = auditor.get_summary()
    
    assert summary["critical_issues"] == 0
    assert summary["warnings"] == 0

def test_leakage_auditor_high_correlation():
    prediction_date = date(2024, 1, 1)
    auditor = LeakageAuditor(prediction_date)
    
    # High correlation leakage
    df = pd.DataFrame({
        "suspicious_feature": [0, 1, 0, 1, 0],
        "churned_in_window": [0, 1, 0, 1, 0]
    })
    
    results = auditor.audit_features(df, ["suspicious_feature"])
    summary = auditor.get_summary()
    
    assert summary["critical_issues"] > 0
    assert any("correlation" in r.check_type for r in results)

def test_leakage_auditor_suspicious_names():
    prediction_date = date(2024, 1, 1)
    auditor = LeakageAuditor(prediction_date)
    
    df = pd.DataFrame({
        "churn_next_month": [0.1, 0.2, 0.3],
        "will_churn": [1, 0, 1]
    })
    
    results = auditor.audit_features(df, ["churn_next_month", "will_churn"])
    summary = auditor.get_summary()
    
    assert summary["warnings"] >= 2
    assert any("suspicious_name" in r.check_type for r in results)

def test_leakage_auditor_raw_dates():
    prediction_date = date(2024, 1, 1)
    auditor = LeakageAuditor(prediction_date)
    
    df = pd.DataFrame({
        "signup_date": ["2023-01-01", "2023-02-01"],
        "churn_at": ["2024-01-15", pd.NaT]
    })
    
    results = auditor.audit_features(df, ["signup_date", "churn_at"])
    summary = auditor.get_summary()
    
    assert summary["warnings"] >= 2
    assert any("raw_date" in r.check_type for r in results)

def test_leakage_auditor_perfect_separation():
    prediction_date = date(2024, 1, 1)
    auditor = LeakageAuditor(prediction_date)
    
    # Perfect separation
    df = pd.DataFrame({
        "leak": [10, 20, 30, 1, 2, 3],
        "target": [1, 1, 1, 0, 0, 0]
    })
    
    results = auditor.audit_features(df, ["leak"], target_column="target")
    summary = auditor.get_summary()
    
    assert summary["critical_issues"] > 0
    assert any("perfect_separation" in r.check_type for r in results)
