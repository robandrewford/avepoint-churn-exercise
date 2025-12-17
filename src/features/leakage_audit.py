"""
Temporal Leakage Audit Module

Implements formal leakage detection for churn prediction features.

Leakage Types:
- Direct: Feature uses data from after prediction point
- Indirect: Feature derived from future data
- Target: Feature correlated with label by definition
- Aggregation: Window extends past prediction point
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LeakageCheckResult:
    """Result of a leakage check."""
    feature: str
    check_type: str
    passed: bool
    message: str
    severity: str  # "critical", "warning", "info"


class LeakageAuditor:
    """
    Auditor for detecting temporal leakage in features.
    
    Usage:
        auditor = LeakageAuditor(prediction_date=date(2024, 12, 1))
        results = auditor.audit_features(df, feature_columns)
    """
    
    def __init__(self, prediction_date: date):
        """
        Initialize auditor.
        
        Args:
            prediction_date: Point-in-time date for feature calculation
        """
        self.prediction_date = prediction_date
        self.results: list[LeakageCheckResult] = []
    
    def audit_features(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str = "churned_in_window",
    ) -> list[LeakageCheckResult]:
        """
        Run all leakage checks on features.
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            target_column: Name of target column
            
        Returns:
            List of LeakageCheckResult
        """
        self.results = []
        
        # Check each feature
        for feature in feature_columns:
            if feature not in df.columns:
                continue
            
            # Run checks
            self._check_target_leakage(df, feature, target_column)
            self._check_future_correlation(df, feature)
            self._check_suspicious_names(feature)
            self._check_perfect_prediction(df, feature, target_column)
        
        # Check for date-based leakage
        self._check_date_columns(df, feature_columns)
        
        return self.results
    
    def _check_target_leakage(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str,
    ) -> None:
        """Check if feature is suspiciously correlated with target."""
        if target not in df.columns:
            return
        
        # Calculate correlation
        if df[feature].dtype in [np.float64, np.int64, float, int]:
            corr = df[feature].corr(df[target].astype(float))
            
            if abs(corr) > 0.95:
                self.results.append(LeakageCheckResult(
                    feature=feature,
                    check_type="target_correlation",
                    passed=False,
                    message=f"Extremely high correlation with target ({corr:.3f}). Likely leakage.",
                    severity="critical",
                ))
            elif abs(corr) > 0.8:
                self.results.append(LeakageCheckResult(
                    feature=feature,
                    check_type="target_correlation",
                    passed=False,
                    message=f"High correlation with target ({corr:.3f}). Review for leakage.",
                    severity="warning",
                ))
    
    def _check_future_correlation(
        self,
        df: pd.DataFrame,
        feature: str,
    ) -> None:
        """Check if feature correlates with future dates."""
        # Look for patterns suggesting future data
        if "churn_date" in df.columns:
            # Feature shouldn't know about churn_date
            churn_dates = pd.to_datetime(df["churn_date"])
            future_churns = churn_dates > pd.Timestamp(self.prediction_date)
            
            if df[feature].dtype in [np.float64, np.int64, float, int]:
                # Check if feature differs significantly for future churners
                if future_churns.any():
                    future_mean = df.loc[future_churns, feature].mean()
                    past_mean = df.loc[~future_churns, feature].mean()
                    
                    if abs(future_mean - past_mean) > 2 * df[feature].std():
                        self.results.append(LeakageCheckResult(
                            feature=feature,
                            check_type="future_pattern",
                            passed=False,
                            message=f"Feature shows different pattern for future churners. May indicate leakage.",
                            severity="warning",
                        ))
    
    def _check_suspicious_names(self, feature: str) -> None:
        """Check for feature names that suggest leakage."""
        suspicious_patterns = [
            ("churn", "Feature name contains 'churn' - may leak target"),
            ("cancel", "Feature name contains 'cancel' - may be target-derived"),
            ("_future", "Feature name suggests future data"),
            ("_next", "Feature name suggests future data"),
            ("will_", "Feature name suggests future knowledge"),
            ("after_", "Feature name suggests post-prediction data"),
        ]
        
        feature_lower = feature.lower()
        for pattern, message in suspicious_patterns:
            if pattern in feature_lower:
                self.results.append(LeakageCheckResult(
                    feature=feature,
                    check_type="suspicious_name",
                    passed=False,
                    message=message,
                    severity="warning",
                ))
    
    def _check_perfect_prediction(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str,
    ) -> None:
        """Check if feature allows perfect prediction (definite leakage)."""
        if target not in df.columns:
            return
        
        if df[feature].dtype in [np.float64, np.int64, float, int]:
            # Check if feature perfectly separates classes
            target_values = df[target].astype(int)
            
            if target_values.nunique() < 2:
                return
            
            pos_values = df.loc[target_values == 1, feature]
            neg_values = df.loc[target_values == 0, feature]
            
            # Check for perfect separation
            if len(pos_values) > 0 and len(neg_values) > 0:
                if pos_values.min() > neg_values.max() or pos_values.max() < neg_values.min():
                    self.results.append(LeakageCheckResult(
                        feature=feature,
                        check_type="perfect_separation",
                        passed=False,
                        message="Feature perfectly separates classes. Definite leakage.",
                        severity="critical",
                    ))
    
    def _check_date_columns(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> None:
        """Check for raw date columns that shouldn't be features."""
        date_patterns = ["_date", "_at", "timestamp", "_time"]
        
        for feature in feature_columns:
            feature_lower = feature.lower()
            for pattern in date_patterns:
                if pattern in feature_lower:
                    self.results.append(LeakageCheckResult(
                        feature=feature,
                        check_type="raw_date",
                        passed=False,
                        message="Raw date column used as feature. Should be engineered into relative features.",
                        severity="warning",
                    ))
    
    def get_summary(self) -> dict:
        """Get summary of audit results."""
        critical = [r for r in self.results if r.severity == "critical"]
        warnings = [r for r in self.results if r.severity == "warning"]
        info = [r for r in self.results if r.severity == "info"]
        
        return {
            "total_checks": len(self.results),
            "critical_issues": len(critical),
            "warnings": len(warnings),
            "passed": len([r for r in self.results if r.passed]),
            "critical_features": [r.feature for r in critical],
            "warning_features": [r.feature for r in warnings],
        }
    
    def print_report(self) -> None:
        """Print human-readable audit report."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("LEAKAGE AUDIT REPORT")
        print("=" * 60)
        print(f"Prediction Date: {self.prediction_date}")
        print(f"Total Issues Found: {summary['total_checks']}")
        print(f"  - Critical: {summary['critical_issues']}")
        print(f"  - Warnings: {summary['warnings']}")
        print()
        
        if summary["critical_issues"] > 0:
            print("ðŸš¨ CRITICAL ISSUES (likely leakage):")
            for r in self.results:
                if r.severity == "critical":
                    print(f"  [{r.feature}] {r.message}")
            print()
        
        if summary["warnings"] > 0:
            print("âš ï¸  WARNINGS (review recommended):")
            for r in self.results:
                if r.severity == "warning":
                    print(f"  [{r.feature}] {r.message}")
            print()
        
        if summary["critical_issues"] == 0 and summary["warnings"] == 0:
            print("âœ… No leakage issues detected!")
        
        print("=" * 60)


def audit_feature_windows(
    feature_definitions: dict[str, dict],
    prediction_date: date,
) -> list[LeakageCheckResult]:
    """
    Audit feature window definitions for temporal correctness.
    
    Args:
        feature_definitions: Dict mapping feature name to window config
            Example: {"logins_7d": {"lookback_days": 7, "includes_today": False}}
        prediction_date: Point-in-time date
        
    Returns:
        List of audit results
    """
    results = []
    
    for feature, config in feature_definitions.items():
        lookback = config.get("lookback_days", 0)
        includes_today = config.get("includes_today", False)
        
        if includes_today:
            results.append(LeakageCheckResult(
                feature=feature,
                check_type="window_includes_today",
                passed=False,
                message=f"Feature window includes prediction date. Should use < not <=",
                severity="warning",
            ))
        
        if lookback <= 0:
            results.append(LeakageCheckResult(
                feature=feature,
                check_type="invalid_window",
                passed=False,
                message=f"Invalid lookback window ({lookback} days)",
                severity="critical",
            ))
    
    return results


def create_leakage_checklist() -> list[dict]:
    """
    Return standard leakage checklist for manual review.
    
    Use this as a template for documenting feature leakage review.
    """
    return [
        {
            "check": "Timestamp validation",
            "question": "Is the feature timestamp strictly before the prediction date?",
            "how_to_verify": "Check the SQL/code generating the feature for date comparisons",
        },
        {
            "check": "Window boundaries",
            "question": "Does the aggregation window use < (not <=) for the end date?",
            "how_to_verify": "Review window function definitions",
        },
        {
            "check": "Target derivation",
            "question": "Is the feature independent of the churn label?",
            "how_to_verify": "Trace feature calculation - should never reference churn status",
        },
        {
            "check": "Production availability",
            "question": "Would this feature be available at inference time in production?",
            "how_to_verify": "Consider data latency and availability in production system",
        },
        {
            "check": "Causal direction",
            "question": "Is the feature a cause (or predictor) of churn, not a consequence?",
            "how_to_verify": "Draw causal diagram - arrows should point TO churn, not FROM",
        },
        {
            "check": "Aggregation leakage",
            "question": "For rolling aggregates, is the window properly aligned?",
            "how_to_verify": "Verify that window_end < prediction_date for all records",
        },
    ]