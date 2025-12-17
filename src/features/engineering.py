"""
Feature Engineering Module for Churn Prediction

Implements cohort-aware feature engineering with:
- Activation features (new users)
- Engagement features (all cohorts)
- Velocity features (change over time)
- Support features
- Contract features

All features are point-in-time correct to prevent leakage.
"""

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def assign_cohort(tenure_days: int) -> str:
    """Assign customer to cohort based on tenure."""
    if tenure_days <= 30:
        return "new_user"
    elif tenure_days <= 180:
        return "established"
    else:
        return "mature"


def get_cohort_windows(cohort: str) -> dict:
    """
    Get appropriate feature windows for each cohort.
    
    Different cohorts need different observation windows:
    - New users: Short windows (behavior is recent)
    - Established: Medium windows
    - Mature: Longer windows (more stable patterns)
    """
    windows = {
        "new_user": {"short": 7, "medium": 14, "long": 14},
        "established": {"short": 7, "medium": 14, "long": 30},
        "mature": {"short": 14, "medium": 30, "long": 60},
    }
    return windows.get(cohort, windows["established"])


def calculate_velocity(
    current_value: float,
    previous_value: float,
) -> float:
    """
    Calculate velocity (rate of change) between two values.
    
    Returns: (current - previous) / previous, or 0 if previous is 0
    """
    if previous_value == 0:
        return 1.0 if current_value > 0 else 0.0
    return (current_value - previous_value) / previous_value


def calculate_trend_slope(values: list[float]) -> float:
    """
    Calculate linear trend slope from a series of values.
    
    Uses simple linear regression slope.
    """
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    x = np.arange(n)
    y = np.array(values)
    
    # Simple linear regression slope
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


class FeatureEngineer:
    """
    Feature engineering class for churn prediction.
    
    Usage:
        engineer = FeatureEngineer(prediction_date=date(2024, 12, 1))
        features = engineer.build_features(customers_df, engagement_df, tickets_df)
    """
    
    def __init__(self, prediction_date: date):
        """
        Initialize feature engineer.
        
        Args:
            prediction_date: Point-in-time date for feature calculation
        """
        self.prediction_date = prediction_date
    
    def build_features(
        self,
        customers: pd.DataFrame,
        daily_engagement: pd.DataFrame,
        support_tickets: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build all features for modeling.
        
        Args:
            customers: Customer master data
            daily_engagement: Daily engagement summaries
            support_tickets: Support ticket data
            
        Returns:
            DataFrame with all features, one row per customer
        """
        # Filter to customers who exist at prediction date
        customers = customers[
            pd.to_datetime(customers["signup_date"]) <= pd.Timestamp(self.prediction_date)
        ].copy()
        
        # Calculate tenure at prediction date
        customers["tenure_at_prediction"] = (
            pd.Timestamp(self.prediction_date) - pd.to_datetime(customers["signup_date"])
        ).dt.days
        
        # Assign cohorts
        customers["cohort"] = customers["tenure_at_prediction"].apply(assign_cohort)
        
        # Build feature sets
        engagement_features = self._build_engagement_features(
            customers, daily_engagement
        )
        velocity_features = self._build_velocity_features(
            customers, daily_engagement
        )
        support_features = self._build_support_features(
            customers, support_tickets
        )
        activation_features = self._build_activation_features(
            customers, daily_engagement
        )
        contract_features = self._build_contract_features(customers)
        
        # Merge all features
        features = customers[["customer_id", "cohort", "ltv_tier"]].copy()
        features["prediction_date"] = self.prediction_date
        
        for feature_df in [
            engagement_features,
            velocity_features,
            support_features,
            activation_features,
            contract_features,
        ]:
            features = features.merge(feature_df, on="customer_id", how="left")
        
        # Add target label
        features = self._add_target_label(features, customers)
        
        # Add sample weights
        features = self._add_sample_weights(features, customers)
        
        return features
    
    def _build_engagement_features(
        self,
        customers: pd.DataFrame,
        daily_engagement: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build engagement frequency and depth features."""
        
        # Filter engagement to before prediction date
        engagement = daily_engagement[
            pd.to_datetime(daily_engagement["activity_date"]) < pd.Timestamp(self.prediction_date)
        ].copy()
        
        features = []
        
        for _, customer in customers.iterrows():
            customer_id = customer["customer_id"]
            cohort = customer["cohort"]
            
            # Get customer's engagement
            cust_engagement = engagement[engagement["customer_id"] == customer_id]
            
            # Get cohort-appropriate windows
            windows = get_cohort_windows(cohort)
            
            # Calculate features for each window
            feature_row = {"customer_id": customer_id}
            
            for window_name, window_days in windows.items():
                window_start = self.prediction_date - timedelta(days=window_days)
                
                window_data = cust_engagement[
                    pd.to_datetime(cust_engagement["activity_date"]) >= pd.Timestamp(window_start)
                ]
                
                suffix = f"_{window_days}d"
                
                feature_row[f"logins{suffix}"] = window_data["login_count"].sum()
                feature_row[f"features_used{suffix}"] = window_data["features_used"].sum()
                feature_row[f"session_minutes{suffix}"] = window_data["session_duration_minutes"].sum()
            
            # Recency features
            if len(cust_engagement) > 0:
                last_login = cust_engagement[
                    cust_engagement["login_count"] > 0
                ]["activity_date"].max()
                
                if pd.notna(last_login):
                    feature_row["days_since_last_login"] = (
                        self.prediction_date - pd.to_datetime(last_login).date()
                    ).days
                else:
                    feature_row["days_since_last_login"] = 999
                
                last_feature = cust_engagement[
                    cust_engagement["features_used"] > 0
                ]["activity_date"].max()
                
                if pd.notna(last_feature):
                    feature_row["days_since_last_feature_use"] = (
                        self.prediction_date - pd.to_datetime(last_feature).date()
                    ).days
                else:
                    feature_row["days_since_last_feature_use"] = 999
            else:
                feature_row["days_since_last_login"] = 999
                feature_row["days_since_last_feature_use"] = 999
            
            # Feature adoption percentage
            max_features = cust_engagement["unique_features"].max() if len(cust_engagement) > 0 else 0
            feature_row["feature_adoption_pct"] = max_features / 15.0  # 15 total features
            
            # Average session duration
            total_logins = cust_engagement["login_count"].sum()
            total_session = cust_engagement["session_duration_minutes"].sum()
            feature_row["avg_session_duration"] = (
                total_session / total_logins if total_logins > 0 else 0
            )
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _build_velocity_features(
        self,
        customers: pd.DataFrame,
        daily_engagement: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build engagement velocity (change over time) features."""
        
        engagement = daily_engagement[
            pd.to_datetime(daily_engagement["activity_date"]) < pd.Timestamp(self.prediction_date)
        ].copy()
        
        features = []
        
        for _, customer in customers.iterrows():
            customer_id = customer["customer_id"]
            
            cust_engagement = engagement[engagement["customer_id"] == customer_id]
            
            feature_row = {"customer_id": customer_id}
            
            # Week-over-week velocity
            this_week_start = self.prediction_date - timedelta(days=7)
            last_week_start = self.prediction_date - timedelta(days=14)
            
            this_week = cust_engagement[
                (pd.to_datetime(cust_engagement["activity_date"]) >= pd.Timestamp(this_week_start))
            ]
            last_week = cust_engagement[
                (pd.to_datetime(cust_engagement["activity_date"]) >= pd.Timestamp(last_week_start)) &
                (pd.to_datetime(cust_engagement["activity_date"]) < pd.Timestamp(this_week_start))
            ]
            
            logins_this_week = this_week["login_count"].sum()
            logins_last_week = last_week["login_count"].sum()
            
            feature_row["login_velocity_wow"] = calculate_velocity(
                logins_this_week, logins_last_week
            )
            
            features_this_week = this_week["features_used"].sum()
            features_last_week = last_week["features_used"].sum()
            
            feature_row["feature_velocity_wow"] = calculate_velocity(
                features_this_week, features_last_week
            )
            
            # 4-week trend
            weekly_logins = []
            for week in range(4):
                week_start = self.prediction_date - timedelta(days=7 * (week + 1))
                week_end = self.prediction_date - timedelta(days=7 * week)
                
                week_data = cust_engagement[
                    (pd.to_datetime(cust_engagement["activity_date"]) >= pd.Timestamp(week_start)) &
                    (pd.to_datetime(cust_engagement["activity_date"]) < pd.Timestamp(week_end))
                ]
                weekly_logins.append(week_data["login_count"].sum())
            
            # Reverse so oldest is first
            weekly_logins = weekly_logins[::-1]
            feature_row["login_trend_4w"] = calculate_trend_slope(weekly_logins)
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _build_support_features(
        self,
        customers: pd.DataFrame,
        support_tickets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build support interaction features."""
        
        tickets = support_tickets[
            pd.to_datetime(support_tickets["created_date"]) < pd.Timestamp(self.prediction_date)
        ].copy()
        
        features = []
        
        for _, customer in customers.iterrows():
            customer_id = customer["customer_id"]
            
            cust_tickets = tickets[tickets["customer_id"] == customer_id]
            
            feature_row = {"customer_id": customer_id}
            
            # Ticket volume (30-day window)
            window_start = self.prediction_date - timedelta(days=30)
            recent_tickets = cust_tickets[
                pd.to_datetime(cust_tickets["created_date"]) >= pd.Timestamp(window_start)
            ]
            
            feature_row["tickets_30d"] = len(recent_tickets)
            
            # Sentiment
            if len(recent_tickets) > 0:
                feature_row["avg_sentiment_30d"] = recent_tickets["sentiment_score"].mean()
                feature_row["min_sentiment_30d"] = recent_tickets["sentiment_score"].min()
            else:
                feature_row["avg_sentiment_30d"] = 0.0
                feature_row["min_sentiment_30d"] = 0.0
            
            # Escalation rate
            if len(recent_tickets) > 0:
                feature_row["escalation_rate_30d"] = recent_tickets["escalated"].mean()
            else:
                feature_row["escalation_rate_30d"] = 0.0
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _build_activation_features(
        self,
        customers: pd.DataFrame,
        daily_engagement: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build activation features (primarily for new users)."""
        
        engagement = daily_engagement.copy()
        
        features = []
        
        for _, customer in customers.iterrows():
            customer_id = customer["customer_id"]
            signup_date = pd.to_datetime(customer["signup_date"]).date()
            cohort = customer["cohort"]
            
            feature_row = {"customer_id": customer_id}
            
            # Only calculate activation features for new users
            if cohort != "new_user":
                feature_row["days_to_first_login"] = None
                feature_row["onboarding_completion_pct"] = None
                feature_row["first_week_logins"] = None
                features.append(feature_row)
                continue
            
            cust_engagement = engagement[
                (engagement["customer_id"] == customer_id) &
                (pd.to_datetime(engagement["activity_date"]) < pd.Timestamp(self.prediction_date))
            ]
            
            # Days to first login
            if len(cust_engagement) > 0:
                first_login = cust_engagement[
                    cust_engagement["login_count"] > 0
                ]["activity_date"].min()
                
                if pd.notna(first_login):
                    feature_row["days_to_first_login"] = (
                        pd.to_datetime(first_login).date() - signup_date
                    ).days
                else:
                    feature_row["days_to_first_login"] = 999
            else:
                feature_row["days_to_first_login"] = 999
            
            # First week activity
            first_week_end = signup_date + timedelta(days=7)
            first_week_data = cust_engagement[
                pd.to_datetime(cust_engagement["activity_date"]) <= pd.Timestamp(first_week_end)
            ]
            
            feature_row["first_week_logins"] = first_week_data["login_count"].sum()
            
            # Onboarding completion (approximated by feature usage in first week)
            first_week_features = first_week_data["unique_features"].max() if len(first_week_data) > 0 else 0
            feature_row["onboarding_completion_pct"] = first_week_features / 15.0
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _build_contract_features(
        self,
        customers: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build contract and billing features."""
        
        features = []
        
        for _, customer in customers.iterrows():
            customer_id = customer["customer_id"]
            
            feature_row = {"customer_id": customer_id}
            
            # Contract info
            contract_type = customer.get("contract_type", "Month-to-month")
            contract_months = {"Month-to-month": 1, "One year": 12, "Two year": 24}.get(
                contract_type, 1
            )
            
            tenure_days = customer.get("tenure_days", 0)
            
            # Days to renewal (simplified)
            if contract_type == "Month-to-month":
                days_to_renewal = 30 - (tenure_days % 30)
            elif contract_type == "One year":
                days_to_renewal = 365 - (tenure_days % 365)
            else:
                days_to_renewal = 730 - (tenure_days % 730)
            
            feature_row["days_to_renewal"] = days_to_renewal
            feature_row["contract_type_encoded"] = {"Month-to-month": 0, "One year": 1, "Two year": 2}.get(
                contract_type, 0
            )
            
            # Contract value remaining
            monthly_charges = customer.get("monthly_charges", 0)
            feature_row["contract_value_remaining"] = monthly_charges * (days_to_renewal / 30)
            
            # Tenure
            feature_row["tenure_days"] = tenure_days
            
            # LTV
            feature_row["estimated_ltv"] = customer.get("estimated_ltv", 0)
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def _add_target_label(
        self,
        features: pd.DataFrame,
        customers: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add target label: churned within prediction window."""
        
        # Merge churn info
        churn_info = customers[["customer_id", "churn_label", "churn_date"]].copy()
        features = features.merge(churn_info, on="customer_id", how="left")
        
        # Calculate if churned within 30-day window
        prediction_window_end = self.prediction_date + timedelta(days=30)
        
        def check_churn_in_window(row):
            if not row["churn_label"]:
                return False
            if pd.isna(row["churn_date"]):
                return False
            
            churn_date = pd.to_datetime(row["churn_date"]).date()
            return (
                churn_date > self.prediction_date and
                churn_date <= prediction_window_end
            )
        
        features["churned_in_window"] = features.apply(check_churn_in_window, axis=1)
        
        # Drop intermediate columns
        features = features.drop(columns=["churn_label", "churn_date"], errors="ignore")
        
        return features
    
    def _add_sample_weights(
        self,
        features: pd.DataFrame,
        customers: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add sample weights based on LTV tier."""
        
        weight_map = {
            "smb": 1.0,
            "mid_market": 3.0,
            "enterprise": 10.0,
        }
        
        if "ltv_tier" in features.columns:
            features["sample_weight"] = features["ltv_tier"].map(weight_map).fillna(1.0)
        else:
            features["sample_weight"] = 1.0
        
        return features


def build_features_for_date(
    prediction_date: date,
    customers: pd.DataFrame,
    daily_engagement: pd.DataFrame,
    support_tickets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convenience function to build features for a single date.
    
    Args:
        prediction_date: Point-in-time date
        customers: Customer master data
        daily_engagement: Daily engagement data
        support_tickets: Support ticket data
        
    Returns:
        Feature DataFrame
    """
    engineer = FeatureEngineer(prediction_date)
    return engineer.build_features(customers, daily_engagement, support_tickets)


def build_features_for_date_range(
    start_date: date,
    end_date: date,
    customers: pd.DataFrame,
    daily_engagement: pd.DataFrame,
    support_tickets: pd.DataFrame,
    frequency_days: int = 7,
) -> pd.DataFrame:
    """
    Build features for multiple prediction dates.
    
    Useful for creating training data with multiple time points.
    
    Args:
        start_date: First prediction date
        end_date: Last prediction date
        customers: Customer master data
        daily_engagement: Daily engagement data
        support_tickets: Support ticket data
        frequency_days: Days between prediction dates
        
    Returns:
        Combined feature DataFrame
    """
    all_features = []
    
    current_date = start_date
    while current_date <= end_date:
        print(f"Building features for {current_date}...")
        
        features = build_features_for_date(
            current_date, customers, daily_engagement, support_tickets
        )
        all_features.append(features)
        
        current_date += timedelta(days=frequency_days)
    
    return pd.concat(all_features, ignore_index=True)