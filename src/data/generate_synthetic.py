"""
Synthetic Data Generator for Churn Prediction

Extends the Kaggle Telco Churn dataset with:
- Scaled to 50K customers
- 12 months of behavioral events (logins, feature usage, support tickets)
- Cohort assignments (New/Established/Mature)
- LTV tiers (SMB/Mid-Market/Enterprise)
- Realistic churn patterns aligned to lifecycle framework

Usage:
    python -m src.data.generate_synthetic
    # Or via uv:
    uv run python -m src.data.generate_synthetic
"""

import hashlib
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config/model_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_customer_id() -> str:
    """Generate unique customer ID."""
    return str(uuid.uuid4()).upper()


def assign_cohort(tenure_days: int) -> str:
    """Assign customer to cohort based on tenure."""
    if tenure_days <= 30:
        return "new_user"
    elif tenure_days <= 180:
        return "established"
    else:
        return "mature"


def assign_ltv_tier(monthly_charges: float, config: dict) -> str:
    """Assign LTV tier based on monthly charges."""
    tiers = config["data"]["ltv_tiers"]
    if monthly_charges < tiers["smb"]["monthly_revenue_range"][1]:
        return "smb"
    elif monthly_charges < tiers["mid_market"]["monthly_revenue_range"][1]:
        return "mid_market"
    else:
        return "enterprise"


def calculate_estimated_ltv(monthly_charges: float, contract_type: str, tenure_days: int) -> float:
    """Calculate estimated customer lifetime value."""
    contract_months = {"Month-to-month": 1, "One year": 12, "Two year": 24}.get(contract_type, 1)
    avg_lifetime_months = max(12, tenure_days / 30 + contract_months)
    return monthly_charges * avg_lifetime_months * 1.2  # 1.2x for expansion potential


def generate_base_customers(
    kaggle_df: pd.DataFrame,
    n_customers: int,
    reference_date: date,
    config: dict,
) -> pd.DataFrame:
    """Generate customer base by scaling Kaggle data."""
    
    # Sample with replacement to scale up
    n_kaggle = len(kaggle_df)
    sample_indices = np.random.choice(n_kaggle, size=n_customers, replace=True)
    customers = kaggle_df.iloc[sample_indices].copy().reset_index(drop=True)
    
    # Generate unique customer IDs
    customers["customer_id"] = [generate_customer_id() for _ in range(n_customers)]
    
    # Generate signup dates (distributed over past 24 months for tenure distribution)
    max_tenure_days = 720  # 24 months
    tenure_days = np.random.exponential(scale=180, size=n_customers).astype(int)
    tenure_days = np.clip(tenure_days, 1, max_tenure_days)
    
    # Override tenure from Kaggle with generated tenure for better distribution
    customers["tenure_days"] = tenure_days
    customers["signup_date"] = [
        reference_date - timedelta(days=int(t)) for t in tenure_days
    ]
    
    # Assign cohorts
    customers["cohort"] = customers["tenure_days"].apply(assign_cohort)
    
    # Scale monthly charges for LTV tier distribution
    ltv_config = config["data"]["ltv_tiers"]
    
    def scale_charges(row):
        # Randomly assign to LTV tier based on configured proportions
        tier_roll = np.random.random()
        if tier_roll < ltv_config["smb"]["proportion"]:
            tier = "smb"
            range_min, range_max = ltv_config["smb"]["monthly_revenue_range"]
        elif tier_roll < ltv_config["smb"]["proportion"] + ltv_config["mid_market"]["proportion"]:
            tier = "mid_market"
            range_min, range_max = ltv_config["mid_market"]["monthly_revenue_range"]
        else:
            tier = "enterprise"
            range_min, range_max = ltv_config["enterprise"]["monthly_revenue_range"]
        
        monthly = np.random.uniform(range_min, range_max)
        return pd.Series({"monthly_charges_scaled": monthly, "ltv_tier": tier})
    
    tier_data = customers.apply(scale_charges, axis=1)
    customers["monthly_charges"] = tier_data["monthly_charges_scaled"]
    customers["ltv_tier"] = tier_data["ltv_tier"]
    
    # Calculate estimated LTV
    customers["estimated_ltv"] = customers.apply(
        lambda r: calculate_estimated_ltv(
            r["monthly_charges"], r["Contract"], r["tenure_days"]
        ),
        axis=1,
    )
    
    # Generate sample weights based on LTV tier
    weight_map = {
        "smb": ltv_config["smb"]["weight"],
        "mid_market": ltv_config["mid_market"]["weight"],
        "enterprise": ltv_config["enterprise"]["weight"],
    }
    customers["sample_weight"] = customers["ltv_tier"].map(weight_map)
    
    # Process churn labels
    customers["churn_label"] = customers["Churn"].map({"Yes": True, "No": False})
    
    # Generate churn dates for churned customers (within observation period)
    def generate_churn_date(row):
        if not row["churn_label"]:
            return None
        # Churn happened sometime between signup and reference date
        days_to_churn = np.random.randint(
            max(1, row["tenure_days"] - 30),
            row["tenure_days"] + 1
        )
        return row["signup_date"] + timedelta(days=days_to_churn)
    
    customers["churn_date"] = customers.apply(generate_churn_date, axis=1)
    
    # Clean column names
    customers = customers.rename(columns={
        "Contract": "contract_type",
        "PaymentMethod": "payment_method",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "PaperlessBilling": "paperless_billing",
    })
    
    # Convert Yes/No to boolean for service columns
    bool_columns = [
        "partner", "dependents", "phone_service", "paperless_billing"
    ]
    for col in bool_columns:
        if col in customers.columns:
            customers[col] = customers[col].map({"Yes": True, "No": False})
    
    # Select final columns
    final_columns = [
        "customer_id", "signup_date", "tenure_days", "cohort", "ltv_tier",
        "contract_type", "monthly_charges", "estimated_ltv", "sample_weight",
        "payment_method", "gender", "senior_citizen", "partner", "dependents",
        "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies", "paperless_billing",
        "churn_label", "churn_date"
    ]
    
    return customers[[c for c in final_columns if c in customers.columns]]


def generate_engagement_events(
    customers: pd.DataFrame,
    reference_date: date,
    config: dict,
) -> pd.DataFrame:
    """Generate daily engagement events for all customers."""
    
    events = []
    n_features_total = 15  # Total product features available
    
    for _, customer in customers.iterrows():
        customer_id = customer["customer_id"]
        signup_date = customer["signup_date"]
        churn_date = customer["churn_date"]
        cohort = customer["cohort"]
        is_churner = customer["churn_label"]
        ltv_tier = customer["ltv_tier"]
        
        # Determine date range for this customer
        start_date = signup_date
        end_date = min(
            churn_date if churn_date else reference_date,
            reference_date
        )
        
        if start_date >= end_date:
            continue
        
        # Base engagement parameters by cohort
        base_params = {
            "new_user": {"login_rate": 0.7, "feature_rate": 0.5, "session_base": 15},
            "established": {"login_rate": 0.5, "feature_rate": 0.6, "session_base": 20},
            "mature": {"login_rate": 0.4, "feature_rate": 0.7, "session_base": 25},
        }
        params = base_params.get(cohort, base_params["established"])
        
        # LTV tier multiplier (enterprise users are more engaged)
        ltv_multiplier = {"smb": 0.8, "mid_market": 1.0, "enterprise": 1.3}.get(ltv_tier, 1.0)
        
        # Churner behavior pattern
        if is_churner and churn_date:
            days_before_churn = (churn_date - signup_date).days
        else:
            days_before_churn = None
        
        # Generate daily events
        current_date = start_date
        while current_date <= end_date:
            days_since_signup = (current_date - signup_date).days
            days_to_churn = (churn_date - current_date).days if churn_date else None
            
            # Calculate engagement decay for churners
            if is_churner and days_to_churn is not None and days_to_churn >= 0:
                # Exponential decay as churn approaches
                decay_factor = max(0.1, min(1.0, days_to_churn / 60))
            else:
                decay_factor = 1.0
            
            # Add some natural variation
            daily_variation = np.random.normal(1.0, 0.2)
            daily_variation = max(0.3, min(1.7, daily_variation))
            
            # Weekend effect (lower engagement)
            is_weekend = current_date.weekday() >= 5
            weekend_factor = 0.6 if is_weekend else 1.0
            
            # Calculate daily metrics
            adjusted_login_rate = (
                params["login_rate"] * ltv_multiplier * decay_factor * 
                daily_variation * weekend_factor
            )
            
            # Determine if user logs in today
            logs_in = np.random.random() < adjusted_login_rate
            
            if logs_in:
                login_count = np.random.poisson(lam=1.5) + 1
                
                # Session duration (minutes)
                session_base = params["session_base"] * ltv_multiplier * decay_factor
                session_duration = max(1, np.random.exponential(session_base))
                
                # Features used
                feature_rate = params["feature_rate"] * decay_factor
                features_used = np.random.binomial(n_features_total, feature_rate)
                unique_features = min(features_used, np.random.poisson(3) + 1)
            else:
                login_count = 0
                session_duration = 0
                features_used = 0
                unique_features = 0
            
            events.append({
                "customer_id": customer_id,
                "activity_date": current_date,
                "login_count": login_count,
                "session_duration_minutes": round(session_duration, 2),
                "features_used": features_used,
                "unique_features": unique_features,
            })
            
            current_date += timedelta(days=1)
    
    return pd.DataFrame(events)


def generate_support_tickets(
    customers: pd.DataFrame,
    reference_date: date,
    config: dict,
) -> pd.DataFrame:
    """Generate support ticket events for all customers."""
    
    tickets = []
    categories = ["billing", "technical", "feature_request", "account", "general"]
    priorities = ["low", "medium", "high", "critical"]
    
    for _, customer in customers.iterrows():
        customer_id = customer["customer_id"]
        signup_date = customer["signup_date"]
        churn_date = customer["churn_date"]
        is_churner = customer["churn_label"]
        ltv_tier = customer["ltv_tier"]
        tenure_days = customer["tenure_days"]
        
        # Base ticket rate (monthly)
        base_rate = {"smb": 0.3, "mid_market": 0.5, "enterprise": 0.8}.get(ltv_tier, 0.4)
        
        # Churners generate more tickets before churning
        if is_churner:
            base_rate *= 1.5
        
        # Generate tickets over customer lifetime
        current_date = signup_date
        end_date = min(
            churn_date if churn_date else reference_date,
            reference_date
        )
        
        while current_date <= end_date:
            days_to_churn = (churn_date - current_date).days if churn_date else None
            
            # Increase ticket rate as churn approaches
            if is_churner and days_to_churn is not None and days_to_churn < 30:
                rate_multiplier = 2.0 - (days_to_churn / 30)
            else:
                rate_multiplier = 1.0
            
            # Poisson process for ticket creation
            monthly_rate = base_rate * rate_multiplier
            daily_rate = monthly_rate / 30
            
            if np.random.random() < daily_rate:
                ticket_id = f"TKT-{uuid.uuid4().hex.upper()}"
                
                # Category distribution
                if is_churner and days_to_churn and days_to_churn < 45:
                    # Churners more likely to have billing/technical issues
                    category = np.random.choice(
                        categories, 
                        p=[0.3, 0.35, 0.1, 0.15, 0.1]
                    )
                else:
                    category = np.random.choice(categories)
                
                # Priority
                priority = np.random.choice(
                    priorities,
                    p=[0.4, 0.35, 0.2, 0.05]
                )
                
                # Sentiment score (-1 to 1)
                if is_churner and days_to_churn and days_to_churn < 30:
                    # Negative sentiment for imminent churners
                    sentiment = np.random.normal(-0.3, 0.3)
                else:
                    sentiment = np.random.normal(0.1, 0.3)
                sentiment = max(-1, min(1, sentiment))
                
                # Resolution time (days)
                resolution_base = {"low": 3, "medium": 2, "high": 1, "critical": 0.5}
                resolution_days = np.random.exponential(resolution_base[priority])
                resolved_at = current_date + timedelta(days=resolution_days)
                
                # Escalation
                escalation_prob = {"low": 0.02, "medium": 0.05, "high": 0.15, "critical": 0.3}
                escalated = np.random.random() < escalation_prob[priority]
                
                tickets.append({
                    "ticket_id": ticket_id,
                    "customer_id": customer_id,
                    "created_at": datetime.combine(current_date, datetime.min.time()),
                    "resolved_at": datetime.combine(resolved_at, datetime.min.time()),
                    "created_date": current_date,
                    "resolution_days": round(resolution_days, 2),
                    "category": category,
                    "priority": priority,
                    "sentiment_score": round(sentiment, 2),
                    "escalated": escalated,
                })
            
            current_date += timedelta(days=1)
    
    return pd.DataFrame(tickets)


def generate_login_events(
    daily_engagement: pd.DataFrame,
) -> pd.DataFrame:
    """Generate individual login events from daily engagement summary."""
    
    events = []
    
    for _, row in daily_engagement.iterrows():
        if row["login_count"] > 0:
            for i in range(row["login_count"]):
                event_id = f"EVT-{uuid.uuid4().hex.upper()}"
                # Distribute logins throughout the day
                hour = int(np.random.choice(range(8, 22)))  # 8am to 10pm
                minute = int(np.random.randint(0, 60))
                
                event_timestamp = datetime.combine(
                    row["activity_date"],
                    datetime.min.time()
                ) + timedelta(hours=hour, minutes=minute)
                
                events.append({
                    "event_id": event_id,
                    "customer_id": row["customer_id"],
                    "event_type": "login",
                    "event_timestamp": event_timestamp,
                })
    
    return pd.DataFrame(events)


def save_to_parquet(
    customers: pd.DataFrame,
    daily_engagement: pd.DataFrame,
    support_tickets: pd.DataFrame,
    login_events: pd.DataFrame,
    output_dir: str,
) -> dict:
    """Save all generated data to Parquet files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save customers
    customers_path = output_path / "customers.parquet"
    customers.to_parquet(customers_path, index=False)
    paths["customers"] = str(customers_path)
    
    # Save daily engagement
    engagement_path = output_path / "daily_engagement.parquet"
    daily_engagement.to_parquet(engagement_path, index=False)
    paths["daily_engagement"] = str(engagement_path)
    
    # Save support tickets
    tickets_path = output_path / "support_tickets.parquet"
    support_tickets.to_parquet(tickets_path, index=False)
    paths["support_tickets"] = str(tickets_path)
    
    # Save login events
    events_path = output_path / "login_events.parquet"
    login_events.to_parquet(events_path, index=False)
    paths["login_events"] = str(events_path)
    
    return paths


def main():
    """Generate synthetic dataset."""
    
    print("=" * 60)
    print("Synthetic Data Generator for Churn Prediction")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    set_seed(config["data"]["random_seed"])
    
    n_customers = config["data"]["n_customers"]
    reference_date = date.today()
    
    print(f"\nConfiguration:")
    print(f"  - Target customers: {n_customers:,}")
    print(f"  - History months: {config['data']['n_months_history']}")
    print(f"  - Reference date: {reference_date}")
    
    # Load Kaggle dataset
    print("\n[1/5] Loading Kaggle Telco dataset...")
    kaggle_path = config["paths"]["data_raw"]
    kaggle_df = pd.read_csv(kaggle_path)
    print(f"  - Loaded {len(kaggle_df):,} records from Kaggle")
    
    # Generate customers
    print("\n[2/5] Generating scaled customer base...")
    customers = generate_base_customers(kaggle_df, n_customers, reference_date, config)
    print(f"  - Generated {len(customers):,} customers")
    print(f"  - Cohort distribution:")
    for cohort, count in customers["cohort"].value_counts().items():
        print(f"      {cohort}: {count:,} ({count/len(customers)*100:.1f}%)")
    print(f"  - LTV tier distribution:")
    for tier, count in customers["ltv_tier"].value_counts().items():
        print(f"      {tier}: {count:,} ({count/len(customers)*100:.1f}%)")
    print(f"  - Churn rate: {customers['churn_label'].mean()*100:.1f}%")
    
    # Generate engagement events
    print("\n[3/5] Generating daily engagement events...")
    daily_engagement = generate_engagement_events(customers, reference_date, config)
    print(f"  - Generated {len(daily_engagement):,} daily records")
    
    # Generate support tickets
    print("\n[4/5] Generating support tickets...")
    support_tickets = generate_support_tickets(customers, reference_date, config)
    print(f"  - Generated {len(support_tickets):,} tickets")
    
    # Generate login events
    print("\n[5/5] Generating individual login events...")
    login_events = generate_login_events(daily_engagement)
    print(f"  - Generated {len(login_events):,} login events")
    
    # Save to Parquet
    print("\nSaving to Parquet files...")
    output_dir = "outputs/synthetic_data"
    paths = save_to_parquet(
        customers, daily_engagement, support_tickets, login_events, output_dir
    )
    
    for name, path in paths.items():
        print(f"  - {name}: {path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"\nDataset Summary:")
    print(f"  - Customers: {len(customers):,}")
    print(f"  - Daily engagement records: {len(daily_engagement):,}")
    print(f"  - Support tickets: {len(support_tickets):,}")
    print(f"  - Login events: {len(login_events):,}")
    print(f"  - Total events: {len(daily_engagement) + len(login_events):,}")
    
    return customers, daily_engagement, support_tickets, login_events


if __name__ == "__main__":
    main()
