# Feature Architecture & Engineering

## Overview

This document outlines the feature architecture and engineering approach for the 
churn prediction system, focusing on cohort-aware design and temporal correctness.

## Core Architecture Principles

### 1. Medallion Data Architecture

```m
┌─────────────────────────────────────────────────────────────┐
│                    DATA FLOW ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────┘

Raw Sources ──► Bronze ──► Silver ──► Gold ──► ML ──► Actions
    │             │        │        │       │        │        │
    ▼             ▼        ▼       ▼       ▼        ▼
Events        Raw      Cleaned   Feature  Model   Interventions
Customers     Data     Validated Matrix Scores & Decisions
Support       Events   Tickets  Explanations  Actions
```

**Layer Responsibilities:**

- **Bronze Layer**: Raw data ingestion, no transformations
- **Silver Layer**: Cleaning, validation, sessionization
- **Gold Layer**: Feature engineering, customer 360 view

### 2. Cohort-Aware Feature Design

#### Cohort Definitions

| Cohort | Tenure Range | Observation Window | Prediction Horizon |
|---------|---------------|-------------------|-------------------|
| New User | 0-30 days | 14 days | Days 15-45 |
| Established | 31-180 days | 30 days | Next 30 days |
| Mature | 181+ days | 60 days | Next 90 days |

#### Feature Categories

```m
FEATURE TAXONOMY
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  ACTIVATION │  │   ENGAGEMENT   │  │   RETENTION    │
│  (Day 1-14)  │  │  (Day 15-90) │  │  (Day 90+)    │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ • First value │  │ • Frequency   │  │ • Renewal timing│
│ • Onboarding │  │ • Depth      │  │ • Contract value│
│ • Setup time  │  │ • Intensity  │  │ • Usage trends  │
└─────────────┘  └─────────────┘  └─────────────┘
                           │
                           ▼
              ┌────────────────────┐
              │   VELOCITY FEATURES │
              │   (Δ between windows)│
              └────────────────────┘
```

### 3. Temporal Correctness Framework

#### Leakage Prevention Protocol

| Type | Description | Prevention Method |
|-------|-------------|------------------|
| Direct | Feature from future | Strict prediction point cutoff |
| Indirect | Feature derived from future data | Temporal validation |
| Target | Feature correlates with outcome | Feature isolation |
| Aggregation | Window crosses prediction point | Window boundary checks |

#### Implementation Checklist

For each feature, verify:

1. **Source Timestamp**: When is data available?
2. **Production Availability**: Would feature exist at prediction time?
3. **Window Respect**: Does aggregation respect temporal boundaries?
4. **Target Isolation**: Is feature independent of churn outcome?

### 4. Feature Engineering Pipeline

```python
class FeaturePipeline:
    """
    Point-in-time correct feature engineering pipeline
    """
    
    def __init__(self, lakehouse):
        self.lakehouse = lakehouse
    
    def build_features(self, prediction_date):
        # 1. Load silver data up to prediction_date
        # 2. Apply cohort-specific windows
        # 3. Calculate temporal features
        # 4. Validate temporal correctness
        # 5. Write to gold layer
        pass
```

## Implementation Details

### SQL Feature Templates

#### Activation Features (New Users Only)

```sql
-- Time to first value moment
WITH first_activity AS (
    SELECT customer_id, 
           MIN(activity_date) AS first_activity_date
    FROM silver.daily_engagement
    WHERE activity_date < prediction_date
    GROUP BY customer_id
),
activation_metrics AS (
    SELECT c.customer_id,
           first_activity_date,
           signup_date,
           DATEDIFF(first_activity_date, signup_date) AS days_to_first_login,
           -- Onboarding completion
           (SELECT COUNT(DISTINCT feature_name) / 15.0) * 100 AS onboarding_pct
           FROM silver.daily_engagement
           WHERE activity_date BETWEEN c.signup_date AND c.signup_date + INTERVAL '7 days'
           AND activity_date < prediction_date
           GROUP BY c.customer_id
    )
SELECT a.customer_id,
       a.days_to_first_login,
       a.onboarding_pct,
       -- First week engagement
       (SELECT COUNT(*) AS first_week_logins
        FROM silver.daily_engagement
        WHERE activity_date BETWEEN a.signup_date AND a.signup_date + INTERVAL '7 days'
        AND activity_date < prediction_date
        GROUP BY a.customer_id
FROM activation_metrics a
```

#### Engagement Features (All Users)

```sql
-- Frequency metrics
WITH engagement_freq AS (
    SELECT customer_id,
           SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '7 days' 
                   AND activity_date < prediction_date THEN login_count ELSE 0 END) AS logins_7d,
           SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '30 days' 
                   AND activity_date < prediction_date THEN login_count ELSE 0 END) AS logins_30d,
           AVG(CASE WHEN activity_date >= prediction_date - INTERVAL '7 days' 
                   AND activity_date < prediction_date THEN session_duration_minutes ELSE NULL END) AS avg_session_duration
    FROM silver.daily_engagement
    WHERE activity_date < prediction_date
    GROUP BY customer_id
),
-- Recency metrics
engagement_recency AS (
    SELECT e.customer_id,
           MAX(CASE WHEN e.login_count > 0 THEN e.activity_date ELSE NULL END) AS last_login_date,
           DATEDIFF(prediction_date, last_login_date) AS days_since_last_login
    FROM engagement_freq e
    GROUP BY e.customer_id
),
-- Velocity metrics
velocity_metrics AS (
    SELECT customer_id,
           -- Week-over-week change
           (logins_current_week - logins_previous_week) / NULLIF(logins_previous_week, 0) AS login_velocity_wow
    FROM (
        SELECT customer_id,
               SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '7 days' 
                        AND activity_date < prediction_date THEN login_count ELSE 0 END) AS logins_current_week,
               SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '14 days' 
                        AND activity_date < prediction_date - INTERVAL '7 days' THEN login_count ELSE 0 END) AS logins_previous_week
        FROM silver.daily_engagement
        WHERE activity_date < prediction_date
        GROUP BY customer_id, 
               FLOOR(DATEDIFF(prediction_date, activity_date) / 7) AS week_num
    )
    GROUP BY customer_id
)
SELECT f.customer_id,
       e.logins_7d,
       e.logins_30d,
       e.avg_session_duration,
       r.days_since_last_login,
       v.login_velocity_wow
FROM engagement_freq f
JOIN engagement_recency r ON f.customer_id = r.customer_id
JOIN velocity_metrics v ON f.customer_id = v.customer_id
```

#### Support Features

```sql
-- Support interaction metrics
WITH support_metrics AS (
    SELECT customer_id,
           COUNT(*) AS tickets_30d,
           AVG(sentiment_score) AS avg_sentiment_30d,
           AVG(CASE WHEN escalated THEN 1.0 ELSE 0.0 END) AS escalation_rate_30d,
           AVG(resolution_days) AS avg_resolution_days_30d
    FROM silver.support_tickets
    WHERE created_date < prediction_date
      AND created_date >= prediction_date - INTERVAL '30 days'
    GROUP BY customer_id
)
SELECT * FROM support_metrics
```

## Data Quality Validation

### Schema Validation

```python
def validate_gold_layer(df, prediction_date):
    """
    Validates gold layer data quality
    """
    issues = []
    
    # Check for temporal leakage
    future_features = df.columns[df.columns.str.contains('future') | 
                           df.columns.str.contains('churn_date')]
    if not future_features.empty:
        issues.append(f"Future features found: {future_features.tolist()}")
    
    # Check for missing values
    missing_rates = df.isnull().mean()
    if missing_rates['churn_probability'] > 0.05:
        issues.append(f"High missing rate in target: {missing_rates['churn_probability']:.2%}")
    
    # Check value ranges
    if df['logins_30d'].min() < 0:
        issues.append("Negative login counts found")
    
    return issues
```

### Performance Optimization

#### Indexing Strategy

```sql
-- Optimize for time-based queries
CREATE INDEX IF NOT EXISTS idx_gold_customer_360_date 
ON gold.customer_360 (customer_id, prediction_date);

-- Optimize for cohort-based queries
CREATE INDEX IF NOT EXISTS idx_gold_customer_360_cohort 
ON gold.customer_360 (cohort, prediction_date);
```

## Testing Strategy

### Unit Tests

```python
class TestFeatureEngineering:
    def test_temporal_correctness(self):
        # Test that features respect prediction point
        pass
    
    def test_cohort_consistency(self):
        # Test cohort-specific window logic
        pass
    
    def test_leakage_audit(self):
        # Test leakage detection functionality
        pass
```

### Integration Tests

```python
class TestEndToEndPipeline:
    def test_feature_pipeline(self):
        # Test complete data flow from bronze to gold
        pass
    
    def test_model_training(self):
        # Test model training with SHAP explanations
        pass
```

This architecture ensures:

1. **Temporal Correctness**: All features respect prediction boundaries
2. **Cohort Awareness**: Different treatment for different lifecycle stages
3. **Scalability**: Optimized for production workloads
4. **Testability**: Comprehensive test coverage
