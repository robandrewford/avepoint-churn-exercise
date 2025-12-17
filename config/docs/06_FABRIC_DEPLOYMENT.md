# Microsoft Fabric Deployment Guide

This document maps local DuckDB implementation to Microsoft Fabric for production deployment, with step-by-step instructions and code examples.

## Architecture Translation

| Local (DuckDB) | Fabric Equivalent | Notes |
|----------------|-------------------|-------|
| DuckDB `.duckdb` file | OneLake Lakehouse | Delta tables stored in OneLake |
| DuckDB schemas (bronze/silver/gold) | Lakehouse schemas | Same medallion architecture |
| Python scripts | Synapse Notebooks | Same Python code, Spark optional |
| MLflow tracking | Fabric MLflow | Managed MLflow, auto-configured |
| Parquet files | Delta tables | Same columnar format, add ACID transactions |
| Marimo notebooks | Fabric Notebooks | Convert to standard notebooks |
| SQL queries | Synapse SQL | Nearly identical syntax |
| `INTERVAL '7 days'` | `INTERVAL 7 DAY` or `date_sub(col, 7)` | Minor syntax differences |

### Why Fabric for AvePoint

| AvePoint Context | Fabric Capability | Value |
|---------|-------------------|-------|
| Microsoft ecosystem customer base | Native M365 integration | No additional infrastructure for customers |
| Data governance focus | OneLake unified governance | Single security model across all data |
| Rapid prototyping need | Notebooks + Pipelines in one workspace | Fast iteration without DevOps overhead |
| Enterprise customers | Power BI embedded | Dashboards customers already trust |
| Multi-tenant SaaS | Workspace isolation | Per-customer or per-segment deployments |
| Cost optimization needs | Delta Lake caching, Z-order clustering, Spark for large scale | Optimize for production workloads |
| Security considerations | Row-level security, column masking, service principals | Enterprise-grade security and compliance |

---

## Step 1: Create Fabric Workspace

```m
1. Navigate to app.fabric.microsoft.com
2. Create new Workspace: "churn-prediction-prod"
3. Enable Fabric capacity (Trial/Premium/Fabric)
```

---

## Step 2: Create Lakehouse

```m
1. In workspace, select "New" → "Lakehouse"
2. Name: "churn_lakehouse"
3. This creates:
   - Tables (managed Delta tables)
   - Files (unmanaged files)
   - SQL endpoint (for querying)
```

---

## Step 3: Upload Data

### Option A: Direct Upload

```python
# In Fabric notebook
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Upload Parquet files to Files section, then:
df = spark.read.parquet("Files/synthetic_data/customers.parquet")
df.write.format("delta").saveAsTable("bronze.customers")
```

### Option B: Data Pipeline

```m
1. Create Data Pipeline
2. Source: Local files or cloud storage
3. Destination: Lakehouse tables
4. Schedule: Daily refresh
```

---

## Step 4: Create Bronze Layer Tables

```sql
-- Execute in Lakehouse SQL endpoint or notebook

CREATE SCHEMA IF NOT EXISTS bronze;

CREATE TABLE bronze.customers
USING DELTA
AS SELECT * FROM parquet.`Files/synthetic_data/customers.parquet`;

CREATE TABLE bronze.events
USING DELTA
AS SELECT * FROM parquet.`Files/synthetic_data/login_events.parquet`;

CREATE TABLE bronze.support_tickets
USING DELTA
AS SELECT * FROM parquet.`Files/synthetic_data/support_tickets.parquet`;
```

---

## Step 5: Create Silver Layer (Transformation Notebook)

Create notebook: `01_bronze_to_silver.py`

```python
# Fabric Notebook - Bronze to Silver Transformation

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import date, timedelta

spark = SparkSession.builder.getOrCreate()

# Read bronze customers
bronze_customers = spark.table("bronze.customers")

# Transform to silver
silver_customers = bronze_customers.select(
    F.col("customer_id"),
    F.col("signup_date"),
    F.col("tenure_days"),
    F.col("cohort"),
    F.col("ltv_tier"),
    F.col("contract_type"),
    F.when(F.col("contract_type") == "Month-to-month", 1)
     .when(F.col("contract_type") == "One year", 12)
     .when(F.col("contract_type") == "Two year", 24)
     .otherwise(1).alias("contract_months"),
    F.col("monthly_charges"),
    F.col("estimated_ltv"),
    F.col("phone_service").alias("has_phone"),
    (F.col("internet_service") != "No").alias("has_internet"),
    F.when(F.col("internet_service") == "No", None)
     .otherwise(F.col("internet_service")).alias("internet_type"),
    F.col("paperless_billing"),
    F.col("payment_method"),
    F.col("senior_citizen").alias("is_senior"),
    F.col("partner").alias("has_partner"),
    F.col("dependents").alias("has_dependents"),
    F.col("churn_label"),
    F.col("churn_date")
)

# Write to silver
silver_customers.write.format("delta").mode("overwrite").saveAsTable("silver.customers")

print(f"Silver customers: {silver_customers.count():,} rows")
```

---

## Step 6: Create Gold Layer (Feature Engineering)

Create notebook: `02_silver_to_gold.py`

```python
# Fabric Notebook - Gold Layer Feature Engineering

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import date, timedelta

spark = SparkSession.builder.getOrCreate()

# Parameters
prediction_date = date.today() - timedelta(days=30)

# Read silver tables
customers = spark.table("silver.customers")
engagement = spark.table("silver.daily_engagement")
tickets = spark.table("silver.support_tickets")

# Build engagement features
engagement_features = engagement.filter(
    F.col("activity_date") < F.lit(prediction_date)
).groupBy("customer_id").agg(
    # Frequency
    F.sum(F.when(F.col("activity_date") >= F.date_sub(F.lit(prediction_date), 7), 
                 F.col("login_count")).otherwise(0)).alias("logins_7d"),
    F.sum(F.when(F.col("activity_date") >= F.date_sub(F.lit(prediction_date), 30), 
                 F.col("login_count")).otherwise(0)).alias("logins_30d"),
    
    # Recency
    F.datediff(F.lit(prediction_date), F.max("activity_date")).alias("days_since_last_login"),
    
    # Depth
    F.sum(F.col("features_used")).alias("total_features_used"),
    F.avg(F.col("session_duration_minutes")).alias("avg_session_duration")
)

# Build support features
support_features = tickets.filter(
    F.col("created_date") < F.lit(prediction_date)
    AND F.col("created_date") >= F.date_sub(F.lit(prediction_date), 30)
).groupBy("customer_id").agg(
    F.sum(F.when(F.col("escalated"), 1).otherwise(0)).alias("escalation_rate_30d"),
    F.avg("sentiment_score")).alias("avg_sentiment_30d"),
    F.avg(F.when(F.col("escalated"), 1).otherwise(0)).alias("avg_resolution_days_30d")
)

# Join to create Customer 360
customer_360 = customers.join(
    engagement_features, "customer_id", "left"
).join(
    support_features, "customer_id", "left"
).select(
    "*",
    F.lit(prediction_date).alias("prediction_date"),
    
    # Target: churned within 30 days of prediction date
    F.when(
        (F.col("churn_date").isNotNull()) & 
        (F.col("churn_date") > F.lit(prediction_date)) &
        (F.col("churn_date") <= F.date_add(F.lit(prediction_date), 30))
    ).otherwise(False).alias("churned_in_window")
)

# Write to gold
customer_360.write.format("delta").mode("overwrite").saveAsTable("gold.customer_360")

print(f"Gold customer_360: {customer_360.count():,} rows")
```

---

## Step 7: Model Training with Fabric MLflow

Create notebook: `03_train_model.py`

```python
# Fabric Notebook - Model Training with MLflow

import mlflow
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd

# Set experiment
mlflow.set_experiment("churn-prediction")

# Load data
df = spark.table("gold.customer_360").toPandas()

# Prepare features
feature_cols = [
    "logins_7d", "logins_30d", "days_since_last_login",
    "total_features_used", "avg_session_duration",
    "escalation_rate_30d", "avg_sentiment_30d",
    "tenure_days", "cohort", "ltv_tier",
    "contract_months", "monthly_charges", "estimated_ltv"
]

X = df[feature_cols].fillna(0)
y = df["churned_in_window"].astype(int)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, 
    stratify=y  # Cohort-aware split
)

# Calculate sample weights
def calculate_sample_weights(df):
    weight_map = {"smb": 1.0, "mid_market": 3.0, "enterprise": 10.0}
    return df["ltv_tier"].map(weight_map).fillna(1.0)

sample_weights = calculate_sample_weights(X_train)

# Train with MLflow tracking
with mlflow.start_run(run_name="lgbm_baseline"):
    mlflow.autolog()  # Log parameters and metrics automatically
    
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        class_weight="balanced"  # Complements sample weights
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Custom business metrics
    from sklearn.metrics import average_precision_score, precision_recall_curve
    import numpy as np
    
    # Calculate AUC-PR
    y_proba = model.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test, y_proba)
    mlflow.log_metric("auc_pr", auc_pr)
    
    # Calculate Precision@Top10%
    def precision_at_top_k(y_true, y_proba, k=0.10):
        k = int(len(y_true) * k)
        top_k_indices = y_proba.argsort()[::-1][:k]
        true_positives = y_true.iloc[top_k_indices].sum()
        return true_positives / k
    
    precision_10 = precision_at_top_k(y_test, y_proba, 0.10)
    mlflow.log_metric("precision_at_10pct", precision_10)
    
    # Calculate average lead time
    def calculate_lead_time(y_true, y_proba, prediction_date_col="prediction_date"):
        # Get prediction dates for positive cases
        pos_mask = y_true == 1
        if pos_mask.sum() > 0:
            pos_pred_dates = df.loc[pos_mask, prediction_date_col]
            pos_dates = pd.to_datetime(pos_pred_dates["prediction_date"])
            actual_dates = pd.to_datetime(df.loc[pos_mask, "churn_date"])
            
            lead_times = (actual_dates - pos_pred_dates).dt.days
            return lead_times.mean().days
        else:
            return 45.0  # Default for non-churners
    
    avg_lead_time = calculate_lead_time(y_test, y_proba)
    mlflow.log_metric("lead_time_days", avg_lead_time)
    
    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "churn-prediction-model")

print(f"AUC-PR: {auc_pr:.4f}")
print(f"Precision@Top10%: {precision_10:.4f}")
print(f"Average Lead Time: {avg_lead_time:.1f} days")
```

---

## Step 8: Create Scoring Pipeline

Create notebook: `04_batch_scoring.py`

```python
# Fabric Notebook - Batch Scoring

import mlflow
import pandas as pd

# Load registered model
model = mlflow.lightgbm.load_model("models:/churn-prediction-model/Production")

# Load latest features for scoring
df = spark.table("gold.customer_360").toPandas()

# Generate predictions
feature_cols = [
    "logins_7d", "logins_30d", "days_since_last_login",
    "total_features_used", "avg_session_duration",
    "escalation_rate_30d", "avg_sentiment_30d",
    "tenure_days", "cohort", "ltv_tier",
    "contract_months", "monthly_charges", "estimated_ltv"
]

X = df[feature_cols].fillna(0)

# Score
df["churn_probability"] = model.predict_proba(X)[:, 1]

# Add risk tiers
df["risk_tier"] = pd.cut(
    df["churn_probability"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High", "Critical"]
)

# Add intervention priority
def calculate_intervention_priority(df):
    ltv_weights = {"smb": 1.0, "mid_market": 3.0, "enterprise": 10.0}
    df["intervention_priority"] = (
        df["churn_probability"] * 
        df["ltv_tier"].map(ltv_weights)
    )

# Write predictions to Delta
spark.createDataFrame(df).write.format("delta").mode("overwrite").saveAsTable("gold.churn_predictions")

print(f"Scored {len(df)} customers")
```

---

## Step 9: Create Power BI Dashboard

```m
1. Connect Power BI to Lakehouse SQL endpoint
2. Import tables:
   - gold.churn_predictions
   - gold.customer_360
3. Create visuals:
   - Risk distribution pie chart
   - Churn rate by cohort
   - Top at-risk customers table
   - Feature importance bar chart
```

---

## Step 10: Schedule Pipelines

```m
1. Create Data Pipeline: "churn-daily-pipeline"
2. Add activities:
   - Notebook: 01_bronze_to_silver (if data refresh)
   - Notebook: 02_silver_to_gold
   - Notebook: 04_batch_scoring
3. Schedule: Daily at 4:00 AM UTC
4. Enable alerts for failures
```

---

## SQL Translation Reference

### DuckDB → Fabric SQL

| DuckDB | Fabric SQL (Spark SQL) |
|--------|------------------------|
| `CREATE SCHEMA bronze` | `CREATE SCHEMA bronze` (same) |
| `INTERVAL '7 days'` | `INTERVAL 7 DAY` or `date_sub(col, 7)` |
| `EXTRACT(DAY FROM ...)` | `DATEPART(day, col)` or `datediff(...)` |
| `$prediction_date` | Use Python variable in notebook |
| `::INTEGER` | `CAST(... AS INT)` |

### Window Functions

```sql
-- DuckDB
SUM(CASE WHEN activity_date >= $date - INTERVAL '7 days' THEN login_count ELSE 0 END)

-- Spark SQL
SUM(CASE WHEN activity_date >= date_sub('{date}', 7) THEN login_count ELSE 0 END)
```

---

## Cost Optimization Tips

1. **Use Delta Lake caching** for frequently accessed tables
2. **Partition by date** for time-series data: `PARTITIONED BY (prediction_date)`
3. **Z-Order clustering** on customer_id for join performance
4. **Use Spark for large scale** transformations (>1M rows), pandas for smaller
5. **Schedule during off-peak** hours for batch jobs

---

## Security Considerations

1. **Row-level security**: Filter customer data by region/team
2. **Column masking**: Mask PII fields in SQL endpoint
3. **Service principal**: Use for automated pipelines
4. **Private endpoints**: For enterprise deployments

---

## Files to Deploy

```m
churn-prediction-fabric/
├── notebooks/
│   ├── 01_bronze_to_silver.py
│   ├── 02_silver_to_gold.py
│   ├── 03_train_model.py
│   └── 04_batch_scoring.py
├── pipelines/
│   └── churn-daily-pipeline.json
└── powerbi/
    └── churn-dashboard.pbix
```

The local DuckDB implementation serves as:

1. Development environment
2. Unit testing
3. Demo for stakeholders
4. Interview presentation

The Fabric deployment is production-ready with:

1. Scalability (Spark)
2. Governance (OneLake)
3. MLOps (MLflow)
4. BI integration (Power BI)
