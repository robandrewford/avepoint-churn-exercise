# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready ML pipeline for SaaS customer churn prediction, demonstrating best practices for the AvePoint Principal Applied Scientist interview exercise. The system uses DuckDB-based medallion architecture that maps 1:1 to Microsoft Fabric Synapse.

## Development Commands

### Environment Setup
```bash
# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Data Generation
```bash
# Generate synthetic data (50K customers with behavioral events)
uv run python -m src.data.generate_synthetic

# Outputs to:
# - outputs/synthetic_data/customers.parquet
# - outputs/synthetic_data/daily_engagement.parquet
# - outputs/synthetic_data/support_tickets.parquet
# - outputs/synthetic_data/login_events.parquet
```

### Running Notebooks
```bash
# EDA notebook (exploratory data analysis)
uv run marimo run notebooks/01_eda.py

# Modeling notebook (training and evaluation)
uv run marimo run notebooks/02_modeling.py

# Monitoring dashboard
uv run marimo run notebooks/03_monitoring.py

# Edit notebooks
uv run marimo edit notebooks/01_eda.py
```

### Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run single test file
uv run pytest tests/test_specific.py -v
```

### Code Quality
```bash
# Run ruff linter
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Architecture Overview

### Medallion Data Architecture (DuckDB → Fabric Translation)

The project uses a three-tier medallion architecture in DuckDB that maps directly to Microsoft Fabric:

**Bronze Layer (Raw Data)**
- `bronze.customers`: Raw customer records
- `bronze.events`: Raw behavioral events
- `bronze.support_tickets`: Raw support ticket data

**Silver Layer (Cleaned/Validated)**
- `silver.customers`: Transformed customers with cohort assignments, tenure calculations, and LTV tiers
- `silver.daily_engagement`: Aggregated daily engagement metrics
- `silver.support_tickets`: Cleaned support data with sentiment scores

**Gold Layer (ML-Ready Features)**
- `gold.customer_360`: Point-in-time feature table combining all sources

All SQL in `src/utils/duckdb_lakehouse.py` translates directly to Fabric Synapse SQL.

### Cohort-Aware Feature Engineering

The system uses cohort-based prediction windows:

- **New Users** (0-30 days): 14-day observation window, 30-day prediction horizon
- **Established** (31-180 days): 30-day observation window, 30-day prediction horizon
- **Mature** (180+ days): 60-day observation window, 90-day prediction horizon

Features are engineered per cohort in `src/features/engineering.py`.

### Core Modules

**`src/data/`**
- `generate_synthetic.py`: Synthetic data generator with realistic churn patterns
- `schema.py`: DuckDB DDL schemas for bronze/silver/gold layers

**`src/features/`**
- `engineering.py`: Cohort-aware feature engineering with activation, engagement, velocity, support, and contract features
- `leakage_audit.py`: Temporal leakage detection to ensure point-in-time correctness

**`src/model/`**
- `train.py`: LightGBM training with MLflow tracking (Fabric-compatible)
- `evaluate.py`: Business metrics evaluation (precision@10%, recall, lead time)
- `explain.py`: SHAP-based global and local explanations

**`src/utils/`**
- `duckdb_lakehouse.py`: Medallion architecture implementation with methods to load data, transform layers, and build the Customer 360 feature table
- `temporal_split.py`: Time-series cross-validation splitter

### Feature Categories

All features are point-in-time correct to prevent temporal leakage:

1. **Activation Features** (new users only): days_to_first_login, onboarding_completion_pct, first_week_logins
2. **Engagement Features**: login counts (7d/14d/30d), features_used, session_minutes, feature_adoption_pct
3. **Recency Features**: days_since_last_login, days_since_last_feature_use
4. **Velocity Features**: login_velocity_wow, feature_velocity_wow, login_trend_4w (week-over-week change)
5. **Support Features**: tickets_30d, avg_sentiment_30d, escalation_rate_30d
6. **Contract Features**: days_to_renewal, contract_value_remaining, monthly_charges, estimated_ltv

### LTV-Weighted Training

The model uses sample weights based on LTV tiers:
- **SMB**: 1.0x weight (60% of customers, $200-800/month)
- **Mid-Market**: 3.0x weight (30% of customers, $1500-5000/month)
- **Enterprise**: 10.0x weight (10% of customers, $10K-30K/month)

This ensures the model prioritizes correctly predicting high-value customer churn.

## Configuration

`config/model_config.yaml` contains:
- Data generation parameters (customer count, churn rates by cohort)
- Cohort definitions (tenure thresholds, observation windows)
- Feature engineering windows (short/medium/long)
- LightGBM hyperparameters
- Evaluation targets (AUC-PR > 0.5, Precision@10% > 0.70)
- MLflow settings

## Key Implementation Patterns

### Point-in-Time Correctness

All feature engineering uses a `prediction_date` parameter to ensure temporal correctness:

```python
# Building Customer 360 features for a specific prediction date
lakehouse.build_customer_360(prediction_date=date(2024, 12, 1))

# Feature queries use: WHERE activity_date < $prediction_date
# Target labels use: WHERE churn_date > $prediction_date AND churn_date <= $prediction_date + INTERVAL '30 days'
```

### DuckDB Lakehouse Usage

```python
from src.utils.duckdb_lakehouse import DuckDBLakehouse

# Initialize lakehouse
lakehouse = DuckDBLakehouse("outputs/churn_lakehouse.duckdb")
lakehouse.initialize_schemas()

# Load bronze data
lakehouse.load_bronze_customers(customers_df)

# Transform to silver
lakehouse.transform_to_silver_customers()

# Build gold features
lakehouse.build_customer_360(prediction_date=date(2024, 12, 1))

# Query features
features_df = lakehouse.get_customer_360(prediction_date=date(2024, 12, 1))
```

### MLflow Integration

The training code uses MLflow tracking compatible with Fabric:

```python
import mlflow

mlflow.set_experiment("churn-prediction")
mlflow.autolog()  # Auto-logs LightGBM params, metrics, model

with mlflow.start_run():
    model = lgb.LGBMClassifier(**config["model"]["lgbm"])
    model.fit(X_train, y_train, sample_weight=weights)
    mlflow.log_metrics({"auc_pr": auc_pr, "precision_at_10pct": p10})
```

## Fabric Deployment Translation

| Local (DuckDB) | Fabric Equivalent |
|----------------|-------------------|
| DuckDB schemas | Lakehouse schemas |
| Parquet files | Delta tables |
| Python scripts | Synapse notebooks |
| MLflow (local) | Fabric-managed MLflow |
| Marimo notebooks | Fabric notebooks |
| `duckdb_lakehouse.py` SQL | Synapse SQL (nearly identical) |

## Important Notes

- **Marimo Notebooks**: Use `marimo run` to execute, `marimo edit` to modify. Marimo notebooks are pure Python files with reactive execution.
- **Temporal Leakage**: Always verify features don't use future information. Use `src/features/leakage_audit.py` to validate.
- **Sample Weights**: Training uses LTV-based sample weights (`sample_weight` column in gold.customer_360).
- **Evaluation Metrics**: Focus on precision@10% (CS team capacity constraint) and lead time > 45 days (intervention window).
- **Database Location**: Default DuckDB file is `outputs/churn_lakehouse.duckdb`. Can be changed via config.

## Project Context

This is an interview exercise demonstrating:
1. Production ML engineering practices
2. Understanding of SaaS churn dynamics
3. Cohort-aware modeling approaches
4. Microsoft Fabric compatibility
5. MLOps best practices (tracking, monitoring, temporal correctness)

See `docs/00_EXERCISE_PLAN.md` for full problem statement and `README.md` for quick start guide.
