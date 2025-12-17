# Technical Context: Churn Prediction System

## Technology Stack Overview

This project uses a modern, production-ready stack designed for scalability, reproducibility, and seamless Microsoft Fabric integration.

### Core Technologies

| Component | Technology | Version | Rationale |
|-----------|-------------|---------|-----------|
| **Language** | Python | 3.11+ | ML ecosystem, performance, Fabric compatibility |
| **Package Manager** | uv | Latest | Fast dependency resolution, lockfile management |
| **Data Layer** | DuckDB | 1.0.0+ | SQL-native, portable, 1:1 Fabric mapping |
| **ML Framework** | LightGBM | 4.0.0+ | SOTA for tabular, native imbalance handling |
| **Explainability** | SHAP | 0.43.0+ | Tree SHAP optimization, industry standard |
| **Tracking** | MLflow | 2.10.0+ | Fabric-managed, experiment tracking |
| **Notebooks** | Marimo | 0.6.0+ | Pure Python, reactive, Git-friendly |
| **Quality** | ruff + pytest | Latest | Fast linting, comprehensive testing |

## Development Environment

### Setup Requirements

```bash
# Prerequisites
- Python 3.11+
- uv (package manager)
- Git

# Quick start
git clone <repo>
cd avepoint-churn-exercise
uv sync                    # Install all dependencies
uv run marimo run notebooks/01_eda.py  # Start exploring
```

### Development Workflow

1. **Local Development**: DuckDB in-memory for rapid iteration
2. **Experimentation**: Marimo notebooks for interactive exploration
3. **Code Quality**: ruff for linting, pytest for testing
4. **Version Control**: Git-friendly pure Python notebooks
5. **Reproducibility**: uv lockfile ensures consistent environments

### Key Development Patterns

#### Configuration Management
```yaml
# config/model_config.yaml - Single source of truth
data:
  n_customers: 50000
  cohorts:
    new_user:
      observation_window_days: 14
      prediction_horizon_days: 30
      base_churn_rate: 0.25
```

#### Modular Architecture
```
src/
├── data/          # Data generation and schema
├── features/      # Feature engineering and leakage audit
├── model/         # Training, evaluation, explanation
└── utils/         # Lakehouse, temporal split, utilities
```

## Data Architecture

### DuckDB Medallion Architecture

**Why DuckDB?**
- **SQL Native**: Complex window functions for cohort-aware features
- **Portable**: Single file database, easy version control
- **Fabric Mapping**: 1:1 translation to Synapse SQL
- **Performance**: Columnar storage, vectorized execution
- **No Dependencies**: Self-contained, no external services

#### Layer Definitions

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Bronze    │───▶│   Silver    │───▶│    Gold     │
│  Raw Events │    │  Cleaned    │    │  Features   │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Bronze Layer**: Raw data as-is from sources
- `bronze.customers`: Raw customer master data
- `bronze.events`: Raw behavioral events
- `bronze.support_tickets`: Raw support interactions

**Silver Layer**: Cleaned, validated, deduplicated
- `silver.customers`: Customer data with tenure, cohorts, LTV tiers
- `silver.daily_engagement`: Sessionized engagement metrics
- `silver.support_tickets`: Cleaned support interactions

**Gold Layer**: Feature tables, ML-ready
- `gold.customer_360`: Point-in-time feature matrix for ML

### Microsoft Fabric Integration

**Translation Strategy**: Every DuckDB pattern maps to Fabric equivalent

| DuckDB Concept | Fabric Equivalent | Translation Complexity |
|----------------|-------------------|------------------------|
| Database | Lakehouse | 1:1 |
| Schema | Schema/Lakehouse folder | 1:1 |
| Table | Delta Table | 1:1 |
| SQL Query | Synapse SQL | Nearly identical |
| Parquet Files | Delta Files | Format conversion |

**Deployment Pattern**:
```python
# Local: DuckDB
lakehouse = DuckDBLakehouse("local.duckdb")

# Fabric: OneLake
lakehouse = FabricLakehouse("fabric_workspace/lakehouse")
```

## Machine Learning Pipeline

### Algorithm Selection: LightGBM

**Why LightGBM over alternatives?**

| Criterion | LightGBM | Logistic Regression | Neural Network |
|-----------|----------|---------------------|----------------|
| Mixed features | ✅ Native | Requires encoding | Requires encoding |
| Non-linear | ✅ Native | ❌ Manual | ✅ Native |
| Interpretability | ✅ SHAP | ✅ Coefficients | ❌ Black box |
| Tabular SOTA | ✅ Yes | ⚠️ Baseline | ⚠️ Underperforms |
| Class imbalance | ✅ Native weights | ✅ Weights | ⚠️ Requires tuning |
| Training speed | ✅ Fast | ✅ Fast | ❌ Slow |

**Key Configuration**:
```python
lgb.LGBMClassifier(
    objective='binary',
    metric='average_precision',  # Better for imbalanced
    num_leaves=31,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    class_weight='balanced',  # Native imbalance handling
    random_state=42
)
```

### Temporal Validation Strategy

**Why not Random Cross-Validation?**
- **Leakage Prevention**: Future data cannot train on past
- **Realistic Performance**: Simulates production deployment
- **Concept Drift Detection**: Multiple time periods test stability

**Time Series CV Design**:
```
Historical Data                              Holdout
◄──────────────────────────────────────────►◄───────────────►

Fold 1: Train [M1-M3] → Test [M4]
Fold 2: Train [M1-M4] → Test [M5]
...
Fold 6: Train [M1-M8] → Test [M9]
Final:  Train [M1-M9] → Evaluate [M10-M12]
```

### Class Imbalance Strategy

**Multi-layered Approach**:
1. **Sample Weights**: LTV-tier weighted during training
2. **Threshold Tuning**: Business-optimal cutoff selection
3. **Metric Focus**: Precision@Top10% over overall accuracy

**Weight Calculation**:
```python
sample_weights = np.where(
    df['ltv_tier'] == 'enterprise', 10.0,
    np.where(df['ltv_tier'] == 'mid_market', 3.0, 1.0)
)
```

## Feature Engineering Architecture

### Cohort-Aware Design

**Why Cohort-Specific Features?**
Different lifecycle stages have different predictive signals:

| Cohort | Key Features | Time Windows |
|--------|-------------|--------------|
| **New User** | Activation metrics | 7d, 14d |
| **Established** | Engagement patterns | 14d, 30d |
| **Mature** | Renewal signals | 30d, 60d |

### Point-in-Time Correctness

**Leakage Prevention Protocol**:
1. **Feature Timestamp**: Every feature has known availability time
2. **Window Respect**: No future data in observation windows
3. **Target Separation**: Churn labels always after prediction point
4. **Audit Trail**: Formal verification for every feature

**Example Audit**:
```
Feature: logins_30d
Source: daily_engagement.login_count
Window: prediction_date - 30 days TO prediction_date
Available: Yes (historical data)
Leakage Risk: None ✅
```

### Feature Categories

#### Activation Features (New Users Only)
- `days_to_first_login`: Time to engagement
- `onboarding_completion_pct`: Setup progress
- `first_week_logins`: Early engagement intensity

#### Engagement Features (All Users)
- `logins_7d/14d/30d`: Frequency patterns
- `features_used_30d`: Adoption breadth
- `session_minutes_30d`: Usage intensity

#### Velocity Features (All Users)
- `login_velocity_wow`: Week-over-week change
- `feature_velocity_wow`: Feature adoption change
- `login_trend_4w`: 4-week engagement trend

#### Support Features (All Users)
- `tickets_30d`: Support interaction frequency
- `avg_sentiment_30d`: Customer satisfaction
- `escalation_rate_30d`: Issue severity

## Model Explainability

### SHAP Integration

**Why SHAP?**
- **Tree SHAP**: Optimized for LightGBM, exact values
- **Local & Global**: Individual predictions + overall patterns
- **Actionable**: Maps features to business drivers
- **Visual**: Clear plots for stakeholder communication

**Implementation**:
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test)

# Individual explanation
shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])
```

### Intervention Mapping

**SHAP → Action Translation**:

| SHAP Driver | Business Meaning | Recommended Action |
|-------------|------------------|-------------------|
| `login_velocity_wow < 0` | Usage declining | Re-engagement campaign |
| `features_used_30d ↓` | Low adoption | Feature training session |
| `tickets_30d ↑` | Product friction | Proactive support check |
| `days_to_renewal < 30` | Contract ending | Strategic account review |

## Monitoring & Production

### MLflow Integration

**Minimal, Fabric-Native Tracking**:
```python
import mlflow

mlflow.set_experiment("churn-prediction")

with mlflow.start_run():
    mlflow.autolog()  # Auto-track params, metrics
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, sample_weight=weights)
    
    # Business metrics
    mlflow.log_metric("precision_at_10pct", precision_10)
    mlflow.log_metric("lead_time_days", lead_time)
```

### Three-Pillar Monitoring

| Pillar | Metrics | Threshold | Action |
|--------|---------|-----------|--------|
| **Data Quality** | Feature freshness | <24h | Pause scoring |
| | Missing rate | >5% or >2x baseline | Investigate |
| | PSI (distribution shift) | >0.2 | Retrain |
| **Model Health** | Prediction drift (KS) | >0.1 | Investigate |
| | Calibration (Brier) | >20% increase | Recalibrate |
| | Cohort AUC-PR | <0.4 any cohort | Retrain |
| **Business Impact** | Intervention rate | <80% | CS operations |
| | Save rate | <30% | Strategy review |
| | Lead time accuracy | <30 days | Retrain |

## Deployment Architecture

### Local → Fabric Translation

**Development Workflow**:
1. **Local Development**: DuckDB + Marimo + uv
2. **Validation**: Temporal CV, leakage audit, performance checks
3. **Translation**: SQL queries port to Fabric Synapse
4. **Deployment**: MLflow models to Fabric ML
5. **Monitoring**: Fabric-native monitoring tools

### CI/CD Integration

**Quality Gates**:
```bash
# Pre-commit hooks
uv run ruff check src/           # Code quality
uv run pytest tests/ -v         # Test suite
uv run python -m src.data.leakage_audit  # Temporal correctness
```

**Deployment Pipeline**:
1. **Code Review**: Architecture and logic validation
2. **Automated Tests**: Unit, integration, temporal tests
3. **Shadow Mode**: Model runs in parallel with champion
4. **A/B Test**: Statistical validation of business impact
5. **Promotion**: Full deployment with rollback capability

## Performance & Scalability

### Optimizations

**Data Layer**:
- DuckDB columnar storage for fast aggregations
- Appropriate indexing on customer_id, timestamps
- Materialized views for complex window functions

**Model Training**:
- LightGBM's native parallelization
- Early stopping to prevent overfitting
- Feature selection based on SHAP importance

**Inference**:
- Batch scoring for efficiency
- Cached SHAP explanations for common patterns
- Pre-computed aggregates for feature engineering

### Scaling Considerations

**From 10K to 1M+ Customers**:
- DuckDB → Fabric Synapse for distributed query
- Single node → Distributed training if needed
- Daily scoring → Real-time scoring for high-value accounts
- Manual monitoring → Automated alerting and retraining

**Resource Requirements**:
- **Development**: 8GB RAM, 4 CPU cores sufficient
- **Production**: Scales with customer base, Fabric handles load
- **Storage**: Parquet/Delta format for columnar efficiency

**This technical architecture ensures the system is both sophisticated enough for the exercise and practical enough for real-world deployment at scale.**
