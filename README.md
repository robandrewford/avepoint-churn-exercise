# avepoint-churn-exercise

Repo for the exercise deliverables

## Churn Prediction - AvePoint Principal Applied Scientist Exercise

A production-ready churn prediction system demonstrating ML engineering best practices, designed for the AvePoint Principal Applied Scientist interview exercise.

## 🎯 Overview

This project implements a complete ML pipeline for SaaS customer churn prediction:

- **Problem Framing**: Multi-tier churn taxonomy with cohort-aware prediction windows
- **Data Engineering**: DuckDB-based medallion architecture (maps 1:1 to Microsoft Fabric)
- **Feature Engineering**: Cohort-aware features with formal leakage audit
- **Modeling**: LightGBM with LTV-weighted cost-sensitive learning
- **Explainability**: SHAP-based global and local explanations
- **Production**: MLflow tracking, monitoring framework, deployment patterns

## 📁 Project Structure

```m
churn_prediction/
├── pyproject.toml          # uv dependencies
├── README.md
├── docs/
│   └── PLAN.md                    # Comprehensive exercise plan
│   └── FABRIC_DEPLOYMENT.md       # Fabric translation guide
├── config/
│   └── model_config.yaml          # Model and data configuration
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py  # Synthetic data generator
│   │   └── schema.py              # DuckDB DDL schemas
│   ├── features/
│   │   ├── engineering.py         # Feature engineering
│   │   └── leakage_audit.py       # Temporal leakage detection
│   ├── model/
│   │   ├── train.py               # Training with MLflow
│   │   ├── evaluate.py            # Business metrics
│   │   └── explain.py             # SHAP explanations
│   └── utils/
│       ├── duckdb_lakehouse.py    # Medallion architecture
│       └── temporal_split.py      # Time-series CV
├── notebooks/                     # Marimo notebooks (pure Python)
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   └── 03_monitoring.py
├── tests/
└── outputs/
    ├── synthetic_data/            # Generated Parquet files
    ├── models/                    # Saved models
    └── figures/                   # Visualizations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd churn_prediction

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Generate Synthetic Data

```bash
# Generate 50K customers with behavioral events
uv run python -m src.data.generate_synthetic

# Output:
# - outputs/synthetic_data/customers.parquet
# - outputs/synthetic_data/daily_engagement.parquet
# - outputs/synthetic_data/support_tickets.parquet
# - outputs/synthetic_data/login_events.parquet
```

### Run Notebooks

```bash
# EDA notebook
uv run marimo run notebooks/01_eda.py

# Modeling notebook
uv run marimo run notebooks/02_modeling.py

# Monitoring dashboard
uv run marimo run notebooks/03_monitoring.py
```

## 🏗️ Architecture

### Data Layer: DuckDB Medallion

```m
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Bronze    │───▶│   Silver    │───▶│    Gold     │
│  Raw Events │    │  Cleaned    │    │  Features   │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Fabric Translation**: Each DuckDB schema maps to a Fabric Lakehouse. SQL is nearly identical.

### Feature Engineering

| Category | Features | Cohort |
|----------|----------|--------|
| Activation | Time-to-first-value, onboarding % | New users only |
| Engagement | Login frequency, feature adoption | All |
| Velocity | Week-over-week change, trend slope | All |
| Support | Ticket volume, sentiment, escalations | All |
| Contract | Days to renewal, value remaining | Mature |

## 📊 Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-PR | > 0.50 | Precision-recall area (imbalanced) |
| Precision@10% | > 0.70 | CS team capacity constraint |
| Recall@30d | > 0.60 | Coverage requirement |
| Lead Time | > 45 days | Intervention window |
| Lift@10% | > 3.0x | Better than random |

## 📈 Microsoft Fabric Integration

This project is designed to translate directly to Fabric:

| Local (DuckDB) | Fabric Equivalent |
|----------------|-------------------|
| DuckDB schemas | Lakehouse schemas |
| Parquet files | Delta tables |
| Python scripts | Synapse notebooks |
| MLflow (local) | Fabric MLflow |
| Marimo | Fabric notebooks |

## 🧪 Testing

```bash
# Run tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

- **[PLAN.md](docs/PLAN.md)**: Complete exercise plan (Parts 1-5)
- **[model_config.yaml](config/model_config.yaml)**: Configuration reference

## 🎤 Interview Presentation

30-minute structure:

- 0:00-0:02: Opening ("Churn is solvable")
- 0:02-0:06: Problem Framing
- 0:06-0:10: Data & Features
- 0:10-0:17: Modeling (technical depth)
- 0:17-0:23: Recommendations (business impact)
- 0:23-0:28: Mentorship & Scale
- 0:28-0:30: Close + Q&A

## 📋 Key Differentiators

1. **Cohort-Aware**: Different prediction windows for New/Established/Mature users
2. **LTV-Weighted**: Enterprise churns weighted 10x vs SMB
3. **Leakage Audit**: Formal temporal correctness verification
4. **Uplift Framework**: Prioritize "Persuadables" over "Sure Things"
5. **Production-Ready**: Monitoring, retraining triggers, escalation paths

## License

MIT
