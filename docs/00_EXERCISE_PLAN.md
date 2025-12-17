# AvePoint Principal Applied Scientist - ML Exercise Plan

## Executive Summary

This document outlines a comprehensive approach to churn prediction
exercise, designed for a 30-minute presentation to AvePoint's CDO,
VP of Data Science, and AI/ML engineering team.

**Technical Stack:**

- **Data Layer:** DuckDB (medallion architecture) → portable, maps 1:1 to Fabric Synapse
- **Feature Engineering:** SQL (DuckDB) + Python
- **ML Framework:** LightGBM with Fabric-managed MLflow
- **Notebooks:** Marimo (pure Python, Git-friendly, reactive)
- **Package Management:** uv + ruff

**Key Differentiators:**

1. Cohort-aware feature engineering (New/Established/Mature users)
2. LTV-weighted cost-sensitive learning
3. Formal temporal leakage audit
4. Uplift-based intervention prioritization
5. Production-ready architecture with monitoring

---

## Part 0: The Exercise

# Scenario

You’ve joined a mid-sized SaaS company as a Principal Data Scientist.

The product team is concerned about user churn and wants to understand:

- Why users are leaving?
- How to predict churn before it happens?
- What actions can improve retention?
- You may use AI-generated or synthetic data for this exercise.

## Objective

You are expected to lead the analysis and modeling effort as an individual contributor,
while also demonstrating how you would mentor a junior team member through this project.

## Project Components

### Part 1: Problem Framing

- Define the business problem
- Propose success metrics
- Identify risks and assumptions

### Part 2: Data Exploration & Feature Engineering

- Perform EDA
- Engineer predictive features
- Document your rationale

### Part 3: Predictive Modeling

Build a churn prediction model
- Justify algorithm choice
- Evaluate performance
- Address class imbalance, data leakage, and interpretability

### Part 4: Strategic Recommendations

- Present 2–3 actionable insights
- Discuss testing approaches

### Part 5: Mentorship & Scalability

- Explain how you’d mentor a junior team member
- Outline high-level deployment architecture
- Describe how you’d monitor model performance

Deliverables:

- Python Script: Clean, modular, and well-documented
- Executive Summary: 5–7 slides or 1–2 pages (PDF or slides)
- GitHub Repo: Include a README

## Part 1: Problem Framing

### Business Problem Definition

**Core Question:** Which users are at risk of churning, why, and what is
the financial impact of inaction?

### Multi-Tier Churn Taxonomy

| Churn Type | Definition | Detection Signal | Intervention Window |
|------------|------------|------------------|---------------------|
| **Contractual (Voluntary)** | Customer cancels subscription | Cancellation request | 0 days (too late) |
| **Contractual (Involuntary)** | Payment failure, lapse | Failed payment | 14-30 days |
| **Engagement Decay** | Active subscription, usage drops | <2 logins/month | 60-90 days |
| **Silent Churn** | Paying but not deriving value | Low engagement + renewal approaching | 90-120 days |

**Primary Modeling Target:** Engagement Decay → Most actionable, longest
intervention window.

### Customer Lifecycle Framework

```m
CUSTOMER LIFECYCLE STAGES
┌───────────┬───────────┬───────────┬───────────┬───────────┬────────────┐
│  Acquire  │  Activate │   Engage  │   Retain  │   Expand  │    Churn   │
│  (Day 0)  │ (Day 1-14)│(Day 15-90)│ (Day 90+) │(Month 6+) │ (Variable) │
└───────────┴───────────┴───────────┴───────────┴───────────┴────────────┘

CHURN RISK WINDOWS:
• Activation Failure: Day 1-14 (never reached first value moment)
• Engagement Decay: Day 30-90 (usage decline after initial peak)
• Renewal Risk: 90 days pre-renewal (low engagement + contract end)
```

### Cohort-Based Prediction Windows

| Cohort | Observation Period | Prediction Horizon | Rationale |
|--------|-------------------|-------------------|-----------|
| **New Users** (0-30 days) | First 14 days | Churn in days 15-45 | Early activation signals strongest |
| **Established** (30-180 days) | Rolling 30-day | Churn in next 30 days | Stable baseline enables deviation detection |
| **Mature** (180+ days) | Rolling 60-day | Churn in next 90 days | Longer patterns, renewal alignment |

### LTV-Based Impact Quantification

```m
Churn Cost = (Remaining Contract Value) + (Lost Expansion Revenue) + (Replacement Cost)
```

| Segment | Avg Monthly Revenue | Avg Remaining Term | Churn Cost/Customer |
|---------|--------------------|--------------------|---------------------|
| SMB | $500 | 6 months | $3,600 |
| Mid-Market | $2,500 | 9 months | $33,750 |
| Enterprise | $15,000 | 12 months | $324,000 |

**Key Insight:** Preventing 1 Enterprise churn = Preventing 90 SMB churns.

### Success Metrics Hierarchy

#### Primary KPIs (Executive)

| KPI | Target | Rationale |
|-----|--------|-----------|
| Churn Rate Reduction | -15% in 6 months | Direct retention impact |
| Revenue Saved | $500K/quarter | Financial justification |
| Intervention Efficiency | >40% | Operational effectiveness |

#### Model Performance KPIs

| KPI | Target | Rationale |
|-----|--------|-----------|
| Precision@Top10% | >70% | CS team capacity constraint |
| Recall@30d | >60% | Coverage requirement |
| Lead Time | >45 days | Intervention window |

### Assumptions & Risks

| Assumption | Validation | Impact if Wrong |
|------------|------------|-----------------|
| Historical behavior predicts future churn | Temporal holdout | Model has no power |
| Engagement metrics logged consistently | Data quality audit | Features unreliable |
| CS team has capacity to act | Stakeholder interview | Predictions don't convert |
| Churn is preventable | Intervention A/B test | Model accurate but useless |

| Risk | Mitigation |
|------|------------|
| Data Leakage | Strict temporal cutoffs, leakage audit |
| Class Imbalance | Class weights, threshold tuning |
| Concept Drift | Monitoring pipeline, scheduled retraining |

---

## Part 2: Data Exploration & Feature Engineering

### Feature Architecture (Lifecycle-Aligned)

```m
FEATURE TAXONOMY
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   ACTIVATION    │  │   ENGAGEMENT    │  │    RETENTION    │
│   (Day 1-14)    │  │   (Day 15-90)   │  │    (Day 90+)    │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ • Time to first │  │ • Login freq    │  │ • Days to       │
│   value moment  │  │   (7d/14d/30d)  │  │   renewal       │
│ • Onboarding %  │  │ • Feature depth │  │ • Contract val  │
│ • First-week    │  │ • Session Δ     │  │ • Expansion hx  │
│   login count   │  │ • Support rate  │  │ • Renewal hx    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │     VELOCITY FEATURES      │
              │     (Δ between windows)    │
              ├────────────────────────────┤
              │ • Login velocity           │
              │ • Feature adoption velocity│
              │ • Engagement trend (slope) │
              └────────────────────────────┘
```

### Feature Categories

| Category | Features | Churn Signal | Cohort |
|----------|----------|--------------|--------|
| **Activation** | Time-to-first-value, onboarding %, setup completion | Never activated → high churn | New only |
| **Engagement Frequency** | Logins/week, DAU/MAU, session count | Declining → decay | All |
| **Engagement Depth** | Features used, % adopted, advanced usage | Shallow → low stickiness | Established+ |
| **Engagement Velocity** | WoW Δ, trend slope, acceleration | Negative → early warning | All |
| **Support Signals** | Ticket count, sentiment, resolution time | High + negative → frustration | All |
| **Contract** | Value, days to renewal, payment failures | High value + renewal → priority | Mature |

### Cohort-Aware Window Functions

| Cohort | Windows |
|--------|---------|
| New User | 7d, 14d |
| Established | 14d, 30d |
| Mature | 30d, 60d |

### Temporal Leakage Audit Protocol

| Leakage Type | Detection | Example |
|--------------|-----------|---------|
| **Direct** | Feature timestamp > prediction point | `cancellation_requested` as feature |
| **Indirect** | Feature derived from future | `total_logins_this_month` mid-month |
| **Target** | Feature correlated by definition | `account_status = churned` |
| **Aggregation** | Window past prediction point | 30-day lookback on day 25 |

**Audit checklist for every feature:**

1. What is the timestamp of source data?
2. Is this available at prediction time in production?
3. Does aggregation window respect prediction point?
4. Would this feature exist for non-churned user?

### DuckDB Medallion Architecture

```sql
-- Bronze: Raw events
CREATE SCHEMA bronze;

-- Silver: Cleaned, sessionized
CREATE SCHEMA silver;

-- Gold: Features, Customer 360
CREATE SCHEMA gold;
CREATE TABLE gold.customer_360 AS ...
```

---

## Part 3: Predictive Modeling

### Design Decisions

#### Single Model with Cohort Features (Recommended)

- Train unified model with cohort as categorical feature
- Evaluate performance stratified by cohort
- Mention cohort-specific models as future iteration

#### Cost-Sensitive Learning

- Sample weights by LTV tier during training
- Segment-specific thresholds at inference
- Separates statistical modeling from business prioritization

#### Class Imbalance Strategy

- Class weights as primary (native to LightGBM)
- Compare against baseline and threshold tuning
- Avoid SMOTE for tree models (limited benefit)

### Temporal Validation Design

```m
TEMPORAL VALIDATION
──────────────────────────────────────────────────────
Historical Data                              Holdout
◄──────────────────────────────────────────►◄───────────────►

┌─────────────────────────────────────────┐ ┌───────────────┐
│              TRAINING                   │ │   VALIDATION  │
│         (Months 1-9)                    │ │  (Months 10-12)│
└─────────────────────────────────────────┘ └───────────────┘

TIME-SERIES CROSS-VALIDATION:
Fold 1: Train [M1-M3] → Test [M4]
Fold 2: Train [M1-M4] → Test [M5]
...
Fold 6: Train [M1-M8] → Test [M9]
Final:  Train [M1-M9] → Evaluate [M10-M12]
```

### Algorithm Selection

| Criterion | LightGBM | Logistic Regression | Neural Network |
|-----------|----------|---------------------|----------------|
| Mixed features | ✅ Native | Requires encoding | Requires encoding |
| Non-linear | ✅ Native | ❌ Manual | ✅ Native |
| Interpretability | ✅ SHAP | ✅ Coefficients | ❌ Black box |
| Tabular SOTA | ✅ Yes | ⚠️ Baseline | ⚠️ Underperforms |
| Class imbalance | ✅ Native weights | ✅ Weights | ⚠️ Requires tuning |

**Justification:** "Gradient boosted trees are empirical SOTA for tabular classification, with native class imbalance support and mature SHAP interpretability."

### Evaluation Framework

| Metric | Target | Business Rationale |
|--------|--------|-------------------|
| AUC-PR | >0.5 | Better than AUC-ROC for imbalanced |
| Precision@Top10% | >70% | CS capacity constraint |
| Recall@30d | >60% | Coverage requirement |
| Lead Time | >45 days | Intervention window |
| Calibration (Brier) | <0.1 | Trustworthy probabilities |

### Threshold Selection

```m
Business Cost Matrix:
                    Predicted: Churn    Predicted: Retain
Actual: Churn       TP: $50 cost        FN: LTV lost ($3K-$300K)
Actual: Retain      FP: $50 cost        TN: $0

Since LTV >> Intervention Cost:
→ Favor lower threshold (catch more churners)
→ But constrained by CS capacity

Practical: Flag top N where N = CS capacity, rank by (P(churn) × LTV)
```

### Fabric MLflow Integration (Minimal)

```python
import mlflow

mlflow.set_experiment("churn-prediction")

with mlflow.start_run():
    mlflow.autolog()  # Handles params, metrics, artifacts
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, sample_weight=weights)
    
    # Custom logging for business metrics
    mlflow.log_metric("precision_at_10pct", precision_10)
    mlflow.log_metric("lead_time_days", lead_time)
```

---

## Part 4: Strategic Recommendations

### SHAP → Root Cause → Intervention

| SHAP Driver | Root Cause | Intervention | Owner |
|-------------|------------|--------------|-------|
| Login frequency ↓ | Product not habit-forming | Re-engagement campaign | Marketing |
| Feature adoption ↓ | Onboarding incomplete | Guided tours, CSM outreach | Product/CS |
| Support tickets ↑ | Product friction | Prioritize fixes, proactive check-in | Support/Product |
| Engagement velocity negative | Lost momentum | "We miss you" campaign | Marketing/CS |

### Three Actionable Recommendations

#### Recommendation 1: Activation SLA for New Users

**Insight:** Onboarding completion % and time-to-first-value are top predictors. Users who don't reach value in 14 days churn at 3x rate.

**Action:** Implement "Day 14 Activation SLA"

- Day 3: Automated check-in if no login
- Day 7: Onboarding assistance if <50% complete
- Day 14: CSM outreach (Enterprise) / guided tour (SMB)

**Test Design:**

- Hypothesis: Reduces Day 30 churn by 20%
- Randomization: By account
- Sample: 500 accounts/arm
- Duration: 60 days
- Primary metric: Day 30 churn rate

#### Recommendation 2: Engagement Velocity Alert System

**Insight:** Negative velocity for 2+ weeks precedes 70% of Established User churns.

**Action:** Real-time velocity monitoring with triggered interventions

- Alert: -20% velocity for 2 consecutive weeks
- Enterprise: CSM call within 48 hours
- SMB: Automated "We noticed you've been less active" campaign

**Test Design:**

- Hypothesis: Reduces 60-day churn by 15%
- Sample: 300 accounts/arm/tier
- Duration: 90 days

#### Recommendation 3: Pre-Renewal Risk Review

**Insight:** Churn decision made 60-90 days before contract end.

**Action:** Mandatory Risk Review at 90 days pre-renewal

- High-risk: Business review meeting + ROI analysis
- Medium-risk: CSM value recap + expansion assessment

**Test Design:**

- Hypothesis: Increases renewal rate by 10%
- Sample: 200 renewals/arm
- Duration: 6 months

### Uplift Prioritization Framework

```m
                    Churn Probability
              Low ◄─────────────────► High
         ┌─────────────────┬─────────────────┐
Will     │  SURE THINGS    │  PERSUADABLES   │ ← HIGHEST ROI
Retain   │  Don't waste    │  Focus here     │
with     │  resources      │                 │
Inter-   ├─────────────────┼─────────────────┤
vention  │  SLEEPING DOGS  │  LOST CAUSES    │
Won't    │  Leave alone    │  Graceful exit  │
Retain   │                 │                 │
         └─────────────────┴─────────────────┘

Priority = P(churn) × P(save|intervention) × LTV
```

---

## Part 5: Mentorship & Scalability

### Mentorship: Graduated Ownership Model

```m
Week 1-2      Week 3-4      Week 5-6      Week 7-8
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ SHADOW  │ → │ ASSIST  │ → │  LEAD   │ → │  OWN    │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
Principal:90% Principal:60% Principal:30% Principal:10%
Junior:10%    Junior:40%    Junior:70%    Junior:90%
```

### Project Phase Mapping

| Phase | Junior's Role | My Role | Learning Objective |
|-------|---------------|---------|-------------------|
| Problem Framing | Shadow meetings, document | Lead, explain "why" | Business → technical translation |
| EDA | Execute notebook, present | Review, challenge | Data intuition |
| Modeling | Implement baseline | Design experiments | Model selection rationale |
| Recommendations | Draft one with test design | Sharpen business framing | Model → action connection |
| Deployment | Implement monitoring | Design architecture | MLOps fundamentals |

### Competency Development Matrix

| Competency | L1 (Novice) | L2 (Developing) | L3 (Competent) | L4 (Proficient) |
|------------|-------------|-----------------|----------------|-----------------|
| Data Intuition | Runs code | Notices anomalies | Proactively validates | Designs frameworks |
| Modeling | Follows tutorials | Understands tradeoffs | Selects with justification | Innovates |
| Business Translation | Technical only | Explains metrics | Connects to outcomes | Frames projects |
| Production Thinking | Notebook code | Considers reproducibility | Production-grade | Designs systems |

**Goal:** Move junior from L1-2 → L2-3 across all dimensions.

### Production Architecture

```m
DATA LAYER (DuckDB → Fabric OneLake)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Bronze    │───▶│   Silver    │───▶│    Gold     │
│  Raw Events │    │  Cleaned    │    │  Features   │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
FEATURE PIPELINE (Daily 2am UTC)
┌─────────────────────────────────────────────────────────┐
│ Extract → Transform (cohort windows) → Load → Validate │
└─────────────────────────────────────────────────────────┘
                                            │
                                            ▼
INFERENCE PIPELINE (Daily 4am UTC)
┌─────────────────────────────────────────────────────────┐
│ Load model → Score → Explain (SHAP) → Enrich → Write   │
└─────────────────────────────────────────────────────────┘
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐    ┌───────────────────┐
        │  CONSUMERS        │    │  MONITORING       │
        │  • Power BI       │    │  • Prediction drift│
        │  • CRM integration│    │  • Feature drift  │
        │  • Alerts         │    │  • Business impact│
        └───────────────────┘    └───────────────────┘
```

### Monitoring Framework: Three Pillars

| Pillar | Metrics | Threshold | Action |
|--------|---------|-----------|--------|
| **Data Quality** | Feature freshness | <24h | Pause scoring |
| | Missing rate | >5% or >2x baseline | Investigate |
| | PSI (distribution shift) | >0.2 | Investigate, retrain |
| **Model Health** | Prediction drift (KS) | >0.1 | Investigate |
| | Calibration (Brier) | >20% increase | Recalibrate |
| | Cohort AUC-PR | <0.4 any cohort | Investigate |
| **Business Impact** | Intervention rate | <80% | CS operations |
| | Save rate | <30% | Strategy review |
| | Lead time accuracy | <30 days | Model retrain |

### Model Lifecycle

```m
Development → Staging → Production → Archived

Promotion Criteria:
• Dev → Staging: AUC-PR > 0.5, Precision@10% > 0.6, code review
• Staging → Prod: 2-week shadow, A/B >= champion, stakeholder sign-off
```

### Retraining Strategy

| Trigger | Frequency | Validation |
|---------|-----------|------------|
| Scheduled | Monthly | Must beat champion |
| Performance-triggered | When AUC-PR drops >10% | Human review + A/B |
| Drift-triggered | When PSI > 0.25 | Investigate first |
| Business-triggered | Major product change | Full revalidation |

---

## Technical Implementation

### Stack Summary

| Component | Tool | Rationale |
|-----------|------|-----------|
| Data Layer | DuckDB | Portable, SQL-native, maps to Fabric |
| Features | DuckDB SQL + Python | Window functions, CTEs |
| ML | LightGBM | SOTA for tabular, SHAP support |
| Tracking | MLflow (Fabric managed) | Minimal, integrated |
| Notebooks | Marimo | Pure Python, reactive, Git-friendly |
| Package Mgmt | uv + ruff | Fast, modern |
| Visualization | Marimo + Plotly | Interactive, exportable |

### Project Structure

```m
churn_prediction/
├── pyproject.toml          # uv dependencies
├── README.md
├── docs/
│   ├── PLAN.md             # This document
│   └── FABRIC_DEPLOYMENT.md # Fabric translation guide
├── config/
│   └── model_config.yaml
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py
│   │   └── schema.py
│   ├── features/
│   │   ├── engineering.py
│   │   └── leakage_audit.py
│   ├── model/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   └── utils/
│       ├── duckdb_lakehouse.py
│       └── temporal_split.py
├── notebooks/
│   ├── 01_eda.py           # Marimo
│   ├── 02_modeling.py      # Marimo
│   └── 03_monitoring.py    # Marimo
├── tests/
└── outputs/
    ├── figures/
    └── models/
```

### Synthetic Data Design

**Base:** Kaggle Telco Churn (7,043 customers, static)

**Synthesized Extensions:**

- 50,000 customers (scaled)
- 2.5M behavioral events (logins, feature usage, support tickets)
- Temporal dimension (12 months of history)
- Cohort assignments (New/Established/Mature)
- LTV tiers (SMB/Mid-Market/Enterprise)
- Realistic churn patterns aligned to lifecycle framework

---

## Presentation Structure (30 minutes)

| Time | Section | Content |
|------|---------|---------|
| 0:00-0:02 | Opening | "Churn is solvable" |
| 0:02-0:06 | Part 1 | Problem Framing (4 min) |
| 0:06-0:10 | Part 2 | Data & Features (4 min) |
| 0:10-0:17 | Part 3 | Modeling (7 min) ← Technical depth |
| 0:17-0:23 | Part 4 | Recommendations (6 min) ← Business impact |
| 0:23-0:28 | Part 5 | Mentorship & Scale (5 min) |
| 0:28-0:30 | Close | Key takeaways |
| +10-15 min | Q&A | |

---

## AvePoint-Specific Framing

| Element | AvePoint Relevance |
|---------|-------------------|
| Activation SLA | Products require config; time-to-value depends on setup |
| Velocity Alerts | M365 usage measurable; detect when protection stops |
| Pre-Renewal Review | Enterprise SaaS cycles predictable; make it data-driven |
| Fabric Integration | Customer ecosystem; DuckDB → Fabric is 1:1 translation |
