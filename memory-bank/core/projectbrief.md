# Project Brief: AvePoint Churn Prediction Exercise

## Executive Summary

This is a comprehensive ML engineering exercise designed for the AvePoint Principal Applied Scientist interview position. The project implements a production-ready churn prediction system demonstrating advanced ML engineering best practices, designed for a 30-minute presentation to AvePoint's CDO, VP of Data Science, and AI/ML engineering team.

## Core Requirements

### Primary Deliverables

1. **Complete ML Pipeline**: End-to-end churn prediction system with production-ready architecture
2. **30-Minute Presentation**: Technical depth paired with business impact, structured for executive audience
3. **Fabric Integration**: Demonstrates Microsoft Fabric deployment capability with 1:1 mapping from local DuckDB
4. **Differentiated Approach**: Cohort-aware modeling with LTV-weighted learning that addresses enterprise SaaS realities

### Success Criteria

#### Technical Excellence
- **Model Performance**: AUC-PR > 0.5, Precision@Top10% > 0.7, Recall@30d > 0.6
- **Temporal Correctness**: Formal leakage audit with point-in-time feature engineering
- **Production Readiness**: Monitoring framework, retraining triggers, deployment patterns
- **Code Quality**: Clean architecture, comprehensive testing, proper documentation

#### Business Impact
- **Actionable Insights**: SHAP explanations mapped to specific interventions
- **ROI Framework**: LTV-weighted prioritization with uplift-based targeting
- **Strategic Recommendations**: Three concrete, testable business recommendations
- **Executive Communication**: Clear framing of technical work in business terms

## Problem Framing

### Business Question
*"Which users are at risk of churning, why, and what is the financial impact of inaction?"*

### Multi-Tier Churn Taxonomy

| Churn Type | Definition | Detection Signal | Intervention Window |
|------------|------------|------------------|---------------------|
| **Contractual (Voluntary)** | Customer cancels subscription | Cancellation request | 0 days (too late) |
| **Contractual (Involuntary)** | Payment failure, lapse | Failed payment | 14-30 days |
| **Engagement Decay** | Active subscription, usage drops | <2 logins/month | 60-90 days |
| **Silent Churn** | Paying but not deriving value | Low engagement + renewal approaching | 90-120 days |

**Primary Modeling Target**: Engagement Decay → Most actionable, longest intervention window.

### Customer Lifecycle Framework

```
CUSTOMER LIFECYCLE STAGES
┌───────────┬───────────┬───────────┬───────────┬───────────┬────────────┐
│  Acquire  │  Activate │   Engage  │   Retain  │   Expand  │    Churn   │
│  (Day 0)  │ (Day 1-14)│(Day 15-90)│ (Day 90+) │(Month 6+) │ (Variable) │
└───────────┴───────────┴───────────┴───────────┴───────────┴────────────┘
```

### Cohort-Based Prediction Strategy

| Cohort | Observation Period | Prediction Horizon | Rationale |
|--------|-------------------|-------------------|-----------|
| **New Users** (0-30 days) | First 14 days | Churn in days 15-45 | Early activation signals strongest |
| **Established** (30-180 days) | Rolling 30-day | Churn in next 30 days | Stable baseline enables deviation detection |
| **Mature** (180+ days) | Rolling 60-day | Churn in next 90 days | Longer patterns, renewal alignment |

## Technical Architecture

### Data Layer: Medallion Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Bronze    │───▶│   Silver    │───▶│    Gold     │
│  Raw Events │    │  Cleaned    │    │  Features   │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Fabric Translation**: Each DuckDB schema maps to a Fabric Lakehouse. SQL is nearly identical.

### Technology Stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| Data Layer | DuckDB | Portable, SQL-native, maps 1:1 to Fabric |
| Features | DuckDB SQL + Python | Window functions, cohort-aware engineering |
| ML Framework | LightGBM | SOTA for tabular, native class imbalance, SHAP support |
| Tracking | MLflow (Fabric managed) | Minimal integration, experiment tracking |
| Notebooks | Marimo | Pure Python, reactive, Git-friendly |
| Package Mgmt | uv + ruff | Fast, modern Python tooling |
| Visualization | Plotly | Interactive, exportable charts |

### Key Differentiators

1. **Cohort-Aware**: Different prediction windows and feature sets for New/Established/Mature users
2. **LTV-Weighted**: Enterprise churn weighted 10x vs SMB during training
3. **Leakage Audit**: Formal temporal correctness verification for every feature
4. **Uplift Framework**: Prioritize "Persuadables" over "Sure Things" and "Lost Causes"
5. **Production-Ready**: Monitoring, retraining triggers, escalation paths built-in

## Presentation Structure (30 minutes)

| Time | Section | Content Focus |
|------|---------|--------------|
| 0:00-0:02 | Opening | "Churn is solvable" - compelling hook |
| 0:02-0:06 | Problem Framing | Business context, churn taxonomy, cohort strategy |
| 0:06-0:10 | Data & Features | Medallion architecture, cohort-aware engineering |
| 0:10-0:17 | Modeling | Technical depth: LightGBM, temporal validation, metrics |
| 0:17-0:23 | Recommendations | Business impact, SHAP insights, testable interventions |
| 0:23-0:28 | Mentorship & Scale | Team development, production architecture, monitoring |
| 0:28-0:30 | Close | Key takeaways, Q&A setup |

## Success Metrics

### Model Performance KPIs
- **AUC-PR**: > 0.50 (better than random for imbalanced data)
- **Precision@Top10%**: > 0.70 (CS team capacity constraint)
- **Recall@30d**: > 0.60 (coverage requirement)
- **Lead Time**: > 45 days (intervention window)
- **Lift@10%**: > 3.0x (better than baseline)

### Business KPIs
- **Churn Rate Reduction**: -15% in 6 months
- **Revenue Saved**: $500K/quarter
- **Intervention Efficiency**: > 40% (interventions that prevent churn)

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|-------------------|
| **Data Leakage** | Strict temporal cutoffs, formal leakage audit process |
| **Class Imbalance** | LTV-weighted learning, cohort-specific thresholds |
| **Concept Drift** | Monitoring pipeline, scheduled retraining triggers |
| **Production Readiness** | Built-in monitoring, deployment patterns, CI/CD integration |

## Interview Context

This exercise demonstrates competency across the Principal Applied Scientist role:

1. **Technical Depth**: Advanced ML engineering, temporal validation, production architecture
2. **Business Acumen**: LTV thinking, ROI framing, strategic recommendations
3. **Leadership**: Mentorship framework, team development, scalable design
4. **Communication**: Executive presentation, technical translation, storytelling

**Target Audience**: CDO, VP of Data Science, AI/ML Engineering Team
**Key Message**: "Churn prediction isn't just about accuracy - it's about actionable, timely interventions that protect revenue."
