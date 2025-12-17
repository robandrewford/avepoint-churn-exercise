# Product Context: Churn Prediction System

## Why This Project Exists

### Business Problem

Churn is the single largest revenue leak in enterprise SaaS, yet most churn prediction systems fail because they treat all customers the same. Traditional approaches miss critical nuances:

- **One-size-fits-all modeling** ignores that new users churn for different reasons than mature customers
- **Equal-weight learning** treats a $500/month SMB the same as a $15,000/month enterprise account
- **Static features** don't capture the temporal patterns that precede churn decisions
- **Accuracy-focused metrics** ignore the operational constraints of customer success teams

**Core Insight**: Churn is not a binary event - it's a lifecycle phenomenon that requires cohort-aware, temporally correct, and business-weighted modeling.

### AvePoint-Specific Context

This project is specifically designed for AvePoint's enterprise SaaS profile:

- **Complex Setup**: AvePoint products require configuration and setup, creating an "activation" phase where early engagement is critical
- **Measurable Usage**: M365 and SharePoint integration provides rich behavioral data that can be tracked over time
- **Enterprise Sales Cycle**: Long sales cycles with predictable renewal patterns create opportunities for proactive intervention
- **Multi-Product Ecosystem**: Customers using multiple products have different engagement patterns and churn dynamics

## How the System Should Work

### User Experience Goals

#### For Customer Success Teams
- **Prioritized Outreach**: Clear, ranked list of at-risk accounts with intervention recommendations
- **Actionable Insights**: Not just "who will churn" but "why they'll churn and what to do about it"
- **Capacity-Aware**: Top N recommendations where N matches team capacity, not overwhelming with alerts
- **ROI-Focused**: Prioritization based on expected revenue impact, not just churn probability

#### For Executive Leadership
- **Financial Impact**: Clear connection between model predictions and revenue at risk
- **Strategic Insights**: Understanding of churn drivers across customer lifecycle stages
- **Performance Tracking**: Monitoring of intervention effectiveness and model business impact
- **Scalable Framework**: Architecture that can grow from 10K to 1M+ customers

### Decision Support Framework

```
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

### Operational Workflow

1. **Daily Scoring**: All customers scored with cohort-appropriate models
2. **SHAP Explanation**: Each prediction includes feature-level explanations
3. **Intervention Mapping**: SHAP drivers mapped to specific playbook recommendations
4. **Capacity Assignment**: Top N accounts assigned to CS team members
5. **Outcome Tracking**: Intervention results fed back to improve future targeting

## Target User Personas

### Primary: Customer Success Manager
- **Needs**: Clear priorities, actionable insights, efficient workflow integration
- **Pain Points**: Too many alerts, unclear action steps, overwhelmed by volume
- **Success Metrics**: Customer retention rates, time to intervene, customer satisfaction

### Secondary: VP of Customer Success
- **Needs**: Team performance insights, revenue impact reporting, strategic churn trends
- **Pain Points**: Lack of visibility into intervention effectiveness, difficulty justifying headcount
- **Success Metrics**: Team productivity, revenue saved, churn rate reduction

### Tertiary: Chief Data Officer
- **Needs**: Model performance monitoring, ROI justification, scalable architecture
- **Pain Points**: Black-box models, production maintenance overhead, business impact attribution
- **Success Metrics**: Model uptime, business KPI improvement, cost of ownership

## Customer Journey Mapping

### Activation Phase (Days 0-14)
**Critical Period**: 70% of customers who churn in first 30 days never reach their first value moment.

**Key Indicators**:
- Time to first login
- Onboarding completion percentage
- Feature setup progress
- Support ticket patterns

**Intervention Triggers**:
- No login by Day 3: Automated check-in
- <50% onboarding by Day 7: Guided tour
- No value moment by Day 14: CSM outreach

### Engagement Phase (Days 15-90)
**Establishment Period**: Building habits and demonstrating ongoing value.

**Key Indicators**:
- Login frequency trends
- Feature adoption velocity
- Usage depth progression
- Session quality metrics

**Intervention Triggers**:
- Declining velocity for 2+ weeks: Re-engagement campaign
- Low feature adoption: Advanced feature training
- Support ticket spikes: Proactive check-in

### Retention Phase (Days 90+)
**Maturity Period**: Maintaining value and identifying expansion opportunities.

**Key Indicators**:
- Long-term engagement patterns
- Feature utilization breadth
- Support dependency ratio
- Renewal risk signals

**Intervention Triggers**:
- 90 days pre-renewal: Strategic account review
- Engagement decline: Value recap session
- High support usage: Efficiency consultation

## Business Impact Framework

### Financial Quantification

**Churn Cost Formula**:
```
Churn Cost = (Remaining Contract Value) + (Lost Expansion Revenue) + (Replacement Cost)
```

**Segment Impact**:

| Segment | Avg Monthly Revenue | Avg Remaining Term | Churn Cost/Customer | Weight in Model |
|---------|--------------------|--------------------|---------------------|-----------------|
| SMB | $500 | 6 months | $3,600 | 1.0x |
| Mid-Market | $2,500 | 9 months | $33,750 | 3.0x |
| Enterprise | $15,000 | 12 months | $324,000 | 10.0x |

**ROI Calculation**:
- **Intervention Cost**: $50 per customer (CS time, resources)
- **Save Rate**: 40% (historical intervention effectiveness)
- **Break-even**: Prevent 1 enterprise churn or 90 SMB churns per quarter

### Success Metrics Hierarchy

#### Level 1: Business Outcomes
- **Churn Rate Reduction**: Target -15% in 6 months
- **Revenue Protected**: $500K/quarter saved from churn prevention
- **Customer Lifetime Value**: +25% increase through better retention

#### Level 2: Operational Metrics
- **Intervention Efficiency**: >40% of interventions prevent churn
- **Lead Time**: >45 days average warning before churn
- **CS Team Productivity**: 2x more effective customer engagement

#### Level 3: Model Performance
- **Precision@Top10%**: >70% (matches CS team capacity)
- **Recall@30d**: >60% (coverage requirement)
- **AUC-PR**: >0.50 (better than random baseline)

## Competitive Differentiators

### vs. Traditional Churn Models
1. **Cohort Awareness**: Different models for different lifecycle stages
2. **Temporal Correctness**: Point-in-time feature engineering prevents leakage
3. **Business Weighting**: LTV-aware training reflects real revenue impact
4. **Actionability**: SHAP explanations mapped to specific interventions
5. **Production Ready**: Built-in monitoring and retraining framework

### vs. Off-the-Shelf Solutions
1. **AvePoint Specific**: Designed for enterprise SaaS with complex setup
2. **Fabric Native**: Seamless integration with Microsoft ecosystem
3. **Customizable**: Adaptable to changing business needs and products
4. **Transparent**: Full model explainability, no black-box decisions
5. **Cost-Effective**: No per-seat licensing, built on open-source stack

## Future Roadmap Considerations

### Phase 1: Foundation (Current)
- Cohort-aware churn prediction
- SHAP-based explainability
- Basic monitoring framework
- Microsoft Fabric integration

### Phase 2: Enhancement
- Uplift modeling for intervention selection
- Multi-product adoption patterns
- Seasonal and trend adjustments
- Automated intervention routing

### Phase 3: Intelligence
- Causal inference for attribution
- Dynamic pricing optimization
- Expansion opportunity identification
- Competitive intelligence integration

**This project establishes the foundation for a comprehensive customer intelligence platform that goes beyond churn prediction to drive overall customer growth and success.**
