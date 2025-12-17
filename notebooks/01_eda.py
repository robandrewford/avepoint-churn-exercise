"""
Exploratory Data Analysis for Churn Prediction

Marimo notebook for interactive EDA.

Run with:
    marimo run notebooks/01_eda.py
    # Or edit mode:
    marimo edit notebooks/01_eda.py
"""

import marimo

__generated_with = "0.6.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        """
        # Churn Prediction: Exploratory Data Analysis
        
        This notebook explores the synthetic churn dataset to understand:
        1. Customer distribution by cohort and LTV tier
        2. Churn rates across segments
        3. Feature distributions
        4. Temporal patterns
        """
    )
    return


@app.cell
def __():
    import sys
    sys.path.insert(0, "..")
    
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, np, pd, px, sys


@app.cell
def __(mo):
    mo.md("## 1. Load Data")
    return


@app.cell
def __(pd):
    # Load synthetic data
    customers = pd.read_parquet("../outputs/synthetic_data/customers.parquet")
    daily_engagement = pd.read_parquet("../outputs/synthetic_data/daily_engagement.parquet")
    support_tickets = pd.read_parquet("../outputs/synthetic_data/support_tickets.parquet")
    
    print(f"Customers: {len(customers):,}")
    print(f"Daily engagement records: {len(daily_engagement):,}")
    print(f"Support tickets: {len(support_tickets):,}")
    return customers, daily_engagement, support_tickets


@app.cell
def __(mo):
    mo.md("## 2. Customer Distribution")
    return


@app.cell
def __(customers, make_subplots, go):
    # Cohort and LTV distribution
    fig_dist = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Customers by Cohort", "Customers by LTV Tier"),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )
    
    cohort_counts = customers["cohort"].value_counts()
    fig_dist.add_trace(
        go.Pie(labels=cohort_counts.index, values=cohort_counts.values, name="Cohort"),
        row=1, col=1
    )
    
    ltv_counts = customers["ltv_tier"].value_counts()
    fig_dist.add_trace(
        go.Pie(labels=ltv_counts.index, values=ltv_counts.values, name="LTV Tier"),
        row=1, col=2
    )
    
    fig_dist.update_layout(title="Customer Distribution", height=400)
    fig_dist
    return cohort_counts, fig_dist, ltv_counts


@app.cell
def __(mo):
    mo.md("## 3. Churn Analysis")
    return


@app.cell
def __(customers, px):
    # Churn rate by cohort
    churn_by_cohort = customers.groupby("cohort").agg(
        total=("customer_id", "count"),
        churned=("churn_label", "sum")
    ).reset_index()
    churn_by_cohort["churn_rate"] = churn_by_cohort["churned"] / churn_by_cohort["total"]
    
    fig_churn_cohort = px.bar(
        churn_by_cohort,
        x="cohort",
        y="churn_rate",
        color="cohort",
        title="Churn Rate by Cohort",
        text=churn_by_cohort["churn_rate"].apply(lambda x: f"{x:.1%}")
    )
    fig_churn_cohort.update_layout(yaxis_tickformat=".0%")
    fig_churn_cohort
    return churn_by_cohort, fig_churn_cohort


@app.cell
def __(customers, px):
    # Churn rate by LTV tier
    churn_by_ltv = customers.groupby("ltv_tier").agg(
        total=("customer_id", "count"),
        churned=("churn_label", "sum"),
        avg_ltv=("estimated_ltv", "mean")
    ).reset_index()
    churn_by_ltv["churn_rate"] = churn_by_ltv["churned"] / churn_by_ltv["total"]
    
    fig_churn_ltv = px.bar(
        churn_by_ltv,
        x="ltv_tier",
        y="churn_rate",
        color="ltv_tier",
        title="Churn Rate by LTV Tier",
        text=churn_by_ltv["churn_rate"].apply(lambda x: f"{x:.1%}")
    )
    fig_churn_ltv.update_layout(yaxis_tickformat=".0%")
    fig_churn_ltv
    return churn_by_ltv, fig_churn_ltv


@app.cell
def __(customers, px):
    # Churn by contract type
    churn_by_contract = customers.groupby("contract_type").agg(
        total=("customer_id", "count"),
        churned=("churn_label", "sum")
    ).reset_index()
    churn_by_contract["churn_rate"] = churn_by_contract["churned"] / churn_by_contract["total"]
    
    fig_contract = px.bar(
        churn_by_contract,
        x="contract_type",
        y="churn_rate",
        color="contract_type",
        title="Churn Rate by Contract Type",
        text=churn_by_contract["churn_rate"].apply(lambda x: f"{x:.1%}")
    )
    fig_contract
    return churn_by_contract, fig_contract


@app.cell
def __(mo):
    mo.md("## 4. Engagement Patterns")
    return


@app.cell
def __(daily_engagement, customers, px):
    # Merge engagement with customer info
    engagement_with_churn = daily_engagement.merge(
        customers[["customer_id", "churn_label", "cohort"]],
        on="customer_id"
    )
    
    # Average daily engagement by churn status
    avg_engagement = engagement_with_churn.groupby("churn_label").agg(
        avg_logins=("login_count", "mean"),
        avg_features=("features_used", "mean"),
        avg_session=("session_duration_minutes", "mean")
    ).reset_index()
    avg_engagement["churn_label"] = avg_engagement["churn_label"].map({True: "Churned", False: "Retained"})
    
    fig_engagement = px.bar(
        avg_engagement.melt(id_vars="churn_label", var_name="metric", value_name="value"),
        x="metric",
        y="value",
        color="churn_label",
        barmode="group",
        title="Average Daily Engagement: Churned vs Retained"
    )
    fig_engagement
    return avg_engagement, engagement_with_churn, fig_engagement


@app.cell
def __(mo):
    mo.md("## 5. Summary Statistics")
    return


@app.cell
def __(customers, mo):
    summary = f"""
    ### Dataset Summary
    
    | Metric | Value |
    |--------|-------|
    | Total Customers | {len(customers):,} |
    | Overall Churn Rate | {customers['churn_label'].mean():.1%} |
    | Avg Monthly Revenue | ${customers['monthly_charges'].mean():,.0f} |
    | Avg Estimated LTV | ${customers['estimated_ltv'].mean():,.0f} |
    
    ### Key Insights
    
    1. **Cohort Effect**: New users have highest churn risk - activation is critical
    2. **Contract Lock-in**: Month-to-month contracts churn at much higher rates
    3. **Engagement Signal**: Churners show lower engagement across all metrics
    4. **LTV Paradox**: Enterprise customers churn less but each loss is catastrophic
    """
    
    mo.md(summary)
    return summary,


if __name__ == "__main__":
    app.run()