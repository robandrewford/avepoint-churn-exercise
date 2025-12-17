"""
Exploratory Data Analysis for Churn Prediction

Marimo notebook for interactive EDA.

Run with:
    marimo run notebooks/01_eda.py
    # Or edit mode:
    marimo edit notebooks/01_eda.py
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="full",
    app_title="Data profiling",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Churn Prediction: Exploratory Data Analysis

    This notebook explores the synthetic churn dataset to understand:
    1. Customer distribution by cohort and LTV tier
    2. Churn rates across segments
    3. Feature distributions
    4. Temporal patterns
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # Get project root relative to this script's location
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return PROJECT_ROOT, go, make_subplots, pd, px


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load Data
    """)
    return


@app.cell
def _(PROJECT_ROOT, pd):
    # Load synthetic data using absolute paths
    DATA_DIR = PROJECT_ROOT / "outputs" / "synthetic_data"

    customers = pd.read_parquet(DATA_DIR / "customers.parquet")
    daily_engagement = pd.read_parquet(DATA_DIR / "daily_engagement.parquet")
    support_tickets = pd.read_parquet(DATA_DIR / "support_tickets.parquet")

    print(f"Customers: {len(customers):,}")
    print(f"Daily engagement records: {len(daily_engagement):,}")
    print(f"Support tickets: {len(support_tickets):,}")
    return customers, daily_engagement


@app.cell
def _(mo):
    mo.md("""
    ## 2. Customer Distribution
    """)
    return


@app.cell
def _(mo):
    # Interactive controls for data profiling
    show_distributions = mo.ui.switch(label="Show Distribution Charts", value=True)
    page_size_slider = mo.ui.slider(
        start=10, stop=100, step=10, value=20, label="Rows per page"
    )

    mo.hstack(
        [
            mo.md("**Data Profiling Controls:**"),
            show_distributions,
            page_size_slider,
        ],
        justify="start",
        gap=2,
    )
    return page_size_slider, show_distributions


@app.cell
def _(customers, mo, page_size_slider, show_distributions):
    # Define columns: exclude ID columns from default view but include all others
    _id_columns = ["customer_id"]
    _profile_columns = [col for col in customers.columns if col not in _id_columns]

    # Prepare profiling dataframe with sortable columns
    _profiling_df = customers[_profile_columns].copy()

    # Display based on toggle
    if show_distributions.value:
        # Use mo.ui.table with distribution charts enabled
        _profile_view = mo.ui.table(
            _profiling_df,
            pagination=True,
            page_size=page_size_slider.value,
            show_column_summaries="chart",
            show_data_types=True,
            selection=None,
            max_columns=None,
        )
    else:
        # Use mo.ui.dataframe for full transform capabilities
        # (sorting, filtering, groupby, etc. built into the UI)
        _profile_view = mo.ui.dataframe(
            _profiling_df,
            page_size=page_size_slider.value,
        )

    mo.vstack(
        [
            mo.md(
                f"**Customer Data Profile** - {len(_profiling_df):,} rows, "
                f"{len(_profile_columns)} columns "
                f"({'with distributions' if show_distributions.value else 'with transforms'})"
            ),
            _profile_view,
        ],
        gap=1,
    )
    return


@app.cell
def _(customers, go, make_subplots):
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
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Churn Analysis
    """)
    return


@app.cell
def _(customers, px):
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
    return


@app.cell
def _(customers, px):
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
    return


@app.cell
def _(customers, px):
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
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Engagement Patterns
    """)
    return


@app.cell
def _(customers, daily_engagement, px):
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
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Summary Statistics
    """)
    return


@app.cell
def _(customers, mo):
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
    return


if __name__ == "__main__":
    app.run()
