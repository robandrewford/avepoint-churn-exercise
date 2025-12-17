"""
Exploratory Data Analysis for Churn Prediction

Marimo notebook for interactive EDA with sophisticated visualizations.

Run with:
    marimo run notebooks/01_eda.py
    # Or edit mode:
    marimo edit notebooks/01_eda.py
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="full",
    app_title="Churn EDA - Data Profiling & Analysis",
    css_file="",
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

    This notebook provides comprehensive exploration of the churn dataset with:

    1. **Data Quality** - Missing values, distributions, correlations
    2. **Customer Segmentation** - Cohort and LTV tier analysis
    3. **Churn Patterns** - Multi-dimensional churn analysis
    4. **Cohort Analysis** - Temporal churn heatmap by signup cohort
    5. **Engagement Insights** - Behavioral patterns of churned vs retained
    6. **Feature Relationships** - Correlation analysis and feature importance
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # Get project root relative to this script's location
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Define consistent color palette
    COLORS = {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "success": "#2ca02c",
        "danger": "#d62728",
        "warning": "#bcbd22",
        "info": "#17becf",
        "churned": "#d62728",
        "retained": "#2ca02c",
        "cohort": {"new_user": "#ff7f0e", "established": "#2ca02c", "mature": "#1f77b4"},
        "ltv": {"smb": "#17becf", "mid_market": "#bcbd22", "enterprise": "#9467bd"},
    }

    # Custom Plotly template for consistent styling
    PLOT_TEMPLATE = "plotly_white"
    return COLORS, PLOT_TEMPLATE, PROJECT_ROOT, go, make_subplots, np, pd, px


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load & Profile Data
    """)
    return


@app.cell
def _(PROJECT_ROOT, pd):
    # Load synthetic data using absolute paths
    DATA_DIR = PROJECT_ROOT / "outputs" / "synthetic_data"

    customers = pd.read_parquet(DATA_DIR / "customers.parquet")
    daily_engagement = pd.read_parquet(DATA_DIR / "daily_engagement.parquet")
    support_tickets = pd.read_parquet(DATA_DIR / "support_tickets.parquet")

    # Add derived columns for analysis
    customers["signup_month"] = pd.to_datetime(customers["signup_date"]).dt.to_period("M").astype(str)
    customers["churn_month"] = pd.to_datetime(customers["churn_date"]).dt.to_period("M").astype(str)
    customers["months_to_churn"] = (
        (pd.to_datetime(customers["churn_date"]) - pd.to_datetime(customers["signup_date"])).dt.days / 30
    ).round(0)

    print(f"✓ Loaded {len(customers):,} customers")
    print(f"✓ Loaded {len(daily_engagement):,} daily engagement records")
    print(f"✓ Loaded {len(support_tickets):,} support tickets")
    return customers, daily_engagement


@app.cell
def _(mo):
    # Interactive controls for data profiling
    show_distributions = mo.ui.switch(label="Show Distribution Charts", value=True)
    page_size_slider = mo.ui.slider(
        start=10, stop=100, step=10, value=15, label="Rows per page"
    )

    mo.hstack(
        [
            mo.md("**📋 Data Profiling Controls:**"),
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
    _profile_columns = [_c for _c in customers.columns if _c not in _id_columns]

    # Prepare profiling dataframe with sortable columns
    _profiling_df = customers[_profile_columns].copy()

    # Display based on toggle
    if show_distributions.value:
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
        _profile_view = mo.ui.dataframe(
            _profiling_df,
            page_size=page_size_slider.value,
        )

    mo.vstack(
        [
            mo.md(
                f"**Customer Data Profile** — {len(_profiling_df):,} rows, "
                f"{len(_profile_columns)} columns "
                f"({'with distributions' if show_distributions.value else 'with transforms'})"
            ),
            _profile_view,
        ],
        gap=1,
    )
    return


@app.cell
def _(customers, mo):
    # Data Quality Summary
    missing_pct = (customers.isnull().sum() / len(customers) * 100).round(2)
    missing_cols = missing_pct[missing_pct > 0]

    quality_md = f"""
    ### Data Quality Summary

    | Metric | Value |
    |--------|-------|
    | Total Records | {len(customers):,} |
    | Total Features | {len(customers.columns)} |
    | Missing Values | {customers.isnull().sum().sum():,} ({(customers.isnull().sum().sum() / customers.size * 100):.2f}%) |
    | Duplicate Rows | {customers.duplicated().sum()} |

    **Columns with Missing Values:** {len(missing_cols)} columns
    """
    if len(missing_cols) > 0:
        quality_md += "\n| Column | Missing % |\n|--------|----------|\n"
        for _col, _pct in missing_cols.head(10).items():
            quality_md += f"| {_col} | {_pct:.1f}% |\n"

    mo.md(quality_md)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Customer Segmentation

    Understanding customer distribution by lifecycle cohort and value tier.
    """)
    return


@app.cell
def _(PLOT_TEMPLATE, customers, px):
    # Replace pie charts with Treemap - Hierarchical view of Cohort → LTV Tier
    segment_data = customers.groupby(["cohort", "ltv_tier"]).agg(
        count=("customer_id", "count"),
        churn_rate=("churn_label", "mean"),
        avg_ltv=("estimated_ltv", "mean")
    ).reset_index()

    segment_data["label"] = segment_data.apply(
        lambda r: f"{r['ltv_tier']}<br>{r['count']:,} customers<br>Churn: {r['churn_rate']:.1%}",
        axis=1
    )

    fig_treemap = px.treemap(
        segment_data,
        path=["cohort", "ltv_tier"],
        values="count",
        color="churn_rate",
        color_continuous_scale=["#2ca02c", "#bcbd22", "#ff7f0e", "#d62728"],
        color_continuous_midpoint=segment_data["churn_rate"].mean(),
        title="Customer Segmentation: Cohort → LTV Tier (Color = Churn Rate)",
        hover_data={"churn_rate": ":.1%", "avg_ltv": ":$,.0f", "count": ":,"},
        template=PLOT_TEMPLATE,
    )

    fig_treemap.update_layout(
        height=500,
        coloraxis_colorbar={"title": "Churn Rate", "tickformat": ".0%"},
    )
    fig_treemap  # noqa: B018
    return


@app.cell
def _(COLORS, PLOT_TEMPLATE, customers, go, make_subplots):
    # Horizontal stacked bar chart - replaces pie charts with more readable format
    cohort_ltv = customers.groupby(["cohort", "ltv_tier"]).size().unstack(fill_value=0)

    fig_stacked = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Customers by Cohort & LTV Tier", "Churn Rate by Segment"),
        horizontal_spacing=0.15,
    )

    # Stacked bar for customer counts
    for _i, ltv in enumerate(cohort_ltv.columns):
        fig_stacked.add_trace(
            go.Bar(
                name=ltv,
                y=cohort_ltv.index,
                x=cohort_ltv[ltv],
                orientation="h",
                marker_color=COLORS["ltv"].get(ltv, "#888"),
                text=cohort_ltv[ltv],
                textposition="inside",
            ),
            row=1, col=1
        )

    # Churn rate by cohort
    churn_by_cohort = customers.groupby("cohort")["churn_label"].mean().sort_values(ascending=True)
    fig_stacked.add_trace(
        go.Bar(
            name="Churn Rate",
            y=churn_by_cohort.index,
            x=churn_by_cohort.values,
            orientation="h",
            marker_color=[COLORS["cohort"].get(c, "#888") for c in churn_by_cohort.index],
            text=[f"{v:.1%}" for v in churn_by_cohort.values],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=2
    )

    fig_stacked.update_layout(
        barmode="stack",
        height=400,
        title="Customer Distribution & Churn by Segment",
        template=PLOT_TEMPLATE,
    )
    fig_stacked.update_xaxes(title_text="Customer Count", row=1, col=1)
    fig_stacked.update_xaxes(title_text="Churn Rate", tickformat=".0%", row=1, col=2)
    fig_stacked  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Multi-Dimensional Churn Analysis

    Examining churn rates across multiple dimensions simultaneously.
    """)
    return


@app.cell
def _(COLORS, PLOT_TEMPLATE, customers, go, make_subplots):
    # Multi-dimensional churn analysis - grouped bar chart with faceting
    _churn_multi = customers.groupby(["cohort", "ltv_tier", "contract_type"]).agg(
        total=("customer_id", "count"),
        churned=("churn_label", "sum"),
        churn_rate=("churn_label", "mean"),
    ).reset_index()

    # Overall churn rate for reference line
    overall_churn = customers["churn_label"].mean()

    # Create faceted bar chart by contract type
    _fig_multi = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Month-to-month", "One year", "Two year"],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    _contract_types = ["Month-to-month", "One year", "Two year"]

    for _i, _contract in enumerate(_contract_types, 1):
        _contract_data = _churn_multi[_churn_multi["contract_type"] == _contract]

        for _cohort in ["new_user", "established", "mature"]:
            _cohort_data = _contract_data[_contract_data["cohort"] == _cohort]
            if len(_cohort_data) > 0:
                _fig_multi.add_trace(
                    go.Bar(
                        name=_cohort if _i == 1 else None,
                        x=_cohort_data["ltv_tier"],
                        y=_cohort_data["churn_rate"],
                        marker_color=COLORS["cohort"].get(_cohort, "#888"),
                        text=[f"{v:.1%}" for v in _cohort_data["churn_rate"]],
                        textposition="outside",
                        legendgroup=_cohort,
                        showlegend=(_i == 1),
                    ),
                    row=1, col=_i
                )

        # Add reference line for overall churn
        _fig_multi.add_hline(
            y=overall_churn, line_dash="dash", line_color="gray",
            annotation_text=f"Avg: {overall_churn:.1%}" if _i == 3 else None,
            row=1, col=_i
        )

    _fig_multi.update_layout(
        barmode="group",
        height=450,
        title="📊 Churn Rate by Cohort × LTV Tier × Contract Type",
        template=PLOT_TEMPLATE,
        legend_title="Cohort",
    )
    _fig_multi.update_yaxes(tickformat=".0%", range=[0, min(_churn_multi["churn_rate"].max() * 1.3, 1)])
    _fig_multi  # noqa: B018
    return (overall_churn,)


@app.cell
def _(PLOT_TEMPLATE, customers, go, overall_churn):
    # Diverging bar chart - churn rate relative to average
    churn_segments = customers.groupby(["cohort", "ltv_tier"]).agg(
        churn_rate=("churn_label", "mean"),
        count=("customer_id", "count")
    ).reset_index()

    churn_segments["segment"] = churn_segments["cohort"] + " / " + churn_segments["ltv_tier"]
    churn_segments["deviation"] = churn_segments["churn_rate"] - overall_churn
    churn_segments = churn_segments.sort_values("deviation")

    # Create diverging bar chart
    fig_diverging = go.Figure()

    fig_diverging.add_trace(go.Bar(
        y=churn_segments["segment"],
        x=churn_segments["deviation"],
        orientation="h",
        marker_color=[
            "#d62728" if d > 0 else "#2ca02c"
            for d in churn_segments["deviation"]
        ],
        text=[f"{d:+.1%}" for d in churn_segments["deviation"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Deviation: %{x:.1%}<br>Churn Rate: %{customdata:.1%}<extra></extra>",
        customdata=churn_segments["churn_rate"],
    ))

    fig_diverging.add_vline(x=0, line_dash="solid", line_color="gray", line_width=2)

    fig_diverging.update_layout(
        title=f"📈 Churn Rate Deviation from Average ({overall_churn:.1%})",
        xaxis_title="Deviation from Average Churn Rate",
        yaxis_title="Customer Segment",
        height=400,
        template=PLOT_TEMPLATE,
        xaxis_tickformat="+.1%",
    )
    fig_diverging  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Cohort Churn Heatmap

    Visualizing churn patterns by signup month and customer age (months since signup).
    """)
    return


@app.cell
def _(PLOT_TEMPLATE, customers, go, pd):
    # Build cohort heatmap data
    # X-axis: Signup month (calendar month)
    # Y-axis: Month number (months since signup when churn occurred)

    # Create cohort analysis data
    cohort_data = []
    for signup_month in customers["signup_month"].unique():
        month_customers = customers[customers["signup_month"] == signup_month]
        total_in_cohort = len(month_customers)

        for months_since in range(0, 13):  # 0-12 months
            churned_in_month = len(
                month_customers[
                    (month_customers["churn_label"]) &
                    (month_customers["months_to_churn"] >= months_since) &
                    (month_customers["months_to_churn"] < months_since + 1)
                ]
            )
            churn_rate = churned_in_month / total_in_cohort if total_in_cohort > 0 else 0

            cohort_data.append({
                "signup_month": signup_month,
                "months_since_signup": f"Month {months_since}",
                "churn_rate": churn_rate,
                "churned_count": churned_in_month,
                "cohort_size": total_in_cohort,
            })

    cohort_df = pd.DataFrame(cohort_data)

    # Pivot for heatmap
    heatmap_data = cohort_df.pivot(
        index="months_since_signup",
        columns="signup_month",
        values="churn_rate"
    ).fillna(0)

    # Sort columns by date
    heatmap_data = heatmap_data[sorted(heatmap_data.columns)]

    # Create heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[
            [0, "#1a9850"],      # Low churn - green
            [0.25, "#91cf60"],
            [0.5, "#ffffbf"],    # Medium - yellow
            [0.75, "#fc8d59"],
            [1, "#d73027"],      # High churn - red
        ],
        colorbar={"title": "Churn Rate", "tickformat": ".1%"},
        hovertemplate="Signup: %{x}<br>%{y}<br>Churn Rate: %{z:.2%}<extra></extra>",
    ))

    fig_heatmap.update_layout(
        title="Cohort Churn Heatmap: When Do Customers Churn?",
        xaxis_title="Signup Month (Cohort)",
        yaxis_title="Months Since Signup",
        height=500,
        template=PLOT_TEMPLATE,
        yaxis={"autorange": "reversed"},  # Month 0 at top
    )

    # Add annotations for high-churn cells
    for _i, _row in enumerate(heatmap_data.index):
        for _j, _col in enumerate(heatmap_data.columns):
            _val = heatmap_data.loc[_row, _col]
            if _val > 0.02:  # Only annotate significant churn
                fig_heatmap.add_annotation(
                    x=_col, y=_row,
                    text=f"{_val:.1%}",
                    showarrow=False,
                    font={"size": 8, "color": "white" if _val > 0.03 else "black"},
                )

    fig_heatmap  # noqa: B018
    return (cohort_df,)


@app.cell
def _(COLORS, PLOT_TEMPLATE, cohort_df, px):
    # Aggregate cohort trends - line chart showing churn by month since signup
    cohort_trend = cohort_df.groupby("months_since_signup").agg(
        avg_churn_rate=("churn_rate", "mean"),
        total_churned=("churned_count", "sum"),
    ).reset_index()

    fig_cohort_trend = px.line(
        cohort_trend,
        x="months_since_signup",
        y="avg_churn_rate",
        markers=True,
        title="Average Churn Rate by Months Since Signup",
        labels={"avg_churn_rate": "Churn Rate", "months_since_signup": "Months Since Signup"},
        template=PLOT_TEMPLATE,
    )

    fig_cohort_trend.update_traces(
        line={"color": COLORS["danger"], "width": 3},
        marker={"size": 10},
    )

    fig_cohort_trend.update_layout(
        height=350,
        yaxis_tickformat=".1%",
    )
    fig_cohort_trend  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Engagement Analysis

    Comparing behavioral patterns between churned and retained customers.
    """)
    return


@app.cell
def _(COLORS, PLOT_TEMPLATE, customers, daily_engagement, go, make_subplots):
    # Merge engagement with customer info
    engagement_with_churn = daily_engagement.merge(
        customers[["customer_id", "churn_label", "cohort", "ltv_tier"]],
        on="customer_id"
    )

    # Calculate customer-level engagement metrics
    customer_engagement = engagement_with_churn.groupby(["customer_id", "churn_label"]).agg(
        avg_logins=("login_count", "mean"),
        avg_features=("features_used", "mean"),
        avg_session=("session_duration_minutes", "mean"),
        total_days=("login_count", "count"),
    ).reset_index()

    customer_engagement["churn_status"] = customer_engagement["churn_label"].map(
        {True: "Churned", False: "Retained"}
    )

    # Create box plots for engagement comparison
    fig_engagement = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Avg Daily Logins", "Avg Features Used", "Avg Session Duration (min)"],
        horizontal_spacing=0.08,
    )

    for _i, (_col, _title) in enumerate([
        ("avg_logins", "Logins"),
        ("avg_features", "Features"),
        ("avg_session", "Session Duration")
    ], 1):
        for _status in ["Retained", "Churned"]:
            _data = customer_engagement[customer_engagement["churn_status"] == _status][_col]
            fig_engagement.add_trace(
                go.Box(
                    y=_data,
                    name=_status,
                    marker_color=COLORS["retained"] if _status == "Retained" else COLORS["churned"],
                    legendgroup=_status,
                    showlegend=(_i == 1),
                    boxmean=True,
                ),
                row=1, col=_i
            )

    fig_engagement.update_layout(
        title="Engagement Distribution: Churned vs Retained Customers",
        height=450,
        template=PLOT_TEMPLATE,
        boxmode="group",
    )
    fig_engagement  # noqa: B018
    return (customer_engagement,)


@app.cell
def _(COLORS, PLOT_TEMPLATE, customer_engagement, go, make_subplots):
    # Violin plots for richer distribution view
    fig_violin = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Avg Daily Logins", "Avg Features Used", "Avg Session Duration"],
        horizontal_spacing=0.08,
    )

    for _i, _col in enumerate(["avg_logins", "avg_features", "avg_session"], 1):
        for _status in ["Retained", "Churned"]:
            _data = customer_engagement[customer_engagement["churn_status"] == _status][_col]
            fig_violin.add_trace(
                go.Violin(
                    y=_data,
                    name=_status,
                    side="positive" if _status == "Retained" else "negative",
                    marker_color=COLORS["retained"] if _status == "Retained" else COLORS["churned"],
                    legendgroup=_status,
                    showlegend=(_i == 1),
                    meanline_visible=True,
                    box_visible=True,
                ),
                row=1, col=_i
            )

    fig_violin.update_layout(
        title="Engagement Distribution (Violin Plot) - Churned vs Retained",
        height=400,
        template=PLOT_TEMPLATE,
        violinmode="overlay",
    )
    fig_violin  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Feature Correlations

    Understanding relationships between key features and churn.
    """)
    return


@app.cell
def _(PLOT_TEMPLATE, customers, go, np):
    # Select numeric columns for correlation analysis
    numeric_cols = [
        "tenure_days", "monthly_charges", "estimated_ltv",
        "churn_label"
    ]

    # Add available engagement-related columns if they exist
    for _c in ["n_addon_services"]:
        if _c in customers.columns:
            numeric_cols.append(_c)

    corr_matrix = customers[numeric_cols].corr()

    # Create correlation heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar={"title": "Correlation"},
    ))

    fig_corr.update_layout(
        title="Feature Correlation Matrix",
        height=450,
        template=PLOT_TEMPLATE,
        xaxis={"tickangle": 45},
    )
    fig_corr  # noqa: B018
    return


@app.cell
def _(COLORS, PLOT_TEMPLATE, customers, go, np, pd):
    # Feature importance for churn - bar chart of correlations with churn_label
    numeric_features = customers.select_dtypes(include=["number"]).columns.tolist()
    exclude = ["churn_label", "customer_id"]
    numeric_features = [f for f in numeric_features if f not in exclude and not f.endswith("_id")]

    correlations = []
    for _feat in numeric_features:
        if _feat in customers.columns:
            _corr = customers[_feat].corr(customers["churn_label"])
            if not np.isnan(_corr):
                correlations.append({"feature": _feat, "correlation": _corr})

    corr_df = pd.DataFrame(correlations).sort_values("correlation", key=abs, ascending=True)

    fig_feature_corr = go.Figure()

    fig_feature_corr.add_trace(go.Bar(
        y=corr_df["feature"],
        x=corr_df["correlation"],
        orientation="h",
        marker_color=[COLORS["danger"] if c > 0 else COLORS["success"] for c in corr_df["correlation"]],
        text=[f"{c:+.3f}" for c in corr_df["correlation"]],
        textposition="outside",
    ))

    fig_feature_corr.add_vline(x=0, line_color="gray", line_width=2)

    fig_feature_corr.update_layout(
        title="Feature Correlation with Churn",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Feature",
        height=max(400, len(corr_df) * 25),
        template=PLOT_TEMPLATE,
    )
    fig_feature_corr  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Summary & Key Insights
    """)
    return


@app.cell
def _(customers, mo, overall_churn):
    # Calculate key statistics
    churn_by_cohort_summary = customers.groupby("cohort")["churn_label"].mean()
    churn_by_contract_summary = customers.groupby("contract_type")["churn_label"].mean()
    churn_by_ltv_summary = customers.groupby("ltv_tier")["churn_label"].mean()

    highest_churn_cohort = churn_by_cohort_summary.idxmax()
    highest_churn_contract = churn_by_contract_summary.idxmax()

    summary = f"""
    ### Dataset Summary

    | Metric | Value |
    |--------|-------|
    | **Total Customers** | {len(customers):,} |
    | **Overall Churn Rate** | {overall_churn:.1%} |
    | **Avg Monthly Revenue** | ${customers['monthly_charges'].mean():,.0f} |
    | **Avg Estimated LTV** | ${customers['estimated_ltv'].mean():,.0f} |
    | **Date Range** | {customers['signup_month'].min()} to {customers['signup_month'].max()} |

    ### Churn Rate by Segment

    | Cohort | Churn Rate | | LTV Tier | Churn Rate |
    |--------|------------|---|----------|------------|
    | New User | {churn_by_cohort_summary.get('new_user', 0):.1%} | | SMB | {churn_by_ltv_summary.get('smb', 0):.1%} |
    | Established | {churn_by_cohort_summary.get('established', 0):.1%} | | Mid-Market | {churn_by_ltv_summary.get('mid_market', 0):.1%} |
    | Mature | {churn_by_cohort_summary.get('mature', 0):.1%} | | Enterprise | {churn_by_ltv_summary.get('enterprise', 0):.1%} |

    ### Key Insights

    1. ** Highest Risk:** `{highest_churn_cohort}` cohort with `{highest_churn_contract}` contracts
    2. ** Early Churn:** Most churn occurs in first 3 months (see cohort heatmap)
    3. ** Engagement Signal:** Churners show significantly lower engagement metrics
    4. ** LTV Paradox:** Enterprise customers churn less but each loss is high-impact
    5. ** Contract Lock-in:** Annual/biennial contracts reduce churn substantially

    ### Next Steps

    - **Feature Engineering:** Create velocity features from engagement trends
    - **Cohort-Specific Models:** Consider separate models for new vs established users
    - **Intervention Timing:** Focus retention efforts in months 1-3
    - **Contract Strategy:** Incentivize longer contract terms for high-risk segments
    """

    mo.md(summary)
    return


if __name__ == "__main__":
    app.run()
