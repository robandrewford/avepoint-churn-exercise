"""
Model Monitoring Dashboard for Churn Prediction

Marimo notebook for production monitoring.

Run with:
    marimo run notebooks/03_monitoring.py
    # Or edit mode:
    marimo edit notebooks/03_monitoring.py
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Churn Prediction: Model Monitoring

    Monitor model health across three pillars:
    1. **Data Quality** - Feature freshness, missing rates, distribution drift
    2. **Model Health** - Prediction drift, calibration, performance decay
    3. **Business Impact** - Intervention rates, save rates, ROI
    4. **Cohort Performance** - Performance trends by customer segment
    5. **Retraining Triggers** - Automated alerts for model refresh
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # Get project root relative to this script's location
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from datetime import date, timedelta

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots, np, pd, px


@app.cell
def _(mo):
    mo.md("""
    ## Data Quality Monitoring
    """)
    return


@app.cell
def _(np, pd):
    # Simulate monitoring data
    np.random.seed(42)
    dates = pd.date_range(start="2024-10-01", end="2024-12-01", freq="D")

    monitoring_data = pd.DataFrame({
        "date": dates,
        "feature_freshness_hours": np.random.normal(12, 3, len(dates)),
        "missing_rate_pct": np.random.exponential(0.5, len(dates)),
        "psi_engagement": np.random.exponential(0.08, len(dates)),
        "psi_support": np.random.exponential(0.05, len(dates)),
    })

    # Add some anomalies
    monitoring_data.loc[45:48, "psi_engagement"] = [0.25, 0.28, 0.22, 0.19]
    monitoring_data.loc[50:52, "missing_rate_pct"] = [3.5, 4.2, 2.8]

    monitoring_data
    return dates, monitoring_data


@app.cell
def _(monitoring_data, px):
    # Feature freshness
    fig_freshness = px.line(
        monitoring_data,
        x="date",
        y="feature_freshness_hours",
        title="Feature Freshness (Target: <24 hours)"
    )
    fig_freshness.add_hline(y=24, line_dash="dash", line_color="red", annotation_text="SLA")
    fig_freshness
    return


@app.cell
def _(monitoring_data, px):
    # PSI (Population Stability Index) tracking
    fig_psi = px.line(
        monitoring_data,
        x="date",
        y=["psi_engagement", "psi_support"],
        title="Feature Distribution Drift (PSI)",
        labels={"value": "PSI", "variable": "Feature Group"}
    )
    fig_psi.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Retrain Threshold")
    fig_psi.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Monitor")
    fig_psi
    return


@app.cell
def _(mo):
    mo.md("""
    ## Model Health Monitoring
    """)
    return


@app.cell
def _(dates, np, pd):
    # Simulate model metrics over time
    model_metrics = pd.DataFrame({
        "date": dates,
        "auc_pr": 0.55 + np.cumsum(np.random.normal(-0.001, 0.01, len(dates))),
        "precision_at_10pct": 0.72 + np.cumsum(np.random.normal(-0.0005, 0.008, len(dates))),
        "prediction_mean": 0.26 + np.cumsum(np.random.normal(0, 0.003, len(dates))),
    })

    # Ensure reasonable bounds
    model_metrics["auc_pr"] = model_metrics["auc_pr"].clip(0.3, 0.8)
    model_metrics["precision_at_10pct"] = model_metrics["precision_at_10pct"].clip(0.4, 0.9)
    model_metrics["prediction_mean"] = model_metrics["prediction_mean"].clip(0.1, 0.5)

    model_metrics
    return (model_metrics,)


@app.cell
def _(model_metrics, px):
    # AUC-PR over time
    fig_auc = px.line(
        model_metrics,
        x="date",
        y="auc_pr",
        title="Model Performance (AUC-PR) Over Time"
    )
    fig_auc.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Target")
    fig_auc.add_hline(y=0.45, line_dash="dash", line_color="orange", annotation_text="Retrain Trigger")
    fig_auc
    return


@app.cell
def _(model_metrics, px):
    # Prediction drift
    fig_pred_drift = px.line(
        model_metrics,
        x="date",
        y="prediction_mean",
        title="Prediction Distribution (Mean Churn Score)"
    )
    fig_pred_drift.add_hline(y=0.26, line_dash="dash", line_color="green", annotation_text="Baseline")
    fig_pred_drift
    return


@app.cell
def _(mo):
    mo.md("""
    ## Business Impact Monitoring
    """)
    return


@app.cell
def _(dates, np, pd):
    # Simulate business metrics
    business_metrics = pd.DataFrame({
        "date": dates,
        "high_risk_flagged": np.random.poisson(150, len(dates)),
        "interventions_attempted": np.random.poisson(120, len(dates)),
        "interventions_successful": np.random.poisson(45, len(dates)),
    })

    business_metrics["intervention_rate"] = (
        business_metrics["interventions_attempted"] / business_metrics["high_risk_flagged"]
    ).clip(0, 1)
    business_metrics["save_rate"] = (
        business_metrics["interventions_successful"] / business_metrics["interventions_attempted"]
    ).clip(0, 1)

    # Add revenue impact
    business_metrics["revenue_at_risk"] = business_metrics["high_risk_flagged"] * np.random.uniform(8000, 15000, len(dates))
    business_metrics["revenue_saved"] = business_metrics["interventions_successful"] * np.random.uniform(10000, 20000, len(dates))

    business_metrics
    return (business_metrics,)


@app.cell
def _(business_metrics, go, make_subplots):
    # Business impact dashboard
    fig_business = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Intervention Rate (Target: >80%)",
            "Save Rate (Target: >40%)",
            "Revenue at Risk",
            "Revenue Saved"
        )
    )

    # Intervention rate
    fig_business.add_trace(
        go.Scatter(x=business_metrics["date"], y=business_metrics["intervention_rate"],
                   mode="lines", name="Intervention Rate"),
        row=1, col=1
    )
    fig_business.add_hline(y=0.8, line_dash="dash", line_color="green", row=1, col=1)

    # Save rate
    fig_business.add_trace(
        go.Scatter(x=business_metrics["date"], y=business_metrics["save_rate"],
                   mode="lines", name="Save Rate"),
        row=1, col=2
    )
    fig_business.add_hline(y=0.4, line_dash="dash", line_color="green", row=1, col=2)

    # Revenue at risk
    fig_business.add_trace(
        go.Bar(x=business_metrics["date"], y=business_metrics["revenue_at_risk"],
               name="Revenue at Risk"),
        row=2, col=1
    )

    # Revenue saved
    fig_business.add_trace(
        go.Bar(x=business_metrics["date"], y=business_metrics["revenue_saved"],
               name="Revenue Saved", marker_color="green"),
        row=2, col=2
    )

    fig_business.update_layout(height=600, title="Business Impact Dashboard")
    fig_business
    return


@app.cell
def _(mo):
    mo.md("""
    ## Alert Summary
    """)
    return


@app.cell
def _(mo, model_metrics, monitoring_data):
    # Check for alerts
    alerts = []

    # Data quality alerts
    if monitoring_data["psi_engagement"].iloc[-7:].max() > 0.2:
        alerts.append(("🔴 CRITICAL", "Engagement feature drift detected (PSI > 0.2)"))
    if monitoring_data["missing_rate_pct"].iloc[-7:].max() > 3.0:
        alerts.append(("🟠 WARNING", "Missing rate spike detected (>3%)"))

    # Model health alerts
    if model_metrics["auc_pr"].iloc[-1] < 0.45:
        alerts.append(("🔴 CRITICAL", "Model performance below threshold (AUC-PR < 0.45)"))
    elif model_metrics["auc_pr"].iloc[-1] < 0.5:
        alerts.append(("🟠 WARNING", "Model performance degrading (AUC-PR < 0.5)"))

    # Generate alert table
    if alerts:
        alert_table = "| Severity | Alert |\n|----------|-------|\n"
        for severity, message in alerts:
            alert_table += f"| {severity} | {message} |\n"
    else:
        alert_table = "✅ No active alerts"

    mo.md(f"""
    ### Active Alerts

    {alert_table}

    **Last Updated:** {monitoring_data['date'].iloc[-1].strftime('%Y-%m-%d')}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cohort Performance Monitoring

    Track model performance by customer lifecycle stage to identify cohort-specific degradation.
    """)
    return


@app.cell
def _(dates, go, make_subplots, np, pd):
    # Simulate cohort-specific performance over time
    cohort_performance = pd.DataFrame({
        "date": dates,
        "new_user_auc_pr": 0.72 + np.cumsum(np.random.normal(-0.002, 0.015, len(dates))),
        "established_auc_pr": 0.66 + np.cumsum(np.random.normal(-0.001, 0.012, len(dates))),
        "mature_auc_pr": 0.64 + np.cumsum(np.random.normal(-0.0005, 0.010, len(dates))),
    })

    # Clip to reasonable bounds
    for col in ["new_user_auc_pr", "established_auc_pr", "mature_auc_pr"]:
        cohort_performance[col] = cohort_performance[col].clip(0.3, 0.85)

    # Create cohort performance chart
    fig_cohort = make_subplots(rows=1, cols=1)

    fig_cohort.add_trace(go.Scatter(
        x=cohort_performance["date"], 
        y=cohort_performance["new_user_auc_pr"],
        mode="lines", name="New Users", line={"color": "#ff7f0e"}
    ))
    fig_cohort.add_trace(go.Scatter(
        x=cohort_performance["date"], 
        y=cohort_performance["established_auc_pr"],
        mode="lines", name="Established", line={"color": "#2ca02c"}
    ))
    fig_cohort.add_trace(go.Scatter(
        x=cohort_performance["date"], 
        y=cohort_performance["mature_auc_pr"],
        mode="lines", name="Mature", line={"color": "#1f77b4"}
    ))

    fig_cohort.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Target Minimum")
    fig_cohort.update_layout(
        title="AUC-PR by Cohort Over Time",
        xaxis_title="Date",
        yaxis_title="AUC-PR",
        height=400
    )
    fig_cohort
    return


@app.cell
def _(mo):
    mo.md("""
    ## Retraining Trigger Status

    Automated evaluation of conditions that warrant model retraining.
    """)
    return


@app.cell
def _(business_metrics, mo, model_metrics, monitoring_data):
    # Define retraining trigger thresholds
    TRIGGERS = {
        "performance_drop": {"threshold": 0.10, "metric": "AUC-PR drop from baseline", "status": "check"},
        "drift_psi": {"threshold": 0.20, "metric": "Feature PSI", "status": "check"},
        "calibration": {"threshold": 0.20, "metric": "Calibration increase", "status": "check"},
        "business_roi": {"threshold": 0.30, "metric": "Save rate minimum", "status": "check"}
    }

    # Check each trigger
    baseline_auc = model_metrics["auc_pr"].iloc[:7].mean()
    current_auc = model_metrics["auc_pr"].iloc[-7:].mean()
    auc_drop = (baseline_auc - current_auc) / baseline_auc

    current_psi = monitoring_data["psi_engagement"].iloc[-7:].max()
    current_save_rate = business_metrics["save_rate"].iloc[-7:].mean()

    # Update trigger status
    trigger_results = []

    # Performance trigger
    if auc_drop > TRIGGERS["performance_drop"]["threshold"]:
        trigger_results.append(("🔴", "Performance Drop", f"{auc_drop:.1%} drop", "TRIGGERED"))
    elif auc_drop > TRIGGERS["performance_drop"]["threshold"] * 0.7:
        trigger_results.append(("🟠", "Performance Drop", f"{auc_drop:.1%} drop", "WARNING"))
    else:
        trigger_results.append(("✅", "Performance Drop", f"{auc_drop:.1%} drop", "OK"))

    # Drift trigger
    if current_psi > TRIGGERS["drift_psi"]["threshold"]:
        trigger_results.append(("🔴", "Feature Drift (PSI)", f"{current_psi:.3f}", "TRIGGERED"))
    elif current_psi > TRIGGERS["drift_psi"]["threshold"] * 0.5:
        trigger_results.append(("🟠", "Feature Drift (PSI)", f"{current_psi:.3f}", "WARNING"))
    else:
        trigger_results.append(("✅", "Feature Drift (PSI)", f"{current_psi:.3f}", "OK"))

    # Business impact trigger
    if current_save_rate < TRIGGERS["business_roi"]["threshold"]:
        trigger_results.append(("🔴", "Business ROI (Save Rate)", f"{current_save_rate:.1%}", "TRIGGERED"))
    elif current_save_rate < TRIGGERS["business_roi"]["threshold"] * 1.2:
        trigger_results.append(("🟠", "Business ROI (Save Rate)", f"{current_save_rate:.1%}", "WARNING"))
    else:
        trigger_results.append(("✅", "Business ROI (Save Rate)", f"{current_save_rate:.1%}", "OK"))

    # Build trigger table
    trigger_table = """
    ### Retraining Trigger Evaluation

    | Status | Trigger | Current Value | Decision |
    |--------|---------|---------------|----------|
    """
    for status, trigger, value, decision in trigger_results:
        trigger_table += f"| {status} | {trigger} | {value} | {decision} |\n"

    # Overall recommendation
    triggered_count = sum(1 for r in trigger_results if r[3] == "TRIGGERED")
    warning_count = sum(1 for r in trigger_results if r[3] == "WARNING")

    if triggered_count > 0:
        recommendation = f"""
    ### 🔴 Recommendation: RETRAIN MODEL

    **{triggered_count} trigger(s) activated.** Initiate model retraining pipeline.

    **Immediate Actions:**
    1. Review feature drift analysis
    2. Assess data quality issues
    3. Schedule model retraining job
    4. Notify stakeholders
    """
    elif warning_count > 0:
        recommendation = f"""
    ### 🟠 Recommendation: MONITOR CLOSELY

    **{warning_count} warning(s) detected.** Increase monitoring frequency.

    **Next Steps:**
    1. Daily performance review
    2. Investigate root cause of warnings
    3. Prepare retraining resources
    """
    else:
        recommendation = """
    ### ✅ Recommendation: NO ACTION REQUIRED

    All systems healthy. Continue standard monitoring schedule.
    """

    mo.md(trigger_table + recommendation)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Threshold Configuration

    Adjust monitoring thresholds to fine-tune alert sensitivity.
    """)
    return


@app.cell
def _(mo):
    # Interactive threshold controls
    psi_threshold = mo.ui.slider(
        start=0.1, stop=0.3, step=0.02, value=0.2,
        label="PSI Alert Threshold"
    )
    auc_threshold = mo.ui.slider(
        start=0.4, stop=0.6, step=0.02, value=0.5,
        label="AUC-PR Minimum Threshold"
    )
    freshness_threshold = mo.ui.slider(
        start=12, stop=48, step=6, value=24,
        label="Feature Freshness SLA (hours)"
    )

    mo.vstack([
        mo.md("**Adjust Alert Thresholds:**"),
        mo.hstack([psi_threshold, auc_threshold, freshness_threshold], justify="start", gap=2)
    ])
    return auc_threshold, freshness_threshold, psi_threshold


@app.cell
def _(
    auc_threshold,
    freshness_threshold,
    mo,
    model_metrics,
    monitoring_data,
    psi_threshold,
):
    # Dynamic alert evaluation based on slider values
    psi_alerts = (monitoring_data["psi_engagement"] > psi_threshold.value).sum()
    auc_alerts = (model_metrics["auc_pr"] < auc_threshold.value).sum()
    freshness_alerts = (monitoring_data["feature_freshness_hours"] > freshness_threshold.value).sum()

    dynamic_alert_md = f"""
    ### Dynamic Alert Summary (Based on Current Thresholds)

    | Alert Type | Threshold | Days Triggered | % of Period |
    |------------|-----------|----------------|-------------|
    | PSI Drift | >{psi_threshold.value:.2f} | {psi_alerts} | {psi_alerts/len(monitoring_data)*100:.1f}% |
    | Low AUC-PR | <{auc_threshold.value:.2f} | {auc_alerts} | {auc_alerts/len(model_metrics)*100:.1f}% |
    | Freshness | >{freshness_threshold.value}h | {freshness_alerts} | {freshness_alerts/len(monitoring_data)*100:.1f}% |

    💡 **Tip:** Adjust thresholds to balance alert sensitivity vs. noise.
    - Too sensitive → Alert fatigue
    - Too lenient → Missed issues
    """

    mo.md(dynamic_alert_md)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Monitoring Configuration

    ### Thresholds

    | Metric | Warning | Critical | Action |
    |--------|---------|----------|--------|
    | Feature Freshness | >18h | >24h | Investigate pipeline |
    | Missing Rate | >2% | >5% | Pause predictions |
    | PSI | >0.1 | >0.2 | Trigger retraining |
    | AUC-PR | <0.5 | <0.45 | Urgent review |
    | Intervention Rate | <70% | <60% | CS capacity review |
    | Save Rate | <35% | <25% | Strategy review |

    ### Escalation Path

    1. **Warning** → DS on-call notified, investigate within 24h
    2. **Critical** → DS Lead + VP DS notified, investigate within 4h
    3. **Business Impact** → Weekly stakeholder review

    ### Monitoring Best Practices

    - **Daily**: Check alert dashboard, review high-risk customer list
    - **Weekly**: Review cohort performance trends, save rate analysis
    - **Monthly**: Full model health assessment, stakeholder report
    - **Quarterly**: Retrain model with latest data, feature engineering review
    """)
    return


if __name__ == "__main__":
    app.run()
