"""
Model Monitoring Dashboard for Churn Prediction

Marimo notebook for production monitoring.

Run with:
    marimo run notebooks/03_monitoring.py
    # Or edit mode:
    marimo edit notebooks/03_monitoring.py
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
        # Churn Prediction: Model Monitoring
        
        Monitor model health across three pillars:
        1. **Data Quality** - Feature freshness, missing rates, distribution drift
        2. **Model Health** - Prediction drift, calibration, performance decay
        3. **Business Impact** - Intervention rates, save rates, ROI
        """
    )
    return


@app.cell
def __():
    import sys
    sys.path.insert(0, "..")
    
    import pandas as pd
    import numpy as np
    from datetime import date, timedelta
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return date, go, make_subplots, np, pd, px, sys, timedelta


@app.cell
def __(mo):
    mo.md("## Data Quality Monitoring")
    return


@app.cell
def __(np, pd):
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
def __(monitoring_data, px):
    # Feature freshness
    fig_freshness = px.line(
        monitoring_data,
        x="date",
        y="feature_freshness_hours",
        title="Feature Freshness (Target: <24 hours)"
    )
    fig_freshness.add_hline(y=24, line_dash="dash", line_color="red", annotation_text="SLA")
    fig_freshness
    return fig_freshness,


@app.cell
def __(monitoring_data, px):
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
    return fig_psi,


@app.cell
def __(mo):
    mo.md("## Model Health Monitoring")
    return


@app.cell
def __(np, pd, dates):
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
    return model_metrics,


@app.cell
def __(model_metrics, px):
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
    return fig_auc,


@app.cell
def __(model_metrics, px):
    # Prediction drift
    fig_pred_drift = px.line(
        model_metrics,
        x="date",
        y="prediction_mean",
        title="Prediction Distribution (Mean Churn Score)"
    )
    fig_pred_drift.add_hline(y=0.26, line_dash="dash", line_color="green", annotation_text="Baseline")
    fig_pred_drift
    return fig_pred_drift,


@app.cell
def __(mo):
    mo.md("## Business Impact Monitoring")
    return


@app.cell
def __(np, pd, dates):
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
    return business_metrics,


@app.cell
def __(business_metrics, make_subplots, go):
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
    return fig_business,


@app.cell
def __(mo):
    mo.md("## Alert Summary")
    return


@app.cell
def __(monitoring_data, model_metrics, mo):
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
    return alert_table, alerts


@app.cell
def __(mo):
    mo.md(
        """
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
        """
    )
    return


if __name__ == "__main__":
    app.run()