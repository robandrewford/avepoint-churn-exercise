"""
Model Explanation Module using SHAP

Implements:
- Global feature importance (SHAP summary)
- Local explanations (per-prediction)
- Cohort-level explanations
- Intervention mapping (SHAP → business actions)
- Visualization helpers
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap

# Suppress SHAP warnings about LightGBM binary classifier output format
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
    category=UserWarning,
)

# =============================================================================
# INTERVENTION MAPPING: Feature → Business Action
# =============================================================================

# Maps feature patterns to recommended interventions
INTERVENTION_MAP = {
    # Engagement frequency features
    "logins_7d": {
        "category": "engagement",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Re-engagement campaign", "trigger": "< 2", "priority": "high"},
            {"action": "Usage tips email", "trigger": "< 5", "priority": "medium"},
        ],
        "business_context": "Low login frequency indicates disengagement risk",
    },
    "logins_14d": {
        "category": "engagement",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "CSM outreach call", "trigger": "< 3", "priority": "high"},
            {"action": "Feature discovery email", "trigger": "< 7", "priority": "medium"},
        ],
        "business_context": "Two-week login trend shows engagement trajectory",
    },
    "logins_30d": {
        "category": "engagement",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Account review meeting", "trigger": "< 5", "priority": "critical"},
            {"action": "Executive sponsor outreach", "trigger": "< 10", "priority": "high"},
        ],
        "business_context": "Monthly engagement critical for retention",
    },
    "days_since_last_login": {
        "category": "engagement",
        "direction": "low_risk_when_low",
        "interventions": [
            {"action": "Win-back campaign", "trigger": "> 14", "priority": "critical"},
            {"action": "Check-in email", "trigger": "> 7", "priority": "high"},
        ],
        "business_context": "Recency of activity is strong churn signal",
    },
    # Feature adoption
    "feature_adoption_pct": {
        "category": "adoption",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Product training session", "trigger": "< 0.3", "priority": "high"},
            {"action": "Feature showcase webinar", "trigger": "< 0.5", "priority": "medium"},
        ],
        "business_context": "Low adoption indicates unrealized value",
    },
    "features_used_7d": {
        "category": "adoption",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "In-app guidance setup", "trigger": "< 3", "priority": "medium"},
            {"action": "Best practices guide", "trigger": "< 5", "priority": "low"},
        ],
        "business_context": "Feature breadth indicates stickiness",
    },
    # Velocity/trend features
    "login_velocity_wow": {
        "category": "velocity",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Proactive CSM call", "trigger": "< -0.3", "priority": "critical"},
            {"action": "Usage decline notification", "trigger": "< 0", "priority": "high"},
        ],
        "business_context": "Declining engagement is early warning signal",
    },
    "login_trend_4w": {
        "category": "velocity",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Quarterly business review", "trigger": "< -0.2", "priority": "high"},
            {"action": "Value realization workshop", "trigger": "< 0", "priority": "medium"},
        ],
        "business_context": "4-week trend shows sustained pattern changes",
    },
    # Support features
    "tickets_30d": {
        "category": "support",
        "direction": "low_risk_when_low",
        "interventions": [
            {"action": "Escalation to support lead", "trigger": "> 3", "priority": "high"},
            {"action": "Proactive support check-in", "trigger": "> 1", "priority": "medium"},
        ],
        "business_context": "High ticket volume indicates product friction",
    },
    "avg_sentiment_30d": {
        "category": "support",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Executive escalation", "trigger": "< -0.3", "priority": "critical"},
            {"action": "Satisfaction survey follow-up", "trigger": "< 0", "priority": "high"},
        ],
        "business_context": "Negative sentiment predicts churn",
    },
    "escalation_rate_30d": {
        "category": "support",
        "direction": "low_risk_when_low",
        "interventions": [
            {"action": "Support process review", "trigger": "> 0.3", "priority": "critical"},
            {"action": "Dedicated support contact", "trigger": "> 0.1", "priority": "high"},
        ],
        "business_context": "Escalations indicate unresolved issues",
    },
    # Contract features
    "days_to_renewal": {
        "category": "contract",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Renewal discussion", "trigger": "< 30", "priority": "critical"},
            {"action": "Contract review prep", "trigger": "< 60", "priority": "high"},
            {"action": "Value summary report", "trigger": "< 90", "priority": "medium"},
        ],
        "business_context": "Approaching renewal requires action",
    },
    "contract_value_remaining": {
        "category": "contract",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Retention discount offer", "trigger": "high_value", "priority": "high"},
            {"action": "Upsell opportunity review", "trigger": "any", "priority": "medium"},
        ],
        "business_context": "Contract value guides intervention investment",
    },
    # Onboarding features (new users)
    "onboarding_completion_pct": {
        "category": "activation",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Onboarding rescue call", "trigger": "< 0.3", "priority": "critical"},
            {"action": "Guided setup session", "trigger": "< 0.6", "priority": "high"},
        ],
        "business_context": "Incomplete onboarding leads to early churn",
    },
    "days_to_first_login": {
        "category": "activation",
        "direction": "low_risk_when_low",
        "interventions": [
            {"action": "Welcome call", "trigger": "> 3", "priority": "high"},
            {"action": "Quick-start guide", "trigger": "> 1", "priority": "medium"},
        ],
        "business_context": "Delayed first login indicates adoption risk",
    },
    "first_week_logins": {
        "category": "activation",
        "direction": "low_risk_when_high",
        "interventions": [
            {"action": "Early success check-in", "trigger": "< 3", "priority": "high"},
            {"action": "Use case discovery call", "trigger": "< 5", "priority": "medium"},
        ],
        "business_context": "First week engagement predicts long-term retention",
    },
}


def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: int | None = 1000,
) -> tuple[shap.Explainer, np.ndarray]:
    """
    Compute SHAP values for model explanations.

    Args:
        model: Trained model (tree-based)
        X: Feature matrix
        sample_size: Number of samples for background (None for all)

    Returns:
        Tuple of (explainer, shap_values)
    """
    # Sample if dataset is large
    if sample_size and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    # Create TreeExplainer for LightGBM
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, get positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values


def get_feature_importance_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Get global feature importance from SHAP values.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance
    """
    # Mean absolute SHAP value per feature
    importance = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    # Add rank
    df["rank"] = range(1, len(df) + 1)

    return df


def get_top_drivers(
    shap_values: np.ndarray,
    feature_names: list[str],
    n_top: int = 10,
) -> list[dict]:
    """
    Get top N churn drivers.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        n_top: Number of top features

    Returns:
        List of dicts with feature info
    """
    importance_df = get_feature_importance_shap(shap_values, feature_names)

    top_features = []
    for _, row in importance_df.head(n_top).iterrows():
        top_features.append({
            "rank": int(row["rank"]),
            "feature": row["feature"],
            "importance": float(row["importance"]),
        })

    return top_features


def explain_prediction(
    model,
    explainer: shap.Explainer,
    X_single: pd.DataFrame,
    n_top: int = 5,
) -> dict:
    """
    Generate local explanation for a single prediction.

    Args:
        model: Trained model
        explainer: SHAP explainer
        X_single: Single row of features
        n_top: Number of top contributing features

    Returns:
        Dictionary with prediction explanation
    """
    # Get prediction
    proba = model.predict_proba(X_single)[0, 1]

    # Get SHAP values
    shap_values = explainer.shap_values(X_single)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = shap_values.flatten()
    feature_names = X_single.columns.tolist()

    # Get top contributors (sorted by absolute value)
    contributions = list(zip(feature_names, shap_values, X_single.values.flatten(), strict=False))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_contributors = []
    for name, shap_val, feature_val in contributions[:n_top]:
        top_contributors.append({
            "feature": name,
            "shap_value": float(shap_val),
            "feature_value": float(feature_val),
            "direction": "increases risk" if shap_val > 0 else "decreases risk",
        })

    return {
        "churn_probability": float(proba),
        "base_value": float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value),
        "top_contributors": top_contributors,
    }


def explain_cohort(
    model,
    X: pd.DataFrame,
    cohort_mask: pd.Series,
    cohort_name: str,
    sample_size: int = 500,
) -> dict:
    """
    Generate SHAP explanation for a cohort.

    Args:
        model: Trained model
        X: Full feature matrix
        cohort_mask: Boolean mask for cohort
        cohort_name: Name of cohort
        sample_size: Sample size for SHAP computation

    Returns:
        Dictionary with cohort explanation
    """
    X_cohort = X[cohort_mask]

    if len(X_cohort) == 0:
        return {"cohort": cohort_name, "n_samples": 0, "top_drivers": []}

    # Sample if needed
    if len(X_cohort) > sample_size:
        X_cohort = X_cohort.sample(n=sample_size, random_state=42)

    # Compute SHAP
    explainer, shap_values = compute_shap_values(model, X_cohort, sample_size=None)

    # Get top drivers
    top_drivers = get_top_drivers(shap_values, X_cohort.columns.tolist(), n_top=10)

    return {
        "cohort": cohort_name,
        "n_samples": len(X_cohort),
        "top_drivers": top_drivers,
    }


def plot_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    n_top: int = 15,
    title: str = "Top Churn Drivers (SHAP Feature Importance)",
) -> go.Figure:
    """
    Create Plotly bar chart of feature importance.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        n_top: Number of features to show
        title: Chart title

    Returns:
        Plotly figure
    """
    importance_df = get_feature_importance_shap(shap_values, feature_names)
    top_df = importance_df.head(n_top).sort_values("importance")

    fig = go.Figure(go.Bar(
        x=top_df["importance"],
        y=top_df["feature"],
        orientation="h",
        marker_color="#1f77b4",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        height=400 + n_top * 20,
        margin={"l": 200},
    )

    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    n_top: int = 15,
) -> go.Figure:
    """
    Create SHAP summary plot using Plotly.

    Shows distribution of SHAP values for each feature,
    colored by feature value.

    Args:
        shap_values: SHAP values array
        X: Feature matrix
        n_top: Number of features to show

    Returns:
        Plotly figure
    """
    # Get top features by importance
    importance = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(importance)[-n_top:]

    feature_names = X.columns.tolist()

    # Build data for plot - order features by importance
    data = []
    feature_order = []
    for rank, idx in enumerate(reversed(top_idx)):  # Reverse so most important is at top
        feature = feature_names[idx]
        feature_order.append(feature)
        values = shap_values[:, idx]
        feature_values = X.iloc[:, idx].values

        # Normalize feature values for color
        fv_min, fv_max = feature_values.min(), feature_values.max()
        fv_norm = (feature_values - fv_min) / (fv_max - fv_min + 1e-8)

        # Add jitter to y-axis for better visualization
        np.random.seed(idx)
        y_jitter = np.random.normal(0, 0.15, len(values))

        for sv, fv, jitter in zip(values, fv_norm, y_jitter, strict=False):
            data.append({
                "feature": feature,
                "feature_rank": rank + y_jitter[0] * 0,  # Use rank for positioning
                "shap_value": sv,
                "feature_value_norm": fv,
                "y_position": rank + jitter,
            })

    df = pd.DataFrame(data)

    # Use scatter plot with continuous color scale
    fig = px.scatter(
        df,
        x="shap_value",
        y="y_position",
        color="feature_value_norm",
        color_continuous_scale="RdBu_r",
        labels={"shap_value": "SHAP Value", "feature_value_norm": "Feature Value"},
    )

    # Update y-axis to show feature names
    fig.update_layout(
        title="SHAP Summary Plot",
        height=500 + n_top * 25,
        coloraxis_colorbar={"title": "Feature Value<br>(normalized)"},
        yaxis={
            "tickmode": "array",
            "tickvals": list(range(len(feature_order))),
            "ticktext": feature_order,
            "title": "Feature",
        },
        xaxis_title="SHAP Value (impact on model output)",
    )

    # Make points smaller for better visibility
    fig.update_traces(marker={"size": 4, "opacity": 0.7})

    return fig


def plot_waterfall_explanation(
    explanation: dict,
    title: str = "Prediction Explanation",
) -> go.Figure:
    """
    Create waterfall plot for single prediction explanation.

    Args:
        explanation: Output from explain_prediction()
        title: Chart title

    Returns:
        Plotly figure
    """
    contributors = explanation["top_contributors"]
    base_value = explanation["base_value"]
    final_value = explanation["churn_probability"]

    # Build waterfall data
    features = [c["feature"] for c in contributors]
    values = [c["shap_value"] for c in contributors]

    # Calculate cumulative for positioning
    cumulative = [base_value]
    for v in values:
        cumulative.append(cumulative[-1] + v)

    ["red" if v > 0 else "blue" for v in values]

    fig = go.Figure(go.Waterfall(
        orientation="h",
        y=["Base Value"] + features + ["Final Prediction"],
        x=[base_value] + values + [final_value - cumulative[-1]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "blue"}},
        increasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "green"}},
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Contribution to Churn Probability",
        height=300 + len(contributors) * 30,
    )

    return fig


def save_shap_artifacts(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_dir: str,
) -> dict:
    """
    Save SHAP artifacts for later use.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        output_dir: Output directory

    Returns:
        Dictionary with saved paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_path = output_path / "shap_values.parquet"
    shap_df.to_parquet(shap_path)
    paths["shap_values"] = str(shap_path)

    # Save feature importance
    importance_df = get_feature_importance_shap(shap_values, feature_names)
    importance_path = output_path / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    paths["feature_importance"] = str(importance_path)

    # Save importance plot
    fig = plot_feature_importance(shap_values, feature_names)
    plot_path = output_path / "feature_importance.html"
    fig.write_html(str(plot_path))
    paths["importance_plot"] = str(plot_path)

    return paths


# =============================================================================
# INTERVENTION MAPPING FUNCTIONS
# =============================================================================


def get_intervention_for_feature(
    feature_name: str,
    feature_value: float,
    shap_value: float,
) -> dict | None:
    """
    Get recommended intervention for a feature based on its value and SHAP contribution.

    Args:
        feature_name: Name of the feature
        feature_value: Current value of the feature
        shap_value: SHAP value (contribution to churn probability)

    Returns:
        Dictionary with intervention details, or None if no mapping exists
    """
    if feature_name not in INTERVENTION_MAP:
        return None

    mapping = INTERVENTION_MAP[feature_name]

    # Only suggest intervention if feature is increasing churn risk
    if shap_value <= 0:
        return None

    # Find applicable intervention based on trigger
    for intervention in mapping["interventions"]:
        trigger = intervention["trigger"]

        # Parse trigger condition
        if trigger.startswith("< "):
            threshold = float(trigger[2:])
            if feature_value < threshold:
                return {
                    "feature": feature_name,
                    "feature_value": feature_value,
                    "shap_contribution": shap_value,
                    "action": intervention["action"],
                    "priority": intervention["priority"],
                    "category": mapping["category"],
                    "business_context": mapping["business_context"],
                }
        elif trigger.startswith("> "):
            threshold = float(trigger[2:])
            if feature_value > threshold:
                return {
                    "feature": feature_name,
                    "feature_value": feature_value,
                    "shap_contribution": shap_value,
                    "action": intervention["action"],
                    "priority": intervention["priority"],
                    "category": mapping["category"],
                    "business_context": mapping["business_context"],
                }
        elif trigger in ["any", "high_value"]:
            return {
                "feature": feature_name,
                "feature_value": feature_value,
                "shap_contribution": shap_value,
                "action": intervention["action"],
                "priority": intervention["priority"],
                "category": mapping["category"],
                "business_context": mapping["business_context"],
            }

    return None


def generate_intervention_plan(
    model,
    explainer: shap.Explainer,
    X_single: pd.DataFrame,
    customer_id: str = "unknown",
    max_interventions: int = 5,
) -> dict:
    """
    Generate a prioritized intervention plan for a single customer.

    This is the main function for operationalizing SHAP explanations
    into actionable business recommendations.

    Args:
        model: Trained model
        explainer: SHAP explainer
        X_single: Single row of features
        customer_id: Customer identifier
        max_interventions: Maximum number of interventions to recommend

    Returns:
        Dictionary with customer risk assessment and intervention plan
    """
    # Get prediction and explanation
    explanation = explain_prediction(model, explainer, X_single, n_top=15)

    # Collect all applicable interventions
    interventions = []

    for contributor in explanation["top_contributors"]:
        feature_name = contributor["feature"]
        feature_value = contributor["feature_value"]
        shap_value = contributor["shap_value"]

        intervention = get_intervention_for_feature(
            feature_name, feature_value, shap_value
        )

        if intervention:
            interventions.append(intervention)

    # Sort by priority and SHAP contribution
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    interventions.sort(
        key=lambda x: (priority_order.get(x["priority"], 4), -x["shap_contribution"])
    )

    # Deduplicate by action (keep highest priority)
    seen_actions = set()
    unique_interventions = []
    for intervention in interventions:
        if intervention["action"] not in seen_actions:
            seen_actions.add(intervention["action"])
            unique_interventions.append(intervention)

    # Limit to max interventions
    final_interventions = unique_interventions[:max_interventions]

    # Categorize interventions
    categories = {}
    for intervention in final_interventions:
        cat = intervention["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(intervention["action"])

    return {
        "customer_id": customer_id,
        "churn_probability": explanation["churn_probability"],
        "risk_level": _get_risk_level(explanation["churn_probability"]),
        "top_risk_factors": [
            {
                "feature": c["feature"],
                "contribution": c["shap_value"],
                "direction": c["direction"],
            }
            for c in explanation["top_contributors"][:5]
        ],
        "recommended_interventions": final_interventions,
        "intervention_categories": categories,
        "total_interventions": len(final_interventions),
    }


def _get_risk_level(probability: float) -> str:
    """Convert probability to risk level category."""
    if probability >= 0.7:
        return "critical"
    elif probability >= 0.5:
        return "high"
    elif probability >= 0.3:
        return "medium"
    else:
        return "low"


def generate_cohort_intervention_summary(
    model,
    X: pd.DataFrame,
    df: pd.DataFrame,
    cohort_column: str = "cohort",
    sample_size: int = 500,
) -> pd.DataFrame:
    """
    Generate intervention summary by cohort.

    Shows which interventions are most relevant for each customer segment.

    Args:
        model: Trained model
        X: Feature matrix
        df: Full dataframe with cohort column
        cohort_column: Name of cohort column
        sample_size: Sample size per cohort for SHAP

    Returns:
        DataFrame with intervention recommendations by cohort
    """
    results = []

    for cohort in df[cohort_column].unique():
        mask = df[cohort_column] == cohort
        X_cohort = X[mask]

        if len(X_cohort) == 0:
            continue

        # Sample for SHAP computation
        if len(X_cohort) > sample_size:
            X_sample = X_cohort.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_cohort

        # Compute SHAP
        explainer, shap_values = compute_shap_values(model, X_sample, sample_size=None)

        # Get top drivers for cohort
        importance_df = get_feature_importance_shap(shap_values, X_sample.columns.tolist())
        top_features = importance_df.head(10)["feature"].tolist()

        # Map to interventions
        cohort_interventions = []
        for feature in top_features:
            if feature in INTERVENTION_MAP:
                mapping = INTERVENTION_MAP[feature]
                for intervention in mapping["interventions"]:
                    cohort_interventions.append({
                        "cohort": cohort,
                        "feature": feature,
                        "action": intervention["action"],
                        "priority": intervention["priority"],
                        "category": mapping["category"],
                    })

        results.extend(cohort_interventions)

    return pd.DataFrame(results)


def plot_intervention_priority_matrix(
    interventions_df: pd.DataFrame,
) -> go.Figure:
    """
    Create a visual matrix of interventions by category and priority.

    Args:
        interventions_df: DataFrame from generate_cohort_intervention_summary

    Returns:
        Plotly figure
    """
    # Count interventions by category and priority
    summary = interventions_df.groupby(["category", "priority"]).size().reset_index(name="count")

    fig = px.treemap(
        summary,
        path=["category", "priority"],
        values="count",
        color="priority",
        color_discrete_map={
            "critical": "#d62728",
            "high": "#ff7f0e",
            "medium": "#2ca02c",
            "low": "#1f77b4",
        },
        title="Intervention Priority Matrix",
    )

    fig.update_layout(height=500)

    return fig


def explain_with_interventions(
    model,
    X: pd.DataFrame,
    df: pd.DataFrame,
    customer_ids: list[str],
    n_top_customers: int = 10,
) -> list[dict]:
    """
    Generate comprehensive explanations with interventions for high-risk customers.

    Args:
        model: Trained model
        X: Feature matrix
        df: Full dataframe with customer metadata
        customer_ids: List of customer IDs
        n_top_customers: Number of highest-risk customers to explain

    Returns:
        List of intervention plans for top risk customers
    """
    # Get predictions
    y_proba = model.predict_proba(X)[:, 1]

    # Get top risk customers
    risk_order = np.argsort(y_proba)[::-1][:n_top_customers]

    # Create explainer once
    explainer, _ = compute_shap_values(model, X, sample_size=1000)

    # Generate plans
    plans = []
    for idx in risk_order:
        customer_id = customer_ids[idx] if idx < len(customer_ids) else f"customer_{idx}"
        X_single = X.iloc[[idx]]

        plan = generate_intervention_plan(
            model, explainer, X_single, customer_id
        )

        # Add cohort info if available
        if "cohort" in df.columns:
            plan["cohort"] = df.iloc[idx]["cohort"]
        if "ltv_tier" in df.columns:
            plan["ltv_tier"] = df.iloc[idx]["ltv_tier"]

        plans.append(plan)

    return plans
