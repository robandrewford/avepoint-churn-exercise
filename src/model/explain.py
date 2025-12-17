"""
Model Explanation Module using SHAP

Implements:
- Global feature importance (SHAP summary)
- Local explanations (per-prediction)
- Cohort-level explanations
- Visualization helpers
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap


def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: Optional[int] = 1000,
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
    contributions = list(zip(feature_names, shap_values, X_single.values.flatten()))
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
        margin=dict(l=200),
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
    
    # Build data for plot
    data = []
    for idx in top_idx:
        feature = feature_names[idx]
        values = shap_values[:, idx]
        feature_values = X.iloc[:, idx].values
        
        # Normalize feature values for color
        fv_norm = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-8)
        
        for i, (sv, fv) in enumerate(zip(values, fv_norm)):
            data.append({
                "feature": feature,
                "shap_value": sv,
                "feature_value_norm": fv,
            })
    
    df = pd.DataFrame(data)
    
    fig = px.strip(
        df,
        x="shap_value",
        y="feature",
        color="feature_value_norm",
        color_continuous_scale="RdBu_r",
        labels={"shap_value": "SHAP Value", "feature_value_norm": "Feature Value"},
    )
    
    fig.update_layout(
        title="SHAP Summary Plot",
        height=500 + n_top * 25,
        coloraxis_colorbar=dict(title="Feature Value<br>(normalized)"),
    )
    
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
    
    colors = ["red" if v > 0 else "blue" for v in values]
    
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
