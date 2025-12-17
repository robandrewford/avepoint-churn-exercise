"""
Model Evaluation Module for Churn Prediction

Implements:
- Standard classification metrics
- Business-specific metrics (Precision@K, Lead Time)
- Cohort-stratified evaluation
- Threshold optimization
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model with comprehensive metrics.
    
    Args:
        model: Trained classifier with predict_proba method
        X: Features
        y: True labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Standard metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y, y_proba) if y.nunique() > 1 else 0,
        "auc_pr": average_precision_score(y, y_proba) if y.nunique() > 1 else 0,
        "brier_score": brier_score_loss(y, y_proba),
    }
    
    # Business metrics
    metrics["precision_at_10pct"] = precision_at_k(y, y_proba, k_pct=0.10)
    metrics["precision_at_20pct"] = precision_at_k(y, y_proba, k_pct=0.20)
    metrics["lift_at_10pct"] = lift_at_k(y, y_proba, k_pct=0.10)
    
    # Confusion matrix - handle single class case
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)
    
    return metrics


def precision_at_k(
    y_true: pd.Series,
    y_proba: np.ndarray,
    k_pct: float = 0.10,
) -> float:
    """
    Calculate precision when flagging top k% of predictions.
    
    This is a key business metric: If CS team can only contact
    top 10% of customers, what fraction are actual churners?
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        k_pct: Top percentage to consider (e.g., 0.10 for top 10%)
        
    Returns:
        Precision at k%
    """
    n_samples = len(y_true)
    k = int(n_samples * k_pct)
    
    if k == 0:
        return 0.0
    
    # Get indices of top k predictions
    top_k_idx = np.argsort(y_proba)[-k:]
    
    # Calculate precision
    y_true_array = np.array(y_true)
    precision = y_true_array[top_k_idx].mean()
    
    return float(precision)


def lift_at_k(
    y_true: pd.Series,
    y_proba: np.ndarray,
    k_pct: float = 0.10,
) -> float:
    """
    Calculate lift at k% - how much better than random.
    
    Lift = Precision@K / Base Rate
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        k_pct: Top percentage to consider
        
    Returns:
        Lift at k%
    """
    base_rate = y_true.mean()
    
    if base_rate == 0:
        return 0.0
    
    precision_k = precision_at_k(y_true, y_proba, k_pct)
    lift = precision_k / base_rate
    
    return float(lift)


def find_optimal_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    method: str = "f1",
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
) -> tuple[float, dict]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        method: Optimization method ('f1', 'precision_recall', 'cost')
        cost_fp: Cost of false positive (for cost method)
        cost_fn: Cost of false negative (for cost method)
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    if method == "f1":
        # Find threshold that maximizes F1
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # Last element has no threshold
        optimal_threshold = thresholds[best_idx]
        
    elif method == "precision_recall":
        # Find threshold where precision >= recall
        for i, (p, r) in enumerate(zip(precisions[:-1], recalls[:-1])):
            if p >= r:
                optimal_threshold = thresholds[i]
                break
        else:
            optimal_threshold = 0.5
            
    elif method == "cost":
        # Minimize expected cost
        n_samples = len(y_true)
        n_positive = y_true.sum()
        n_negative = n_samples - n_positive
        
        best_cost = float("inf")
        optimal_threshold = 0.5
        
        for thresh in np.linspace(0.1, 0.9, 81):
            y_pred = (y_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Total cost
            total_cost = fp * cost_fp + fn * cost_fn
            
            if total_cost < best_cost:
                best_cost = total_cost
                optimal_threshold = thresh
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get metrics at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    metrics = {
        "threshold": optimal_threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    return optimal_threshold, metrics


def evaluate_by_cohort(
    model,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    cohort_column: str = "cohort",
) -> pd.DataFrame:
    """
    Evaluate model performance by cohort.
    
    Args:
        model: Trained classifier
        df: Full dataframe with cohort column
        X: Features
        y: True labels
        cohort_column: Name of cohort column
        
    Returns:
        DataFrame with metrics per cohort
    """
    results = []
    
    for cohort in df[cohort_column].unique():
        mask = df[cohort_column] == cohort
        
        if mask.sum() == 0:
            continue
        
        X_cohort = X[mask]
        y_cohort = y[mask]
        
        metrics = evaluate_model(model, X_cohort, y_cohort)
        metrics["cohort"] = cohort
        metrics["n_samples"] = mask.sum()
        metrics["churn_rate"] = y_cohort.mean()
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def evaluate_by_ltv_tier(
    model,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    ltv_column: str = "ltv_tier",
) -> pd.DataFrame:
    """
    Evaluate model performance by LTV tier.
    
    Important for understanding if model performs well on high-value customers.
    """
    results = []
    
    for tier in df[ltv_column].unique():
        mask = df[ltv_column] == tier
        
        if mask.sum() == 0:
            continue
        
        X_tier = X[mask]
        y_tier = y[mask]
        
        metrics = evaluate_model(model, X_tier, y_tier)
        metrics["ltv_tier"] = tier
        metrics["n_samples"] = mask.sum()
        metrics["churn_rate"] = y_tier.mean()
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def calculate_business_impact(
    y_true: pd.Series,
    y_proba: np.ndarray,
    ltv_values: pd.Series,
    intervention_cost: float = 50.0,
    save_rate: float = 0.40,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate business impact of model predictions.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        ltv_values: LTV for each customer
        intervention_cost: Cost per intervention
        save_rate: Expected rate of successful saves
        threshold: Classification threshold
        
    Returns:
        Dictionary with business metrics
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # True positives: churners we caught
    # These have a chance to be saved
    tp_ltv = ltv_values[((y_true == 1) & (y_pred == 1))].sum()
    expected_saved_revenue = tp_ltv * save_rate
    
    # False negatives: churners we missed
    fn_ltv = ltv_values[((y_true == 1) & (y_pred == 0))].sum()
    lost_revenue = fn_ltv
    
    # Intervention costs (all predicted positives)
    total_interventions = tp + fp
    total_intervention_cost = total_interventions * intervention_cost
    
    # Net benefit
    net_benefit = expected_saved_revenue - total_intervention_cost
    
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "total_interventions": int(total_interventions),
        "intervention_cost": total_intervention_cost,
        "expected_saved_revenue": expected_saved_revenue,
        "lost_revenue": lost_revenue,
        "net_benefit": net_benefit,
        "roi": net_benefit / max(total_intervention_cost, 1),
    }


def generate_evaluation_report(
    model,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Generate comprehensive evaluation report.
    
    Args:
        model: Trained classifier
        df: Full dataframe with metadata
        X: Features
        y: True labels
        
    Returns:
        Dictionary with all evaluation results
    """
    # Overall metrics
    overall_metrics = evaluate_model(model, X, y)
    
    # Cohort metrics
    cohort_metrics = evaluate_by_cohort(model, df, X, y)
    
    # LTV tier metrics
    ltv_metrics = evaluate_by_ltv_tier(model, df, X, y)
    
    # Optimal threshold
    y_proba = model.predict_proba(X)[:, 1]
    optimal_threshold, threshold_metrics = find_optimal_threshold(y, y_proba)
    
    # Business impact (if LTV available)
    if "estimated_ltv" in df.columns:
        business_impact = calculate_business_impact(
            y, y_proba, df["estimated_ltv"], threshold=optimal_threshold
        )
    else:
        business_impact = None
    
    return {
        "overall": overall_metrics,
        "by_cohort": cohort_metrics.to_dict(orient="records"),
        "by_ltv_tier": ltv_metrics.to_dict(orient="records"),
        "optimal_threshold": optimal_threshold,
        "threshold_metrics": threshold_metrics,
        "business_impact": business_impact,
    }
