"""
Model Evaluation Module for Churn Prediction

Implements:
- Standard classification metrics
- Business-specific metrics (Precision@K, Recall@Window, Lead Time)
- Cohort-stratified evaluation
- Threshold optimization (F1, cost-based, capacity-aware)
- Comprehensive evaluation reports
"""


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


def recall_at_window(
    y_true: pd.Series,
    y_proba: np.ndarray,
    prediction_dates: pd.Series,
    churn_dates: pd.Series,
    window_days: int = 30,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate recall within a specific time window.

    This answers: "Of customers who churned within 30 days,
    what percentage did we correctly identify?"

    Args:
        y_true: True labels (churned_in_window)
        y_proba: Predicted probabilities
        prediction_dates: Date when prediction was made
        churn_dates: Actual churn dates (NaT for non-churners)
        window_days: Prediction window in days
        threshold: Classification threshold

    Returns:
        Dictionary with recall metrics at different windows
    """
    y_pred = (y_proba >= threshold).astype(int)

    # Customers who actually churned
    churned_mask = y_true == 1

    if churned_mask.sum() == 0:
        return {
            "recall_in_window": 0.0,
            "churners_caught": 0,
            "churners_missed": 0,
            "total_churners": 0,
        }

    # Count true positives (churners we predicted)
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    false_negatives = ((y_true == 1) & (y_pred == 0)).sum()

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    return {
        "recall_in_window": float(recall),
        "churners_caught": int(true_positives),
        "churners_missed": int(false_negatives),
        "total_churners": int(churned_mask.sum()),
        "window_days": window_days,
    }


def calculate_lead_time(
    y_true: pd.Series,
    y_proba: np.ndarray,
    prediction_dates: pd.Series,
    churn_dates: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate lead time - days of warning before churn.

    Lead time is critical for interventions: more lead time =
    more time for CS to act.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        prediction_dates: Date when prediction was made
        churn_dates: Actual churn dates (NaT for non-churners)
        threshold: Classification threshold

    Returns:
        Dictionary with lead time statistics
    """
    y_pred = (y_proba >= threshold).astype(int)

    # True positives: churners we correctly identified
    tp_mask = (y_true == 1) & (y_pred == 1)

    if tp_mask.sum() == 0:
        return {
            "mean_lead_time_days": 0.0,
            "median_lead_time_days": 0.0,
            "min_lead_time_days": 0.0,
            "max_lead_time_days": 0.0,
            "std_lead_time_days": 0.0,
            "n_true_positives": 0,
        }

    # Calculate lead time for true positives
    pred_dates = pd.to_datetime(prediction_dates[tp_mask])
    actual_churn = pd.to_datetime(churn_dates[tp_mask])

    # Lead time = churn_date - prediction_date
    lead_times = (actual_churn - pred_dates).dt.days

    # Remove any negative lead times (shouldn't happen but safety check)
    lead_times = lead_times[lead_times >= 0]

    if len(lead_times) == 0:
        return {
            "mean_lead_time_days": 0.0,
            "median_lead_time_days": 0.0,
            "min_lead_time_days": 0.0,
            "max_lead_time_days": 0.0,
            "std_lead_time_days": 0.0,
            "n_true_positives": 0,
        }

    return {
        "mean_lead_time_days": float(lead_times.mean()),
        "median_lead_time_days": float(lead_times.median()),
        "min_lead_time_days": float(lead_times.min()),
        "max_lead_time_days": float(lead_times.max()),
        "std_lead_time_days": float(lead_times.std()),
        "n_true_positives": int(len(lead_times)),
    }


def capacity_aware_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    max_interventions: int,
    min_precision: float = 0.5,
) -> tuple[float, dict]:
    """
    Find threshold that respects CS team capacity constraints.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        max_interventions: Maximum number of customers CS can contact
        min_precision: Minimum precision required (avoid wasting CS time)

    Returns:
        Tuple of (optimal_threshold, metrics)
    """
    len(y_true)

    # Try thresholds from high to low
    best_threshold = 0.9
    best_metrics = None
    best_recall = 0.0

    for thresh in np.linspace(0.9, 0.1, 81):
        y_pred = (y_proba >= thresh).astype(int)
        n_predicted = y_pred.sum()

        # Skip if exceeds capacity
        if n_predicted > max_interventions:
            continue

        # Calculate precision
        if n_predicted == 0:
            continue

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        precision = tp / n_predicted

        # Skip if below minimum precision
        if precision < min_precision:
            continue

        # Calculate recall
        total_positives = y_true.sum()
        recall = tp / total_positives if total_positives > 0 else 0.0

        # Keep the threshold that maximizes recall within constraints
        if recall > best_recall:
            best_recall = recall
            best_threshold = thresh
            best_metrics = {
                "threshold": float(thresh),
                "n_interventions": int(n_predicted),
                "precision": float(precision),
                "recall": float(recall),
                "true_positives": int(tp),
                "capacity_utilization": float(n_predicted / max_interventions),
            }

    if best_metrics is None:
        # No threshold meets constraints, return highest threshold
        best_metrics = {
            "threshold": 0.9,
            "n_interventions": 0,
            "precision": 0.0,
            "recall": 0.0,
            "true_positives": 0,
            "capacity_utilization": 0.0,
            "warning": "No threshold meets constraints",
        }

    return best_threshold, best_metrics


def evaluate_at_multiple_thresholds(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate model at multiple thresholds for threshold selection.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: List of thresholds to evaluate (default: 0.1 to 0.9)

    Returns:
        DataFrame with metrics at each threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    base_rate = y_true.mean()

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        # Calculate metrics
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()

        n_predicted = tp + fp
        precision = tp / n_predicted if n_predicted > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": thresh,
            "n_predicted_positive": int(n_predicted),
            "pct_flagged": float(n_predicted / len(y_true)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "lift": float(precision / base_rate) if base_rate > 0 else 0.0,
        })

    return pd.DataFrame(results)


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
        for i, (p, r) in enumerate(zip(precisions[:-1], recalls[:-1], strict=False)):
            if p >= r:
                optimal_threshold = thresholds[i]
                break
        else:
            optimal_threshold = 0.5

    elif method == "cost":
        # Minimize expected cost
        n_samples = len(y_true)
        n_positive = y_true.sum()
        n_samples - n_positive

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


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from src.model.train import load_model, prepare_training_data
    from src.utils.duckdb_lakehouse import create_lakehouse
    from datetime import date as date_module, timedelta
    
    parser = argparse.ArgumentParser(description="Evaluate churn prediction model")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--days-offset", type=int, default=60, help="Days offset for test data")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    
    # 1. Load Model
    model_path = args.model or str(project_root / "outputs" / "models" / "churn_model_v1.joblib")
    print(f"Loading model from {model_path}...")
    model, features = load_model(model_path)
    
    # 2. Setup Lakehouse and Get Data
    db_path = str(project_root / "outputs" / "churn_lakehouse.duckdb")
    lakehouse = create_lakehouse(db_path)
    
    prediction_date = date_module.today() - timedelta(days=args.days_offset)
    print(f"Loading test data for {prediction_date}...")
    df = lakehouse.get_training_data(prediction_date)
    
    if len(df) == 0:
        print("Error: No data found for the specified date.")
        exit(1)
        
    # 3. Prepare Data
    X, y, weights = prepare_training_data(df)
    
    # 4. Generate Report
    print("Generating evaluation report...")
    report = generate_evaluation_report(model, df, X, y)
    
    # 5. Print Summary
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Overall AUC-PR: {report['overall']['auc_pr']:.3f}")
    print(f"Overall Recall: {report['overall']['recall']:.3f}")
    print(f"Precision@10%:  {report['overall']['precision_at_10pct']:.3f}")
    print(f"Optimal Threshold: {report['optimal_threshold']:.2f}")
    
    if report['business_impact']:
        print("-"*40)
        print(f"Expected ROI: {report['business_impact']['roi']:.1f}x")
        print(f"Net Benefit:  ${report['business_impact']['net_benefit']:,.0f}")
    print("="*40)
    
    # 6. Save Report
    report_path = project_root / "outputs" / "evaluation_report.json"
    # Convert types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Deeply nested serialization
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        else:
            return convert_to_serializable(obj)

    with open(report_path, "w") as f:
        json.dump(deep_convert(report), f, indent=2)
    
    print(f"Full report saved to {report_path}")
