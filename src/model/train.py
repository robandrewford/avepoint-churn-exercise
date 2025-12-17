"""
Model Training Module for Churn Prediction

Implements:
- LightGBM training with class weights and sample weights
- Hyperparameter tuning via Optuna
- MLflow integration (Fabric-compatible)
- Temporal cross-validation

Fabric Translation:
- MLflow calls work identically in Fabric managed MLflow
- Use mlflow.autolog() for automatic tracking
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.utils.temporal_split import TimeSeriesSplitter, validate_no_leakage


def load_config(config_path: str = "config/model_config.yaml") -> dict:
    """Load model configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns for modeling."""
    exclude_columns = {
        "customer_id",
        "prediction_date",
        "churned_in_window",
        "churn_date",
        "sample_weight",
        "cohort",  # Used for stratification, not as feature (or encode it)
        "ltv_tier",  # Same as above
        "contract_type",  # Needs encoding
    }
    
    # Get numeric columns only for now
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_columns]
    
    return feature_cols


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for modeling."""
    df = df.copy()
    
    # Encode cohort
    cohort_map = {"new_user": 0, "established": 1, "mature": 2}
    if "cohort" in df.columns:
        df["cohort_encoded"] = df["cohort"].map(cohort_map).fillna(1)
    
    # Encode LTV tier
    ltv_map = {"smb": 0, "mid_market": 1, "enterprise": 2}
    if "ltv_tier" in df.columns:
        df["ltv_tier_encoded"] = df["ltv_tier"].map(ltv_map).fillna(0)
    
    # Encode contract type
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    if "contract_type" in df.columns:
        df["contract_type_encoded"] = df["contract_type"].map(contract_map).fillna(0)
    
    return df


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = "churned_in_window",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for training.
    
    Returns:
        Tuple of (features, target, sample_weights)
    """
    # Encode categoricals
    df = encode_categorical_features(df)
    
    # Get feature columns (including encoded ones)
    feature_cols = get_feature_columns(df)
    feature_cols.extend(["cohort_encoded", "ltv_tier_encoded", "contract_type_encoded"])
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    y = df[target_column].astype(int)
    
    # Sample weights
    if "sample_weight" in df.columns:
        weights = df["sample_weight"]
    else:
        weights = pd.Series(np.ones(len(df)))
    
    # Fill NaN values
    X = X.fillna(0)
    
    return X, y, weights


def calculate_class_weight(y: pd.Series) -> dict:
    """Calculate balanced class weights."""
    n_samples = len(y)
    n_classes = 2
    
    class_counts = y.value_counts()
    weights = {}
    
    for cls in [0, 1]:
        count = class_counts.get(cls, 1)
        weights[cls] = n_samples / (n_classes * count)
    
    return weights


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    sample_weights: Optional[pd.Series] = None,
    config: Optional[dict] = None,
) -> lgb.LGBMClassifier:
    """
    Train LightGBM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        sample_weights: Sample weights for cost-sensitive learning
        config: Model configuration
        
    Returns:
        Trained LGBMClassifier
    """
    if config is None:
        config = load_config()
    
    lgbm_params = config["model"]["lgbm"].copy()
    
    # Extract params that aren't LightGBM native
    early_stopping = lgbm_params.pop("early_stopping_rounds", 50)
    
    # Calculate class weights
    class_weights = calculate_class_weight(y_train)
    scale_pos_weight = class_weights[1] / class_weights[0]
    lgbm_params["scale_pos_weight"] = scale_pos_weight
    
    # Create model
    model = lgb.LGBMClassifier(**lgbm_params)
    
    # Fit with optional validation
    fit_params = {}
    
    if sample_weights is not None:
        fit_params["sample_weight"] = sample_weights.values
    
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["callbacks"] = [
            lgb.early_stopping(stopping_rounds=early_stopping),
            lgb.log_evaluation(period=100),
        ]
    
    model.fit(X_train, y_train, **fit_params)
    
    return model


def train_with_mlflow(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weights: Optional[pd.Series] = None,
    experiment_name: str = "churn-prediction",
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
) -> tuple[lgb.LGBMClassifier, str]:
    """
    Train model with MLflow tracking.
    
    Fabric Integration:
    - In Fabric, MLflow is automatically configured
    - Just call mlflow.set_experiment() and tracking works
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        sample_weights: Sample weights
        experiment_name: MLflow experiment name
        run_name: Optional run name
        config: Model configuration
        
    Returns:
        Tuple of (trained model, run_id)
    """
    if config is None:
        config = load_config()
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    if run_name is None:
        run_name = f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Enable autologging
        mlflow.autolog()
        
        # Log custom parameters
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_val_samples", len(y_val))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("churn_rate_train", y_train.mean())
        mlflow.log_param("churn_rate_val", y_val.mean())
        mlflow.log_param("use_sample_weights", sample_weights is not None)
        
        # Train model
        model = train_lightgbm(
            X_train, y_train,
            X_val, y_val,
            sample_weights=sample_weights,
            config=config,
        )
        
        # Log additional metrics
        from src.model.evaluate import evaluate_model
        
        metrics = evaluate_model(model, X_val, y_val)
        
        mlflow.log_metric("val_auc_pr", metrics["auc_pr"])
        mlflow.log_metric("val_precision_at_10pct", metrics["precision_at_10pct"])
        mlflow.log_metric("val_recall", metrics["recall"])
        mlflow.log_metric("val_f1", metrics["f1"])
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        
        mlflow.log_table(feature_importance, "feature_importance.json")
        
        run_id = mlflow.active_run().info.run_id
    
    return model, run_id


def cross_validate_temporal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    n_splits: int = 5,
) -> dict:
    """
    Perform temporal cross-validation.
    
    Args:
        df: Full dataset with prediction_date column
        config: Model configuration
        n_splits: Number of CV folds
        
    Returns:
        Dictionary with CV metrics
    """
    if config is None:
        config = load_config()
    
    splitter = TimeSeriesSplitter(
        n_splits=n_splits,
        test_size_days=30,
        gap_days=0,
        min_train_size_days=60,
    )
    
    # Prepare full data
    X_full, y_full, weights_full = prepare_training_data(df)
    
    metrics_list = []
    
    for split in splitter.split(df, date_column="prediction_date"):
        # Validate no leakage
        assert validate_no_leakage(df, split.train_idx, split.test_idx)
        
        # Split data
        X_train = X_full.iloc[split.train_idx]
        y_train = y_full.iloc[split.train_idx]
        X_val = X_full.iloc[split.test_idx]
        y_val = y_full.iloc[split.test_idx]
        weights_train = weights_full.iloc[split.train_idx]
        
        # Train
        model = train_lightgbm(
            X_train, y_train,
            X_val, y_val,
            sample_weights=weights_train,
            config=config,
        )
        
        # Evaluate
        from src.model.evaluate import evaluate_model
        
        fold_metrics = evaluate_model(model, X_val, y_val)
        fold_metrics["fold"] = split.fold
        fold_metrics["train_size"] = len(X_train)
        fold_metrics["val_size"] = len(X_val)
        
        metrics_list.append(fold_metrics)
        
        print(f"Fold {split.fold}: AUC-PR={fold_metrics['auc_pr']:.3f}, "
              f"Precision@10%={fold_metrics['precision_at_10pct']:.3f}")
    
    # Aggregate metrics
    metrics_df = pd.DataFrame(metrics_list)
    
    cv_results = {
        "n_folds": len(metrics_list),
        "mean_auc_pr": metrics_df["auc_pr"].mean(),
        "std_auc_pr": metrics_df["auc_pr"].std(),
        "mean_precision_at_10pct": metrics_df["precision_at_10pct"].mean(),
        "std_precision_at_10pct": metrics_df["precision_at_10pct"].std(),
        "mean_recall": metrics_df["recall"].mean(),
        "std_recall": metrics_df["recall"].std(),
        "fold_metrics": metrics_list,
    }
    
    return cv_results


def save_model(
    model: lgb.LGBMClassifier,
    path: str,
    feature_columns: list[str],
) -> None:
    """Save model and feature columns."""
    import joblib
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, path)
    
    # Save feature columns
    feature_path = path.replace(".joblib", "_features.txt")
    with open(feature_path, "w") as f:
        f.write("\n".join(feature_columns))
    
    print(f"Saved model to {path}")


def load_model(path: str) -> tuple[lgb.LGBMClassifier, list[str]]:
    """Load model and feature columns."""
    import joblib
    
    model = joblib.load(path)
    
    feature_path = path.replace(".joblib", "_features.txt")
    with open(feature_path) as f:
        feature_columns = f.read().strip().split("\n")
    
    return model, feature_columns
