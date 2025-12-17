#!/usr/bin/env python3
"""
End-to-End Churn Prediction Pipeline

Runs the complete pipeline:
1. Generate synthetic data
2. Build lakehouse
3. Train model
4. Evaluate and generate SHAP
5. Save artifacts

Usage:
    uv run python scripts/run_pipeline.py
"""

import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 70)
    print("CHURN PREDICTION PIPELINE")
    print("=" * 70)
    
    # Step 1: Generate synthetic data
    print("\n" + "=" * 70)
    print("STEP 1: Generate Synthetic Data")
    print("=" * 70)
    
    from src.data.generate_synthetic import main as generate_data
    generate_data()
    
    # Step 2: Build lakehouse
    print("\n" + "=" * 70)
    print("STEP 2: Build Lakehouse")
    print("=" * 70)
    
    from scripts.build_lakehouse import main as build_lakehouse
    build_lakehouse()
    
    # Step 3: Train model
    print("\n" + "=" * 70)
    print("STEP 3: Train Model")
    print("=" * 70)
    
    from src.utils.duckdb_lakehouse import create_lakehouse
    from src.model.train import prepare_training_data, train_lightgbm, load_config
    from src.model.evaluate import evaluate_model, evaluate_by_cohort, find_optimal_threshold
    from src.model.explain import compute_shap_values, get_feature_importance_shap, save_shap_artifacts
    
    # Load data
    lakehouse = create_lakehouse("outputs/churn_lakehouse.duckdb")
    df = lakehouse.query("SELECT * FROM gold.customer_360")
    
    print(f"Loaded {len(df):,} records for training")
    
    # Prepare data
    X, y, weights = prepare_training_data(df)
    
    # Temporal split (80/20)
    dates = pd.to_datetime(df["prediction_date"])
    split_date = dates.quantile(0.8)
    
    train_mask = dates <= split_date
    test_mask = dates > split_date
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    weights_train = weights[train_mask]
    
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Train
    config = load_config()
    model = train_lightgbm(
        X_train, y_train,
        X_test, y_test,
        sample_weights=weights_train,
        config=config
    )
    
    # Step 4: Evaluate
    print("\n" + "=" * 70)
    print("STEP 4: Evaluate Model")
    print("=" * 70)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\nTest Set Performance:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  Precision@10%: {metrics['precision_at_10pct']:.4f}")
    print(f"  Lift@10%: {metrics['lift_at_10pct']:.2f}x")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    
    # Cohort evaluation
    df_test = df[test_mask].copy()
    cohort_metrics = evaluate_by_cohort(model, df_test, X_test, y_test)
    
    print("\nPerformance by Cohort:")
    print(cohort_metrics[["cohort", "n_samples", "auc_pr", "precision_at_10pct"]].to_string(index=False))
    
    # Optimal threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    opt_thresh, _ = find_optimal_threshold(y_test, y_proba)
    print(f"\nOptimal Threshold (F1): {opt_thresh:.3f}")
    
    # Step 5: SHAP Analysis
    print("\n" + "=" * 70)
    print("STEP 5: SHAP Analysis")
    print("=" * 70)
    
    explainer, shap_values = compute_shap_values(model, X_test, sample_size=1000)
    importance_df = get_feature_importance_shap(shap_values, X_test.columns.tolist())
    
    print("\nTop 10 Features by SHAP Importance:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save artifacts
    output_dir = Path("outputs")
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, models_dir / "churn_model.joblib")
    
    # Save feature columns
    with open(models_dir / "feature_columns.txt", "w") as f:
        f.write("\n".join(X_train.columns.tolist()))
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    
    # Save SHAP artifacts
    save_shap_artifacts(shap_values, X_test.columns.tolist(), str(figures_dir))
    
    # Save cohort metrics
    cohort_metrics.to_csv(output_dir / "cohort_metrics.csv", index=False)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nArtifacts saved to: {output_dir}")
    print(f"  - Model: {models_dir / 'churn_model.joblib'}")
    print(f"  - Features: {models_dir / 'feature_columns.txt'}")
    print(f"  - Metrics: {output_dir / 'test_metrics.csv'}")
    print(f"  - SHAP: {figures_dir}")
    
    lakehouse.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
