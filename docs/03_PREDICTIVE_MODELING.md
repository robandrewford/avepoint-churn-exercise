# Predictive Modeling Strategy

## Overview

This document outlines the modeling approach for the churn prediction system, focusing on algorithm selection, validation methodology, and evaluation framework.

## Algorithm Selection

### LightGBM Choice

#### Rationale

Gradient Boosted Trees are the empirical State-of-the-Art (SOTA) for tabular classification tasks, particularly for imbalanced datasets like churn prediction.

**Key Advantages:**

1. **Native Class Imbalance Handling**: Built-in support for weighted learning
2. **Mixed Feature Types**: Handles categorical and numerical features without encoding
3. **Non-linear Relationships**: Captures complex feature interactions automatically
4. **Interpretability**: Excellent SHAP integration with exact Tree SHAP values
5. **Performance**: Fast training and inference, handles large datasets efficiently

#### Comparison to Alternatives

| Criterion | LightGBM | Logistic Regression | Neural Networks |
|-----------|------------|---------------------|----------------|
| Mixed Features | ✅ Native | Requires encoding | Requires encoding |
| Non-linearity | ✅ Native | ❌ Manual | ✅ Native |
| Interpretability | ✅ SHAP | ✅ Coefficients | ❌ Black box |
| Tabular SOTA | ✅ Yes | ⚠️ Baseline | ⚠️ Underperforms |
| Class Imbalance | ✅ Native | ✅ Weights | ⚠️ Requires tuning |
| Training Speed | ✅ Fast | ✅ Fast | ❌ Slow |
| Inference Speed | ✅ Fast | ✅ Fast | ❌ Slow |

## Training Strategy

### Cost-Sensitive Learning

#### LTV Weighting

Enterprise customers have 10x the churn cost of SMB customers. The model should reflect this business reality through weighted learning.

```python
# LTV weight mapping
WEIGHT_MAP = {
    'smb': 1.0,
    'mid_market': 3.0,
    'enterprise': 10.0
}

def calculate_sample_weights(df):
    """
    Calculate sample weights based on LTV tier
    """
    return df['ltv_tier'].map(WEIGHT_MAP).fillna(1.0)
```

#### Training Implementation

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, precision_recall_curve
import mlflow

# Train with business weights
sample_weights = calculate_sample_weights(train_df)

model = lgb.LGBMClassifier(
    objective='binary',
    metric='average_precision',  # Better for imbalanced
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    class_weight='balanced',  # Complements sample weights
    random_state=42
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50)]
)
```

### Threshold Optimization

#### Business-Conscious Threshold Selection

The optimal threshold balances catch more churners against the constraint of CS team capacity.

```python
def find_optimal_threshold(y_true, y_proba, capacity_pct=0.10):
    """
    Find threshold that maximizes expected value within capacity constraint
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find threshold within capacity constraint
    capacity_threshold_idx = int(len(y_true) * capacity_pct)
    if capacity_threshold_idx >= len(thresholds):
        return thresholds[capacity_threshold_idx - 1]
    else:
        return thresholds[capacity_threshold_idx]
```

## Validation Methodology

### Time Series Cross-Validation

#### Why Not Random CV?

Random cross-validation can lead to data leakage in time-series problems. Training on future data to predict past events doesn't reflect real-world deployment.

#### Temporal CV Design

```python
class TemporalTimeSeriesSplit:
    """
    Time-aware cross-validation for churn prediction
    """
    
    def __init__(self, n_splits=6, test_size_days=30, gap_days=0):
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
    
    def split(self, X, y=None, groups=None):
        dates = X['prediction_date'].unique()
        dates.sort()
        
        for i in range(self.n_splits):
            # Calculate split points
            test_start_idx = len(dates) - self.n_splits + i
            test_start = dates[test_start_idx]
            test_end = test_start + pd.Timedelta(days=self.test_size_days)
            
            train_end = test_start - pd.Timedelta(days=self.gap_days)
            
            # Create masks
            train_mask = X['prediction_date'] < train_end
            test_mask = (X['prediction_date'] >= test_start) & (X['prediction_date'] < test_end)
            
            yield X[train_mask], X[test_mask]
```

## Model Evaluation Framework

### Business Metrics

#### Precision@Top10%

Measures the precision when flagging the top 10% highest-risk customers. Critical metric because CS team has limited capacity.

```python
def precision_at_top_k(y_true, y_proba, k=0.10):
    """
    Calculate precision at top k percentage of customers
    """
    # Get top k riskiest customers
    k = int(len(y_true) * k)
    top_k_indices = y_proba.argsort()[::-1][:k]
    
    # Calculate precision
    true_positives = y_true.iloc[top_k_indices].sum()
    return true_positives / k
```

#### Recall@30d

Measures the proportion of customers who actually churn within 30 days and were flagged as high-risk.

#### Lead Time Analysis

Average time between prediction and actual churn for customers who were correctly predicted to churn.

### Calibration Metrics

#### Brier Score

Measures the accuracy of predicted probabilities. Lower scores indicate better-calibrated models.

```python
def brier_score(y_true, y_proba):
    """
    Calculate Brier skill score
    """
    return np.mean((y_proba - y_true) ** 2)
```

## Model Explainability

### SHAP Integration

SHAP (SHapley Additive exPlanations) provides both global and local model interpretability.

#### Global Feature Importance

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

#### Local Explanations

```python
# Individual customer explanations
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

### Intervention Mapping

Translate SHAP values to actionable business recommendations.

```python
INTERVENTION_MAP = {
    'login_velocity_wow': {
        'negative': {
            'action': 'Re-engagement campaign',
            'owner': 'Customer Success',
            'priority': 'HIGH'
        }
    },
    'features_used_30d': {
        'low': {
            'action': 'Feature training session',
            'owner': 'Product Team',
            'priority': 'MEDIUM'
        }
    },
    'days_to_renewal': {
        'critical': {
            'action': 'Strategic account review',
            'owner': 'Account Executive',
            'priority': 'HIGH'
        }
    }
}
```

## Model Registry & Deployment

### MLflow Integration

```python
import mlflow

# Set experiment
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():
    # Log parameters automatically
    mlflow.autolog()
    
    # Train model
    model.fit(X_train, y_train, sample_weight=weights)
    
    # Log custom metrics
    mlflow.log_metric("auc_pr", auc_pr_score)
    mlflow.log_metric("precision_at_10pct", precision_at_top_k)
    mlflow.log_metric("recall_at_30d", recall_at_30d)
    mlflow.log_metric("lead_time_days", avg_lead_time)
    
    # Log model
    mlflow.log_model(model, "churn_model")
```

### Production Scoring Pipeline

```python
def score_production_model(model, features, prediction_date):
    """
    Generate churn predictions for production deployment
    """
    # Generate probabilities
    churn_proba = model.predict_proba(features)
    
    # Add risk tiers
    features['churn_score'] = churn_proba[:, 1]
    features['risk_tier'] = pd.cut(
        features['churn_score'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    # Calculate intervention priority
    ltv_weights = {'smb': 1.0, 'mid_market': 3.0, 'enterprise': 10.0}
    features['intervention_priority'] = (
        features['churn_score'] * 
        features['ltv_tier'].map(ltv_weights)
    )
    
    return features
```

This modeling approach ensures:

1. **Business Alignment**: LTV-weighted learning reflects real revenue impact
2. **Temporal Correctness**: Time series CV prevents leakage
3. **Actionability**: SHAP explanations translate to interventions
4. **Production Ready**: Complete MLflow integration and scoring pipeline
