# System Patterns: Churn Prediction Architecture

## Architecture Overview

This system implements a cohort-aware, temporally correct churn prediction platform designed for enterprise SaaS scale. The architecture follows a medallion pattern with clear separation of concerns and production-ready monitoring.

## Core Architectural Patterns

### 1. Medallion Data Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA FLOW ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────┘

Raw Sources ──► Bronze ──► Silver ──► Gold ──► ML ──► Actions
    │             │        │        │       │        │
    │             │        │        │       │        │
    ▼             ▼        ▼        ▼       ▼        ▼
Events        Raw      Clean   Feature  Model   Interventions
Customers     Data     Validated Matrix Scores & Decisions
Support       │        │        │       │        │
Tickets       │        │        │       │        │
              │        │        │       │        │
              ▼        ▼        ▼       ▼        ▼
          Point-in-  Cohort- SHAP    Business
          Time       Aware   Explanations Impact
          Correct    Windows Analysis
```

**Pattern Benefits**:
- **Temporal Correctness**: Each layer maintains point-in-time integrity
- **Reproducibility**: Clear data lineage from raw to features
- **Scalability**: Layered processing enables incremental updates
- **Quality Gates**: Validation at each transformation step

### 2. Cohort-Aware Feature Engineering

**Multi-Temporal Window Pattern**:

```
CUSTOMER LIFECYCLE → COHORT SEGMENTATION → TIME WINDOWS

┌─────────────────┬─────────────────┬─────────────────┐
│   NEW USER      │   ESTABLISHED   │     MATURE      │
│   (0-30 days)   │   (30-180 days) │   (180+ days)   │
├─────────────────┼─────────────────┼─────────────────┤
│ Day 0-14       │ Rolling 30d   │ Rolling 60d     │
│ Prediction:    │ Prediction:    │ Prediction:     │
│ Days 15-45     │ Next 30d       │ Next 90d        │
│                 │                │                 │
│ Features:       │ Features:      │ Features:       │
│ • Activation    │ • Engagement   │ • Retention     │
│ • Onboarding    │ • Velocity     │ • Renewal       │
│ • First Value   │ • Patterns     │ • Expansion     │
└─────────────────┴─────────────────┴─────────────────┘
```

**Implementation Pattern**:
```sql
-- Cohort-specific window selection
CASE 
    WHEN cohort = 'new_user' THEN 
        SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '14 days' 
                 AND activity_date < prediction_date THEN metric END)
    WHEN cohort = 'established' THEN 
        SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '30 days' 
                 AND activity_date < prediction_date THEN metric END)
    WHEN cohort = 'mature' THEN 
        SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '60 days' 
                 AND activity_date < prediction_date THEN metric END)
END AS feature_name
```

### 3. Temporal Leakage Prevention Pattern

**Four-Layer Verification System**:

```
FEATURE AUDIT FRAMEWORK
┌─────────────────────────────────────────────────────────────┐
│ 1. SOURCE VALIDATION                                          │
│    • Raw data timestamp verification                         │
│    • Data availability at prediction time                   │
├─────────────────────────────────────────────────────────────┤
│ 2. WINDOW CORRECTNESS                                        │
│    • Observation window end < Prediction point              │
│    • No future data in feature calculation                  │
├─────────────────────────────────────────────────────────────┤
│ 3. TARGET ISOLATION                                          │
│    • Churn label period > Prediction horizon                │
│    • No target leakage into features                        │
├─────────────────────────────────────────────────────────────┤
│ 4. AGGREGATION VALIDATION                                    │
│    • All aggregates respect temporal boundaries            │
│    • No rolling windows that cross prediction point        │
└─────────────────────────────────────────────────────────────┘
```

**Audit Implementation**:
```python
def audit_feature(feature_name, source_query, window_start, window_end):
    """
    Audits a single feature for temporal correctness.
    
    Returns: {
        'feature': feature_name,
        'leakage_risk': 'LOW' | 'MEDIUM' | 'HIGH',
        'issues': [list of potential problems],
        'mitigation': [recommended fixes]
    }
    """
```

## Key Technical Patterns

### 1. Point-in-Time Feature Construction

**Pattern**: Build features as if you're standing at the prediction date

```sql
-- Template for point-in-time correct features
WITH customer_features AS (
    SELECT 
        customer_id,
        prediction_date,
        -- Engagement in lookback window (ONLY historical data)
        SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '30 days' 
                 AND activity_date < prediction_date 
                 THEN login_count ELSE 0 END) AS logins_30d,
        
        -- Recency from last activity (respecting prediction point)
        DATEDIFF(
            prediction_date,
            MAX(CASE WHEN activity_date < prediction_date 
                     THEN activity_date END)
        ) AS days_since_last_login,
        
        -- Velocity: change between historical periods
        (SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '7 days' 
                  AND activity_date < prediction_date THEN login_count END) -
         SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '14 days' 
                  AND activity_date < prediction_date - INTERVAL '7 days' 
                  THEN login_count END)) /
        NULLIF(SUM(CASE WHEN activity_date >= prediction_date - INTERVAL '14 days' 
                       AND activity_date < prediction_date - INTERVAL '7 days' 
                       THEN login_count END), 0) AS login_velocity_wow
        
    FROM silver.daily_engagement
    WHERE activity_date < prediction_date  -- Critical: NO future data
    GROUP BY customer_id, prediction_date
)
```

### 2. LTV-Weighted Learning Pattern

**Pattern**: Align model learning with business impact

```python
# Business weight calculation
def calculate_sample_weights(df):
    """
    Converts LTV tiers to model training weights.
    Higher LTV = higher weight = more influence on learning.
    """
    weight_map = {
        'smb': 1.0,        # Baseline
        'mid_market': 3.0, # 3x influence
        'enterprise': 10.0 # 10x influence
    }
    
    return df['ltv_tier'].map(weight_map).fillna(1.0)

# Training with business weights
model.fit(
    X_train, 
    y_train, 
    sample_weight=calculate_sample_weights(train_df)
)
```

### 3. Time Series Cross-Validation Pattern

**Pattern**: Validate temporal performance and detect concept drift

```python
class TemporalTimeSeriesSplit:
    """
    Implements time-aware cross-validation for churn prediction.
    
    Each fold respects temporal boundaries to prevent leakage.
    """
    
    def __init__(self, n_splits=6, test_size_days=30, gap_days=0):
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
    
    def split(self, X, y=None, groups=None):
        """Generate temporal train/test splits."""
        dates = X['prediction_date'].unique()
        dates.sort()
        
        for i in range(self.n_splits):
            # Calculate split points
            test_start = dates[-(self.n_splits - i) * self.test_size_days]
            test_end = test_start + pd.Timedelta(days=self.test_size_days)
            train_end = test_start - pd.Timedelta(days=self.gap_days)
            
            # Create masks
            train_mask = X['prediction_date'] < train_end
            test_mask = (X['prediction_date'] >= test_start) & \
                       (X['prediction_date'] < test_end)
            
            yield X[train_mask], X[test_mask]
```

### 4. SHAP-to-Action Mapping Pattern

**Pattern**: Convert model explanations to business actions

```python
class InterventionMapper:
    """
    Maps SHAP explanations to specific intervention playbooks.
    
    Pattern: Feature Driver → Business Meaning → Recommended Action
    """
    
    INTERVENTION_PLAYBOOK = {
        'login_velocity_wow': {
            'negative': {
                'meaning': 'Customer usage is declining',
                'action': 'Re-engagement campaign',
                'priority': 'HIGH',
                'owner': 'Customer Success',
                'template': 'usage_decline_campaign'
            }
        },
        'features_used_30d': {
            'low': {
                'meaning': 'Poor feature adoption',
                'action': 'Feature training session',
                'priority': 'MEDIUM',
                'owner': 'Product Specialist',
                'template': 'feature_adoption_training'
            }
        },
        'days_to_renewal': {
            'critical': {
                'meaning': 'Renewal approaching soon',
                'action': 'Strategic account review',
                'priority': 'HIGH',
                'owner': 'Account Executive',
                'template': 'renewal_review_meeting'
            }
        }
    }
    
    def map_shap_to_intervention(self, shap_values, feature_names, threshold=0.1):
        """
        Converts SHAP values to actionable recommendations.
        """
        interventions = []
        
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
            if abs(shap_val) > threshold:  # Only significant drivers
                playbook = self.INTERVENTION_PLAYBOOK.get(feature, {})
                
                # Determine condition based on SHAP value sign
                condition = 'negative' if shap_val < 0 else 'positive'
                if feature == 'features_used_30d':
                    condition = 'low' if shap_val < 0 else 'high'
                elif feature == 'days_to_renewal':
                    condition = 'critical' if shap_val < 0 else 'normal'
                
                if condition in playbook:
                    interventions.append({
                        'feature': feature,
                        'shap_value': shap_val,
                        'impact_score': abs(shap_val),
                        **playbook[condition]
                    })
        
        return sorted(interventions, key=lambda x: x['impact_score'], reverse=True)
```

## Production Patterns

### 1. Monitoring Framework Pattern

**Three-Pillar Monitoring Architecture**:

```python
class ChurnModelMonitor:
    """
    Comprehensive monitoring for production churn models.
    
    Pillars:
    1. Data Quality: Freshness, completeness, distribution
    2. Model Health: Performance, calibration, drift
    3. Business Impact: Intervention effectiveness, ROI
    """
    
    def __init__(self, config):
        self.data_quality_checker = DataQualityChecker(config.data_quality)
        self.model_health_checker = ModelHealthChecker(config.model_health)
        self.business_impact_tracker = BusinessImpactTracker(config.business)
    
    def run_daily_checks(self):
        """Execute all monitoring checks and alert on issues."""
        results = {}
        
        # Pillar 1: Data Quality
        results['data_quality'] = self.data_quality_checker.check_freshness()
        results['data_quality'].update(self.data_quality_checker.check_completeness())
        results['data_quality'].update(self.data_quality_checker.check_distribution_shift())
        
        # Pillar 2: Model Health
        results['model_health'] = self.model_health_checker.check_prediction_drift()
        results['model_health'].update(self.model_health_checker.check_calibration())
        results['model_health'].update(self.model_health_checker.check_cohort_performance())
        
        # Pillar 3: Business Impact
        results['business_impact'] = self.business_impact_tracker.check_intervention_rates()
        results['business_impact'].update(self.business_impact_tracker.check_save_rates())
        
        # Alert on critical issues
        self._check_alert_conditions(results)
        
        return results
```

### 2. Retraining Trigger Pattern

**Automated Retraining Decision Logic**:

```python
class RetrainingTrigger:
    """
    Determines when to retrain the churn model.
    
    Trigger Types:
    1. Scheduled: Time-based retraining
    2. Performance: Model degradation
    3. Drift: Data distribution changes
    4. Business: ROI thresholds
    """
    
    TRIGGER_THRESHOLDS = {
        'performance_drop': 0.10,      # 10% drop in AUC-PR
        'drift_psi': 0.25,           # Population Stability Index
        'calibration_increase': 0.20, # 20% increase in Brier score
        'business_roi_min': 0.30      # Minimum intervention save rate
    }
    
    def should_retrain(self, monitoring_results):
        """Evaluates all trigger conditions."""
        triggers = []
        
        # Performance-based triggers
        if monitoring_results['model_health']['auc_pr_drop'] > self.TRIGGER_THRESHOLDS['performance_drop']:
            triggers.append({
                'type': 'performance',
                'severity': 'HIGH',
                'reason': f"AUC-PR dropped by {monitoring_results['model_health']['auc_pr_drop']:.2%}"
            })
        
        # Drift-based triggers
        if monitoring_results['data_quality']['psi_max'] > self.TRIGGER_THRESHOLDS['drift_psi']:
            triggers.append({
                'type': 'drift',
                'severity': 'MEDIUM',
                'reason': f"PSI detected: {monitoring_results['data_quality']['psi_max']:.3f}"
            })
        
        # Business impact triggers
        if monitoring_results['business_impact']['save_rate'] < self.TRIGGER_THRESHOLDS['business_roi_min']:
            triggers.append({
                'type': 'business',
                'severity': 'HIGH',
                'reason': f"Save rate below threshold: {monitoring['business_impact']['save_rate']:.2%}"
            })
        
        return triggers
```

### 3. Deployment Safety Pattern

**Gradual Rollout with Automatic Rollback**:

```python
class ModelDeployer:
    """
    Safe model deployment with gradual rollout and rollback capability.
    
    Deployment Stages:
    1. Shadow: Model runs in parallel, no impact
    2. Canary: Small percentage of traffic
    3. Full: Complete rollout
    4. Rollback: Automatic if issues detected
    """
    
    def __init__(self, config):
        self.stages = ['shadow', 'canary', 'full']
        self.current_stage = None
        self.performance_monitor = PerformanceMonitor(config)
    
    def deploy_model(self, new_model, champion_model, stage='shadow'):
        """Deploy model with safety checks."""
        self.current_stage = stage
        
        if stage == 'shadow':
            return self._shadow_deploy(new_model, champion_model)
        elif stage == 'canary':
            return self._canary_deploy(new_model, champion_model)
        elif stage == 'full':
            return self._full_deploy(new_model)
        else:
            raise ValueError(f"Unknown deployment stage: {stage}")
    
    def _shadow_deploy(self, new_model, champion_model):
        """Run new model in parallel without affecting decisions."""
        # Both models score, but champion provides recommendations
        new_scores = new_model.predict(features)
        champion_scores = champion_model.predict(features)
        
        # Log comparison for performance evaluation
        self._log_shadow_comparison(new_scores, champion_scores, true_labels)
        
        return {"status": "shadow_mode", "comparison_logged": True}
    
    def _canary_deploy(self, new_model, champion_model, traffic_percentage=0.10):
        """Deploy to small percentage of traffic."""
        if random.random() < traffic_percentage:
            scores = new_model.predict(features)
            model_version = "candidate"
        else:
            scores = champion_model.predict(features)
            model_version = "champion"
        
        # Monitor performance differences
        self._monitor_canary_performance(scores, model_version)
        
        return {"status": "canary_mode", "traffic_percentage": traffic_percentage}
```

## Integration Patterns

### 1. Microsoft Fabric Integration Pattern

**Seamless Local-to-Production Translation**:

```python
class FabricDeployer:
    """
    Translates local DuckDB implementation to Microsoft Fabric.
    
    Translation Map:
    - DuckDB → Synapse SQL (nearly identical)
    - Parquet files → Delta tables
    - Local MLflow → Fabric MLflow
    - Marimo notebooks → Fabric notebooks
    """
    
    TRANSLATION_MAP = {
        'duckdb_to_fabric': {
            'bronze.customers': 'lakehouse.bronze.customers',
            'silver.daily_engagement': 'lakehouse.silver.daily_engagement',
            'gold.customer_360': 'lakehouse.gold.customer_360'
        },
        'file_formats': {
            'parquet': 'delta',
            'csv': 'delta'
        }
    }
    
    def translate_sql_to_fabric(self, duckdb_sql):
        """Converts DuckDB SQL to Fabric Synapse SQL."""
        # Most SQL is identical, only minor syntax differences
        fabric_sql = duckdb_sql
        
        # Replace DuckDB-specific functions with Synapse equivalents
        replacements = {
            'EXTRACT(DAY FROM': 'DATEPART(day,',
            'INTERVAL': 'DATEADD',
            '$prediction_date': '@prediction_date'
        }
        
        for duckdb_func, fabric_func in replacements.items():
            fabric_sql = fabric_sql.replace(duckdb_func, fabric_func)
        
        return fabric_sql
```

### 2. Configuration Management Pattern

**Centralized Configuration with Environment Overrides**:

```python
class ConfigManager:
    """
    Manages configuration across development, staging, and production.
    
    Hierarchy:
    1. Base configuration (config/model_config.yaml)
    2. Environment-specific overrides
    3. Runtime parameters
    """
    
    def __init__(self, base_config_path, environment='development'):
        self.base_config = self._load_yaml(base_config_path)
        self.environment = environment
        self._apply_environment_overrides()
    
    def get(self, key, default=None):
        """Get configuration value with environment awareness."""
        keys = key.split('.')
        value = self.base_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration."""
        env_overrides = {
            'production': {
                'data.n_customers': 1000000,
                'model.lgbm.n_estimators': 1000,
                'monitoring.data_quality.max_freshness_hours': 6
            },
            'staging': {
                'data.n_customers': 100000,
                'model.lgbm.n_estimators': 500
            },
            'development': {
                'data.n_customers': 50000,
                'model.lgbm.n_estimators': 200
            }
        }
        
        if self.environment in env_overrides:
            self._deep_update(self.base_config, env_overrides[self.environment])
```

**These system patterns provide a robust foundation for scalable, maintainable, and production-ready churn prediction systems.**
