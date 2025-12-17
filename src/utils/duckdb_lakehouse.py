"""
DuckDB Lakehouse Utility

Implements medallion architecture (Bronze/Silver/Gold) using DuckDB.
Maps 1:1 to Microsoft Fabric Synapse SQL patterns.

Fabric Translation:
- DuckDB schemas → Fabric Lakehouse schemas
- DuckDB tables → Delta tables in OneLake
- DuckDB SQL → Synapse SQL (nearly identical)
"""

from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from src.data.schema import BRONZE_DDL, GOLD_DDL, SILVER_DDL, get_all_ddl


class DuckDBLakehouse:
    """
    DuckDB-based lakehouse with medallion architecture.
    
    Bronze: Raw data as-is from source
    Silver: Cleaned, validated, deduplicated
    Gold: Feature tables, aggregates, ML-ready
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize lakehouse connection.
        
        Args:
            db_path: Path to DuckDB file, or ":memory:" for in-memory
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._initialized = False
    
    def initialize_schemas(self) -> None:
        """Create all schemas and tables."""
        if self._initialized:
            return
        
        ddl = get_all_ddl()
        self.conn.execute(ddl)
        self._initialized = True
        print(f"Initialized lakehouse schemas at {self.db_path}")
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =========================================================================
    # Bronze Layer Operations
    # =========================================================================
    
    def load_bronze_customers(self, df: pd.DataFrame) -> int:
        """Load raw customer data into bronze layer."""
        self.initialize_schemas()
        
        # Insert into bronze.customers
        self.conn.execute("DELETE FROM bronze.customers")
        self.conn.execute("""
            INSERT INTO bronze.customers 
            SELECT * FROM df
        """)
        
        count = self.conn.execute("SELECT COUNT(*) FROM bronze.customers").fetchone()[0]
        print(f"Loaded {count:,} customers into bronze.customers")
        return count
    
    def load_bronze_events(self, df: pd.DataFrame) -> int:
        """Load raw events into bronze layer."""
        self.initialize_schemas()
        
        self.conn.execute("DELETE FROM bronze.events")
        self.conn.execute("""
            INSERT INTO bronze.events (event_id, customer_id, event_type, event_timestamp, event_properties)
            SELECT event_id, customer_id, event_type, event_timestamp, NULL
            FROM df
        """)
        
        count = self.conn.execute("SELECT COUNT(*) FROM bronze.events").fetchone()[0]
        print(f"Loaded {count:,} events into bronze.events")
        return count
    
    def load_bronze_tickets(self, df: pd.DataFrame) -> int:
        """Load raw support tickets into bronze layer."""
        self.initialize_schemas()
        
        self.conn.execute("DELETE FROM bronze.support_tickets")
        self.conn.execute("""
            INSERT INTO bronze.support_tickets
            SELECT ticket_id, customer_id, created_at, resolved_at, 
                   category, priority, sentiment_score, escalated
            FROM df
        """)
        
        count = self.conn.execute("SELECT COUNT(*) FROM bronze.support_tickets").fetchone()[0]
        print(f"Loaded {count:,} tickets into bronze.support_tickets")
        return count
    
    # =========================================================================
    # Silver Layer Operations
    # =========================================================================
    
    def transform_to_silver_customers(self) -> int:
        """
        Transform bronze customers to silver layer.
        
        Transformations:
        - Calculate tenure from signup_date
        - Assign cohort based on tenure
        - Normalize service flags
        - Calculate contract months
        """
        self.initialize_schemas()
        
        self.conn.execute("DELETE FROM silver.customers")
        self.conn.execute("""
            INSERT INTO silver.customers
            SELECT 
                customer_id,
                signup_date,
                tenure_days,
                cohort,
                ltv_tier,
                contract_type,
                CASE contract_type
                    WHEN 'Month-to-month' THEN 1
                    WHEN 'One year' THEN 12
                    WHEN 'Two year' THEN 24
                    ELSE 1
                END AS contract_months,
                monthly_charges,
                estimated_ltv,
                COALESCE(phone_service, FALSE) AS has_phone,
                internet_service != 'No' AND internet_service IS NOT NULL AS has_internet,
                CASE WHEN internet_service = 'No' THEN NULL ELSE internet_service END AS internet_type,
                (
                    (CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN online_backup = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN streaming_tv = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END)
                ) AS n_addon_services,
                COALESCE(paperless_billing, FALSE) AS paperless_billing,
                payment_method,
                COALESCE(senior_citizen, FALSE) AS is_senior,
                COALESCE(partner, FALSE) AS has_partner,
                COALESCE(dependents, FALSE) AS has_dependents,
                churn_label,
                churn_date
            FROM bronze.customers
        """)
        
        count = self.conn.execute("SELECT COUNT(*) FROM silver.customers").fetchone()[0]
        print(f"Transformed {count:,} customers to silver.customers")
        return count
    
    def load_silver_daily_engagement(self, df: pd.DataFrame) -> int:
        """Load daily engagement data directly to silver (pre-aggregated)."""
        self.initialize_schemas()
        
        self.conn.execute("DELETE FROM silver.daily_engagement")
        self.conn.execute("""
            INSERT INTO silver.daily_engagement
            SELECT customer_id, activity_date, login_count, 
                   session_duration_minutes, features_used, unique_features,
                   0 AS support_tickets_opened, 0 AS support_tickets_resolved
            FROM df
        """)
        
        count = self.conn.execute("SELECT COUNT(*) FROM silver.daily_engagement").fetchone()[0]
        print(f"Loaded {count:,} records into silver.daily_engagement")
        return count
    
    def transform_to_silver_tickets(self) -> int:
        """Transform bronze tickets to silver layer."""
        self.initialize_schemas()
        
        self.conn.execute("DELETE FROM silver.support_tickets")
        self.conn.execute("""
            INSERT INTO silver.support_tickets
            SELECT 
                ticket_id,
                customer_id,
                CAST(created_at AS DATE) AS created_date,
                CASE 
                    WHEN resolved_at IS NOT NULL 
                    THEN EXTRACT(EPOCH FROM (resolved_at - created_at)) / 86400.0
                    ELSE NULL 
                END AS resolution_days,
                category,
                priority,
                COALESCE(sentiment_score, 0) AS sentiment_score,
                COALESCE(escalated, FALSE) AS was_escalated
            FROM bronze.support_tickets
        """)
        
        count = self.conn.execute("SELECT COUNT(*) FROM silver.support_tickets").fetchone()[0]
        print(f"Transformed {count:,} tickets to silver.support_tickets")
        return count
    
    # =========================================================================
    # Gold Layer Operations
    # =========================================================================
    
    def build_customer_360(self, prediction_date: date) -> int:
        """
        Build Customer 360 feature table for a given prediction date.
        
        This is the main feature engineering query that combines all sources
        into a single ML-ready table.
        
        Args:
            prediction_date: Point-in-time date for feature calculation
        """
        self.initialize_schemas()
        
        # Delete existing records for this prediction date
        self.conn.execute(
            "DELETE FROM gold.customer_360 WHERE prediction_date = ?",
            [prediction_date]
        )
        
        # Build features using point-in-time correct windows
        query = """
        INSERT INTO gold.customer_360
        WITH customer_base AS (
            SELECT 
                customer_id,
                signup_date,
                tenure_days,
                cohort,
                ltv_tier,
                contract_type,
                contract_months,
                monthly_charges,
                estimated_ltv,
                churn_label,
                churn_date,
                -- Calculate days to renewal (approximate)
                CASE 
                    WHEN contract_type = 'Month-to-month' THEN 30 - (tenure_days % 30)
                    WHEN contract_type = 'One year' THEN 365 - (tenure_days % 365)
                    WHEN contract_type = 'Two year' THEN 730 - (tenure_days % 730)
                    ELSE NULL
                END AS days_to_renewal_calc,
                -- Sample weight based on LTV tier
                CASE ltv_tier
                    WHEN 'smb' THEN 1.0
                    WHEN 'mid_market' THEN 3.0
                    WHEN 'enterprise' THEN 10.0
                    ELSE 1.0
                END AS sample_weight
            FROM silver.customers
            WHERE signup_date <= $prediction_date
        ),
        
        -- Engagement features with cohort-aware windows
        engagement_features AS (
            SELECT 
                e.customer_id,
                -- Frequency features
                SUM(CASE WHEN e.activity_date >= $prediction_date - INTERVAL '7 days' THEN e.login_count ELSE 0 END) AS logins_7d,
                SUM(CASE WHEN e.activity_date >= $prediction_date - INTERVAL '14 days' THEN e.login_count ELSE 0 END) AS logins_14d,
                SUM(CASE WHEN e.activity_date >= $prediction_date - INTERVAL '30 days' THEN e.login_count ELSE 0 END) AS logins_30d,
                
                -- Depth features
                SUM(CASE WHEN e.activity_date >= $prediction_date - INTERVAL '7 days' THEN e.features_used ELSE 0 END) AS features_used_7d,
                SUM(CASE WHEN e.activity_date >= $prediction_date - INTERVAL '30 days' THEN e.features_used ELSE 0 END) AS features_used_30d,
                
                -- Feature adoption (unique features / total available)
                COALESCE(MAX(e.unique_features) / 15.0, 0) AS feature_adoption_pct,
                
                -- Intensity features
                SUM(CASE WHEN e.activity_date >= $prediction_date - INTERVAL '7 days' THEN e.session_duration_minutes ELSE 0 END) AS session_minutes_7d,
                SUM(CASE WHEN e.activity_date >= $prediction_date - INTERVAL '30 days' THEN e.session_duration_minutes ELSE 0 END) AS session_minutes_30d,
                AVG(CASE WHEN e.login_count > 0 THEN e.session_duration_minutes / e.login_count ELSE NULL END) AS avg_session_duration,
                
                -- Recency features
                COALESCE(
                    EXTRACT(DAY FROM ($prediction_date - MAX(CASE WHEN e.login_count > 0 THEN e.activity_date END))),
                    999
                )::INTEGER AS days_since_last_login,
                COALESCE(
                    EXTRACT(DAY FROM ($prediction_date - MAX(CASE WHEN e.features_used > 0 THEN e.activity_date END))),
                    999
                )::INTEGER AS days_since_last_feature_use
                
            FROM silver.daily_engagement e
            WHERE e.activity_date < $prediction_date
            GROUP BY e.customer_id
        ),
        
        -- Velocity features (week-over-week change)
        velocity_features AS (
            SELECT 
                customer_id,
                -- Login velocity (this week vs last week)
                CASE 
                    WHEN logins_last_week > 0 
                    THEN (logins_this_week - logins_last_week) / logins_last_week::FLOAT
                    WHEN logins_this_week > 0 THEN 1.0
                    ELSE 0.0
                END AS login_velocity_wow,
                -- Feature velocity
                CASE 
                    WHEN features_last_week > 0 
                    THEN (features_this_week - features_last_week) / features_last_week::FLOAT
                    WHEN features_this_week > 0 THEN 1.0
                    ELSE 0.0
                END AS feature_velocity_wow,
                -- 4-week trend (simplified slope)
                CASE 
                    WHEN logins_week_4 + logins_week_3 > 0 
                    THEN (logins_week_1 + logins_week_2 - logins_week_3 - logins_week_4) / 
                         (logins_week_4 + logins_week_3 + logins_week_2 + logins_week_1 + 1)::FLOAT
                    ELSE 0.0
                END AS login_trend_4w
            FROM (
                SELECT 
                    customer_id,
                    SUM(CASE WHEN activity_date >= $prediction_date - INTERVAL '7 days' THEN login_count ELSE 0 END) AS logins_this_week,
                    SUM(CASE WHEN activity_date >= $prediction_date - INTERVAL '14 days' AND activity_date < $prediction_date - INTERVAL '7 days' THEN login_count ELSE 0 END) AS logins_last_week,
                    SUM(CASE WHEN activity_date >= $prediction_date - INTERVAL '7 days' THEN features_used ELSE 0 END) AS features_this_week,
                    SUM(CASE WHEN activity_date >= $prediction_date - INTERVAL '14 days' AND activity_date < $prediction_date - INTERVAL '7 days' THEN features_used ELSE 0 END) AS features_last_week,
                    SUM(CASE WHEN activity_date >= $prediction_date - INTERVAL '7 days' THEN login_count ELSE 0 END) AS logins_week_1,
                    SUM(CASE WHEN activity_date >= $prediction_date - INTERVAL '14 days' AND activity_date < $prediction_date - INTERVAL '7 days' THEN login_count ELSE 0 END) AS logins_week_2,
                    SUM(CASE WHEN activity_date >= $prediction_date - INTERVAL '21 days' AND activity_date < $prediction_date - INTERVAL '14 days' THEN login_count ELSE 0 END) AS logins_week_3,
                    SUM(CASE WHEN activity_date >= $prediction_date - INTERVAL '28 days' AND activity_date < $prediction_date - INTERVAL '21 days' THEN login_count ELSE 0 END) AS logins_week_4
                FROM silver.daily_engagement
                WHERE activity_date < $prediction_date
                GROUP BY customer_id
            ) weekly_agg
        ),
        
        -- Support features
        support_features AS (
            SELECT 
                customer_id,
                SUM(CASE WHEN created_date >= $prediction_date - INTERVAL '30 days' THEN 1 ELSE 0 END) AS tickets_30d,
                AVG(CASE WHEN created_date >= $prediction_date - INTERVAL '30 days' THEN sentiment_score ELSE NULL END) AS avg_sentiment_30d,
                AVG(CASE WHEN created_date >= $prediction_date - INTERVAL '30 days' AND was_escalated THEN 1.0 ELSE 0.0 END) AS escalation_rate_30d
            FROM silver.support_tickets
            WHERE created_date < $prediction_date
            GROUP BY customer_id
        ),
        
        -- Activation features (for new users only)
        activation_features AS (
            SELECT 
                c.customer_id,
                MIN(CASE WHEN e.login_count > 0 THEN EXTRACT(DAY FROM (e.activity_date - c.signup_date)) END)::INTEGER AS days_to_first_login,
                -- Onboarding completion approximated by first week feature usage
                COALESCE(
                    SUM(CASE WHEN e.activity_date <= c.signup_date + INTERVAL '7 days' THEN e.unique_features ELSE 0 END) / 15.0,
                    0
                ) AS onboarding_completion_pct,
                SUM(CASE WHEN e.activity_date <= c.signup_date + INTERVAL '7 days' THEN e.login_count ELSE 0 END) AS first_week_logins
            FROM silver.customers c
            LEFT JOIN silver.daily_engagement e ON c.customer_id = e.customer_id
            WHERE c.cohort = 'new_user'
              AND c.signup_date <= $prediction_date
              AND (e.activity_date IS NULL OR e.activity_date < $prediction_date)
            GROUP BY c.customer_id
        )
        
        SELECT 
            cb.customer_id,
            $prediction_date AS prediction_date,
            cb.cohort,
            cb.ltv_tier,
            -- Activation (NULL for non-new users)
            af.days_to_first_login,
            af.onboarding_completion_pct,
            af.first_week_logins,
            -- Engagement frequency
            COALESCE(ef.logins_7d, 0) AS logins_7d,
            COALESCE(ef.logins_14d, 0) AS logins_14d,
            COALESCE(ef.logins_30d, 0) AS logins_30d,
            -- Engagement depth
            COALESCE(ef.features_used_7d, 0) AS features_used_7d,
            COALESCE(ef.features_used_30d, 0) AS features_used_30d,
            COALESCE(ef.feature_adoption_pct, 0) AS feature_adoption_pct,
            -- Engagement intensity
            COALESCE(ef.session_minutes_7d, 0) AS session_minutes_7d,
            COALESCE(ef.session_minutes_30d, 0) AS session_minutes_30d,
            COALESCE(ef.avg_session_duration, 0) AS avg_session_duration,
            -- Recency
            COALESCE(ef.days_since_last_login, 999) AS days_since_last_login,
            COALESCE(ef.days_since_last_feature_use, 999) AS days_since_last_feature_use,
            -- Velocity
            COALESCE(vf.login_velocity_wow, 0) AS login_velocity_wow,
            COALESCE(vf.feature_velocity_wow, 0) AS feature_velocity_wow,
            COALESCE(vf.login_trend_4w, 0) AS login_trend_4w,
            -- Support
            COALESCE(sf.tickets_30d, 0) AS tickets_30d,
            COALESCE(sf.avg_sentiment_30d, 0) AS avg_sentiment_30d,
            COALESCE(sf.escalation_rate_30d, 0) AS escalation_rate_30d,
            -- Contract
            cb.contract_type,
            cb.days_to_renewal_calc AS days_to_renewal,
            cb.monthly_charges * COALESCE(cb.days_to_renewal_calc, cb.contract_months * 30) / 30.0 AS contract_value_remaining,
            cb.tenure_days,
            cb.estimated_ltv,
            -- Training metadata
            cb.sample_weight,
            -- Target: Did customer churn within 30 days of prediction date?
            CASE 
                WHEN cb.churn_date IS NOT NULL 
                     AND cb.churn_date > $prediction_date 
                     AND cb.churn_date <= $prediction_date + INTERVAL '30 days'
                THEN TRUE
                ELSE FALSE
            END AS churned_in_window,
            cb.churn_date
            
        FROM customer_base cb
        LEFT JOIN engagement_features ef ON cb.customer_id = ef.customer_id
        LEFT JOIN velocity_features vf ON cb.customer_id = vf.customer_id
        LEFT JOIN support_features sf ON cb.customer_id = sf.customer_id
        LEFT JOIN activation_features af ON cb.customer_id = af.customer_id
        """
        
        self.conn.execute(query, {"prediction_date": prediction_date})
        
        count = self.conn.execute(
            "SELECT COUNT(*) FROM gold.customer_360 WHERE prediction_date = ?",
            [prediction_date]
        ).fetchone()[0]
        
        print(f"Built Customer 360 for {prediction_date}: {count:,} records")
        return count
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        if params:
            return self.conn.execute(sql, params).df()
        return self.conn.execute(sql).df()
    
    def get_customer_360(
        self,
        prediction_date: Optional[date] = None,
        cohort: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get Customer 360 data with optional filters."""
        
        conditions = []
        params = {}
        
        if prediction_date:
            conditions.append("prediction_date = $prediction_date")
            params["prediction_date"] = prediction_date
        
        if cohort:
            conditions.append("cohort = $cohort")
            params["cohort"] = cohort
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"SELECT * FROM gold.customer_360 {where_clause}"
        return self.query(query, params)
    
    def get_training_data(self, prediction_date: date) -> pd.DataFrame:
        """Get ML-ready training data for a specific prediction date."""
        return self.get_customer_360(prediction_date=prediction_date)
    
    def get_schema_info(self) -> dict:
        """Get information about all tables in the lakehouse."""
        schemas = {}
        for schema in ["bronze", "silver", "gold"]:
            tables = self.conn.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema}'
            """).fetchall()
            schemas[schema] = [t[0] for t in tables]
        return schemas
    
    def get_table_counts(self) -> dict:
        """Get row counts for all tables."""
        counts = {}
        schema_info = self.get_schema_info()
        
        for schema, tables in schema_info.items():
            for table in tables:
                count = self.conn.execute(
                    f"SELECT COUNT(*) FROM {schema}.{table}"
                ).fetchone()[0]
                counts[f"{schema}.{table}"] = count
        
        return counts


def create_lakehouse(db_path: str = "outputs/churn_lakehouse.duckdb") -> DuckDBLakehouse:
    """Factory function to create and initialize lakehouse."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    lakehouse = DuckDBLakehouse(db_path)
    lakehouse.initialize_schemas()
    return lakehouse
