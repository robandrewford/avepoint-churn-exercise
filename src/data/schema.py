"""
Data Schema Definitions for Churn Prediction

Defines the structure of all tables in the medallion architecture:
- Bronze: Raw events
- Silver: Cleaned and sessionized
- Gold: Feature tables and Customer 360
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional


class Cohort(Enum):
    """Customer lifecycle cohort based on tenure."""
    NEW_USER = "new_user"           # 0-30 days
    ESTABLISHED = "established"     # 31-180 days
    MATURE = "mature"               # 181+ days


class LTVTier(Enum):
    """Customer value tier based on contract size."""
    SMB = "smb"
    MID_MARKET = "mid_market"
    ENTERPRISE = "enterprise"


class ChurnType(Enum):
    """Multi-tier churn taxonomy."""
    CONTRACTUAL_VOLUNTARY = "contractual_voluntary"
    CONTRACTUAL_INVOLUNTARY = "contractual_involuntary"
    ENGAGEMENT_DECAY = "engagement_decay"
    SILENT_CHURN = "silent_churn"


class EventType(Enum):
    """Types of behavioral events."""
    LOGIN = "login"
    FEATURE_USE = "feature_use"
    SUPPORT_TICKET = "support_ticket"
    PAYMENT = "payment"
    CONTRACT_EVENT = "contract_event"


# =============================================================================
# Bronze Layer Schemas (Raw Events)
# =============================================================================

@dataclass
class CustomerRaw:
    """Raw customer record from source system."""
    customer_id: str
    signup_date: date
    contract_type: str          # Month-to-month, One year, Two year
    monthly_charges: float
    total_charges: float
    payment_method: str
    gender: Optional[str]
    senior_citizen: bool
    partner: bool
    dependents: bool
    # Service flags
    phone_service: bool
    multiple_lines: Optional[str]
    internet_service: str       # DSL, Fiber optic, No
    online_security: Optional[str]
    online_backup: Optional[str]
    device_protection: Optional[str]
    tech_support: Optional[str]
    streaming_tv: Optional[str]
    streaming_movies: Optional[str]
    paperless_billing: bool


@dataclass
class EventRaw:
    """Raw behavioral event."""
    event_id: str
    customer_id: str
    event_type: str             # login, feature_use, support_ticket, etc.
    event_timestamp: datetime
    event_properties: Optional[dict]  # JSON blob for event-specific data


@dataclass
class SupportTicketRaw:
    """Raw support ticket record."""
    ticket_id: str
    customer_id: str
    created_at: datetime
    resolved_at: Optional[datetime]
    category: str
    priority: str
    sentiment_score: Optional[float]  # -1 to 1
    escalated: bool


# =============================================================================
# Silver Layer Schemas (Cleaned & Sessionized)
# =============================================================================

@dataclass
class CustomerClean:
    """Cleaned customer record with derived fields."""
    customer_id: str
    signup_date: date
    tenure_days: int
    cohort: str                 # Cohort enum value
    ltv_tier: str               # LTVTier enum value
    contract_type: str
    contract_months: int        # Derived from contract_type
    monthly_charges: float
    estimated_ltv: float        # Calculated LTV
    # Simplified service flags
    has_phone: bool
    has_internet: bool
    internet_type: Optional[str]
    n_addon_services: int       # Count of add-on services
    paperless_billing: bool
    payment_method: str
    # Demographics
    is_senior: bool
    has_partner: bool
    has_dependents: bool


@dataclass
class DailyEngagement:
    """Daily engagement summary per customer."""
    customer_id: str
    activity_date: date
    login_count: int
    session_duration_minutes: float
    features_used: int
    unique_features: int
    support_tickets_opened: int
    support_tickets_resolved: int


@dataclass
class SupportTicketClean:
    """Cleaned support ticket with derived fields."""
    ticket_id: str
    customer_id: str
    created_date: date
    resolution_days: Optional[float]
    category: str
    priority: str
    sentiment_score: float
    was_escalated: bool


# =============================================================================
# Gold Layer Schemas (Features & Customer 360)
# =============================================================================

@dataclass
class ActivationFeatures:
    """Features for new user activation tracking."""
    customer_id: str
    prediction_date: date
    # Time to first value
    days_to_first_login: Optional[int]
    days_to_first_feature_use: Optional[int]
    # Onboarding completion
    onboarding_completion_pct: float
    setup_steps_completed: int
    # First week activity
    first_week_logins: int
    first_week_features_used: int
    first_week_session_minutes: float


@dataclass
class EngagementFeatures:
    """Features for engagement tracking."""
    customer_id: str
    prediction_date: date
    cohort: str
    # Frequency (window-based)
    logins_7d: int
    logins_14d: int
    logins_30d: int
    # Depth
    features_used_7d: int
    features_used_14d: int
    features_used_30d: int
    feature_adoption_pct: float
    # Intensity
    session_minutes_7d: float
    session_minutes_14d: float
    session_minutes_30d: float
    avg_session_duration: float
    # Recency
    days_since_last_login: int
    days_since_last_feature_use: int


@dataclass
class VelocityFeatures:
    """Features for engagement velocity (change over time)."""
    customer_id: str
    prediction_date: date
    # Week-over-week changes
    login_velocity_wow: float       # (this_week - last_week) / last_week
    feature_velocity_wow: float
    session_velocity_wow: float
    # Trend (slope over 4 weeks)
    login_trend_4w: float
    feature_trend_4w: float
    # Acceleration (change in velocity)
    login_acceleration: float


@dataclass
class SupportFeatures:
    """Features from support interactions."""
    customer_id: str
    prediction_date: date
    # Volume
    tickets_7d: int
    tickets_30d: int
    tickets_90d: int
    # Sentiment
    avg_sentiment_30d: float
    min_sentiment_30d: float
    negative_ticket_pct: float
    # Resolution
    avg_resolution_days_30d: float
    escalation_rate_30d: float


@dataclass
class ContractFeatures:
    """Features related to contract and billing."""
    customer_id: str
    prediction_date: date
    # Contract
    contract_type: str
    days_to_renewal: Optional[int]
    contract_value_remaining: float
    # Payment
    payment_failures_90d: int
    late_payments_90d: int
    # History
    tenure_days: int
    previous_renewals: int
    expansion_events: int
    ltv_tier: str
    estimated_ltv: float


@dataclass
class Customer360:
    """Unified customer view with all features for modeling."""
    customer_id: str
    prediction_date: date
    # Identifiers
    cohort: str
    ltv_tier: str
    # Activation (null for non-new users)
    days_to_first_login: Optional[int]
    onboarding_completion_pct: Optional[float]
    first_week_logins: Optional[int]
    # Engagement frequency
    logins_7d: int
    logins_14d: int
    logins_30d: int
    # Engagement depth
    features_used_7d: int
    features_used_30d: int
    feature_adoption_pct: float
    # Engagement intensity
    session_minutes_7d: float
    session_minutes_30d: float
    avg_session_duration: float
    # Recency
    days_since_last_login: int
    days_since_last_feature_use: int
    # Velocity
    login_velocity_wow: float
    feature_velocity_wow: float
    login_trend_4w: float
    # Support
    tickets_30d: int
    avg_sentiment_30d: float
    escalation_rate_30d: float
    # Contract
    contract_type: str
    days_to_renewal: Optional[int]
    contract_value_remaining: float
    tenure_days: int
    estimated_ltv: float
    # Sample weight (for training)
    sample_weight: float
    # Target (for training)
    churned_in_window: Optional[bool]
    churn_date: Optional[date]


# =============================================================================
# DuckDB Schema DDL
# =============================================================================

BRONZE_DDL = """
-- Bronze Layer: Raw Events

CREATE SCHEMA IF NOT EXISTS bronze;

CREATE TABLE IF NOT EXISTS bronze.customers (
    customer_id VARCHAR PRIMARY KEY,
    signup_date DATE NOT NULL,
    tenure_days INTEGER,
    cohort VARCHAR,
    ltv_tier VARCHAR,
    contract_type VARCHAR,
    monthly_charges DECIMAL(10,2),
    estimated_ltv DECIMAL(12,2),
    sample_weight DECIMAL(8,4),
    payment_method VARCHAR,
    gender VARCHAR,
    senior_citizen BOOLEAN,
    partner BOOLEAN,
    dependents BOOLEAN,
    phone_service BOOLEAN,
    multiple_lines VARCHAR,
    internet_service VARCHAR,
    online_security VARCHAR,
    online_backup VARCHAR,
    device_protection VARCHAR,
    tech_support VARCHAR,
    streaming_tv VARCHAR,
    streaming_movies VARCHAR,
    paperless_billing BOOLEAN,
    churn_label BOOLEAN,
    churn_date DATE
);

CREATE TABLE IF NOT EXISTS bronze.events (
    event_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    event_timestamp TIMESTAMP NOT NULL,
    event_properties JSON
);

CREATE TABLE IF NOT EXISTS bronze.support_tickets (
    ticket_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    category VARCHAR,
    priority VARCHAR,
    sentiment_score DECIMAL(3,2),
    escalated BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_events_customer ON bronze.events(customer_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON bronze.events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON bronze.support_tickets(customer_id);
"""

SILVER_DDL = """
-- Silver Layer: Cleaned & Sessionized

CREATE SCHEMA IF NOT EXISTS silver;

CREATE TABLE IF NOT EXISTS silver.customers (
    customer_id VARCHAR PRIMARY KEY,
    signup_date DATE NOT NULL,
    tenure_days INTEGER,
    cohort VARCHAR,
    ltv_tier VARCHAR,
    contract_type VARCHAR,
    contract_months INTEGER,
    monthly_charges DECIMAL(10,2),
    estimated_ltv DECIMAL(12,2),
    has_phone BOOLEAN,
    has_internet BOOLEAN,
    internet_type VARCHAR,
    n_addon_services INTEGER,
    paperless_billing BOOLEAN,
    payment_method VARCHAR,
    is_senior BOOLEAN,
    has_partner BOOLEAN,
    has_dependents BOOLEAN,
    churn_label BOOLEAN,
    churn_date DATE
);

CREATE TABLE IF NOT EXISTS silver.daily_engagement (
    customer_id VARCHAR NOT NULL,
    activity_date DATE NOT NULL,
    login_count INTEGER DEFAULT 0,
    session_duration_minutes DECIMAL(10,2) DEFAULT 0,
    features_used INTEGER DEFAULT 0,
    unique_features INTEGER DEFAULT 0,
    support_tickets_opened INTEGER DEFAULT 0,
    support_tickets_resolved INTEGER DEFAULT 0,
    PRIMARY KEY (customer_id, activity_date)
);

CREATE TABLE IF NOT EXISTS silver.support_tickets (
    ticket_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR NOT NULL,
    created_date DATE NOT NULL,
    resolution_days DECIMAL(5,2),
    category VARCHAR,
    priority VARCHAR,
    sentiment_score DECIMAL(3,2),
    was_escalated BOOLEAN
);
"""

GOLD_DDL = """
-- Gold Layer: Features & Customer 360

CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS gold.customer_360 (
    customer_id VARCHAR NOT NULL,
    prediction_date DATE NOT NULL,
    -- Identifiers
    cohort VARCHAR,
    ltv_tier VARCHAR,
    -- Activation features
    days_to_first_login INTEGER,
    onboarding_completion_pct DECIMAL(5,2),
    first_week_logins INTEGER,
    -- Engagement frequency
    logins_7d INTEGER,
    logins_14d INTEGER,
    logins_30d INTEGER,
    -- Engagement depth
    features_used_7d INTEGER,
    features_used_30d INTEGER,
    feature_adoption_pct DECIMAL(5,2),
    -- Engagement intensity
    session_minutes_7d DECIMAL(10,2),
    session_minutes_30d DECIMAL(10,2),
    avg_session_duration DECIMAL(10,2),
    -- Recency
    days_since_last_login INTEGER,
    days_since_last_feature_use INTEGER,
    -- Velocity
    login_velocity_wow DECIMAL(8,4),
    feature_velocity_wow DECIMAL(8,4),
    login_trend_4w DECIMAL(8,4),
    -- Support
    tickets_30d INTEGER,
    avg_sentiment_30d DECIMAL(5,2),
    escalation_rate_30d DECIMAL(5,2),
    -- Contract
    contract_type VARCHAR,
    days_to_renewal INTEGER,
    contract_value_remaining DECIMAL(12,2),
    tenure_days INTEGER,
    estimated_ltv DECIMAL(12,2),
    -- Training metadata
    sample_weight DECIMAL(8,4),
    churned_in_window BOOLEAN,
    churn_date DATE,
    PRIMARY KEY (customer_id, prediction_date)
);

CREATE INDEX IF NOT EXISTS idx_c360_date ON gold.customer_360(prediction_date);
CREATE INDEX IF NOT EXISTS idx_c360_cohort ON gold.customer_360(cohort);
"""


def get_all_ddl() -> str:
    """Return combined DDL for all layers."""
    return f"{BRONZE_DDL}\n{SILVER_DDL}\n{GOLD_DDL}"
