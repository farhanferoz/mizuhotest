from enum import StrEnum
from typing import List


class Models(StrEnum):
    LOGISTIC = "logistic"
    XGB = "xgb"


class Columns(StrEnum):
    # Company Metrics
    COMPANY_AGE = "company_age"
    EMPLOYEES = "employees"
    REVENUE_MILLIONS = "revenue_millions"
    REVENUE_GROWTH_RATE = "revenue_growth_rate"
    EBITDA_MARGIN = "ebitda_margin"
    INDUSTRY_SECTOR = "industry_sector"

    # IPO Characteristics
    OFFER_PRICE = "offer_price"
    SHARES_OFFERED_MILLIONS = "shares_offered_millions"
    UNDERWRITER_RANK = "underwriter_rank"
    VENTURE_BACKED = "venture_backed"
    LOCKUP_PERIOD_DAYS = "lockup_period_days"

    # Market Conditions
    MARKET_VOLATILITY_INDEX = "market_volatility_index"
    SECTOR_PERFORMANCE_30D = "sector_performance_30d"
    IPO_MONTH = "ipo_month"

    # Target Variable
    Q1_RETURN = "q1_return"

    # Engineered Features
    IPO_SIZE_M = "ipo_size_M"
    AGE_X_GROWTH = "age_x_growth"


# --- Feature Engineering Configuration ---
SKEWNESS_THRESHOLD: float = 1.0

# --- Model Training Configuration ---
TARGET_COLUMN: str = Columns.Q1_RETURN
TEST_SET_SIZE: float = 0.2
RANDOM_STATE: int = 2
CV_SPLITS: int = 5

# --- File Paths ---
SHAP_SUMMARY_PLOT_PATH: str = "shap_summary.png"

# --- Type Hinting Aliases ---
NumericFeatures = List[str]
CategoricalFeatures = List[str]
