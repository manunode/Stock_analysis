#!/usr/bin/env python3
"""
Stock Analysis from Parquet Files
Reads screener.in parquet data (6 files, ~5275 stocks) and produces
a multi-sheet stock_analysis.xlsx report with scoring, red flags, and decision buckets.

Usage:
    python create_analysis_from_parquet.py
    python create_analysis_from_parquet.py --output custom_output.xlsx
    python create_analysis_from_parquet.py --log-level DEBUG
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Union, Optional

# ── Logging Configuration ──────────────────────────────────────────────────
def setup_logging(level: str = "INFO") -> None:
    """Configure logging with timestamp and severity level."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Parquet file paths ──────────────────────────────────────────────────────
PARQUET_FILES = {
    'annual': os.path.join(BASE_DIR, 'all_stocks_annual_financials.parquet'),
    'quarterly': os.path.join(BASE_DIR, 'all_stocks_quarterly_financials.parquet'),
    'balance': os.path.join(BASE_DIR, 'all_stocks_balance_sheet.parquet'),
    'cashflow': os.path.join(BASE_DIR, 'all_stocks_cash_flow.parquet'),
    'ratios': os.path.join(BASE_DIR, 'all_stocks_ratios.parquet'),
    'user_ratios': os.path.join(BASE_DIR, 'all_stocks_user_ratios.parquet'),
}

# ── Column Name Mapping (Single Source of Truth) ────────────────────────────
COLS = {
    # Identifiers
    'NAME': 'Name',
    'BSE_CODE': 'BSE Code',
    'NSE_CODE': 'NSE Code',
    'ISIN_CODE': 'ISIN Code',
    'INDUSTRY_GROUP': 'Industry Group',
    'INDUSTRY': 'Industry',
    'CURRENT_PRICE': 'Current Price',
    'MARKET_CAP': 'Market Capitalization',
    
    # Valuation Metrics
    'PE': 'Price to Earning',
    'PBV': 'Price to book value',
    'EVEBITDA': 'EVEBITDA',
    'PEG_RATIO': 'PEG Ratio',
    'PRICE_TO_SALES': 'Price to Sales',
    'BOOK_VALUE': 'Book value',
    'INDUSTRY_PE': 'Industry PE',
    'INDUSTRY_PBV': 'Industry PBV',
    'HISTORICAL_PE_3Y': 'Historical PE 3Years',
    'HISTORICAL_PE_5Y': 'Historical PE 5Years',
    'HISTORICAL_PE_10Y': 'Historical PE 10Years',
    'DOWN_52W_HIGH': 'Down from 52w high',
    'UP_52W_LOW': 'Up from 52w low',
    'PRICE_QUARTERLY_EARNING': 'Price to Quarterly Earning',
    'MARKET_CAP_3Y_BACK': 'Market Capitalization 3years back',
    'MARKET_CAP_5Y_BACK': 'Market Capitalization 5years back',
    'MARKET_CAP_10Y_BACK': 'Market Capitalization 10years back',
    
    # Quality Metrics
    'ROE': 'Return on equity',
    'ROE_3Y_AVG': 'Average return on equity 3Years',
    'ROE_5Y_AVG': 'Average return on equity 5Years',
    'ROE_10Y_AVG': 'Average return on equity 10Years',
    'ROCE': 'Return on capital employed',
    'ROCE_3Y_AVG': 'Average return on capital employed 3Years',
    'ROCE_5Y_AVG': 'Average return on capital employed 5Years',
    'ROCE_7Y_AVG': 'Average return on capital employed 7Years',
    'ROCE_10Y_AVG': 'Average return on capital employed 10Years',
    'ROA': 'Return on assets',
    'ROA_3Y': 'Return on assets 3years',
    'ROA_5Y': 'Return on assets 5years',
    'OPM': 'OPM',
    'OPM_LAST_YEAR': 'OPM last year',
    'OPM_PRECEDING_YEAR': 'OPM preceding year',
    'OPM_5Y': 'OPM 5Year',
    'OPM_10Y': 'OPM 10Year',
    'NPM_LAST_YEAR': 'NPM last year',
    'NPM_PRECEDING_YEAR': 'NPM preceding year',
    'PIOTROSKI': 'Piotroski score',
    'ASSET_TURNOVER': 'Asset Turnover Ratio',
    
    # Profit & Income
    'PAT': 'Profit after tax',
    'PAT_LAST_YEAR': 'Profit after tax last year',
    'PAT_PRECEDING_YEAR': 'Profit after tax preceding year',
    'PBT_LAST_YEAR': 'Profit before tax last year',
    'PBT_PRECEDING_YEAR': 'Profit before tax preceding year',
    'OTHER_INCOME': 'Other income',

    # Quarterly Profit (for TTM PAT)
    'NP_Q1': 'Net Profit latest quarter',
    'NP_Q2': 'Net Profit preceding quarter',
    'NP_Q3': 'Net profit 2quarters back',
    'NP_Q4': 'Net profit 3quarters back',
    
    # Cash Flow Metrics
    'CFO_LAST_YEAR': 'Cash from operations last year',
    'CFO_PRECEDING_YEAR': 'Cash from operations preceding year',
    'CFO_3Y_CUMULATIVE': 'Operating cash flow 3years',
    'CFO_5Y_CUMULATIVE': 'Operating cash flow 5years',
    'CFO_7Y_CUMULATIVE': 'Operating cash flow 7years',
    'CFO_10Y_CUMULATIVE': 'Operating cash flow 10years',
    'FCF_LAST_YEAR': 'Free cash flow last year',
    'FCF_PRECEDING_YEAR': 'Free cash flow preceding year',
    'FCF_3Y_CUMULATIVE': 'Free cash flow 3years',
    'FCF_5Y_CUMULATIVE': 'Free cash flow 5years',
    'FCF_7Y_CUMULATIVE': 'Free cash flow 7years',
    'FCF_10Y_CUMULATIVE': 'Free cash flow 10years',
    'CFI_LAST_YEAR': 'Cash from investing last year',
    'CFI_PRECEDING_YEAR': 'Cash from investing preceding year',
    'CFI_3Y_CUMULATIVE': 'Investing cash flow 3years',
    'CFI_5Y_CUMULATIVE': 'Investing cash flow 5years',
    'CFF_LAST_YEAR': 'Cash from financing last year',
    'CFF_PRECEDING_YEAR': 'Cash from financing preceding year',
    
    # Balance Sheet Metrics
    'TOTAL_ASSETS': 'Total Assets',
    'WORKING_CAPITAL': 'Working capital',
    'WORKING_CAPITAL_PY': 'Working capital preceding year',
    'DEBT': 'Debt',
    'DEBT_PY': 'Debt preceding year',
    'DEBT_3Y_BACK': 'Debt 3Years back',
    'DEBT_5Y_BACK': 'Debt 5Years back',
    'RESERVES': 'Reserves',
    'CURRENT_ASSETS': 'Current assets',
    'CURRENT_LIABILITIES': 'Current liabilities',
    'NET_BLOCK': 'Net block',
    'GROSS_BLOCK': 'Gross block',
    'GROSS_BLOCK_PY': 'Gross block preceding year',
    'NET_BLOCK_PY': 'Net block preceding year',
    'CONTINGENT_LIABILITIES': 'Contingent liabilities',
    'CASH_EQUIVALENTS': 'Cash Equivalents',
    'TRADE_PAYABLES': 'Trade Payables',
    'FACE_VALUE': 'Face value',
    
    # Leverage Metrics
    'DEBT_EQUITY': 'Debt to equity',
    'INTEREST_COVERAGE': 'Interest Coverage Ratio',
    'AVG_WC_DAYS_3Y': 'Average Working Capital Days 3years',
    
    # Growth Metrics
    'SALES': 'Sales',
    'SALES_LAST_YEAR': 'Sales last year',
    'SALES_GROWTH_3Y': 'Sales growth 3Years',
    'SALES_GROWTH_5Y': 'Sales growth 5Years',
    'SALES_GROWTH_10Y': 'Sales growth 10Years',
    'PROFIT_GROWTH_3Y': 'Profit growth 3Years',
    'PROFIT_GROWTH_5Y': 'Profit growth 5Years',
    'PROFIT_GROWTH_7Y': 'Profit growth 7Years',
    'PROFIT_GROWTH_10Y': 'Profit growth 10Years',
    'EPS': 'EPS',
    'EPS_GROWTH_3Y': 'EPS growth 3Years',
    'EPS_GROWTH_5Y': 'EPS growth 5Years',
    'EPS_GROWTH_10Y': 'EPS growth 10Years',
    'EBITDA_GROWTH_3Y': 'EBITDA growth 3Years',
    'EBITDA_GROWTH_5Y': 'EBITDA growth 5Years',
    'EBITDA_GROWTH_10Y': 'EBITDA growth 10Years',
    
    # Quarterly Metrics
    'QOQ_SALES': 'QoQ Sales',
    'QOQ_PROFITS': 'QoQ Profits',
    'YOY_Q_SALES': 'YOY Quarterly sales growth',
    'YOY_Q_PROFIT': 'YOY Quarterly profit growth',
    'OPM_Q_LATEST': 'OPM latest quarter',
    'OPM_Q_PRECEDING': 'OPM preceding quarter',
    'OPM_Q_YOY': 'OPM preceding year quarter',
    'SALES_Q_LATEST': 'Sales latest quarter',
    'SALES_Q_PRECEDING': 'Sales preceding quarter',
    'SALES_Q_YOY': 'Sales preceding year quarter',
    'OP_Q_LATEST': 'Operating profit latest quarter',
    'OP_Q_PRECEDING': 'Operating profit preceding quarter',
    'OP_Q_YOY': 'Operating profit preceding year quarter',
    
    # Shareholding Metrics
    'PROMOTER_HOLDING': 'Promoter holding',
    'FII_HOLDING': 'FII holding',
    'DII_HOLDING': 'DII holding',
    'FII_CHANGE_QTR': 'Change in FII holding',
    'DII_CHANGE_QTR': 'Change in DII holding',
    'FII_CHANGE_3Y': 'Change in FII holding 3Years',
    'DII_CHANGE_3Y': 'Change in DII holding 3Years',
    'PROMOTER_CHANGE_3Y': 'Change in promoter holding 3Years',
    'NUM_SHAREHOLDERS': 'Number of Shareholders preceding quarter',
    
    # Dividend Metrics
    'DIVIDEND_LAST_YEAR': 'Dividend last year',
    'DIVIDEND_PRECEDING_YEAR': 'Dividend preceding year',
    'DIVIDEND_PAYOUT': 'Dividend Payout',
    
    # Additional
    'BOOK_VALUE_3Y_BACK': 'Book value 3years back',
    'BOOK_VALUE_5Y_BACK': 'Book value 5years back',
    'BOOK_VALUE_10Y_BACK': 'Book value 10years back',

    # User Ratios (pre-computed by screener.in)
    'CROIC': 'CROIC',
    'PRICE_TO_FCF': 'Price to Free Cash Flow',
    'NCAVPS': 'NCAVPS',
    'DEBT_CAPACITY': 'Debt Capacity',
    'WC_TO_SALES': 'Working Capital to Sales ratio',
    'PB_X_PE': 'PB X PE',
    'PRICE_TO_CF': 'Price to Cash Flow',
}

# Column aliases for handling typos/variants in source data
# Maps canonical key to list of possible column names (in order of preference)
COL_ALIASES = {
    'EBITDA_GROWTH_3Y': ['EBITDA growth 3Years', 'EBIDT growth 3Years'],
    'EBITDA_GROWTH_5Y': ['EBITDA growth 5Years', 'EBIDT growth 5Years'],
    'EBITDA_GROWTH_10Y': ['EBITDA growth 10Years', 'EBIDT growth 10Years'],
}

# Column suffixes for different data sources
SUFFIX = {
    'ANNUAL': '_ann',
    'QUARTERLY': '_qtr',
    'BALANCE': '_bal',
    'CASHFLOW': '_cf',
    'RATIOS': '_rat',
    'USER_RATIOS': '_ur',
}

# Standard identifier columns for all sheets
IDENTIFIER_COLS = ['ISIN', 'NSE_Code', 'BSE_Code']

# ── Configuration Constants ─────────────────────────────────────────────────
CONFIG = {
    'ROE_LOW_THRESHOLD': 10.0,
    'ROE_DECLINING_THRESHOLD': 15.0,
    'ROE_EXCELLENT_THRESHOLD': 25.0,
    'ROCE_LOW_THRESHOLD': 10.0,
    'ROCE_DECLINING_THRESHOLD': 15.0,
    'CFO_PAT_LOW_THRESHOLD': 0.7,
    'CFO_PAT_EXCELLENT_THRESHOLD': 1.2,
    'DEBT_GROWTH_THRESHOLD': 1.5,
    'DEBT_MIN_NEW_THRESHOLD': 50,
    'PE_HIGH_THRESHOLD': 50.0,
    'EV_EBITDA_HIGH_THRESHOLD': 25.0,
    'OTHER_INCOME_PCT_THRESHOLD': 30.0,
    'SCORE_BAND_A': 80.0,
    'SCORE_BAND_B': 65.0,
    'SCORE_BAND_C': 50.0,
    'SCORE_BAND_D': 30.0,
    'SEVERITY_FAILED_THRESHOLD': 2.0,
    'SEVERITY_FLAGS_THRESHOLD': 1.0,
    'SEVERITY_MINOR_THRESHOLD': 0.5,
    'CFO_PAT_WORST_YEAR_THRESHOLD': 0.2,
    'CFO_PAT_BORDERLINE_THRESHOLD': 0.75,
    'CFO_5YR_3YR_RATIO_THRESHOLD': 0.5,
    'ASSET_TURNOVER_CAPITAL_INTENSIVE': 0.8,
    'ASSET_TURNOVER_VERY_CAPITAL_INTENSIVE': 0.5,
    'NPM_OPM_GAP_THRESHOLD': 5.0,
    'ACCRUALS_AGGRESSIVE_THRESHOLD': 0.5,
    'CFO_PAT_EARNINGS_QUALITY_THRESHOLD': 0.8,
    'PROMOTER_BUY_THRESHOLD': 3.0,
    'PROMOTER_SELL_THRESHOLD': -3.0,
    'ETR_LOW_THRESHOLD': 0.15,  # 15% — suspicious when statutory rate is ~25%
}

# ── Scoring Bins for pd.cut vectorization ───────────────────────────────────
SCORING_BINS = {
    'ROE': {'bins': [-np.inf, 5, 10, 12, 15, 20, 25, np.inf], 'labels': [2, 6, 10, 14, 18, 22, 25], 'default': 5},
    'ROCE': {'bins': [-np.inf, 5, 10, 12, 15, 20, 25, np.inf], 'labels': [2, 6, 10, 14, 18, 22, 25], 'default': 5},
    'OPM': {'bins': [-np.inf, 5, 10, 15, 20, 25, np.inf], 'labels': [2, 6, 10, 14, 17, 20], 'default': 5},
    'PIOTROSKI': {'bins': [-np.inf, 2, 4, 6, 8, np.inf], 'labels': [1, 4, 8, 12, 15], 'default': 5},
    'PE': {'bins': [-np.inf, 0, 10, 15, 20, 25, 35, 50, 80, np.inf], 'labels': [5, 40, 35, 30, 25, 20, 15, 10, 5], 'default': 10},
    'PBV': {'bins': [-np.inf, 1, 2, 3, 5, 8, np.inf], 'labels': [20, 16, 12, 8, 5, 2], 'default': 5},
    'EV_EBITDA': {'bins': [-np.inf, 0, 8, 12, 16, 20, 25, np.inf], 'labels': [3, 20, 16, 12, 8, 5, 2], 'default': 5},
    'PEG': {'bins': [-np.inf, 0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf], 'labels': [3, 20, 16, 12, 8, 5, 2], 'default': 5},
    'REVENUE_GROWTH': {'bins': [-np.inf, 0, 5, 10, 15, 20, np.inf], 'labels': [3, 8, 14, 20, 25, 30], 'default': 5},
    'PROFIT_GROWTH': {'bins': [-np.inf, 0, 5, 10, 15, 25, np.inf], 'labels': [3, 8, 14, 20, 25, 30], 'default': 5},
    'EPS_GROWTH': {'bins': [-np.inf, 0, 5, 12, 20, np.inf], 'labels': [2, 7, 12, 16, 20], 'default': 5},
    'CFO_PAT': {'bins': [-np.inf, 0, 0.5, 0.8, 1.0, 1.2, np.inf], 'labels': [10, 30, 50, 70, 85, 100], 'default': 20},
    'DEBT_EQUITY': {'bins': [-np.inf, 0, 0.3, 0.5, 1.0, 1.5, 2.0, np.inf], 'labels': [30, 28, 24, 18, 12, 6, 2], 'default': 10},
    'INTEREST_COVERAGE': {'bins': [-np.inf, 1, 2, 3, 5, 10, np.inf], 'labels': [0, 5, 10, 15, 20, 25], 'default': 8},
    'CURRENT_RATIO': {'bins': [-np.inf, 0.8, 1.0, 1.2, 1.5, 2.0, np.inf], 'labels': [1, 4, 8, 12, 16, 20], 'default': 5},
}

# ── Financial sector identification ─────────────────────────────────────────
FINANCIAL_SECTORS_LOWER = frozenset(s.lower() for s in [
    'Banks', 'Private Banks', 'Public Banks', 'Foreign Banks',
    'Finance', 'Financial Services', 'NBFC', 'Non Banking Financial Company',
    'Insurance', 'Life Insurance', 'General Insurance',
    'Housing Finance', 'Consumer Finance', 'Asset Management',
    'Diversified Financials', 'Financial Institution', 'Capital Markets',
])

# ── Red flag definitions ────────────────────────────────────────────────────
STRUCTURAL_RED_FLAGS = {
    "LOW_ROE": {"severity": "MAJOR", "weight": 1.0, "meaning": "Company generates weak returns on shareholder capital"},
    "DECLINING_ROE": {"severity": "MAJOR", "weight": 1.0, "meaning": "Return on equity falling to mediocre level"},
    "LOW_ROCE": {"severity": "MAJOR", "weight": 1.0, "meaning": "Weak returns on total capital employed"},
    "DECLINING_ROCE": {"severity": "MAJOR", "weight": 1.0, "meaning": "Capital efficiency worsening"},
    "POOR_CASH_CONVERSION": {"severity": "CRITICAL", "weight": 2.0, "meaning": "Reported profits not converting to cash"},
    "NEGATIVE_CFO": {"severity": "CRITICAL", "weight": 2.0, "meaning": "Core operations burning cash"},
    "INCONSISTENT_CFO": {"severity": "CRITICAL", "weight": 2.0, "meaning": "Operating cash flow not reliably positive"},
    "HIGH_OTHER_INCOME": {"severity": "MAJOR", "weight": 1.0, "meaning": "Significant profit from non-core activities"},
    "MARGIN_COMPRESSION": {"severity": "MINOR", "weight": 0.5, "meaning": "Profit margins shrinking over time"},
    "RISING_DEBT": {"severity": "CRITICAL", "weight": 2.0, "meaning": "Financial risk increasing"},
    "WC_DIVERGENCE": {"severity": "MINOR", "weight": 0.5, "meaning": "Working capital growing much faster than sales"},
    "NPM_OPM_DIVERGENCE": {"severity": "MAJOR", "weight": 1.0, "meaning": "Bottom line improving faster than operating performance"},
    "LOW_EFFECTIVE_TAX": {"severity": "MINOR", "weight": 0.5, "meaning": "Effective tax rate persistently below statutory rate—may indicate tax havens, deferred liabilities, or creative accounting"},
    "ASSET_MILKING": {"severity": "MINOR", "weight": 0.5, "meaning": "Capex well below depreciation—company under-investing in its asset base"},
    "DEBT_SPIKE_1YR": {"severity": "MAJOR", "weight": 1.0, "meaning": "Debt jumped >50% in a single year—sudden leveraging needs investigation"},
}

PRICING_RED_FLAGS = {
    "HIGH_PE": {"severity": "MINOR", "weight": 0.5, "meaning": "Very high earnings multiple"},
    "NEGATIVE_PE": {"severity": "MAJOR", "weight": 1.0, "meaning": "Company is loss-making"},
    "HIGH_EV_EBITDA": {"severity": "MINOR", "weight": 0.5, "meaning": "Expensive on cash flow basis"},
    "NEGATIVE_EBITDA": {"severity": "CRITICAL", "weight": 2.0, "meaning": "Negative operating profit before depreciation"},
    "HIGH_PBV_ROE": {"severity": "MINOR", "weight": 0.5, "meaning": "Price implies returns company cannot deliver"},
}

RED_FLAG_DEFINITIONS = {**STRUCTURAL_RED_FLAGS, **PRICING_RED_FLAGS}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def col(metric_key: str, suffix_key: str = '') -> str:
    """Build full column name from metric key and suffix key."""
    base = COLS.get(metric_key, metric_key)
    suffix = SUFFIX.get(suffix_key, '') if suffix_key else ''
    return f"{base}{suffix}"


def get_col(df: pd.DataFrame, metric_key: str, suffix_key: str = '', default=np.nan):
    """
    Get column from DataFrame with fallback to default and alias support.
    
    Handles column aliases for typos/variants in source data (e.g., EBITDA vs EBIDT).
    """
    # Check if this metric has aliases defined
    if metric_key in COL_ALIASES:
        for alias in COL_ALIASES[metric_key]:
            col_name = f"{alias}{SUFFIX.get(suffix_key, '')}" if suffix_key else alias
            if col_name in df.columns:
                return df[col_name]
    
    # Standard lookup
    col_name = col(metric_key, suffix_key)
    if col_name in df.columns:
        return df[col_name]
    return default


def safe_div(a: Union[pd.Series, np.ndarray, float], 
             b: Union[pd.Series, np.ndarray, float], 
             default: float = np.nan) -> Union[pd.Series, np.ndarray, float]:
    """Safe division handling NaN and zero."""
    is_series = isinstance(a, pd.Series) or isinstance(b, pd.Series)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((pd.isna(b)) | (b == 0), default, a / b)
    if is_series:
        return pd.Series(result, index=a.index if isinstance(a, pd.Series) else None)
    return result


def safe_str_lower(series: pd.Series) -> pd.Series:
    """Safely convert Series to lowercase."""
    # ERROR WAS HERE: .str.lower needed parentheses ()
    return series.fillna('').astype(str).str.lower()


def vectorized_score(series: pd.Series, bin_config: dict) -> pd.Series:
    """Apply vectorized scoring using pd.cut."""
    result = pd.cut(series, bins=bin_config['bins'], labels=bin_config['labels'], ordered=False).astype(float)
    return result.fillna(bin_config['default'])


def vectorized_string_build(n: int, conditions: List[np.ndarray], strings: List[str], separator: str = ', ') -> List[str]:
    """
    Build strings by concatenating based on conditions - TRUE VECTORIALIZATION.
    
    Args:
        n: Number of rows
        conditions: List of boolean numpy arrays
        strings: List of strings to concatenate when condition is True
        separator: String to join parts
        
    Returns:
        List of concatenated strings
    """
    # Start with empty strings
    result = np.full(n, '', dtype=object)
    
    for cond, s in zip(conditions, strings):
        # Use numpy where to add string where condition is True
        result = np.where(cond, np.where(result == '', s, result + separator + s), result)
    
    return result.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_parquet_files() -> Dict[str, pd.DataFrame]:
    """Load all parquet files into a dict of DataFrames."""
    dfs = {}
    for key, path in PARQUET_FILES.items():
        if not os.path.exists(path):
            logging.warning(f"Missing parquet file: {path}")
            continue
        df = pd.read_parquet(path)
        logging.info(f"Loaded {key}: {df.shape[0]} stocks, {df.shape[1]} columns")
        dfs[key] = df
    return dfs


def validate_required_columns(df: pd.DataFrame, required_cols: List[str], file_key: str) -> List[str]:
    """Validate that required columns exist in DataFrame."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.warning(f"{file_key} is missing columns: {missing}")
    return missing


def merge_all(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all parquet DataFrames on common identifiers."""
    required_base_cols = [COLS['NAME'], COLS['BSE_CODE'], COLS['NSE_CODE'], COLS['ISIN_CODE'],
                          COLS['INDUSTRY_GROUP'], COLS['INDUSTRY'], COLS['CURRENT_PRICE'], COLS['MARKET_CAP']]
    
    if 'annual' not in dfs:
        raise ValueError("Annual financials file is required but not loaded")
    
    missing_cols = validate_required_columns(dfs['annual'], required_base_cols, 'annual')
    if missing_cols:
        available_cols = [c for c in required_base_cols if c in dfs['annual'].columns]
        if COLS['ISIN_CODE'] not in available_cols:
            raise ValueError("ISIN Code is required but missing from annual file")
        required_base_cols = available_cols
    
    merged = dfs['annual'][required_base_cols].copy()
    drop_cols = {COLS['NAME'], COLS['BSE_CODE'], COLS['NSE_CODE'], COLS['ISIN_CODE'],
                 COLS['INDUSTRY_GROUP'], COLS['INDUSTRY'], COLS['CURRENT_PRICE'], COLS['MARKET_CAP']}

    for key, df in dfs.items():
        if key == 'annual':
            for c in df.columns:
                if c not in drop_cols:
                    merged[col(c, 'ANNUAL')] = df[c].values
            continue

        if COLS['ISIN_CODE'] not in df.columns:
            logging.warning(f"{key} file missing 'ISIN Code', skipping merge")
            continue

        cols_to_keep = [COLS['ISIN_CODE']] + [c for c in df.columns if c not in drop_cols]
        df_subset = df[cols_to_keep].copy()
        suffix = SUFFIX[key.upper()]
        rename = {c: f"{c}{suffix}" for c in df_subset.columns if c != COLS['ISIN_CODE']}
        df_subset = df_subset.rename(columns=rename)
        merged = merged.merge(df_subset, on=COLS['ISIN_CODE'], how='left')

    logging.info(f"Merged dataset: {merged.shape[0]} stocks, {merged.shape[1]} columns")
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# SHEET BUILDING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_master_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build the Master sheet with stock identification info."""
    is_fin = safe_str_lower(m[COLS['INDUSTRY_GROUP']]).isin(FINANCIAL_SECTORS_LOWER) | \
             safe_str_lower(m[COLS['INDUSTRY']]).isin(FINANCIAL_SECTORS_LOWER)
    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'Stock_Name': m[COLS['NAME']], 'Sector': m[COLS['INDUSTRY_GROUP']], 'Industry': m[COLS['INDUSTRY']],
        'Market_Cap': m[COLS['MARKET_CAP']], 'Current_Price': m[COLS['CURRENT_PRICE']],
        'Is_Financial_Sector': is_fin.map({True: 'Yes', False: 'No'}),
        'Face_Value': get_col(m, 'FACE_VALUE', 'BALANCE'),
    })


def build_valuation_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build valuation metrics sheet."""
    pe, pbv = get_col(m, 'PE', 'RATIOS'), get_col(m, 'PBV', 'RATIOS')
    val_comfort = (vectorized_score(pd.Series(pe), SCORING_BINS['PE']) +
                   vectorized_score(pd.Series(pbv), SCORING_BINS['PBV']) +
                   vectorized_score(pd.Series(get_col(m, 'EVEBITDA', 'USER_RATIOS')), SCORING_BINS['EV_EBITDA']) +
                   vectorized_score(pd.Series(get_col(m, 'PEG_RATIO', 'USER_RATIOS')), SCORING_BINS['PEG']))
    val_band = np.where(pd.isna(val_comfort), 'N/A',
               np.where(val_comfort >= 70, 'CHEAP', np.where(val_comfort >= 50, 'FAIR',
               np.where(val_comfort >= 30, 'EXPENSIVE', 'VERY_EXPENSIVE'))))
    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'PE': pe, 'PBV': pbv, 'EV_EBITDA': get_col(m, 'EVEBITDA', 'USER_RATIOS'),
        'PEG': get_col(m, 'PEG_RATIO', 'USER_RATIOS'), 'Price_To_Sales': get_col(m, 'PRICE_TO_SALES', 'USER_RATIOS'),
        'Earnings_Yield': safe_div(100.0, pe), 'Valuation_Band': val_band, 'Valuation_Comfort_Score': val_comfort,
        'Market_Cap': m[COLS['MARKET_CAP']], 'LTP': m[COLS['CURRENT_PRICE']],
        'Book_Value': get_col(m, 'BOOK_VALUE', 'RATIOS'), 'Industry_PE': get_col(m, 'INDUSTRY_PE', 'RATIOS'),
        'Industry_PBV': get_col(m, 'INDUSTRY_PBV', 'RATIOS'),
        'Historical_PE_3Yr': get_col(m, 'HISTORICAL_PE_3Y', 'RATIOS'),
        'Historical_PE_5Yr': get_col(m, 'HISTORICAL_PE_5Y', 'RATIOS'),
        'Historical_PE_10Yr': get_col(m, 'HISTORICAL_PE_10Y', 'RATIOS'),
        'Down_From_52W_High': get_col(m, 'DOWN_52W_HIGH', 'USER_RATIOS'),
        'Up_From_52W_Low': get_col(m, 'UP_52W_LOW', 'USER_RATIOS'),
        'Price_To_Quarterly_Earning': get_col(m, 'PRICE_QUARTERLY_EARNING', 'RATIOS'),
        'Market_Cap_3Yr_Back': get_col(m, 'MARKET_CAP_3Y_BACK', 'RATIOS'),
        'Market_Cap_5Yr_Back': get_col(m, 'MARKET_CAP_5Y_BACK', 'RATIOS'),
        'Market_Cap_10Yr_Back': get_col(m, 'MARKET_CAP_10Y_BACK', 'RATIOS'),
        'Price_To_FCF': get_col(m, 'PRICE_TO_FCF', 'USER_RATIOS'),
        'Price_To_CF': get_col(m, 'PRICE_TO_CF', 'USER_RATIOS'),
        'PB_X_PE': get_col(m, 'PB_X_PE', 'USER_RATIOS'),
        'NCAVPS': get_col(m, 'NCAVPS', 'USER_RATIOS'),
    })


def build_quality_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build quality metrics sheet."""
    roe = pd.Series(get_col(m, 'ROE', 'RATIOS'), dtype=float)
    roe_3 = pd.Series(get_col(m, 'ROE_3Y_AVG', 'RATIOS'), dtype=float)
    roe_5 = pd.Series(get_col(m, 'ROE_5Y_AVG', 'RATIOS'), dtype=float)
    roce = pd.Series(get_col(m, 'ROCE', 'ANNUAL'), dtype=float)
    roce_3 = pd.Series(get_col(m, 'ROCE_3Y_AVG', 'RATIOS'), dtype=float)
    opm = pd.Series(get_col(m, 'OPM', 'ANNUAL'), dtype=float)
    opm_ly = pd.Series(get_col(m, 'OPM_LAST_YEAR', 'ANNUAL'), dtype=float)
    opm_py = pd.Series(get_col(m, 'OPM_PRECEDING_YEAR', 'ANNUAL'), dtype=float)
    piotroski = pd.Series(get_col(m, 'PIOTROSKI', 'RATIOS'), dtype=float)
    asset_turnover = pd.Series(get_col(m, 'ASSET_TURNOVER', 'RATIOS'), dtype=float)
    
    bq_score = (vectorized_score(roe, SCORING_BINS['ROE']) + vectorized_score(roce, SCORING_BINS['ROCE']) +
                vectorized_score(opm, SCORING_BINS['OPM']) + vectorized_score(piotroski, SCORING_BINS['PIOTROSKI']) +
                pd.Series(np.where(pd.isna(roe_5) | pd.isna(roe_3), 5,
                         np.where(np.abs(roe_5 - roe_3) <= 2, 15, np.where(np.abs(roe_5 - roe_3) <= 5, 10, 
                         np.where(np.abs(roe_5 - roe_3) <= 10, 6, 3)))), dtype=float))
    
    pat = pd.Series(get_col(m, 'PAT', 'ANNUAL'), dtype=float)
    sales = pd.Series(get_col(m, 'SALES', 'ANNUAL'), dtype=float)
    other_inc = pd.Series(get_col(m, 'OTHER_INCOME', 'ANNUAL'), dtype=float)
    npm_current = safe_div(pat * 100, sales)  # computed current-year NPM
    npm_ly = pd.Series(get_col(m, 'NPM_LAST_YEAR', 'ANNUAL'), dtype=float)
    npm_py = pd.Series(get_col(m, 'NPM_PRECEDING_YEAR', 'ANNUAL'), dtype=float)

    # Effective Tax Rate: ETR = 1 - (PAT / PBT)
    # Only compute when PBT > 1 Cr to avoid extreme ratios from near-zero denominators
    pat_ly = pd.Series(get_col(m, 'PAT_LAST_YEAR', 'ANNUAL'), dtype=float)
    pbt_ly = pd.Series(get_col(m, 'PBT_LAST_YEAR', 'ANNUAL'), dtype=float)
    pbt_py = pd.Series(get_col(m, 'PBT_PRECEDING_YEAR', 'ANNUAL'), dtype=float)
    pat_py = pd.Series(get_col(m, 'PAT_PRECEDING_YEAR', 'ANNUAL'), dtype=float)
    etr_ly = pd.Series(np.where((~pd.isna(pbt_ly)) & (pbt_ly > 1), 1.0 - safe_div(pat_ly, pbt_ly), np.nan), dtype=float)
    etr_py = pd.Series(np.where((~pd.isna(pbt_py)) & (pbt_py > 1), 1.0 - safe_div(pat_py, pbt_py), np.nan), dtype=float)
    # Clamp to [-1, 1] range — anything outside is data noise
    etr_ly = etr_ly.clip(-1.0, 1.0)
    etr_py = etr_py.clip(-1.0, 1.0)

    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'Business_Quality_Score': bq_score,
        'Earnings_Quality': np.where(pd.isna(bq_score), 'N/A', np.where(bq_score >= 70, 'HIGH',
                                    np.where(bq_score >= 50, 'MODERATE', np.where(bq_score >= 30, 'LOW', 'POOR')))),
        'Piotroski_Score': piotroski,
        'Piotroski_Assessment': np.where(pd.isna(piotroski), 'N/A', np.where(piotroski >= 7, 'STRONG',
                                         np.where(piotroski >= 4, 'NEUTRAL', 'WEAK'))),
        'ROE_Latest': roe, 'ROE_3Yr_Avg': roe_3, 'ROE_5Yr_Avg': roe_5,
        'ROE_10Yr_Avg': get_col(m, 'ROE_10Y_AVG', 'RATIOS'),
        'ROE_Trend': np.where(pd.isna(roe) | pd.isna(roe_3), 'N/A',
                     np.where(roe > roe_3 + 2, 'IMPROVING', np.where(roe < roe_3 - 2, 'DECLINING', 'STABLE'))),
        'ROCE_Latest': roce, 'ROCE_3Yr_Avg': roce_3,
        'ROCE_5Yr_Avg': get_col(m, 'ROCE_5Y_AVG', 'RATIOS'),
        'ROCE_7Yr_Avg': get_col(m, 'ROCE_7Y_AVG', 'RATIOS'),
        'ROCE_10Yr_Avg': get_col(m, 'ROCE_10Y_AVG', 'RATIOS'),
        'ROCE_Trend': np.where(pd.isna(roce) | pd.isna(roce_3), 'N/A',
                      np.where(roce > roce_3 + 2, 'IMPROVING', np.where(roce < roce_3 - 2, 'DECLINING', 'STABLE'))),
        'ROA_Latest': get_col(m, 'ROA', 'RATIOS'),
        'OPM_Latest': opm, 'OPM_Last_Year': opm_ly, 'OPM_Preceding_Year': opm_py,
        'OPM_Trend': np.where(pd.isna(opm) | pd.isna(opm_ly), 'N/A',
                     np.where(opm > opm_ly + 1, 'IMPROVING', np.where(opm < opm_ly - 1, 'DECLINING', 'STABLE'))),
        'Asset_Turnover': asset_turnover,
        'NPM_Latest': npm_current, 'NPM_Last_Year': npm_ly, 'NPM_Preceding_Year': npm_py,
        'Other_Income_Pct_PAT': safe_div(np.abs(other_inc) * 100, np.abs(pat)),
        'ETR_Last_Year': np.round(etr_ly * 100, 1),
        'ETR_Preceding_Year': np.round(etr_py * 100, 1),
        'CROIC': get_col(m, 'CROIC', 'USER_RATIOS'),
    })


def build_cashflow_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build cash flow metrics sheet with TTM PAT and 3yr avg-of-ratios."""
    cfo = pd.Series(get_col(m, 'CFO_LAST_YEAR', 'CASHFLOW'), dtype=float)
    pat_annual = pd.Series(get_col(m, 'PAT', 'ANNUAL'), dtype=float)
    cfo_py = pd.Series(get_col(m, 'CFO_PRECEDING_YEAR', 'CASHFLOW'), dtype=float)
    total_assets = pd.Series(get_col(m, 'TOTAL_ASSETS', 'BALANCE'), dtype=float)
    wc = pd.Series(get_col(m, 'WORKING_CAPITAL', 'BALANCE'), dtype=float)
    wc_py = pd.Series(get_col(m, 'WORKING_CAPITAL_PY', 'BALANCE'), dtype=float)
    sales = pd.Series(get_col(m, 'SALES', 'ANNUAL'), dtype=float)
    sales_ly = pd.Series(get_col(m, 'SALES_LAST_YEAR', 'ANNUAL'), dtype=float)

    # TTM PAT from quarterly data (sum of last 4 quarters)
    np_q1 = pd.Series(get_col(m, 'NP_Q1', 'QUARTERLY'), dtype=float)
    np_q2 = pd.Series(get_col(m, 'NP_Q2', 'QUARTERLY'), dtype=float)
    np_q3 = pd.Series(get_col(m, 'NP_Q3', 'QUARTERLY'), dtype=float)
    np_q4 = pd.Series(get_col(m, 'NP_Q4', 'QUARTERLY'), dtype=float)
    pat_ttm = np_q1 + np_q2 + np_q3 + np_q4
    pat = pd.Series(np.where(pd.notna(pat_ttm) & (pat_ttm != 0), pat_ttm, pat_annual), dtype=float)

    cfo_pat = safe_div(cfo, pat)
    cfo_3yr = pd.Series(get_col(m, 'CFO_3Y_CUMULATIVE', 'CASHFLOW'), dtype=float)
    cfo_5yr = pd.Series(get_col(m, 'CFO_5Y_CUMULATIVE', 'CASHFLOW'), dtype=float)

    # 3yr CFO/PAT: average of individual year ratios (weights each year equally)
    pat_ly = pd.Series(get_col(m, 'PAT_LAST_YEAR', 'ANNUAL'), dtype=float)
    pat_py = pd.Series(get_col(m, 'PAT_PRECEDING_YEAR', 'ANNUAL'), dtype=float)
    cfo_yr3_inferred = cfo_3yr - cfo - cfo_py  # inferred 3rd year back
    r1 = pd.Series(safe_div(cfo, pat_annual), dtype=float)
    r2 = pd.Series(safe_div(cfo_py, pat_ly), dtype=float)
    r3 = pd.Series(safe_div(cfo_yr3_inferred, pat_py), dtype=float)
    ratio_sum = r1.fillna(0) + r2.fillna(0) + r3.fillna(0)
    ratio_count = (~pd.isna(r1)).astype(int) + (~pd.isna(r2)).astype(int) + (~pd.isna(r3)).astype(int)
    pat_3yr_cum = pat_annual + pat_ly + pat_py
    cfo_pat_3yr_avg = pd.Series(
        np.where(ratio_count >= 2, ratio_sum / ratio_count, safe_div(cfo_3yr, pat_3yr_cum)), dtype=float)

    # Worst individual year CFO/PAT ratio
    worst_yr = np.minimum(np.minimum(
        np.where(pd.isna(r1), 999, r1), np.where(pd.isna(r2), 999, r2)),
        np.where(pd.isna(r3), 999, r3))
    worst_yr = np.where(worst_yr == 999, np.nan, worst_yr)

    # Count positive CFO years (from reconstructed latest, preceding, inferred yr3)
    pos_cfo_count = (cfo > 0).astype(int) + (cfo_py > 0).astype(int) + (cfo_yr3_inferred > 0).astype(int)

    wc_growth = safe_div(wc - wc_py, np.abs(wc_py)) * 100
    rev_growth = safe_div(sales - sales_ly, np.abs(sales_ly)) * 100

    # FCF trend (using preceding year data)
    fcf = pd.Series(get_col(m, 'FCF_LAST_YEAR', 'CASHFLOW'), dtype=float)
    fcf_py = pd.Series(get_col(m, 'FCF_PRECEDING_YEAR', 'CASHFLOW'), dtype=float)

    # Cash from financing (positive = raising capital, negative = returning capital)
    cff = pd.Series(get_col(m, 'CFF_LAST_YEAR', 'CASHFLOW'), dtype=float)
    cff_py = pd.Series(get_col(m, 'CFF_PRECEDING_YEAR', 'CASHFLOW'), dtype=float)

    # Investing cash flow cumulatives
    cfi = pd.Series(get_col(m, 'CFI_LAST_YEAR', 'CASHFLOW'), dtype=float)
    cfi_py = pd.Series(get_col(m, 'CFI_PRECEDING_YEAR', 'CASHFLOW'), dtype=float)
    cfi_3yr = pd.Series(get_col(m, 'CFI_3Y_CUMULATIVE', 'CASHFLOW'), dtype=float)
    cfi_5yr = pd.Series(get_col(m, 'CFI_5Y_CUMULATIVE', 'CASHFLOW'), dtype=float)

    # CFI/CFO ratio: how much of operating cash is reinvested
    cfi_cfo_ratio = safe_div(np.abs(cfi_3yr), cfo_3yr)

    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'CFO_Latest': cfo, 'PAT_Latest': pat, 'PAT_TTM': pat_ttm, 'PAT_Annual': pat_annual,
        'CFO_PAT_Latest': cfo_pat, 'CFO_PAT_3Yr_Avg': cfo_pat_3yr_avg,
        'Positive_CFO_Years_3Yr': pos_cfo_count, 'CFO_Year3_Inferred': cfo_yr3_inferred,
        'Worst_Year_CFO_PAT': worst_yr,
        'FCF_Latest': fcf, 'FCF_Preceding_Year': fcf_py,
        'FCF_Trend': np.where(pd.isna(fcf) | pd.isna(fcf_py), 'N/A',
                     np.where(fcf > fcf_py * 1.1, 'IMPROVING', np.where(fcf < fcf_py * 0.9, 'DECLINING', 'STABLE'))),
        'CFO_Preceding_Year': cfo_py,
        'CFO_Trend': np.where(pd.isna(cfo) | pd.isna(cfo_py), 'N/A',
                     np.where(cfo > cfo_py * 1.1, 'IMPROVING', np.where(cfo < cfo_py * 0.9, 'DECLINING', 'STABLE'))),
        'CFO_3Yr_Cumulative': cfo_3yr, 'CFO_5Yr_Cumulative': cfo_5yr,
        'CFROA': safe_div(cfo, total_assets), 'Accruals': safe_div(pat - cfo, total_assets),
        'WC_Growth_Pct': wc_growth, 'Rev_Growth_Pct': rev_growth,
        'WC_Rev_Growth_Ratio': safe_div(wc_growth, rev_growth),
        'CFF_Latest': cff, 'CFF_Preceding_Year': cff_py,
        'CFF_Signal': np.where(pd.isna(cff), 'N/A',
                      np.where(cff > 0, 'RAISING_CAPITAL', np.where(cff < 0, 'RETURNING_CAPITAL', 'NEUTRAL'))),
        'CFI_Latest': cfi, 'CFI_Preceding_Year': cfi_py,
        'CFI_3Yr_Cumulative': cfi_3yr, 'CFI_5Yr_Cumulative': cfi_5yr,
        'CFI_CFO_Ratio_3Yr': np.round(cfi_cfo_ratio, 2),
        'WC_To_Sales_Ratio': get_col(m, 'WC_TO_SALES', 'USER_RATIOS'),
    })


def build_leverage_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build leverage sheet."""
    de = pd.Series(get_col(m, 'DEBT_EQUITY', 'RATIOS'), dtype=float)
    ic = pd.Series(get_col(m, 'INTEREST_COVERAGE', 'USER_RATIOS'), dtype=float)
    debt = pd.Series(get_col(m, 'DEBT', 'BALANCE'), dtype=float)
    debt_3yr = pd.Series(get_col(m, 'DEBT_3Y_BACK', 'BALANCE'), dtype=float)
    ca = pd.Series(get_col(m, 'CURRENT_ASSETS', 'BALANCE'), dtype=float)
    cl = pd.Series(get_col(m, 'CURRENT_LIABILITIES', 'BALANCE'), dtype=float)
    current_ratio = safe_div(ca, cl)
    
    debt_trend = np.where(pd.isna(debt) | pd.isna(debt_3yr), 'N/A',
                 np.where(debt_3yr == 0, np.where(debt == 0, 'ZERO_DEBT', 'RISING'),
                 np.where(debt < debt_3yr * 0.8, 'DECLINING', np.where(debt > debt_3yr * 1.2, 'RISING', 'STABLE'))))
    
    fin_strength = (vectorized_score(de, SCORING_BINS['DEBT_EQUITY']) +
                    vectorized_score(ic, SCORING_BINS['INTEREST_COVERAGE']) +
                    vectorized_score(pd.Series(current_ratio), SCORING_BINS['CURRENT_RATIO']) +
                    pd.Series(np.where(debt_trend == 'ZERO_DEBT', 25, np.where(debt_trend == 'DECLINING', 22,
                              np.where(debt_trend == 'STABLE', 15, np.where(debt_trend == 'RISING', 5, 8)))), dtype=float))

    # Capex vs Depreciation analysis
    gross_block = pd.Series(get_col(m, 'GROSS_BLOCK', 'BALANCE'), dtype=float)
    gross_block_py = pd.Series(get_col(m, 'GROSS_BLOCK_PY', 'BALANCE'), dtype=float)
    net_block = pd.Series(get_col(m, 'NET_BLOCK', 'BALANCE'), dtype=float)
    net_block_py = pd.Series(get_col(m, 'NET_BLOCK_PY', 'BALANCE'), dtype=float)
    capex = gross_block - gross_block_py
    depreciation = capex - (net_block - net_block_py)
    capex_to_dep = safe_div(capex, depreciation)

    # Debt preceding year for spike detection
    debt_py = pd.Series(get_col(m, 'DEBT_PY', 'BALANCE'), dtype=float)

    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'Debt_Equity': de, 'Interest_Coverage': ic, 'Debt_Trend': debt_trend,
        'Financial_Strength_Score': fin_strength, 'Current_Ratio': current_ratio,
        'Total_Debt': debt, 'Debt_3Yr_Back': debt_3yr, 'Debt_Preceding_Year': debt_py,
        'Total_Assets': get_col(m, 'TOTAL_ASSETS', 'BALANCE'),
        'Debt_Capacity': get_col(m, 'DEBT_CAPACITY', 'USER_RATIOS'),
        'Capex': capex, 'Depreciation': np.round(depreciation, 1), 'Capex_To_Depreciation': np.round(capex_to_dep, 2),
    })


def build_growth_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build growth metrics sheet - uses get_col with alias support for EBITDA."""
    sales_g3 = pd.Series(get_col(m, 'SALES_GROWTH_3Y', 'ANNUAL'), dtype=float)
    profit_g3 = pd.Series(get_col(m, 'PROFIT_GROWTH_3Y', 'ANNUAL'), dtype=float)
    eps_g3 = pd.Series(get_col(m, 'EPS_GROWTH_3Y', 'ANNUAL'), dtype=float)
    profit_g5 = pd.Series(get_col(m, 'PROFIT_GROWTH_5Y', 'ANNUAL'), dtype=float)
    sales = pd.Series(get_col(m, 'SALES', 'ANNUAL'), dtype=float)
    sales_ly = pd.Series(get_col(m, 'SALES_LAST_YEAR', 'ANNUAL'), dtype=float)
    
    pg_consistent = np.where(pd.isna(profit_g3) | pd.isna(profit_g5), 'N/A',
                   np.where((profit_g3 > 0) & (profit_g5 > 0), 'CONSISTENT',
                   np.where((profit_g3 > 0) | (profit_g5 > 0), 'MIXED', 'DECLINING')))
    
    growth_durability = (vectorized_score(sales_g3, SCORING_BINS['REVENUE_GROWTH']) +
                         vectorized_score(profit_g3, SCORING_BINS['PROFIT_GROWTH']) +
                         vectorized_score(eps_g3, SCORING_BINS['EPS_GROWTH']) +
                         pd.Series(np.where(pg_consistent == 'CONSISTENT', 20,
                                   np.where(pg_consistent == 'MIXED', 10, 3)), dtype=float))
    
    # Quarterly deterioration signals
    opm_q = pd.Series(get_col(m, 'OPM_Q_LATEST', 'QUARTERLY'), dtype=float)
    opm_q_prev = pd.Series(get_col(m, 'OPM_Q_PRECEDING', 'QUARTERLY'), dtype=float)
    opm_q_yoy = pd.Series(get_col(m, 'OPM_Q_YOY', 'QUARTERLY'), dtype=float)
    sales_q = pd.Series(get_col(m, 'SALES_Q_LATEST', 'QUARTERLY'), dtype=float)
    sales_q_yoy = pd.Series(get_col(m, 'SALES_Q_YOY', 'QUARTERLY'), dtype=float)
    op_q = pd.Series(get_col(m, 'OP_Q_LATEST', 'QUARTERLY'), dtype=float)
    op_q_yoy = pd.Series(get_col(m, 'OP_Q_YOY', 'QUARTERLY'), dtype=float)

    # QoQ OPM change and YoY OPM change
    opm_q_change = opm_q - opm_q_prev
    opm_yoy_change = opm_q - opm_q_yoy

    # Quarterly revenue YoY growth
    q_rev_yoy_growth = safe_div(sales_q - sales_q_yoy, np.abs(sales_q_yoy)) * 100

    # Quarterly operating profit YoY growth
    q_op_yoy_growth = safe_div(op_q - op_q_yoy, np.abs(op_q_yoy)) * 100

    # Quarterly deterioration flag: OPM falling QoQ AND YoY
    q_deteriorating = (
        (~pd.isna(opm_q_change)) & (~pd.isna(opm_yoy_change))
        & (opm_q_change < -2) & (opm_yoy_change < -3)
    )

    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'Revenue_Growth_1Yr': safe_div(sales - sales_ly, np.abs(sales_ly)) * 100,
        'Revenue_Growth_3Yr': sales_g3, 'Profit_Growth_3Yr': profit_g3,
        'EBITDA_Growth_3Yr': get_col(m, 'EBITDA_GROWTH_3Y', 'ANNUAL'),
        'Profit_Growth_Consistency': pg_consistent, 'Growth_Durability_Score': growth_durability,
        'Total_Revenue': sales,
        'OPM_Q_Latest': opm_q, 'OPM_QoQ_Change': np.round(opm_q_change, 1),
        'OPM_YoY_Q_Change': np.round(opm_yoy_change, 1),
        'Q_Rev_YoY_Growth': np.round(q_rev_yoy_growth, 1),
        'Q_OP_YoY_Growth': np.round(q_op_yoy_growth, 1),
        'Q_Deteriorating': np.where(q_deteriorating, 'YES', 'NO'),
    })


def build_shareholding_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build shareholding sheet."""
    promoter = pd.Series(get_col(m, 'PROMOTER_HOLDING', 'RATIOS'), dtype=float)
    fii = pd.Series(get_col(m, 'FII_HOLDING', 'RATIOS'), dtype=float)
    dii = pd.Series(get_col(m, 'DII_HOLDING', 'RATIOS'), dtype=float)
    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'Promoter_Holding': promoter, 'Promoter_Change_3Yr': get_col(m, 'PROMOTER_CHANGE_3Y', 'ANNUAL'),
        'Institutional_Holding': fii.fillna(0) + dii.fillna(0),
        'Num_Shareholders': get_col(m, 'NUM_SHAREHOLDERS', 'RATIOS'),
    })


def build_neglected_firm_sheet(m: pd.DataFrame, quality_df: pd.DataFrame, leverage_df: pd.DataFrame,
                                shareholding_df: pd.DataFrame) -> pd.DataFrame:
    """Build neglected firm sheet with TRUE vectorization."""
    n = len(m)
    inst = shareholding_df['Institutional_Holding'].fillna(0).values
    mcap = m[COLS['MARKET_CAP']].values.astype(float)
    num_sh = pd.Series(get_col(m, 'NUM_SHAREHOLDERS', 'RATIOS'), dtype=float).values
    roe_5 = quality_df['ROE_5Yr_Avg'].fillna(0).values
    de = leverage_df['Debt_Equity'].fillna(999).values
    
    # Scores
    inst_score = np.where(inst <= 1, 40, np.where(inst <= 3, 30, np.where(inst <= 5, 20, np.where(inst <= 10, 10, 0))))
    mcap_score = np.where(pd.isna(mcap), 0, np.where(mcap <= 500, 30, np.where(mcap <= 2000, 20, np.where(mcap <= 5000, 10, 0))))
    sh_score = np.where(pd.isna(num_sh), 10, np.where(num_sh <= 5000, 20, np.where(num_sh <= 20000, 10, np.where(num_sh <= 50000, 5, 0))))
    neglect_score = inst_score + mcap_score + sh_score
    
    generic_candidate = np.where((neglect_score >= 50) & (roe_5 >= 10) & (de <= 1.0), 'Yes',
                        np.where(neglect_score >= 40, 'Maybe', 'No'))
    
    # TRUE vectorized string building for reasons
    # Note: String formatting cannot be truly vectorized at C level, but list comprehension
    # is ~3-5x faster than pd.Series.apply(lambda) and avoids pandas overhead
    inst_strs = np.array([f"Low institutional holding ({x:.1f}%)" if not np.isnan(x) else "" for x in inst], dtype=object)
    mcap_strs = np.array([f"Small cap (₹{x:.0f} Cr)" if not np.isnan(x) else "" for x in mcap], dtype=object)
    numsh_strs = np.array([f"Few shareholders ({int(x)})" if not np.isnan(x) else "" for x in num_sh], dtype=object)
    
    reasons = vectorized_string_build(
        n,
        conditions=[inst <= 5, ~pd.isna(mcap) & (mcap <= 2000), ~pd.isna(num_sh) & (num_sh <= 10000)],
        strings=[inst_strs, mcap_strs, numsh_strs],
        separator='; '
    )
    
    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'Generic_Stock_Candidate': generic_candidate, 'Neglect_Score': neglect_score,
        'Neglect_Reasons': reasons, 'Institutional_Holding': inst, 'ROE_5Yr_Avg': roe_5,
        'Debt_Equity': de, 'Market_Cap': mcap, 'Num_Shareholders': num_sh,
    })


def build_dividends_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build dividends sheet."""
    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'Dividend_Last_Year': get_col(m, 'DIVIDEND_LAST_YEAR', 'ANNUAL'),
        'Dividend_Payout': get_col(m, 'DIVIDEND_PAYOUT', 'USER_RATIOS'),
    })


def build_red_flags_sheet(m: pd.DataFrame, quality_df: pd.DataFrame, cashflow_df: pd.DataFrame,
                          leverage_df: pd.DataFrame, valuation_df: pd.DataFrame,
                          growth_df: pd.DataFrame) -> pd.DataFrame:
    """Build red flags sheet with sector-adjusted severity, enhanced CFO checks, and cyclic peak risk."""
    n = len(m)

    # Extract values
    roe = quality_df['ROE_Latest'].values.astype(float)
    roe_3 = quality_df['ROE_3Yr_Avg'].values.astype(float)
    roce = quality_df['ROCE_Latest'].values.astype(float)
    roce_3 = quality_df['ROCE_3Yr_Avg'].values.astype(float)
    opm = quality_df['OPM_Latest'].values.astype(float)
    opm_ly = quality_df['OPM_Last_Year'].values.astype(float)  # year-1 (not year-2)
    opm_py = quality_df['OPM_Preceding_Year'].values.astype(float)
    other_inc_pct = quality_df['Other_Income_Pct_PAT'].values.astype(float)
    asset_turnover = quality_df['Asset_Turnover'].values.astype(float)
    npm_current = quality_df['NPM_Latest'].values.astype(float)
    npm_py = quality_df['NPM_Preceding_Year'].values.astype(float)

    cfo = cashflow_df['CFO_Latest'].values.astype(float)
    cfo_pat = cashflow_df['CFO_PAT_Latest'].values.astype(float)
    cfo_pat_3yr = cashflow_df['CFO_PAT_3Yr_Avg'].values.astype(float)
    worst_yr_cfo_pat = cashflow_df['Worst_Year_CFO_PAT'].values.astype(float)
    cfo_3yr = cashflow_df['CFO_3Yr_Cumulative'].values.astype(float)
    cfo_5yr = cashflow_df['CFO_5Yr_Cumulative'].values.astype(float)
    cfo_yr3 = cashflow_df['CFO_Year3_Inferred'].values.astype(float)
    pos_cfo_years = cashflow_df['Positive_CFO_Years_3Yr'].values.astype(float)
    wc_rev_ratio = cashflow_df['WC_Rev_Growth_Ratio'].values.astype(float)

    debt = leverage_df['Total_Debt'].values.astype(float)
    debt_3yr = leverage_df['Debt_3Yr_Back'].values.astype(float)
    pe = valuation_df['PE'].values.astype(float)
    pbv = valuation_df['PBV'].values.astype(float)
    ev_ebitda = valuation_df['EV_EBITDA'].values.astype(float)

    # ── Enhanced POOR_CASH_CONVERSION ──────────────────────────────────────
    # Primary: 3yr avg-of-ratios < 0.7 (matches original script intent)
    # Secondary: worst year < 0.2 BUT only when 3yr avg also borderline (< 0.75)
    flag_poor_cash = (
        ((~np.isnan(cfo_pat_3yr)) & (cfo_pat_3yr < CONFIG['CFO_PAT_LOW_THRESHOLD']) & (cfo_pat_3yr >= 0))
        | ((np.isnan(cfo_pat_3yr)) & (~np.isnan(cfo_pat)) & (cfo_pat < CONFIG['CFO_PAT_LOW_THRESHOLD']) & (cfo_pat >= 0))
        | ((~np.isnan(worst_yr_cfo_pat)) & (worst_yr_cfo_pat < CONFIG['CFO_PAT_WORST_YEAR_THRESHOLD']) & (worst_yr_cfo_pat >= 0)
           & (~np.isnan(cfo_pat_3yr)) & (cfo_pat_3yr < CONFIG['CFO_PAT_BORDERLINE_THRESHOLD']))
    )

    # ── Enhanced INCONSISTENT_CFO ──────────────────────────────────────────
    # 5yr vs 3yr check only when recent years aren't all positive (turnarounds are real)
    avg_cfo_3yr = np.where(~np.isnan(cfo_3yr), cfo_3yr / 3.0, np.nan)
    avg_cfo_5yr = np.where(~np.isnan(cfo_5yr), cfo_5yr / 5.0, np.nan)
    cfo_5yr_vs_3yr = safe_div(avg_cfo_5yr, avg_cfo_3yr)

    flag_inconsistent_cfo = (
        ((~np.isnan(cfo_3yr)) & (cfo_3yr < 0))
        | ((~np.isnan(cfo_yr3)) & (cfo_yr3 < 0))
        | ((~np.isnan(pos_cfo_years)) & (pos_cfo_years < 2))
        | ((~np.isnan(cfo_5yr_vs_3yr)) & np.isfinite(cfo_5yr_vs_3yr)
           & (avg_cfo_3yr > 0) & (cfo_5yr_vs_3yr < CONFIG['CFO_5YR_3YR_RATIO_THRESHOLD'])
           & (pos_cfo_years < 3))
    )

    # ── NPM vs OPM Divergence ─────────────────────────────────────────────
    # NPM improving faster than OPM suggests earnings from non-operating sources
    # Approximate multi-year slope: (current - 2yr_back) / 2
    npm_slope = safe_div(npm_current - npm_py, 2.0)
    opm_slope = safe_div(opm - opm_py, 2.0)
    npm_opm_gap = npm_slope - opm_slope
    flag_npm_opm_divergence = (
        (~np.isnan(npm_slope)) & (~np.isnan(opm_slope))
        & np.isfinite(npm_opm_gap) & (npm_opm_gap > CONFIG['NPM_OPM_GAP_THRESHOLD'])
    )

    # ── Effective Tax Rate flag ──────────────────────────────────────────
    # Flag only when BOTH years show low ETR on profitable companies
    etr_ly = quality_df['ETR_Last_Year'].values.astype(float) / 100.0  # convert back to ratio
    etr_py = quality_df['ETR_Preceding_Year'].values.astype(float) / 100.0
    flag_low_effective_tax = (
        (~np.isnan(etr_ly)) & (~np.isnan(etr_py))
        & (etr_ly < CONFIG['ETR_LOW_THRESHOLD']) & (etr_py < CONFIG['ETR_LOW_THRESHOLD'])
        & (etr_ly >= 0) & (etr_py >= 0)  # exclude negative ETR (loss-makers)
    )

    # ── Asset Milking flag ────────────────────────────────────────────────
    # Capex < 50% of Depreciation = company not reinvesting in its asset base
    # Only flag when depreciation > 5 Cr (company has meaningful fixed assets)
    capex_to_dep = leverage_df['Capex_To_Depreciation'].values.astype(float)
    depreciation = leverage_df['Depreciation'].values.astype(float)
    flag_asset_milking = (
        (~np.isnan(capex_to_dep)) & (~np.isnan(depreciation))
        & (depreciation > 5) & (capex_to_dep < 0.5) & (capex_to_dep >= 0)
    )

    # ── Debt spike detection ─────────────────────────────────────────────
    # Debt increased > 50% in just 1 year AND absolute increase > 50 Cr
    debt_py = leverage_df['Debt_Preceding_Year'].values.astype(float)
    flag_debt_spike_1yr = (
        (~np.isnan(debt)) & (~np.isnan(debt_py)) & (debt_py > 0)
        & (debt > debt_py * 1.5) & ((debt - debt_py) > 50)
    )

    # Calculate all flags
    structural_flags = {
        'FLAG_LOW_ROE': (~np.isnan(roe)) & (roe < CONFIG['ROE_LOW_THRESHOLD']),
        'FLAG_DECLINING_ROE': (~np.isnan(roe)) & (~np.isnan(roe_3)) & (roe < roe_3 - 2) & (roe < CONFIG['ROE_DECLINING_THRESHOLD']),
        'FLAG_LOW_ROCE': (~np.isnan(roce)) & (roce < CONFIG['ROCE_LOW_THRESHOLD']),
        'FLAG_DECLINING_ROCE': (~np.isnan(roce)) & (~np.isnan(roce_3)) & (roce < roce_3 - 2) & (roce < CONFIG['ROCE_DECLINING_THRESHOLD']),
        'FLAG_POOR_CASH_CONVERSION': flag_poor_cash,
        'FLAG_NEGATIVE_CFO': (~np.isnan(cfo)) & (cfo < 0),
        'FLAG_INCONSISTENT_CFO': flag_inconsistent_cfo,
        'FLAG_HIGH_OTHER_INCOME': (~np.isnan(other_inc_pct)) & (other_inc_pct > CONFIG['OTHER_INCOME_PCT_THRESHOLD']),
        'FLAG_MARGIN_COMPRESSION': (~np.isnan(opm)) & (~np.isnan(opm_ly)) & (opm < opm_ly - 2),
        'FLAG_RISING_DEBT': ((~np.isnan(debt)) & (~np.isnan(debt_3yr)) & (debt_3yr > 0) & (debt > debt_3yr * CONFIG['DEBT_GROWTH_THRESHOLD'])) |
                            ((~np.isnan(debt)) & (debt_3yr == 0) & (debt > CONFIG['DEBT_MIN_NEW_THRESHOLD'])),
        'FLAG_WC_DIVERGENCE': (~np.isnan(wc_rev_ratio)) & (wc_rev_ratio > 1.5) & np.isfinite(wc_rev_ratio),
        'FLAG_NPM_OPM_DIVERGENCE': flag_npm_opm_divergence,
        'FLAG_LOW_EFFECTIVE_TAX': flag_low_effective_tax,
        'FLAG_ASSET_MILKING': flag_asset_milking,
        'FLAG_DEBT_SPIKE_1YR': flag_debt_spike_1yr,
    }
    pricing_flags = {
        'FLAG_HIGH_PE': (~np.isnan(pe)) & (pe > CONFIG['PE_HIGH_THRESHOLD']),
        'FLAG_NEGATIVE_PE': (~np.isnan(pe)) & (pe < 0),
        'FLAG_HIGH_EV_EBITDA': (~np.isnan(ev_ebitda)) & (ev_ebitda > CONFIG['EV_EBITDA_HIGH_THRESHOLD']),
        'FLAG_NEGATIVE_EBITDA': (~np.isnan(ev_ebitda)) & (ev_ebitda < 0),
        'FLAG_HIGH_PBV_ROE': (~np.isnan(pbv)) & (~np.isnan(roe)) & (roe > 0) & (pbv > roe / 2),
    }
    all_flags = {**structural_flags, **pricing_flags}

    # ── Sector-based severity adjustments (matching original script) ───────
    is_cap_intensive = (~np.isnan(asset_turnover)) & (asset_turnover < CONFIG['ASSET_TURNOVER_CAPITAL_INTENSIVE'])
    is_very_cap_intensive = (~np.isnan(asset_turnover)) & (asset_turnover < CONFIG['ASSET_TURNOVER_VERY_CAPITAL_INTENSIVE'])

    weight_overrides = {
        'FLAG_POOR_CASH_CONVERSION': np.where(is_cap_intensive & flag_poor_cash, 0.5, 2.0),
        'FLAG_NEGATIVE_CFO': np.where(is_very_cap_intensive & structural_flags['FLAG_NEGATIVE_CFO'], 1.0, 2.0),
        'FLAG_INCONSISTENT_CFO': np.where(is_cap_intensive & flag_inconsistent_cfo, 1.0, 2.0),
    }

    # Track sector adjustments for reporting
    sector_adj = np.full(n, '', dtype=object)
    adj_text = [
        (is_cap_intensive & flag_poor_cash, "POOR_CASH_CONVERSION: CRITICAL→MINOR"),
        (is_very_cap_intensive & structural_flags['FLAG_NEGATIVE_CFO'], "NEGATIVE_CFO: CRITICAL→MAJOR"),
        (is_cap_intensive & flag_inconsistent_cfo, "INCONSISTENT_CFO: CRITICAL→MAJOR"),
    ]
    for cond, text in adj_text:
        sector_adj = np.where(cond, np.where(sector_adj == '', text, sector_adj + ' | ' + text), sector_adj)

    # Calculate severity with sector-adjusted weights
    critical_count, major_count, minor_count = np.zeros(n), np.zeros(n), np.zeros(n)
    total_count, quality_severity, pricing_severity = np.zeros(n), np.zeros(n), np.zeros(n)

    for fname, farr in structural_flags.items():
        defn = RED_FLAG_DEFINITIONS.get(fname.replace('FLAG_', ''), {})
        total_count += farr.astype(int)
        if fname in weight_overrides:
            weight = weight_overrides[fname]  # per-stock array
            is_flagged = farr.astype(bool)
            critical_count += (is_flagged & (weight >= 2.0)).astype(int)
            major_count += (is_flagged & (weight >= 1.0) & (weight < 2.0)).astype(int)
            minor_count += (is_flagged & (weight < 1.0)).astype(int)
        else:
            weight = defn.get('weight', 0.5)
            if defn.get('severity') == 'CRITICAL': critical_count += farr.astype(int)
            elif defn.get('severity') == 'MAJOR': major_count += farr.astype(int)
            else: minor_count += farr.astype(int)
        quality_severity += farr.astype(float) * weight

    for fname, farr in pricing_flags.items():
        defn = RED_FLAG_DEFINITIONS.get(fname.replace('FLAG_', ''), {})
        total_count += farr.astype(int)
        pricing_severity += farr.astype(float) * defn.get('weight', 0.5)
        if defn.get('severity') == 'CRITICAL': critical_count += farr.astype(int)
        elif defn.get('severity') == 'MAJOR': major_count += farr.astype(int)
        else: minor_count += farr.astype(int)

    quality_risk = np.where(critical_count >= 2, 'CRITICAL', np.where(critical_count >= 1, 'HIGH',
                  np.where(major_count >= 2, 'ELEVATED', np.where(major_count >= 1, 'MODERATE', 'LOW'))))

    # Build flag name strings using numpy
    quality_flag_names = np.full(n, '', dtype=object)
    for fname, farr in structural_flags.items():
        name = fname.replace('FLAG_', '')
        quality_flag_names = np.where(farr, np.where(quality_flag_names == '', name, quality_flag_names + ', ' + name), quality_flag_names)

    pricing_flag_names = np.full(n, '', dtype=object)
    for fname, farr in pricing_flags.items():
        name = fname.replace('FLAG_', '')
        pricing_flag_names = np.where(farr, np.where(pricing_flag_names == '', name, pricing_flag_names + ', ' + name), pricing_flag_names)

    all_flag_names = np.where((quality_flag_names != '') & (pricing_flag_names != ''),
                              quality_flag_names + ', ' + pricing_flag_names,
                              np.where(quality_flag_names != '', quality_flag_names, pricing_flag_names))

    explained = np.full(n, '', dtype=object)
    for fname, farr in all_flags.items():
        defn = RED_FLAG_DEFINITIONS.get(fname.replace('FLAG_', ''), {})
        meaning = f"{fname.replace('FLAG_', '')}: {defn.get('meaning', '')}"
        explained = np.where(farr, np.where(explained == '', meaning, explained + ' | ' + meaning), explained)

    # ── Earnings Quality classification (Clean/Mixed/Aggressive) ─────────
    # Mirrors JSON script logic (minus FREQUENT_EXCEPTIONALS which isn't in parquet)
    accruals = cashflow_df['Accruals'].values.astype(float)
    eq_issues = np.zeros(n, dtype=int)
    eq_issues += ((~np.isnan(cfo_pat_3yr)) & (cfo_pat_3yr < CONFIG['CFO_PAT_EARNINGS_QUALITY_THRESHOLD'])).astype(int)
    eq_issues += ((~np.isnan(accruals)) & (accruals > CONFIG['ACCRUALS_AGGRESSIVE_THRESHOLD'])).astype(int)
    eq_issues += structural_flags.get('FLAG_HIGH_OTHER_INCOME', np.zeros(n, dtype=bool)).astype(int)
    eq_issues += ((~np.isnan(pos_cfo_years)) & (pos_cfo_years < 2)).astype(int)
    eq_issues += structural_flags.get('FLAG_LOW_EFFECTIVE_TAX', np.zeros(n, dtype=bool)).astype(int)
    earnings_quality_label = np.where(eq_issues >= 3, 'Aggressive',
                             np.where(eq_issues >= 1, 'Mixed', 'Clean'))

    # ── Cyclic Peak Risk Detection ────────────────────────────────────────
    # Detects if company might be at earnings peak (dangerous for entry)
    rev_g1 = growth_df['Revenue_Growth_1Yr'].values.astype(float)
    rev_g3 = growth_df['Revenue_Growth_3Yr'].values.astype(float)
    opm_avg_3 = (opm + opm_ly + opm_py) / 3.0

    cyclic_signals = np.zeros(n, dtype=int)
    # Signal 1: OPM at peak (latest > 3yr avg by 20%)
    cyclic_signals += ((~np.isnan(opm)) & (~np.isnan(opm_avg_3)) & (opm > opm_avg_3 * 1.2)).astype(int)
    # Signal 2: Revenue decelerating (1Y growth < half of 3Y growth)
    cyclic_signals += ((~np.isnan(rev_g1)) & (~np.isnan(rev_g3)) & (rev_g3 > 0) & (rev_g1 < rev_g3 * 0.5)).astype(int)
    # Signal 3: High ROCE + low growth (reinvestment saturation)
    cyclic_signals += ((~np.isnan(roce)) & (roce > 25) & (~np.isnan(rev_g1)) & (rev_g1 < 10)).astype(int)

    cyclic_peak_risk = np.where(cyclic_signals >= 2, 'HIGH',
                       np.where(cyclic_signals >= 1, 'MODERATE', 'LOW'))

    rf_data = {
        'ISIN': m[COLS['ISIN_CODE']].values, 'NSE_Code': m[COLS['NSE_CODE']].values, 'BSE_Code': m[COLS['BSE_CODE']].values,
        'Quality_Risk': quality_risk, 'Quality_Severity': np.round(quality_severity, 1),
        'Pricing_Severity': np.round(pricing_severity, 1), 'Total_Severity': np.round(quality_severity + pricing_severity, 1),
        'Critical_Flags': critical_count.astype(int), 'Major_Flags': major_count.astype(int),
        'Minor_Flags': minor_count.astype(int), 'Red_Flag_Count': total_count.astype(int),
        'Quality_Flags': quality_flag_names.tolist(), 'Pricing_Flags': pricing_flag_names.tolist(),
        'Red_Flags': all_flag_names.tolist(), 'Red_Flags_Explained': explained.tolist(),
        'Sector_Adjustments': sector_adj.tolist(),
        'Earnings_Quality_Label': earnings_quality_label.tolist(),
        'Cyclic_Peak_Risk': cyclic_peak_risk.tolist(),
    }
    for fname, farr in all_flags.items():
        rf_data[fname] = farr.astype(int)

    return pd.DataFrame(rf_data)


def build_analysis_sheet(m: pd.DataFrame, quality_df: pd.DataFrame, valuation_df: pd.DataFrame,
                         leverage_df: pd.DataFrame, growth_df: pd.DataFrame, cashflow_df: pd.DataFrame,
                         red_flags_df: pd.DataFrame, shareholding_df: pd.DataFrame) -> pd.DataFrame:
    """Build analysis sheet."""
    n = len(m)
    bq = quality_df['Business_Quality_Score'].values.astype(float)
    vc = valuation_df['Valuation_Comfort_Score'].values.astype(float)
    fs = leverage_df['Financial_Strength_Score'].values.astype(float)
    gd = growth_df['Growth_Durability_Score'].values.astype(float)
    
    cfo_pat = pd.Series(cashflow_df['CFO_PAT_Latest'].values, dtype=float)
    cfo_latest = pd.Series(cashflow_df['CFO_Latest'].values, dtype=float)
    cfo_3yr = pd.Series(cashflow_df['CFO_3Yr_Cumulative'].values, dtype=float)
    cf_score = vectorized_score(cfo_pat, SCORING_BINS['CFO_PAT'])
    cf_score = np.where(cfo_latest < 0, np.minimum(cf_score, 15), cf_score)
    cf_score = np.where((~pd.isna(cfo_3yr)) & (cfo_3yr < 0), np.minimum(cf_score, 10), cf_score)
    
    composite = 0.35 * bq + 0.25 * gd + 0.20 * vc + 0.10 * fs + 0.10 * cf_score
    score_band = np.where(pd.isna(composite), 'N/A', np.where(composite >= CONFIG['SCORE_BAND_A'], 'A',
                 np.where(composite >= CONFIG['SCORE_BAND_B'], 'B', np.where(composite >= CONFIG['SCORE_BAND_C'], 'C',
                 np.where(composite >= CONFIG['SCORE_BAND_D'], 'D', 'F')))))
    
    critical = red_flags_df['Critical_Flags'].values.astype(int)
    quality_severity = red_flags_df['Quality_Severity'].values.astype(float)
    quality_flags_str = red_flags_df['Quality_Flags'].values
    pricing_flags_str = red_flags_df['Pricing_Flags'].values
    earnings_quality_label = red_flags_df['Earnings_Quality_Label'].values
    cyclic_peak_risk = red_flags_df['Cyclic_Peak_Risk'].values
    mcap = m[COLS['MARKET_CAP']].values
    pe = valuation_df['PE'].values
    inst = shareholding_df['Institutional_Holding'].values
    promoter_chg = shareholding_df['Promoter_Change_3Yr'].values.astype(float)

    is_aggressive = (earnings_quality_label == 'Aggressive')
    has_pricing_flags = (pricing_flags_str != '') & (~pd.isna(pricing_flags_str))

    # Decision logic (mirrors JSON: severity → pricing → aggressive → gates)
    # Note: Cyclic_Peak_Risk is informational only; not used as decision gate
    # because parquet data differences cause false positives vs JSON
    decision = np.where(quality_severity >= CONFIG['SEVERITY_FAILED_THRESHOLD'], 'SCREEN_FAILED',
               np.where(quality_severity >= CONFIG['SEVERITY_FLAGS_THRESHOLD'], 'SCREEN_PASSED_FLAGS',
               np.where(quality_severity >= CONFIG['SEVERITY_MINOR_THRESHOLD'], 'SCREEN_PASSED_FLAGS',
               np.where(has_pricing_flags & np.isin(score_band, ['A', 'B']), 'SCREEN_PASSED_EXPENSIVE',
               np.where(is_aggressive & np.isin(score_band, ['A', 'B']), 'SCREEN_PASSED_FLAGS',
               np.where(np.isin(score_band, ['A', 'B']), 'GATES_CLEARED',
               np.where(score_band == 'C', 'SCREEN_MARGINAL', 'SCREEN_FAILED')))))))

    reject_reason = np.where(decision == 'SCREEN_FAILED', np.where(critical >= 1,
                            "Critical flag detected", "Multiple major flags"),
                   np.where(decision == 'SCREEN_PASSED_FLAGS',
                            np.where(is_aggressive & (quality_severity < CONFIG['SEVERITY_MINOR_THRESHOLD']),
                                     "Aggressive earnings quality", "Quality concerns"),
                   np.where(decision == 'SCREEN_PASSED_EXPENSIVE', "Valuation concern",
                   np.where(decision == 'GATES_CLEARED', "Passed",
                   np.where(decision == 'SCREEN_MARGINAL', "Marginal score", "Low score")))))
    
    # Conviction overrides
    conviction_override = np.full(n, 'None', dtype=object)
    for i in range(n):
        pchg = promoter_chg[i]
        if pd.isna(pchg): continue
        if decision[i] in ('SCREEN_FAILED', 'SCREEN_MARGINAL') and pchg >= CONFIG['PROMOTER_BUY_THRESHOLD'] and quality_severity[i] < 4.0:
            decision[i], conviction_override[i] = 'CONTRARIAN_BET', f"Promoter buying +{pchg:.1f}%"
        elif decision[i] in ('GATES_CLEARED', 'SCREEN_PASSED_EXPENSIVE') and pchg <= CONFIG['PROMOTER_SELL_THRESHOLD']:
            decision[i], conviction_override[i] = 'VALUE_TRAP', f"Promoter selling {pchg:.1f}%"
    
    thesis = np.where(decision == 'GATES_CLEARED', 'Passed quality gates',
             np.where(decision == 'SCREEN_PASSED_EXPENSIVE', 'Good quality but expensive',
             np.where(decision == 'SCREEN_PASSED_FLAGS', 'Passed with concerns',
             np.where(decision == 'SCREEN_MARGINAL', 'Borderline metrics',
             np.where(decision == 'CONTRARIAN_BET', 'Promoter buying despite low score',
             np.where(decision == 'VALUE_TRAP', 'Good score but insiders exiting', ''))))))
    
    return pd.DataFrame({
        'ISIN': m[COLS['ISIN_CODE']], 'NSE_Code': m[COLS['NSE_CODE']], 'BSE_Code': m[COLS['BSE_CODE']],
        'Decision_Bucket': decision, 'MCAP': mcap, 'Conviction_Override': conviction_override,
        'SCREEN_ELIGIBLE': np.where(quality_severity >= CONFIG['SEVERITY_FAILED_THRESHOLD'], 'NO', 'YES'),
        'Investment_Thesis': thesis, 'Reject_Reason': reject_reason,
        'Composite_Score': np.round(composite, 1), 'Score_Band': score_band,
        'Quality_Risk': red_flags_df['Quality_Risk'].values, 'Quality_Severity': np.round(quality_severity, 1),
        'Critical_Flags': critical, 'Business_Quality_Score': np.round(bq, 1),
        'Growth_Durability_Score': np.round(gd, 1), 'Valuation_Comfort_Score': np.round(vc, 1),
        'Financial_Strength_Score': np.round(fs, 1), 'Cash_Flow_Score': np.round(cf_score.astype(float), 1),
    })


def build_summary_sheet(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary sheet."""
    total = len(analysis_df)
    dc = analysis_df['Decision_Bucket'].value_counts()
    rows = [['STOCK SCREENING SUMMARY', '', ''], [f'Total: {total}', f'Generated: {datetime.now():%Y-%m-%d %H:%M}', ''],
            ['Decision', 'Count', 'Pct']]
    for b in ['GATES_CLEARED', 'SCREEN_PASSED_EXPENSIVE', 'SCREEN_PASSED_FLAGS', 'SCREEN_MARGINAL', 'SCREEN_FAILED', 'CONTRARIAN_BET', 'VALUE_TRAP']:
        c = dc.get(b, 0)
        rows.append([b, c, f'{c/total*100:.1f}%'])
    return pd.DataFrame(rows, columns=['A', 'B', 'C'])


def build_decision_audit_sheet(m: pd.DataFrame, analysis_df: pd.DataFrame,
                                quality_df: pd.DataFrame, valuation_df: pd.DataFrame,
                                leverage_df: pd.DataFrame, growth_df: pd.DataFrame,
                                cashflow_df: pd.DataFrame, red_flags_df: pd.DataFrame,
                                shareholding_df: pd.DataFrame) -> pd.DataFrame:
    """Build Decision_Audit sheet: per-stock trace of how each Decision_Bucket was reached."""
    n = len(analysis_df)

    # Extract all the values we need
    bq = analysis_df['Business_Quality_Score'].values.astype(float)
    gd = analysis_df['Growth_Durability_Score'].values.astype(float)
    vc = analysis_df['Valuation_Comfort_Score'].values.astype(float)
    fs = analysis_df['Financial_Strength_Score'].values.astype(float)
    cf = analysis_df['Cash_Flow_Score'].values.astype(float)
    composite = analysis_df['Composite_Score'].values.astype(float)
    band = analysis_df['Score_Band'].values
    decision = analysis_df['Decision_Bucket'].values
    override = analysis_df['Conviction_Override'].values

    quality_severity = red_flags_df['Quality_Severity'].values.astype(float)
    pricing_severity = red_flags_df['Pricing_Severity'].values.astype(float)
    quality_flags = red_flags_df['Quality_Flags'].values
    pricing_flags = red_flags_df['Pricing_Flags'].values
    earnings_quality = red_flags_df['Earnings_Quality_Label'].values
    cyclic_risk = red_flags_df['Cyclic_Peak_Risk'].values
    sector_adj = red_flags_df['Sector_Adjustments'].values

    # Build per-stock strings using loops (audit trail needs rich formatting)
    score_breakdown = np.empty(n, dtype=object)
    severity_gate = np.empty(n, dtype=object)
    pricing_gate = np.empty(n, dtype=object)
    earnings_gate = np.empty(n, dtype=object)
    score_gate = np.empty(n, dtype=object)
    override_col = np.empty(n, dtype=object)
    flags_detail = np.empty(n, dtype=object)
    narrative = np.empty(n, dtype=object)

    for i in range(n):
        # 1. Score breakdown
        score_breakdown[i] = (
            f"BQ:{bq[i]:.0f}(35%) + GD:{gd[i]:.0f}(25%) + "
            f"VC:{vc[i]:.0f}(20%) + FS:{fs[i]:.0f}(10%) + "
            f"CF:{cf[i]:.0f}(10%) = {composite[i]:.1f} [{band[i]}]"
        )

        # 2. Trace through the decision cascade
        sev = quality_severity[i]
        psev = pricing_severity[i]
        qf = quality_flags[i] if pd.notna(quality_flags[i]) else ''
        pf = pricing_flags[i] if pd.notna(pricing_flags[i]) else ''
        eq = earnings_quality[i] if pd.notna(earnings_quality[i]) else ''
        b = band[i]
        d = decision[i]
        ov = override[i] if pd.notna(override[i]) and override[i] != 'None' else ''

        # Gate 1: Severity check
        if sev >= CONFIG['SEVERITY_FAILED_THRESHOLD']:
            severity_gate[i] = f"Severity {sev:.1f} >= {CONFIG['SEVERITY_FAILED_THRESHOLD']:.1f} -> SCREEN_FAILED"
            gate_stopped = 'severity_failed'
        elif sev >= CONFIG['SEVERITY_FLAGS_THRESHOLD']:
            severity_gate[i] = f"Severity {sev:.1f} >= {CONFIG['SEVERITY_FLAGS_THRESHOLD']:.1f} -> SCREEN_PASSED_FLAGS"
            gate_stopped = 'severity_flags'
        elif sev >= CONFIG['SEVERITY_MINOR_THRESHOLD']:
            severity_gate[i] = f"Severity {sev:.1f} >= {CONFIG['SEVERITY_MINOR_THRESHOLD']:.1f} -> SCREEN_PASSED_FLAGS"
            gate_stopped = 'severity_minor'
        else:
            severity_gate[i] = f"Severity {sev:.1f} < {CONFIG['SEVERITY_MINOR_THRESHOLD']:.1f} -> PASSED"
            gate_stopped = None

        # Gate 2: Pricing flags (only reached if severity passed)
        if gate_stopped:
            pricing_gate[i] = "Not reached (stopped at severity gate)"
        elif pf and b in ('A', 'B'):
            pricing_gate[i] = f"Flags [{pf}] + Band {b} -> SCREEN_PASSED_EXPENSIVE"
            gate_stopped = 'pricing'
        elif pf:
            pricing_gate[i] = f"Flags [{pf}] present but Band {b} not A/B -> continued"
        else:
            pricing_gate[i] = "No pricing flags -> PASSED"

        # Gate 3: Earnings quality (only reached if pricing passed)
        if gate_stopped and gate_stopped != 'pricing':
            earnings_gate[i] = "Not reached (stopped at severity gate)"
        elif gate_stopped == 'pricing':
            earnings_gate[i] = "Not reached (stopped at pricing gate)"
        elif eq == 'Aggressive' and b in ('A', 'B'):
            earnings_gate[i] = f"Aggressive earnings + Band {b} -> SCREEN_PASSED_FLAGS"
            gate_stopped = 'earnings'
        elif eq == 'Aggressive':
            earnings_gate[i] = f"Aggressive earnings but Band {b} not A/B -> continued"
        else:
            earnings_gate[i] = f"Earnings quality: {eq} -> PASSED"

        # Gate 4: Score band (only reached if all prior gates passed)
        if gate_stopped:
            score_gate[i] = f"Not reached (stopped at {'severity' if 'severity' in str(gate_stopped) else gate_stopped} gate)"
        elif b in ('A', 'B'):
            score_gate[i] = f"Band {b} (score {composite[i]:.1f}) -> GATES_CLEARED"
        elif b == 'C':
            score_gate[i] = f"Band C (score {composite[i]:.1f}) -> SCREEN_MARGINAL"
        else:
            score_gate[i] = f"Band {b} (score {composite[i]:.1f}) -> SCREEN_FAILED"

        # Override
        if ov:
            override_col[i] = f"{ov} -> {d}"
        else:
            override_col[i] = "No override"

        # Active flags with weights
        flag_parts = []
        if qf:
            for f in qf.split(', '):
                defn = RED_FLAG_DEFINITIONS.get(f, {})
                flag_parts.append(f"{f}({defn.get('severity', '?')},{defn.get('weight', '?')})")
        if pf:
            for f in pf.split(', '):
                defn = RED_FLAG_DEFINITIONS.get(f, {})
                flag_parts.append(f"{f}({defn.get('severity', '?')},{defn.get('weight', '?')})")
        flags_detail[i] = ' | '.join(flag_parts) if flag_parts else 'No flags'

        # Full narrative
        parts = [f"Composite {composite[i]:.1f} (Band {b})."]

        if sev > 0:
            parts.append(f"Quality severity {sev:.1f} from: {qf}.")
        else:
            parts.append("No quality flags.")

        if psev > 0:
            parts.append(f"Pricing severity {psev:.1f} from: {pf}.")

        if eq != 'Clean':
            parts.append(f"Earnings quality: {eq}.")

        sa = sector_adj[i] if pd.notna(sector_adj[i]) and sector_adj[i] else ''
        if sa:
            parts.append(f"Sector adjustments: {sa}.")

        cr = cyclic_risk[i] if pd.notna(cyclic_risk[i]) else 'LOW'
        if cr != 'LOW':
            parts.append(f"Cyclic peak risk: {cr}.")

        if ov:
            parts.append(f"Override: {ov}.")

        parts.append(f"Final: {d}.")
        narrative[i] = ' '.join(parts)

    return pd.DataFrame({
        'ISIN': analysis_df['ISIN'].values,
        'NSE_Code': analysis_df['NSE_Code'].values,
        'BSE_Code': analysis_df['BSE_Code'].values,
        'Decision_Bucket': decision,
        'Composite_Score': composite,
        'Score_Breakdown': score_breakdown,
        'Gate_1_Severity': severity_gate,
        'Gate_2_Pricing': pricing_gate,
        'Gate_3_Earnings_Quality': earnings_gate,
        'Gate_4_Score_Band': score_gate,
        'Override_Applied': override_col,
        'Active_Flags_Detail': flags_detail,
        'Earnings_Quality': earnings_quality,
        'Cyclic_Peak_Risk': cyclic_risk,
        'Decision_Narrative': narrative,
    })


def format_excel(writer: pd.ExcelWriter, sheets: Dict[str, pd.DataFrame]) -> None:
    """Apply formatting."""
    wb = writer.book
    hdr = wb.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white', 'border': 1})
    bucket_fmts = {
        'GATES_CLEARED': wb.add_format({'bg_color': '#92D050'}), 'SCREEN_PASSED_EXPENSIVE': wb.add_format({'bg_color': '#FFC000'}),
        'SCREEN_PASSED_FLAGS': wb.add_format({'bg_color': '#FFEB9C'}), 'SCREEN_FAILED': wb.add_format({'bg_color': '#FF6B6B'}),
        'CONTRARIAN_BET': wb.add_format({'bg_color': '#7030A0', 'font_color': 'white'}), 'VALUE_TRAP': wb.add_format({'bg_color': '#C00000', 'font_color': 'white'}),
    }
    for name, df in sheets.items():
        ws = writer.sheets[name]
        for i, c in enumerate(df.columns): ws.write(0, i, c, hdr)
        max_width = 80 if name == 'Decision_Audit' else 30
        for i, c in enumerate(df.columns):
            mx = max(len(str(c)), df[c].fillna('').astype(str).str.len().max() if len(df) > 0 else 0)
            ws.set_column(i, i, min(mx + 2, max_width))
        ws.freeze_panes(1, 1)
        if name == 'Analysis' and 'Decision_Bucket' in df.columns:
            bc = df.columns.get_loc('Decision_Bucket')
            for ri in range(len(df)):
                v = df.iloc[ri, bc]
                if v in bucket_fmts: ws.write(ri + 1, bc, v, bucket_fmts[v])


def main():
    parser = argparse.ArgumentParser(description='Stock analysis')
    parser.add_argument('--output', default=os.path.join(BASE_DIR, 'stock_analysis.xlsx'))
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logging.info("Starting stock analysis...")
    
    dfs = load_parquet_files()
    if len(dfs) < 6:
        logging.error(f"Missing parquet files. Found {len(dfs)}, expected 6.")
        sys.exit(1)
    
    merged = merge_all(dfs)
    
    master_df = build_master_sheet(merged)
    valuation_df = build_valuation_sheet(merged)
    quality_df = build_quality_sheet(merged)
    cashflow_df = build_cashflow_sheet(merged)
    leverage_df = build_leverage_sheet(merged)
    growth_df = build_growth_sheet(merged)
    shareholding_df = build_shareholding_sheet(merged)
    neglected_df = build_neglected_firm_sheet(merged, quality_df, leverage_df, shareholding_df)
    dividends_df = build_dividends_sheet(merged)
    red_flags_df = build_red_flags_sheet(merged, quality_df, cashflow_df, leverage_df, valuation_df, growth_df)
    analysis_df = build_analysis_sheet(merged, quality_df, valuation_df, leverage_df, growth_df, cashflow_df, red_flags_df, shareholding_df)
    decision_audit_df = build_decision_audit_sheet(merged, analysis_df, quality_df, valuation_df, leverage_df, growth_df, cashflow_df, red_flags_df, shareholding_df)
    summary_df = build_summary_sheet(analysis_df)

    sort_idx = analysis_df['Composite_Score'].astype(float).argsort()[::-1]
    analysis_df = analysis_df.iloc[sort_idx].reset_index(drop=True)
    sheet_dfs = {n: df.iloc[sort_idx].reset_index(drop=True) for n, df in
                 {'Master': master_df, 'Valuation': valuation_df, 'Quality': quality_df, 'Cash_Flow': cashflow_df,
                  'Leverage': leverage_df, 'Growth': growth_df, 'Shareholding': shareholding_df,
                  'Neglected_Firm': neglected_df, 'Dividends': dividends_df, 'Red_Flags': red_flags_df,
                  'Decision_Audit': decision_audit_df}.items()}

    all_sheets = {'Analysis': analysis_df, **sheet_dfs, 'Summary': summary_df}
    with pd.ExcelWriter(args.output, engine='xlsxwriter') as writer:
        for name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
        format_excel(writer, all_sheets)
    
    logging.info(f"Done! Output: {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == '__main__':
    main()