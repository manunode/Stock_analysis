"""
Data I/O layer: parquet loading, merging, column access helpers, sanity checks.
Depends on: rules (COLS, COL_ALIASES, SUFFIX)
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List

from .rules import COLS, COL_ALIASES, SUFFIX


# ── Parquet file paths (relative to caller-supplied base_dir) ───────────────

PARQUET_FILENAMES = {
    'annual': 'all_stocks_annual_financials.parquet',
    'quarterly': 'all_stocks_quarterly_financials.parquet',
    'balance': 'all_stocks_balance_sheet.parquet',
    'cashflow': 'all_stocks_cash_flow.parquet',
    'ratios': 'all_stocks_ratios.parquet',
    'user_ratios': 'all_stocks_user_ratios.parquet',
}


def parquet_paths(base_dir: str) -> Dict[str, str]:
    """Return absolute parquet paths for a given base directory."""
    return {k: os.path.join(base_dir, v) for k, v in PARQUET_FILENAMES.items()}


# ── Column helpers ──────────────────────────────────────────────────────────

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
    if metric_key in COL_ALIASES:
        for alias in COL_ALIASES[metric_key]:
            col_name = f"{alias}{SUFFIX.get(suffix_key, '')}" if suffix_key else alias
            if col_name in df.columns:
                return df[col_name]

    col_name = col(metric_key, suffix_key)
    if col_name in df.columns:
        return df[col_name]
    return default


# ── Loading & merging ───────────────────────────────────────────────────────

def load_parquet_files(base_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all parquet files into a dict of DataFrames."""
    paths = parquet_paths(base_dir)
    dfs = {}
    for key, path in paths.items():
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


# ── Sanity checks ───────────────────────────────────────────────────────────

def run_sanity_checks(m: pd.DataFrame) -> None:
    """Log warnings for data integrity issues that could corrupt downstream calculations."""
    pat = pd.Series(get_col(m, 'PAT', 'ANNUAL'), dtype=float)
    sales = pd.Series(get_col(m, 'SALES', 'ANNUAL'), dtype=float)
    other_income = pd.Series(get_col(m, 'OTHER_INCOME', 'ANNUAL'), dtype=float)
    debt = pd.Series(get_col(m, 'DEBT', 'BALANCE'), dtype=float)
    roe = pd.Series(get_col(m, 'ROE', 'RATIOS'), dtype=float)
    total_assets = pd.Series(get_col(m, 'TOTAL_ASSETS', 'BALANCE'), dtype=float)
    names = m[COLS['NAME']].values

    warnings = []
    info = []

    # PAT > Sales: distinguish Holding Companies from data glitches
    pat_gt_sales = (~pd.isna(pat)) & (~pd.isna(sales)) & (sales > 0) & (pat > sales)
    if pat_gt_sales.sum() > 0:
        holding_co = pat_gt_sales & (~pd.isna(other_income)) & (other_income > 0.8 * pat)
        data_glitch = pat_gt_sales & ((pd.isna(other_income)) | (other_income < 0.5 * pat))

        if holding_co.sum() > 0:
            examples = ', '.join(str(n) for n in names[holding_co][:5])
            info.append(f"{holding_co.sum()} likely Holding Companies (PAT > Sales from Other Income): {examples}")
        if data_glitch.sum() > 0:
            examples = ', '.join(str(n) for n in names[data_glitch][:5])
            warnings.append(f"{data_glitch.sum()} stocks have PAT > Sales without supporting Other Income (likely data error): {examples}")

    # Negative debt
    neg_debt = (~pd.isna(debt)) & (debt < 0)
    if neg_debt.sum() > 0:
        warnings.append(f"{neg_debt.sum()} stocks have negative debt values")

    # Extreme ROE (|ROE| > 200%)
    extreme_roe = (~pd.isna(roe)) & (np.abs(roe) > 200)
    if extreme_roe.sum() > 0:
        warnings.append(f"{extreme_roe.sum()} stocks have |ROE| > 200%")

    # Zero or negative total assets
    bad_assets = (~pd.isna(total_assets)) & (total_assets <= 0)
    if bad_assets.sum() > 0:
        warnings.append(f"{bad_assets.sum()} stocks have zero/negative total assets")

    for msg in info:
        logging.info(f"  {msg}")
    if warnings:
        logging.warning("Data integrity warnings:")
        for w in warnings:
            logging.warning(f"  - {w}")
    if not warnings and not info:
        logging.info("Data integrity checks: all passed")
