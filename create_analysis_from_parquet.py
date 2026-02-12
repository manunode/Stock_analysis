#!/usr/bin/env python3
"""
Stock Analysis from Parquet Files
Reads screener.in parquet data (6 files, ~5275 stocks) and produces
a multi-sheet stock_analysis.xlsx report with scoring, red flags, and decision buckets.

Usage:
    python create_analysis_from_parquet.py
    python create_analysis_from_parquet.py --output custom_output.xlsx
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime

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

# ── Financial sector identification ─────────────────────────────────────────
FINANCIAL_SECTORS_LOWER = frozenset(s.lower() for s in [
    'Banks', 'Private Banks', 'Public Banks', 'Foreign Banks',
    'Finance', 'Financial Services', 'NBFC', 'Non Banking Financial Company',
    'Insurance', 'Life Insurance', 'General Insurance',
    'Housing Finance', 'Consumer Finance', 'Asset Management',
    'Diversified Financials', 'Financial Institution',
    'Capital Markets',
])

# ── Red flag definitions ────────────────────────────────────────────────────
STRUCTURAL_RED_FLAGS = {
    "LOW_ROE": {"trigger": "ROE < 10%", "severity": "MAJOR", "weight": 1.0,
                "meaning": "Company generates weak returns on shareholder capital"},
    "DECLINING_ROE": {"trigger": "ROE avg declining AND latest < 15%", "severity": "MAJOR", "weight": 1.0,
                      "meaning": "Return on equity falling to mediocre level"},
    "LOW_ROCE": {"trigger": "ROCE < 10%", "severity": "MAJOR", "weight": 1.0,
                 "meaning": "Weak returns on total capital employed"},
    "DECLINING_ROCE": {"trigger": "ROCE avg declining AND latest < 15%", "severity": "MAJOR", "weight": 1.0,
                       "meaning": "Capital efficiency worsening"},
    "POOR_CASH_CONVERSION": {"trigger": "CFO/PAT < 0.7", "severity": "CRITICAL", "weight": 2.0,
                             "meaning": "Reported profits not converting to cash"},
    "NEGATIVE_CFO": {"trigger": "Latest CFO < 0", "severity": "CRITICAL", "weight": 2.0,
                     "meaning": "Core operations burning cash"},
    "INCONSISTENT_CFO": {"trigger": "Cumulative 3yr CFO < 0", "severity": "CRITICAL", "weight": 2.0,
                         "meaning": "Operating cash flow not reliably positive"},
    "HIGH_OTHER_INCOME": {"trigger": "Other income > 30% of PAT", "severity": "MAJOR", "weight": 1.0,
                          "meaning": "Significant profit from non-core activities"},
    "MARGIN_COMPRESSION": {"trigger": "OPM declining trend", "severity": "MINOR", "weight": 0.5,
                           "meaning": "Profit margins shrinking over time"},
    "RISING_DEBT": {"trigger": "Debt growing significantly vs 3yr ago", "severity": "CRITICAL", "weight": 2.0,
                    "meaning": "Financial risk increasing"},
    "WC_DIVERGENCE": {"trigger": "WC growth >> Revenue growth", "severity": "MINOR", "weight": 0.5,
                      "meaning": "Working capital growing much faster than sales"},
}

PRICING_RED_FLAGS = {
    "HIGH_PE": {"trigger": "P/E > 50", "severity": "MINOR", "weight": 0.5,
                "meaning": "Very high earnings multiple"},
    "NEGATIVE_PE": {"trigger": "P/E < 0", "severity": "MAJOR", "weight": 1.0,
                    "meaning": "Company is loss-making"},
    "HIGH_EV_EBITDA": {"trigger": "EV/EBITDA > 25", "severity": "MINOR", "weight": 0.5,
                       "meaning": "Expensive on cash flow basis"},
    "NEGATIVE_EBITDA": {"trigger": "EV/EBITDA < 0", "severity": "CRITICAL", "weight": 2.0,
                        "meaning": "Negative operating profit before depreciation"},
    "HIGH_PBV_ROE": {"trigger": "PBV > ROE/2", "severity": "MINOR", "weight": 0.5,
                     "meaning": "Price implies returns company cannot deliver"},
}

RED_FLAG_DEFINITIONS = {**STRUCTURAL_RED_FLAGS, **PRICING_RED_FLAGS}


def safe_div(a, b, default=np.nan):
    """Safe division handling NaN and zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((pd.isna(b)) | (b == 0), default, a / b)
    return result


def load_parquet_files():
    """Load all parquet files into a dict of DataFrames."""
    dfs = {}
    for key, path in PARQUET_FILES.items():
        if not os.path.exists(path):
            print(f"WARNING: Missing parquet file: {path}")
            continue
        df = pd.read_parquet(path)
        print(f"  Loaded {key}: {df.shape[0]} stocks, {df.shape[1]} columns")
        dfs[key] = df
    return dfs


def merge_all(dfs):
    """Merge all parquet DataFrames on common identifiers."""
    # Use ISIN Code as primary key (most reliable), fallback to Name
    merge_cols = ['Name', 'ISIN Code']

    # Start with annual financials as base
    merged = dfs['annual'][['Name', 'BSE Code', 'NSE Code', 'ISIN Code',
                            'Industry Group', 'Industry', 'Current Price',
                            'Market Capitalization']].copy()

    # Add suffixes to avoid column collision
    suffix_map = {
        'annual': '_ann',
        'quarterly': '_qtr',
        'balance': '_bal',
        'cashflow': '_cf',
        'ratios': '_rat',
        'user_ratios': '_ur',
    }

    for key, df in dfs.items():
        if key == 'annual':
            # Annual is already the base — just add suffixed versions of non-base columns
            drop_cols = {'Name', 'BSE Code', 'NSE Code', 'ISIN Code',
                         'Industry Group', 'Industry', 'Current Price', 'Market Capitalization'}
            for c in df.columns:
                if c not in drop_cols:
                    merged[f"{c}_ann"] = df[c].values
            continue

        suffix = suffix_map[key]
        # Drop common identifier/price columns from the merge source to avoid duplication
        drop_cols = {'Name', 'BSE Code', 'NSE Code', 'ISIN Code', 'Industry Group', 'Industry',
                     'Current Price', 'Market Capitalization'}
        cols_to_keep = ['ISIN Code'] + [c for c in df.columns if c not in drop_cols]
        df_subset = df[cols_to_keep].copy()

        # Rename columns with suffix (except merge key)
        rename = {c: f"{c}{suffix}" for c in df_subset.columns if c != 'ISIN Code'}
        df_subset = df_subset.rename(columns=rename)

        merged = merged.merge(df_subset, on='ISIN Code', how='left')

    print(f"  Merged dataset: {merged.shape[0]} stocks, {merged.shape[1]} columns")
    return merged


def build_master_sheet(m):
    """Build the Master sheet with stock identification info."""
    is_fin = m['Industry Group'].fillna('').str.lower().isin(FINANCIAL_SECTORS_LOWER) | \
             m['Industry'].fillna('').str.lower().isin(FINANCIAL_SECTORS_LOWER)

    master = pd.DataFrame({
        'NSE_Code': m['NSE Code'],
        'Stock_Name': m['Name'],
        'Sector': m['Industry Group'],
        'Industry': m['Industry'],
        'BSE_Code': m['BSE Code'],
        'ISIN': m['ISIN Code'],
        'Market_Cap': m['Market Capitalization'],
        'Current_Price': m['Current Price'],
        'Is_Financial_Sector': is_fin.map({True: 'Yes', False: 'No'}),
        'Face_Value': m.get('Face value_bal', np.nan),
    })
    return master


def build_valuation_sheet(m):
    """Build valuation metrics sheet."""
    pe = m.get('Price to Earning_rat', np.nan)
    pbv = m.get('Price to book value_rat', np.nan)
    roe = m.get('Return on equity_rat', np.nan)

    # Earnings yield = 1/PE (as percentage)
    ey = safe_div(100.0, pe)

    # Valuation comfort score (0-100)
    # PE score: lower is better (capped at 0-40)
    pe_s = pd.Series(pe)
    pe_score = np.where(pd.isna(pe_s), 10,
               np.where(pe_s < 0, 5,
               np.where(pe_s <= 10, 40,
               np.where(pe_s <= 15, 35,
               np.where(pe_s <= 20, 30,
               np.where(pe_s <= 25, 25,
               np.where(pe_s <= 35, 20,
               np.where(pe_s <= 50, 15,
               np.where(pe_s <= 80, 10, 5)))))))))

    # PBV score (0-20): lower is better
    pbv_s = pd.Series(pbv)
    pbv_score = np.where(pd.isna(pbv_s), 5,
                np.where(pbv_s <= 1, 20,
                np.where(pbv_s <= 2, 16,
                np.where(pbv_s <= 3, 12,
                np.where(pbv_s <= 5, 8,
                np.where(pbv_s <= 8, 5, 2))))))

    # EV/EBITDA score (0-20)
    ev = pd.Series(m.get('EVEBITDA_ur', np.nan))
    ev_score = np.where(pd.isna(ev), 5,
               np.where(ev < 0, 3,
               np.where(ev <= 8, 20,
               np.where(ev <= 12, 16,
               np.where(ev <= 16, 12,
               np.where(ev <= 20, 8,
               np.where(ev <= 25, 5, 2)))))))

    # PEG score (0-20)
    peg = pd.Series(m.get('PEG Ratio_ur', np.nan))
    peg_score = np.where(pd.isna(peg), 5,
                np.where(peg < 0, 3,
                np.where(peg <= 0.5, 20,
                np.where(peg <= 1.0, 16,
                np.where(peg <= 1.5, 12,
                np.where(peg <= 2.0, 8,
                np.where(peg <= 3.0, 5, 2)))))))

    val_comfort = (pd.Series(pe_score) + pd.Series(pbv_score) +
                   pd.Series(ev_score) + pd.Series(peg_score))

    # Valuation band
    val_band = np.where(val_comfort >= 70, 'CHEAP',
               np.where(val_comfort >= 50, 'FAIR',
               np.where(val_comfort >= 30, 'EXPENSIVE', 'VERY_EXPENSIVE')))

    valuation = pd.DataFrame({
        'NSE_Code': m['NSE Code'],
        'ISIN': m['ISIN Code'],
        'PE': pe,
        'PBV': pbv,
        'EV_EBITDA': m.get('EVEBITDA_ur', np.nan),
        'PEG': m.get('PEG Ratio_ur', np.nan),
        'Price_To_Sales': m.get('Price to Sales_ur', np.nan),
        'Earnings_Yield': ey,
        'Valuation_Band': val_band,
        'Valuation_Comfort_Score': val_comfort,
        'Market_Cap': m['Market Capitalization'],
        'LTP': m['Current Price'],
        'Book_Value': m.get('Book value_rat', np.nan),
        'Industry_PE': m.get('Industry PE_rat', np.nan),
        'Industry_PBV': m.get('Industry PBV_rat', np.nan),
        'Historical_PE_3Yr': m.get('Historical PE 3Years_rat', np.nan),
        'Historical_PE_5Yr': m.get('Historical PE 5Years_rat', np.nan),
        'Historical_PE_10Yr': m.get('Historical PE 10Years_rat', np.nan),
        'Down_From_52W_High': m.get('Down from 52w high_ur', np.nan),
        'Up_From_52W_Low': m.get('Up from 52w low_ur', np.nan),
        'Price_To_Quarterly_Earning': m.get('Price to Quarterly Earning_rat', np.nan),
        'Market_Cap_3Yr_Back': m.get('Market Capitalization 3years back_rat', np.nan),
        'Market_Cap_5Yr_Back': m.get('Market Capitalization 5years back_rat', np.nan),
        'Market_Cap_10Yr_Back': m.get('Market Capitalization 10years back_rat', np.nan),
    })
    return valuation


def build_quality_sheet(m):
    """Build quality metrics sheet."""
    roe = pd.Series(m.get('Return on equity_rat', np.nan), dtype=float)
    roe_3 = pd.Series(m.get('Average return on equity 3Years_rat', np.nan), dtype=float)
    roe_5 = pd.Series(m.get('Average return on equity 5Years_rat', np.nan), dtype=float)
    roe_10 = pd.Series(m.get('Average return on equity 10Years_rat', np.nan), dtype=float)
    roce = pd.Series(m.get('Return on capital employed_ann', np.nan), dtype=float)
    roce_3 = pd.Series(m.get('Average return on capital employed 3Years_rat', np.nan), dtype=float)
    roce_5 = pd.Series(m.get('Average return on capital employed 5Years_rat', np.nan), dtype=float)
    roce_7 = pd.Series(m.get('Average return on capital employed 7Years_rat', np.nan), dtype=float)
    roce_10 = pd.Series(m.get('Average return on capital employed 10Years_rat', np.nan), dtype=float)
    roa = pd.Series(m.get('Return on assets_rat', np.nan), dtype=float)
    opm = pd.Series(m.get('OPM_ann', np.nan), dtype=float)
    opm_ly = pd.Series(m.get('OPM last year_ann', np.nan), dtype=float)
    opm_py = pd.Series(m.get('OPM preceding year_ann', np.nan), dtype=float)
    npm_ly = pd.Series(m.get('NPM last year_ann', np.nan), dtype=float)
    piotroski = pd.Series(m.get('Piotroski score_rat', np.nan), dtype=float)
    asset_turnover = pd.Series(m.get('Asset Turnover Ratio_rat', np.nan), dtype=float)

    # ROE trend: compare latest to 3yr avg
    roe_trend = np.where(pd.isna(roe) | pd.isna(roe_3), 'N/A',
                np.where(roe > roe_3 + 2, 'IMPROVING',
                np.where(roe < roe_3 - 2, 'DECLINING', 'STABLE')))

    # ROCE trend: compare latest to 3yr avg
    roce_trend = np.where(pd.isna(roce) | pd.isna(roce_3), 'N/A',
                 np.where(roce > roce_3 + 2, 'IMPROVING',
                 np.where(roce < roce_3 - 2, 'DECLINING', 'STABLE')))

    # OPM trend: compare current to last year and preceding year
    opm_trend = np.where(pd.isna(opm) | pd.isna(opm_py), 'N/A',
                np.where(opm > opm_py + 1, 'IMPROVING',
                np.where(opm < opm_py - 1, 'DECLINING', 'STABLE')))

    # NPM trend
    npm = pd.Series(m.get('NPM last year_ann', np.nan), dtype=float)
    npm_py_val = pd.Series(m.get('NPM preceding year_ann', np.nan), dtype=float)
    npm_trend = np.where(pd.isna(npm) | pd.isna(npm_py_val), 'N/A',
                np.where(npm > npm_py_val + 1, 'IMPROVING',
                np.where(npm < npm_py_val - 1, 'DECLINING', 'STABLE')))

    # Piotroski assessment
    piotroski_assess = np.where(pd.isna(piotroski), 'N/A',
                      np.where(piotroski >= 7, 'STRONG',
                      np.where(piotroski >= 4, 'NEUTRAL', 'WEAK')))

    # ROE-ROA gap (high gap = leverage-driven)
    roe_roa_gap = roe - roa
    leverage_driven = np.where(pd.isna(roe_roa_gap), 'N/A',
                     np.where(roe_roa_gap > 15, 'Yes', 'No'))

    # Other income as % of PAT
    pat = pd.Series(m.get('Profit after tax_ann', np.nan), dtype=float)
    other_inc = pd.Series(m.get('Other income_ann', np.nan), dtype=float)
    other_inc_pct = safe_div(np.abs(other_inc) * 100, np.abs(pat))

    # Business Quality Score (0-100)
    # ROE component (0-25)
    roe_score = np.where(pd.isna(roe), 5,
                np.where(roe >= 25, 25,
                np.where(roe >= 20, 22,
                np.where(roe >= 15, 18,
                np.where(roe >= 12, 14,
                np.where(roe >= 10, 10,
                np.where(roe >= 5, 6, 2)))))))

    # ROCE component (0-25)
    roce_score = np.where(pd.isna(roce), 5,
                 np.where(roce >= 25, 25,
                 np.where(roce >= 20, 22,
                 np.where(roce >= 15, 18,
                 np.where(roce >= 12, 14,
                 np.where(roce >= 10, 10,
                 np.where(roce >= 5, 6, 2)))))))

    # Margin component (0-20): OPM level
    opm_score = np.where(pd.isna(opm), 5,
                np.where(opm >= 25, 20,
                np.where(opm >= 20, 17,
                np.where(opm >= 15, 14,
                np.where(opm >= 10, 10,
                np.where(opm >= 5, 6, 2))))))

    # Piotroski component (0-15)
    pio_score = np.where(pd.isna(piotroski), 5,
                np.where(piotroski >= 8, 15,
                np.where(piotroski >= 6, 12,
                np.where(piotroski >= 4, 8,
                np.where(piotroski >= 2, 4, 1)))))

    # Consistency component (0-15): ROE stability (3yr vs 5yr vs 10yr avg)
    consistency = np.where(
        pd.isna(roe_5) | pd.isna(roe_3), 5,
        np.where(np.abs(roe_5 - roe_3) <= 2, 15,
        np.where(np.abs(roe_5 - roe_3) <= 5, 10,
        np.where(np.abs(roe_5 - roe_3) <= 10, 6, 3))))

    bq_score = (pd.Series(roe_score) + pd.Series(roce_score) +
                pd.Series(opm_score) + pd.Series(pio_score) +
                pd.Series(consistency))

    # Earnings quality
    eq = np.where(bq_score >= 70, 'HIGH',
         np.where(bq_score >= 50, 'MODERATE',
         np.where(bq_score >= 30, 'LOW', 'POOR')))

    quality = pd.DataFrame({
        'NSE_Code': m['NSE Code'],
        'ISIN': m['ISIN Code'],
        'Business_Quality_Score': bq_score,
        'Earnings_Quality': eq,
        'Piotroski_Score': piotroski,
        'Piotroski_Assessment': piotroski_assess,
        'ROE_Latest': roe,
        'ROE_3Yr_Avg': roe_3,
        'ROE_5Yr_Avg': roe_5,
        'ROE_10Yr_Avg': roe_10,
        'ROE_Trend': roe_trend,
        'ROCE_Latest': roce,
        'ROCE_3Yr_Avg': roce_3,
        'ROCE_5Yr_Avg': roce_5,
        'ROCE_7Yr_Avg': roce_7,
        'ROCE_10Yr_Avg': roce_10,
        'ROCE_Trend': roce_trend,
        'ROA_Latest': roa,
        'ROA_3Yr': m.get('Return on assets 3years_rat', np.nan),
        'ROA_5Yr': m.get('Return on assets 5years_rat', np.nan),
        'OPM_Latest': opm,
        'OPM_Last_Year': opm_ly,
        'OPM_Preceding_Year': opm_py,
        'OPM_5Year': m.get('OPM 5Year_rat', np.nan),
        'OPM_10Year': m.get('OPM 10Year_rat', np.nan),
        'OPM_Trend': opm_trend,
        'NPM_Latest': npm_ly,
        'NPM_Trend': npm_trend,
        'Asset_Turnover': asset_turnover,
        'ROE_ROA_Gap': roe_roa_gap,
        'Leverage_Driven': leverage_driven,
        'Other_Income_Pct_PAT': other_inc_pct,
    })
    return quality


def build_cashflow_sheet(m):
    """Build cash flow metrics sheet."""
    cfo = pd.Series(m.get('Cash from operations last year_cf', np.nan), dtype=float)
    fcf = pd.Series(m.get('Free cash flow last year_cf', np.nan), dtype=float)
    pat_annual = pd.Series(m.get('Profit after tax_ann', np.nan), dtype=float)

    # TTM PAT from quarterly data (sum of last 4 quarters)
    np_q1 = pd.Series(m.get('Net Profit latest quarter_qtr', np.nan), dtype=float)
    np_q2 = pd.Series(m.get('Net Profit preceding quarter_qtr', np.nan), dtype=float)
    np_q3 = pd.Series(m.get('Net profit 2quarters back_qtr', np.nan), dtype=float)
    np_q4 = pd.Series(m.get('Net profit 3quarters back_qtr', np.nan), dtype=float)
    pat_ttm = np_q1 + np_q2 + np_q3 + np_q4

    # Use TTM PAT where available, fallback to annual
    pat = np.where(pd.notna(pat_ttm) & (pat_ttm != 0), pat_ttm, pat_annual)
    pat = pd.Series(pat, dtype=float)

    cfo_pat = safe_div(cfo, pat)

    # 3yr cumulative CFO and FCF
    cfo_3yr = pd.Series(m.get('Operating cash flow 3years_cf', np.nan), dtype=float)
    fcf_3yr = pd.Series(m.get('Free cash flow 3years_cf', np.nan), dtype=float)
    cfo_5yr = pd.Series(m.get('Operating cash flow 5years_cf', np.nan), dtype=float)
    fcf_5yr = pd.Series(m.get('Free cash flow 5years_cf', np.nan), dtype=float)
    cfo_7yr = pd.Series(m.get('Operating cash flow 7years_cf', np.nan), dtype=float)
    fcf_7yr = pd.Series(m.get('Free cash flow 7years_cf', np.nan), dtype=float)
    cfo_10yr = pd.Series(m.get('Operating cash flow 10years_cf', np.nan), dtype=float)
    fcf_10yr = pd.Series(m.get('Free cash flow 10years_cf', np.nan), dtype=float)

    # 3yr CFO/PAT: Use average of individual year ratios (not ratio of sums)
    # This weights each year equally, preventing one blockbuster year from masking
    # a terrible year — matching the original script's approach.
    pat_ly = pd.Series(m.get('Profit after tax last year_ann', np.nan), dtype=float)
    pat_py = pd.Series(m.get('Profit after tax preceding year_ann', np.nan), dtype=float)
    pat_3yr_cum = pat_annual + pat_ly + pat_py

    # Individual year CFO/PAT ratios
    cfo_py_series = pd.Series(m.get('Cash from operations preceding year_cf', np.nan), dtype=float)
    cfo_yr3_raw = cfo_3yr - cfo - cfo_py_series  # inferred 3rd year back
    ratio_yr1 = safe_div(cfo, pat_annual)
    ratio_yr2 = safe_div(cfo_py_series, pat_ly)
    ratio_yr3 = safe_div(cfo_yr3_raw, pat_py)

    # Average of individual year ratios (more conservative than ratio of sums)
    # Only average where individual ratios are not NaN
    r1 = pd.Series(ratio_yr1, dtype=float)
    r2 = pd.Series(ratio_yr2, dtype=float)
    r3 = pd.Series(ratio_yr3, dtype=float)
    ratio_sum = r1.fillna(0) + r2.fillna(0) + r3.fillna(0)
    ratio_count = (~pd.isna(r1)).astype(int) + (~pd.isna(r2)).astype(int) + (~pd.isna(r3)).astype(int)
    cfo_pat_3yr_avg = np.where(ratio_count >= 2, ratio_sum / ratio_count, safe_div(cfo_3yr, pat_3yr_cum))
    cfo_pat_3yr_avg = pd.Series(cfo_pat_3yr_avg, dtype=float)

    # Track the worst individual year ratio (for flagging)
    worst_yr_ratio = np.minimum(np.minimum(
        np.where(pd.isna(r1), 999, r1),
        np.where(pd.isna(r2), 999, r2)),
        np.where(pd.isna(r3), 999, r3))
    worst_yr_ratio = np.where(worst_yr_ratio == 999, np.nan, worst_yr_ratio)

    # Infer Year 3 CFO: cumulative_3yr - latest - preceding
    cfo_py = cfo_py_series  # already computed above
    cfo_yr3 = cfo_yr3_raw  # already computed above

    # Count positive CFO years (from what we can reconstruct: latest, preceding, inferred yr3)
    pos_cfo_count = ((cfo > 0).astype(int) +
                     (cfo_py > 0).astype(int) +
                     (cfo_yr3 > 0).astype(int))

    # CFO trend: compare latest to preceding year
    cfo_trend = np.where(pd.isna(cfo) | pd.isna(cfo_py), 'N/A',
                np.where(cfo > cfo_py * 1.1, 'IMPROVING',
                np.where(cfo < cfo_py * 0.9, 'DECLINING', 'STABLE')))

    # CFI (investing)
    cfi = pd.Series(m.get('Cash from investing last year_cf', np.nan), dtype=float)

    # CFROA = CFO / Total Assets
    total_assets = pd.Series(m.get('Total Assets_bal', np.nan), dtype=float)
    cfroa = safe_div(cfo, total_assets)

    # Accruals = (PAT - CFO) / Total Assets — lower is better
    accruals = safe_div(pat - cfo, total_assets)

    # Working capital growth vs revenue growth
    wc = pd.Series(m.get('Working capital_bal', np.nan), dtype=float)
    wc_py = pd.Series(m.get('Working capital preceding year_bal', np.nan), dtype=float)
    wc_growth = safe_div(wc - wc_py, np.abs(wc_py)) * 100

    sales = pd.Series(m.get('Sales_ann', np.nan), dtype=float)
    sales_ly = pd.Series(m.get('Sales last year_ann', np.nan), dtype=float)
    rev_growth = safe_div(sales - sales_ly, np.abs(sales_ly)) * 100

    wc_rev_ratio = safe_div(wc_growth, rev_growth)

    cashflow = pd.DataFrame({
        'NSE_Code': m['NSE Code'],
        'ISIN': m['ISIN Code'],
        'CFO_Latest': cfo,
        'PAT_Latest': pat,
        'PAT_TTM': pat_ttm,
        'PAT_Annual': pat_annual,
        'CFO_PAT_Latest': cfo_pat,
        'CFO_PAT_3Yr_Avg': cfo_pat_3yr_avg,
        'Positive_CFO_Years_3Yr': pos_cfo_count,
        'CFO_Year3_Inferred': cfo_yr3,
        'Worst_Year_CFO_PAT': worst_yr_ratio,
        'FCF_Latest': fcf,
        'CFO_Preceding_Year': cfo_py,
        'CFO_Trend': cfo_trend,
        'CFO_3Yr_Cumulative': cfo_3yr,
        'CFO_5Yr_Cumulative': cfo_5yr,
        'CFO_7Yr_Cumulative': cfo_7yr,
        'CFO_10Yr_Cumulative': cfo_10yr,
        'FCF_3Yr_Cumulative': fcf_3yr,
        'FCF_5Yr_Cumulative': fcf_5yr,
        'CFO_7Yr_Cumulative': cfo_7yr,
        'CFO_10Yr_Cumulative': cfo_10yr,
        'FCF_3Yr_Cumulative': fcf_3yr,
        'FCF_5Yr_Cumulative': fcf_5yr,
        'FCF_7Yr_Cumulative': fcf_7yr,
        'FCF_10Yr_Cumulative': fcf_10yr,
        'CFI_Latest': cfi,
        'CFROA': cfroa,
        'Accruals': accruals,
        'WC_Growth_Pct': wc_growth,
        'Rev_Growth_Pct': rev_growth,
        'WC_Rev_Growth_Ratio': wc_rev_ratio,
    })
    return cashflow


def build_leverage_sheet(m):
    """Build leverage and financial strength sheet."""
    de = pd.Series(m.get('Debt to equity_rat', np.nan), dtype=float)
    ic = pd.Series(m.get('Interest Coverage Ratio_ur', np.nan), dtype=float)
    debt = pd.Series(m.get('Debt_bal', np.nan), dtype=float)
    debt_3yr = pd.Series(m.get('Debt 3Years back_bal', np.nan), dtype=float)
    debt_5yr = pd.Series(m.get('Debt 5Years back_bal', np.nan), dtype=float)
    reserves = pd.Series(m.get('Reserves_bal', np.nan), dtype=float)
    total_assets = pd.Series(m.get('Total Assets_bal', np.nan), dtype=float)
    ca = pd.Series(m.get('Current assets_bal', np.nan), dtype=float)
    cl = pd.Series(m.get('Current liabilities_bal', np.nan), dtype=float)
    wc = pd.Series(m.get('Working capital_bal', np.nan), dtype=float)
    net_block = pd.Series(m.get('Net block_bal', np.nan), dtype=float)
    gross_block = pd.Series(m.get('Gross block_bal', np.nan), dtype=float)

    current_ratio = safe_div(ca, cl)

    # Debt trend
    debt_trend = np.where(pd.isna(debt) | pd.isna(debt_3yr), 'N/A',
                 np.where(debt_3yr == 0,
                          np.where(debt == 0, 'ZERO_DEBT', 'RISING'),
                 np.where(debt < debt_3yr * 0.8, 'DECLINING',
                 np.where(debt > debt_3yr * 1.2, 'RISING', 'STABLE'))))

    # WC Turnover = Sales / Working Capital
    sales = pd.Series(m.get('Sales_ann', np.nan), dtype=float)
    wc_turnover = safe_div(sales, np.abs(wc))

    # Asset turnover = Sales / Total Assets
    asset_turnover = pd.Series(m.get('Asset Turnover Ratio_rat', np.nan), dtype=float)

    # Average working capital days
    avg_wc_days = pd.Series(m.get('Average Working Capital Days 3years_rat', np.nan), dtype=float)

    # Financial Strength Score (0-100)
    # D/E component (0-30): lower is better
    de_score = np.where(pd.isna(de), 10,
               np.where(de <= 0, 30,
               np.where(de <= 0.3, 28,
               np.where(de <= 0.5, 24,
               np.where(de <= 1.0, 18,
               np.where(de <= 1.5, 12,
               np.where(de <= 2.0, 6, 2)))))))

    # Interest coverage component (0-25): higher is better
    ic_score = np.where(pd.isna(ic), 8,
               np.where(ic >= 10, 25,
               np.where(ic >= 5, 20,
               np.where(ic >= 3, 15,
               np.where(ic >= 2, 10,
               np.where(ic >= 1, 5, 0))))))

    # Current ratio component (0-20)
    cr = pd.Series(current_ratio)
    cr_score = np.where(pd.isna(cr), 5,
               np.where(cr >= 2.0, 20,
               np.where(cr >= 1.5, 16,
               np.where(cr >= 1.2, 12,
               np.where(cr >= 1.0, 8,
               np.where(cr >= 0.8, 4, 1))))))

    # Debt trend component (0-25)
    dt_score = np.where(np.array(debt_trend) == 'ZERO_DEBT', 25,
               np.where(np.array(debt_trend) == 'DECLINING', 22,
               np.where(np.array(debt_trend) == 'STABLE', 15,
               np.where(np.array(debt_trend) == 'RISING', 5, 8))))

    fin_strength = (pd.Series(de_score) + pd.Series(ic_score) +
                    pd.Series(cr_score) + pd.Series(dt_score))

    leverage = pd.DataFrame({
        'NSE_Code': m['NSE Code'],
        'ISIN': m['ISIN Code'],
        'Debt_Equity': de,
        'Interest_Coverage': ic,
        'Debt_Trend': debt_trend,
        'Financial_Strength_Score': fin_strength,
        'Current_Ratio': current_ratio,
        'Total_Debt': debt,
        'Debt_3Yr_Back': debt_3yr,
        'Debt_5Yr_Back': debt_5yr,
        'Reserves': reserves,
        'Total_Assets': total_assets,
        'Working_Capital': wc,
        'WC_Turnover': wc_turnover,
        'Asset_Turnover': asset_turnover,
        'Avg_Working_Capital_Days_3Yr': avg_wc_days,
        'Current_Assets': ca,
        'Current_Liabilities': cl,
        'Net_Block': net_block,
        'Gross_Block': gross_block,
        'Contingent_Liabilities': m.get('Contingent liabilities_bal', np.nan),
        'Cash_Equivalents': m.get('Cash Equivalents_bal', np.nan),
        'Trade_Payables': m.get('Trade Payables_bal', np.nan),
    })
    return leverage


def build_growth_sheet(m):
    """Build growth metrics sheet."""
    sales_g3 = pd.Series(m.get('Sales growth 3Years_ann', np.nan), dtype=float)
    sales_g5 = pd.Series(m.get('Sales growth 5Years_ann', np.nan), dtype=float)
    sales_g10 = pd.Series(m.get('Sales growth 10Years_ann', np.nan), dtype=float)
    profit_g3 = pd.Series(m.get('Profit growth 3Years_ann', np.nan), dtype=float)
    profit_g5 = pd.Series(m.get('Profit growth 5Years_ann', np.nan), dtype=float)
    profit_g7 = pd.Series(m.get('Profit growth 7Years_ann', np.nan), dtype=float)
    profit_g10 = pd.Series(m.get('Profit growth 10Years_ann', np.nan), dtype=float)
    eps_g3 = pd.Series(m.get('EPS growth 3Years_ann', np.nan), dtype=float)
    eps_g5 = pd.Series(m.get('EPS growth 5Years_ann', np.nan), dtype=float)
    eps_g10 = pd.Series(m.get('EPS growth 10Years_ann', np.nan), dtype=float)

    # Quarterly growth
    qoq_sales = pd.Series(m.get('QoQ Sales_ur', np.nan), dtype=float)
    qoq_profit = pd.Series(m.get('QoQ Profits_ur', np.nan), dtype=float)
    yoy_q_sales = pd.Series(m.get('YOY Quarterly sales growth_qtr', np.nan), dtype=float)
    yoy_q_profit = pd.Series(m.get('YOY Quarterly profit growth_qtr', np.nan), dtype=float)

    # Revenue 1yr growth from annual data
    sales = pd.Series(m.get('Sales_ann', np.nan), dtype=float)
    sales_ly = pd.Series(m.get('Sales last year_ann', np.nan), dtype=float)
    rev_growth_1yr = safe_div(sales - sales_ly, np.abs(sales_ly)) * 100

    # Profit growth consistency: both 3yr and 5yr positive
    pg_consistent = np.where(pd.isna(profit_g3) | pd.isna(profit_g5), 'N/A',
                   np.where((profit_g3 > 0) & (profit_g5 > 0), 'CONSISTENT',
                   np.where((profit_g3 > 0) | (profit_g5 > 0), 'MIXED', 'DECLINING')))

    # Growth Durability Score (0-100)
    # Revenue growth (0-30)
    rg_score = np.where(pd.isna(sales_g3), 5,
               np.where(sales_g3 >= 20, 30,
               np.where(sales_g3 >= 15, 25,
               np.where(sales_g3 >= 10, 20,
               np.where(sales_g3 >= 5, 14,
               np.where(sales_g3 >= 0, 8, 3))))))

    # Profit growth (0-30)
    pg_score = np.where(pd.isna(profit_g3), 5,
               np.where(profit_g3 >= 25, 30,
               np.where(profit_g3 >= 15, 25,
               np.where(profit_g3 >= 10, 20,
               np.where(profit_g3 >= 5, 14,
               np.where(profit_g3 >= 0, 8, 3))))))

    # EPS growth (0-20)
    eg_score = np.where(pd.isna(eps_g3), 5,
               np.where(eps_g3 >= 20, 20,
               np.where(eps_g3 >= 12, 16,
               np.where(eps_g3 >= 5, 12,
               np.where(eps_g3 >= 0, 7, 2)))))

    # Consistency bonus (0-20)
    cons_score = np.where(np.array(pg_consistent) == 'CONSISTENT', 20,
                 np.where(np.array(pg_consistent) == 'MIXED', 10, 3))

    growth_durability = (pd.Series(rg_score) + pd.Series(pg_score) +
                         pd.Series(eg_score) + pd.Series(cons_score))

    # Book value
    bv = pd.Series(m.get('Book value_rat', np.nan), dtype=float)
    bv_3 = pd.Series(m.get('Book value 3years back_rat', np.nan), dtype=float)
    bv_5 = pd.Series(m.get('Book value 5years back_rat', np.nan), dtype=float)
    bv_10 = pd.Series(m.get('Book value 10years back_rat', np.nan), dtype=float)

    growth = pd.DataFrame({
        'NSE_Code': m['NSE Code'],
        'ISIN': m['ISIN Code'],
        'Revenue_Growth_1Yr': rev_growth_1yr,
        'Revenue_Growth_3Yr': sales_g3,
        'Revenue_Growth_5Yr': sales_g5,
        'Revenue_Growth_10Yr': sales_g10,
        'Profit_Growth_3Yr': profit_g3,
        'Profit_Growth_5Yr': profit_g5,
        'Profit_Growth_7Yr': profit_g7,
        'Profit_Growth_10Yr': profit_g10,
        'EPS_Growth_3Yr': eps_g3,
        'EPS_Growth_5Yr': eps_g5,
        'EPS_Growth_10Yr': eps_g10,
        'EBIDT_Growth_3Yr': m.get('EBIDT growth 3Years_ann', np.nan),
        'EBIDT_Growth_5Yr': m.get('EBIDT growth 5Years_ann', np.nan),
        'EBIDT_Growth_10Yr': m.get('EBIDT growth 10Years_ann', np.nan),
        'YoY_Quarterly_Sales_Growth': yoy_q_sales,
        'YoY_Quarterly_Profit_Growth': yoy_q_profit,
        'QoQ_Sales': qoq_sales,
        'QoQ_Profits': qoq_profit,
        'Profit_Growth_Consistency': pg_consistent,
        'Growth_Durability_Score': growth_durability,
        'Total_Revenue': sales,
        'EPS': m.get('EPS_ann', np.nan),
        'Book_Value': bv,
        'Book_Value_3Yr': bv_3,
        'Book_Value_5Yr': bv_5,
        'Book_Value_10Yr': bv_10,
        'Dividend_Payout': m.get('Dividend Payout_ur', np.nan),
    })
    return growth


def build_shareholding_sheet(m):
    """Build shareholding sheet."""
    promoter = pd.Series(m.get('Promoter holding_rat', np.nan), dtype=float)
    fii = pd.Series(m.get('FII holding_rat', np.nan), dtype=float)
    dii = pd.Series(m.get('DII holding_rat', np.nan), dtype=float)
    fii_chg = pd.Series(m.get('Change in FII holding_rat', np.nan), dtype=float)
    dii_chg = pd.Series(m.get('Change in DII holding_rat', np.nan), dtype=float)
    fii_chg_3yr = pd.Series(m.get('Change in FII holding 3Years_rat', np.nan), dtype=float)
    dii_chg_3yr = pd.Series(m.get('Change in DII holding 3Years_rat', np.nan), dtype=float)
    promoter_chg_3yr = pd.Series(m.get('Change in promoter holding 3Years_ann', np.nan), dtype=float)

    institutional = fii.fillna(0) + dii.fillna(0)
    public = 100 - promoter.fillna(0) - institutional

    shareholding = pd.DataFrame({
        'NSE_Code': m['NSE Code'],
        'ISIN': m['ISIN Code'],
        'Promoter_Holding': promoter,
        'Promoter_Change_3Yr': promoter_chg_3yr,
        'FII_Holding': fii,
        'FII_Change_Qtr': fii_chg,
        'FII_Change_3Yr': fii_chg_3yr,
        'DII_Holding': dii,
        'DII_Change_Qtr': dii_chg,
        'DII_Change_3Yr': dii_chg_3yr,
        'Institutional_Holding': institutional,
        'Public_Holding': public,
        'Num_Shareholders': m.get('Number of Shareholders preceding quarter_rat', np.nan),
    })
    return shareholding


def build_neglected_firm_sheet(m, quality_df, leverage_df, cashflow_df, growth_df, shareholding_df):
    """Build neglected firm analysis sheet."""
    inst = shareholding_df['Institutional_Holding'].fillna(0)
    mcap = pd.Series(m['Market Capitalization'], dtype=float)
    roe_5 = quality_df['ROE_5Yr_Avg'].fillna(0)
    de = leverage_df['Debt_Equity'].fillna(999)

    # Neglect score: higher = more neglected
    # Low institutional holding contributes most
    inst_score = np.where(inst <= 1, 40,
                 np.where(inst <= 3, 30,
                 np.where(inst <= 5, 20,
                 np.where(inst <= 10, 10, 0))))

    # Small cap bonus
    mcap_score = np.where(pd.isna(mcap), 0,
                 np.where(mcap <= 500, 30,
                 np.where(mcap <= 2000, 20,
                 np.where(mcap <= 5000, 10, 0))))

    # Low shareholder count
    num_sh = pd.Series(m.get('Number of Shareholders preceding quarter_rat', np.nan), dtype=float)
    sh_score = np.where(pd.isna(num_sh), 10,
               np.where(num_sh <= 5000, 20,
               np.where(num_sh <= 20000, 10,
               np.where(num_sh <= 50000, 5, 0))))

    neglect_score = pd.Series(inst_score) + pd.Series(mcap_score) + pd.Series(sh_score)

    # Generic stock candidate: neglected + reasonable quality + low debt
    generic_candidate = np.where(
        (neglect_score >= 50) & (roe_5 >= 10) & (de <= 1.0), 'Yes',
        np.where(neglect_score >= 40, 'Maybe', 'No'))

    # Neglect reasons
    reasons = []
    for i in range(len(m)):
        r = []
        if inst.iloc[i] <= 5:
            r.append(f"Low institutional holding ({inst.iloc[i]:.1f}%)")
        if not pd.isna(mcap.iloc[i]) and mcap.iloc[i] <= 2000:
            r.append(f"Small cap (₹{mcap.iloc[i]:.0f} Cr)")
        if not pd.isna(num_sh.iloc[i]) and num_sh.iloc[i] <= 10000:
            r.append(f"Few shareholders ({int(num_sh.iloc[i])})")
        reasons.append('; '.join(r) if r else '')

    neglected = pd.DataFrame({
        'NSE_Code': m['NSE Code'],
        'ISIN': m['ISIN Code'],
        'Generic_Stock_Candidate': generic_candidate,
        'Neglect_Score': neglect_score,
        'Neglect_Reasons': reasons,
        'Institutional_Holding': inst,
        'ROE_5Yr_Avg': roe_5,
        'Debt_Equity': de,
        'Market_Cap': mcap,
        'Num_Shareholders': num_sh,
    })
    return neglected


def build_dividends_sheet(m):
    """Build dividends sheet from available data."""
    dividends = pd.DataFrame({
        'NSE_Code': m['NSE Code'],
        'ISIN': m['ISIN Code'],
        'Dividend_Last_Year': m.get('Dividend last year_ann', np.nan),
        'Dividend_Preceding_Year': m.get('Dividend preceding year_ann', np.nan),
        'Dividend_Payout': m.get('Dividend Payout_ur', np.nan),
    })
    return dividends


def build_red_flags_sheet(m, quality_df, cashflow_df, leverage_df, valuation_df):
    """Build red flags sheet with individual flag columns."""
    n = len(m)

    roe = quality_df['ROE_Latest'].values.astype(float)
    roe_3 = quality_df['ROE_3Yr_Avg'].values.astype(float)
    roce = quality_df['ROCE_Latest'].values.astype(float)
    roce_3 = quality_df['ROCE_3Yr_Avg'].values.astype(float)
    opm = quality_df['OPM_Latest'].values.astype(float)
    opm_py = quality_df['OPM_Preceding_Year'].values.astype(float)
    other_inc_pct = quality_df['Other_Income_Pct_PAT'].values.astype(float)

    cfo = cashflow_df['CFO_Latest'].values.astype(float)
    cfo_pat = cashflow_df['CFO_PAT_Latest'].values.astype(float)
    cfo_pat_3yr = cashflow_df['CFO_PAT_3Yr_Avg'].values.astype(float)
    worst_yr_cfo_pat = cashflow_df['Worst_Year_CFO_PAT'].values.astype(float)
    cfo_3yr = cashflow_df['CFO_3Yr_Cumulative'].values.astype(float)
    cfo_yr3 = cashflow_df['CFO_Year3_Inferred'].values.astype(float)
    pos_cfo_years = cashflow_df['Positive_CFO_Years_3Yr'].values.astype(float)
    wc_rev_ratio = cashflow_df['WC_Rev_Growth_Ratio'].values.astype(float)

    de = leverage_df['Debt_Equity'].values.astype(float)
    debt = leverage_df['Total_Debt'].values.astype(float)
    debt_3yr = leverage_df['Debt_3Yr_Back'].values.astype(float)

    pe = valuation_df['PE'].values.astype(float)
    pbv = valuation_df['PBV'].values.astype(float)
    ev_ebitda = valuation_df['EV_EBITDA'].values.astype(float)

    # Calculate each flag
    flag_low_roe = (~np.isnan(roe)) & (roe < 10)
    flag_declining_roe = (~np.isnan(roe)) & (~np.isnan(roe_3)) & (roe < roe_3 - 2) & (roe < 15)
    flag_low_roce = (~np.isnan(roce)) & (roce < 10)
    flag_declining_roce = (~np.isnan(roce)) & (~np.isnan(roce_3)) & (roce < roce_3 - 2) & (roce < 15)

    # POOR_CASH_CONVERSION: Primary check matches original script: CFO_PAT_3Yr < 0.7
    # Secondary: worst individual year < 0.2 BUT only when 3yr average is also borderline
    # (< 0.75) — a single bad year in an otherwise healthy average is less concerning.
    flag_poor_cash = (
        # 3yr average-of-ratios CFO/PAT < 0.7 (primary check, matches original)
        ((~np.isnan(cfo_pat_3yr)) & (cfo_pat_3yr < 0.7) & (cfo_pat_3yr >= 0))
        |
        # Or single-year < 0.7 when 3yr not available
        ((np.isnan(cfo_pat_3yr)) & (~np.isnan(cfo_pat)) & (cfo_pat < 0.7) & (cfo_pat >= 0))
        |
        # Worst year terrible AND 3yr average also borderline (not healthy)
        ((~np.isnan(worst_yr_cfo_pat)) & (worst_yr_cfo_pat < 0.2) & (worst_yr_cfo_pat >= 0)
         & (~np.isnan(cfo_pat_3yr)) & (cfo_pat_3yr < 0.75))
    )

    flag_neg_cfo = (~np.isnan(cfo)) & (cfo < 0)

    # INCONSISTENT_CFO: Flag if:
    # 1. Cumulative 3yr CFO is negative, OR
    # 2. Inferred Year 3 CFO is negative, OR
    # 3. Fewer than 2 of 3 reconstructed years have positive CFO, OR
    # 4. 5yr per-year avg CFO is less than 50% of 3yr per-year avg
    #    (indicates much weaker/negative older years we can't see individually)
    cfo_5yr_series = cashflow_df['CFO_5Yr_Cumulative'].values.astype(float)
    avg_cfo_3yr = np.where(~np.isnan(cfo_3yr), cfo_3yr / 3.0, np.nan)
    avg_cfo_5yr = np.where(~np.isnan(cfo_5yr_series), cfo_5yr_series / 5.0, np.nan)
    cfo_5yr_vs_3yr_ratio = safe_div(avg_cfo_5yr, avg_cfo_3yr)

    pos_cfo_count = cashflow_df['Positive_CFO_Years_3Yr'].values.astype(float)

    flag_inconsistent_cfo = (
        ((~np.isnan(cfo_3yr)) & (cfo_3yr < 0))
        |
        ((~np.isnan(cfo_yr3)) & (cfo_yr3 < 0))
        |
        ((~np.isnan(pos_cfo_years)) & (pos_cfo_years < 2))
        |
        # 5yr per-year avg < 50% of 3yr per-year avg → weak older years
        # BUT only when recent years aren't all positive (turnarounds are real)
        ((~np.isnan(cfo_5yr_vs_3yr_ratio)) & np.isfinite(cfo_5yr_vs_3yr_ratio)
         & (avg_cfo_3yr > 0) & (cfo_5yr_vs_3yr_ratio < 0.5)
         & (pos_cfo_count < 3))
    )
    flag_high_other = (~np.isnan(other_inc_pct)) & (other_inc_pct > 30)
    flag_margin_comp = (~np.isnan(opm)) & (~np.isnan(opm_py)) & (opm < opm_py - 2)
    flag_rising_debt = (~np.isnan(debt)) & (~np.isnan(debt_3yr)) & (debt_3yr > 0) & (debt > debt_3yr * 1.5)
    # Also flag if debt was 0 and is now > 0 significantly
    flag_rising_debt = flag_rising_debt | ((~np.isnan(debt)) & (debt_3yr == 0) & (debt > 50))
    flag_wc_div = (~np.isnan(wc_rev_ratio)) & (wc_rev_ratio > 1.5) & np.isfinite(wc_rev_ratio)

    flag_high_pe = (~np.isnan(pe)) & (pe > 50)
    flag_neg_pe = (~np.isnan(pe)) & (pe < 0)
    flag_high_ev = (~np.isnan(ev_ebitda)) & (ev_ebitda > 25)
    flag_neg_ebitda = (~np.isnan(ev_ebitda)) & (ev_ebitda < 0)
    flag_high_pbv_roe = (~np.isnan(pbv)) & (~np.isnan(roe)) & (roe > 0) & (pbv > roe / 2)

    # Aggregate
    structural_flags = {
        'FLAG_LOW_ROE': flag_low_roe, 'FLAG_DECLINING_ROE': flag_declining_roe,
        'FLAG_LOW_ROCE': flag_low_roce, 'FLAG_DECLINING_ROCE': flag_declining_roce,
        'FLAG_POOR_CASH_CONVERSION': flag_poor_cash, 'FLAG_NEGATIVE_CFO': flag_neg_cfo,
        'FLAG_INCONSISTENT_CFO': flag_inconsistent_cfo,
        'FLAG_HIGH_OTHER_INCOME': flag_high_other,
        'FLAG_MARGIN_COMPRESSION': flag_margin_comp,
        'FLAG_RISING_DEBT': flag_rising_debt, 'FLAG_WC_DIVERGENCE': flag_wc_div,
    }
    pricing_flags = {
        'FLAG_HIGH_PE': flag_high_pe, 'FLAG_NEGATIVE_PE': flag_neg_pe,
        'FLAG_HIGH_EV_EBITDA': flag_high_ev, 'FLAG_NEGATIVE_EBITDA': flag_neg_ebitda,
        'FLAG_HIGH_PBV_ROE': flag_high_pbv_roe,
    }

    # ── Sector-based severity adjustments (matching original script) ─────
    # Capital-intensive businesses (Asset_Turnover < 0.8) get reduced severity
    # on CFO-related flags because lumpy project-based cash flows are normal.
    asset_turnover = quality_df['Asset_Turnover'].values.astype(float)
    is_capital_intensive = (~np.isnan(asset_turnover)) & (asset_turnover < 0.8)

    # Per-flag weight overrides: default to original weight, reduce for capital-intensive
    # POOR_CASH_CONVERSION: CRITICAL(2.0) → MINOR(0.5) if capital-intensive
    # NEGATIVE_CFO: CRITICAL(2.0) → MAJOR(1.0) if very capital-intensive (AT < 0.5)
    # INCONSISTENT_CFO: CRITICAL(2.0) → MAJOR(1.0) if capital-intensive
    weight_overrides = {}
    weight_overrides['FLAG_POOR_CASH_CONVERSION'] = np.where(
        is_capital_intensive & flag_poor_cash, 0.5, 2.0)
    weight_overrides['FLAG_NEGATIVE_CFO'] = np.where(
        (~np.isnan(asset_turnover)) & (asset_turnover < 0.5) & flag_neg_cfo, 1.0, 2.0)
    weight_overrides['FLAG_INCONSISTENT_CFO'] = np.where(
        is_capital_intensive & flag_inconsistent_cfo, 1.0, 2.0)

    # Track sector adjustments for reporting
    sector_adj_made = []
    for i in range(n):
        adjs = []
        if is_capital_intensive[i] and flag_poor_cash[i]:
            adjs.append("POOR_CASH_CONVERSION: CRITICAL→MINOR (Capital-intensive)")
        if (~np.isnan(asset_turnover[i])) and asset_turnover[i] < 0.5 and flag_neg_cfo[i]:
            adjs.append("NEGATIVE_CFO: CRITICAL→MAJOR (Very capital-intensive)")
        if is_capital_intensive[i] and flag_inconsistent_cfo[i]:
            adjs.append("INCONSISTENT_CFO: CRITICAL→MAJOR (Capital-intensive)")
        sector_adj_made.append(' | '.join(adjs) if adjs else '')

    # Count flags by severity and compute Quality_Severity (structural only)
    # and Pricing_Severity (pricing only) — matching original script separation
    # Weights: CRITICAL=2.0, MAJOR=1.0, MINOR=0.5
    critical_count = np.zeros(n)
    major_count = np.zeros(n)
    minor_count = np.zeros(n)
    total_count = np.zeros(n)
    quality_severity = np.zeros(n, dtype=float)   # structural flags only
    pricing_severity = np.zeros(n, dtype=float)    # pricing flags only

    all_flags = {**structural_flags, **pricing_flags}
    flag_name_to_def = {}
    for fname, farr in all_flags.items():
        key = fname.replace('FLAG_', '')
        if key in RED_FLAG_DEFINITIONS:
            flag_name_to_def[fname] = RED_FLAG_DEFINITIONS[key]

    # Structural flags → quality_severity (with sector-based weight overrides)
    for fname, farr in structural_flags.items():
        total_count += farr.astype(int)
        defn = flag_name_to_def.get(fname, {})
        if fname in weight_overrides:
            # Per-stock weight array (sector-adjusted for CFO flags)
            weight = weight_overrides[fname]
            is_flagged = farr.astype(bool)
            critical_count += (is_flagged & (weight >= 2.0)).astype(int)
            major_count += (is_flagged & (weight >= 1.0) & (weight < 2.0)).astype(int)
            minor_count += (is_flagged & (weight < 1.0)).astype(int)
        else:
            weight = defn.get('weight', 0.5)
            sev = defn.get('severity', 'MINOR')
            if sev == 'CRITICAL':
                critical_count += farr.astype(int)
            elif sev == 'MAJOR':
                major_count += farr.astype(int)
            else:
                minor_count += farr.astype(int)
        quality_severity += farr.astype(float) * weight

    # Pricing flags → pricing_severity
    for fname, farr in pricing_flags.items():
        total_count += farr.astype(int)
        defn = flag_name_to_def.get(fname, {})
        sev = defn.get('severity', 'MINOR')
        weight = defn.get('weight', 0.5)
        pricing_severity += farr.astype(float) * weight
        if sev == 'CRITICAL':
            critical_count += farr.astype(int)
        elif sev == 'MAJOR':
            major_count += farr.astype(int)
        else:
            minor_count += farr.astype(int)

    # Quality risk level
    quality_risk = np.where(critical_count >= 2, 'CRITICAL',
                  np.where(critical_count >= 1, 'HIGH',
                  np.where(major_count >= 2, 'ELEVATED',
                  np.where(major_count >= 1, 'MODERATE', 'LOW'))))

    # Build quality and pricing flag name lists
    quality_flag_names = []
    pricing_flag_names = []
    all_flag_names = []
    for i in range(n):
        qf = [k.replace('FLAG_', '') for k, v in structural_flags.items() if v[i]]
        pf = [k.replace('FLAG_', '') for k, v in pricing_flags.items() if v[i]]
        quality_flag_names.append(', '.join(qf) if qf else '')
        pricing_flag_names.append(', '.join(pf) if pf else '')
        all_flag_names.append(', '.join(qf + pf) if (qf or pf) else '')

    # Red flags explained
    explained = []
    for i in range(n):
        parts = []
        for fname, farr in all_flags.items():
            if farr[i]:
                defn = flag_name_to_def.get(fname, {})
                parts.append(f"{fname.replace('FLAG_', '')}: {defn.get('meaning', '')}")
        explained.append(' | '.join(parts) if parts else '')

    rf_data = {
        'NSE_Code': m['NSE Code'].values,
        'ISIN': m['ISIN Code'].values,
        'Quality_Risk': quality_risk,
        'Quality_Severity': np.round(quality_severity, 1),
        'Pricing_Severity': np.round(pricing_severity, 1),
        'Total_Severity': np.round(quality_severity + pricing_severity, 1),
        'Critical_Flags': critical_count.astype(int),
        'Major_Flags': major_count.astype(int),
        'Minor_Flags': minor_count.astype(int),
        'Red_Flag_Count': total_count.astype(int),
        'Quality_Flags': quality_flag_names,
        'Pricing_Flags': pricing_flag_names,
        'Red_Flags': all_flag_names,
        'Red_Flags_Explained': explained,
        'Sector_Adjustments': sector_adj_made,
    }

    # Add individual flag columns
    for fname, farr in all_flags.items():
        rf_data[fname] = farr.astype(int)

    return pd.DataFrame(rf_data)


def build_analysis_sheet(m, quality_df, valuation_df, leverage_df, growth_df,
                         cashflow_df, red_flags_df, shareholding_df):
    """Build the main Analysis sheet with composite scores and decision buckets."""
    bq = quality_df['Business_Quality_Score'].values.astype(float)
    vc = valuation_df['Valuation_Comfort_Score'].values.astype(float)
    fs = leverage_df['Financial_Strength_Score'].values.astype(float)
    gd = growth_df['Growth_Durability_Score'].values.astype(float)

    # Cash flow quality score (0-100)
    cfo_pat = pd.Series(cashflow_df['CFO_PAT_Latest'].values, dtype=float)
    cfo_latest = pd.Series(cashflow_df['CFO_Latest'].values, dtype=float)
    cfo_3yr = pd.Series(cashflow_df['CFO_3Yr_Cumulative'].values, dtype=float)

    cf_score = np.where(pd.isna(cfo_pat), 20,
               np.where(cfo_pat >= 1.2, 100,
               np.where(cfo_pat >= 1.0, 85,
               np.where(cfo_pat >= 0.8, 70,
               np.where(cfo_pat >= 0.5, 50,
               np.where(cfo_pat >= 0, 30, 10))))))
    # Adjust for negative CFO
    cf_score = np.where(cfo_latest < 0, np.minimum(cf_score, 15), cf_score)
    # Adjust for inconsistent 3yr CFO
    cf_score = np.where((~pd.isna(cfo_3yr)) & (cfo_3yr < 0), np.minimum(cf_score, 10), cf_score)

    # Composite Score: Quality 35%, Growth 25%, Valuation 20%, Leverage 10%, Cash Flow 10%
    composite = (0.35 * bq + 0.25 * gd + 0.20 * vc + 0.10 * fs + 0.10 * cf_score)

    # Score band (matches original: A/B/C/D/F)
    score_band = np.where(composite >= 80, 'A',
                 np.where(composite >= 65, 'B',
                 np.where(composite >= 50, 'C',
                 np.where(composite >= 30, 'D', 'F'))))

    # Extract red flag data
    critical = red_flags_df['Critical_Flags'].values.astype(int)
    major = red_flags_df['Major_Flags'].values.astype(int)
    quality_severity = red_flags_df['Quality_Severity'].values.astype(float)
    quality_risk = red_flags_df['Quality_Risk'].values
    quality_flags_str = red_flags_df['Quality_Flags'].values
    pricing_flags_str = red_flags_df['Pricing_Flags'].values

    pe = valuation_df['PE'].values.astype(float)
    mcap = pd.Series(m['Market Capitalization'].values, dtype=float)
    piotroski = quality_df['Piotroski_Score'].values.astype(float)

    # Promoter holding data for conviction overrides
    promoter_chg = shareholding_df['Promoter_Change_3Yr'].values.astype(float)
    inst = shareholding_df['Institutional_Holding'].values.astype(float)

    n = len(m)

    # ── SCREEN_ELIGIBLE (hard mechanical gate, YES/NO) ──────────────────
    screen_eligible = np.where(quality_severity >= 2.0, 'NO', 'YES')

    # ── Decision Bucket (severity-based, neutral language) ──────────────
    # Matches original script logic exactly:
    #   GATES_CLEARED           → Score A/B, no flags, clean slate
    #   SCREEN_PASSED_EXPENSIVE → Good quality but expensive valuation
    #   SCREEN_PASSED_FLAGS     → Passed with quality concerns (severity 0.5-1.9)
    #   SCREEN_MARGINAL         → Score Band C (borderline)
    #   SCREEN_FAILED           → Severity >= 2.0 or low score (D/F)
    decision = []
    reject_reason = []
    for i in range(n):
        sev = quality_severity[i]
        band = score_band[i]
        has_pricing = bool(pricing_flags_str[i])
        crit_list = quality_flags_str[i]

        if sev >= 2.0:
            decision.append('SCREEN_FAILED')
            if critical[i] >= 1:
                reject_reason.append(f"Critical flag: {crit_list}")
            else:
                reject_reason.append(f"Multiple major flags (severity {sev:.1f})")
        elif sev >= 1.0:
            decision.append('SCREEN_PASSED_FLAGS')
            reject_reason.append(f"Quality concerns (severity {sev:.1f})")
        elif sev >= 0.5:
            decision.append('SCREEN_PASSED_FLAGS')
            minor_list = quality_flags_str[i] if quality_flags_str[i] else 'N/A'
            reject_reason.append(f"Minor flag: {minor_list}")
        elif has_pricing and band in ('A', 'B'):
            decision.append('SCREEN_PASSED_EXPENSIVE')
            reject_reason.append(f"Valuation concern: {pricing_flags_str[i]}")
        elif band in ('A', 'B'):
            decision.append('GATES_CLEARED')
            reject_reason.append('None')
        elif band == 'C':
            decision.append('SCREEN_MARGINAL')
            reject_reason.append('Marginal score')
        else:
            decision.append('SCREEN_FAILED')
            reject_reason.append('Low score')

    decision = np.array(decision)
    reject_reason = np.array(reject_reason)

    # ── Conviction Overrides (CONTRARIAN_BET / VALUE_TRAP) ──────────────
    conviction_override = np.full(n, 'None', dtype=object)

    for i in range(n):
        pchg = promoter_chg[i]
        if pd.isna(pchg):
            continue
        # CONTRARIAN_BET: Failed/marginal + promoter buying > 3%
        if (decision[i] in ('SCREEN_FAILED', 'SCREEN_MARGINAL')
                and pchg >= 3.0
                and quality_severity[i] < 4.0):
            decision[i] = 'CONTRARIAN_BET'
            conviction_override[i] = f"Promoter buying +{pchg:.1f}% despite low score"
            reject_reason[i] = (
                f"Score={composite[i]:.0f} (was {reject_reason[i]}) BUT "
                f"promoter increased stake by {pchg:.1f}%—special situation"
            )
        # VALUE_TRAP: Good score + promoter selling > 3%
        elif (decision[i] in ('GATES_CLEARED', 'SCREEN_PASSED_EXPENSIVE')
                and pchg <= -3.0):
            decision[i] = 'VALUE_TRAP'
            conviction_override[i] = f"Promoter selling {pchg:.1f}% despite high score"
            reject_reason[i] = (
                f"Score={composite[i]:.0f} BUT "
                f"promoter reduced stake by {abs(pchg):.1f}%—insider exit signal"
            )

    # ── Primary concern ─────────────────────────────────────────────────
    primary_concern = []
    for i in range(n):
        flags = red_flags_df['Red_Flags'].iloc[i]
        if critical[i] >= 2:
            primary_concern.append(f"Multiple critical flags: {flags}")
        elif critical[i] >= 1:
            primary_concern.append(f"Critical flag: {flags}")
        elif major[i] >= 2:
            primary_concern.append(f"Multiple quality issues: {flags}")
        elif composite[i] < 30:
            primary_concern.append("Low composite score")
        elif flags:
            primary_concern.append(flags)
        else:
            primary_concern.append('')

    # ── Investment thesis ────────────────────────────────────────────────
    thesis = np.where(decision == 'GATES_CLEARED', 'Passed quality gates—clean slate',
             np.where(decision == 'SCREEN_PASSED_EXPENSIVE', 'Good quality but expensive—wait for better price',
             np.where(decision == 'SCREEN_PASSED_FLAGS', 'Passed with concerns—needs deeper analysis',
             np.where(decision == 'SCREEN_MARGINAL', 'Borderline metrics—monitor for improvement',
             np.where(decision == 'CONTRARIAN_BET', 'Promoter buying despite low score—special situation',
             np.where(decision == 'VALUE_TRAP', 'Good score but insiders exiting—verify before investing',
             ''))))))

    generic = np.where(inst <= 5, 'Potential', 'No')

    analysis = pd.DataFrame({
        'NSE_Code': m['NSE Code'].values,
        'ISIN': m['ISIN Code'].values,
        'Decision_Bucket': decision,
        'MCAP': mcap,
        'Conviction_Override': conviction_override,
        'SCREEN_ELIGIBLE': screen_eligible,
        'Investment_Thesis': thesis,
        'Reject_Reason': reject_reason,
        'Composite_Score': np.round(composite, 1),
        'Sector_Relative_Adj': 0.0,
        'Score_Band': score_band,
        'Quality_Risk': quality_risk,
        'Quality_Severity': np.round(quality_severity, 1),
        'Critical_Flags': critical,
        'Major_Flags': major,
        'Primary_Concern': primary_concern,
        'Market_Cap': mcap,
        'PE': pe,
        'Piotroski_Score': piotroski,
        'Generic_Stock_Candidate': generic,
        'Business_Quality_Score': np.round(bq, 1),
        'Growth_Durability_Score': np.round(gd, 1),
        'Valuation_Comfort_Score': np.round(vc, 1),
        'Financial_Strength_Score': np.round(fs, 1),
        'Cash_Flow_Score': np.round(cf_score.astype(float), 1),
    })
    return analysis


def build_summary_sheet(analysis_df, master_df):
    """Build summary statistics sheet."""
    total = len(analysis_df)
    decision_counts = analysis_df['Decision_Bucket'].value_counts()

    rows = [
        ['STOCK SCREENING DECISION SUMMARY', '', '', '', '', ''],
        [f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
         f'Total Stocks: {total}', '', '', '', ''],
        ['', '', '', '', '', ''],
        ['Decision Bucket', 'Count', 'Percentage', '', '', ''],
    ]

    for bucket in ['GATES_CLEARED', 'SCREEN_PASSED_EXPENSIVE', 'SCREEN_PASSED_FLAGS',
                    'SCREEN_MARGINAL', 'SCREEN_FAILED', 'CONTRARIAN_BET', 'VALUE_TRAP']:
        cnt = decision_counts.get(bucket, 0)
        pct = f"{cnt/total*100:.1f}%"
        rows.append([bucket, cnt, pct, '', '', ''])

    rows.append(['', '', '', '', '', ''])
    rows.append(['SCORE DISTRIBUTION', '', '', '', '', ''])
    rows.append(['Score Band', 'Count', 'Percentage', '', '', ''])

    band_counts = analysis_df['Score_Band'].value_counts()
    for band in ['A', 'B', 'C', 'D', 'F']:
        cnt = band_counts.get(band, 0)
        pct = f"{cnt/total*100:.1f}%"
        rows.append([band, cnt, pct, '', '', ''])

    rows.append(['', '', '', '', '', ''])
    rows.append(['QUALITY RISK DISTRIBUTION', '', '', '', '', ''])
    rows.append(['Risk Level', 'Count', 'Percentage', '', '', ''])

    risk_counts = analysis_df['Quality_Risk'].value_counts()
    for risk in ['LOW', 'MODERATE', 'ELEVATED', 'HIGH', 'CRITICAL']:
        cnt = risk_counts.get(risk, 0)
        pct = f"{cnt/total*100:.1f}%"
        rows.append([risk, cnt, pct, '', '', ''])

    rows.append(['', '', '', '', '', ''])
    rows.append(['COMPOSITE SCORE STATISTICS', '', '', '', '', ''])
    cs = analysis_df['Composite_Score'].astype(float)
    rows.append(['Mean', f'{cs.mean():.1f}', '', '', '', ''])
    rows.append(['Median', f'{cs.median():.1f}', '', '', '', ''])
    rows.append(['Std Dev', f'{cs.std():.1f}', '', '', '', ''])
    rows.append(['Min', f'{cs.min():.1f}', '', '', '', ''])
    rows.append(['Max', f'{cs.max():.1f}', '', '', '', ''])

    rows.append(['', '', '', '', '', ''])
    rows.append(['TOP SECTORS BY GATES_CLEARED COUNT', '', '', '', '', ''])

    invest_stocks = analysis_df[analysis_df['Decision_Bucket'] == 'GATES_CLEARED']['NSE_Code']
    if len(invest_stocks) > 0:
        invest_master = master_df[master_df['NSE_Code'].isin(invest_stocks)]
        sector_counts = invest_master['Sector'].value_counts().head(10)
        for sector, cnt in sector_counts.items():
            rows.append([sector, cnt, '', '', '', ''])

    rows.append(['', '', '', '', '', ''])
    rows.append(['RED FLAG SUMMARY', '', '', '', '', ''])
    rows.append(['Flag', 'Count', 'Percentage', '', '', ''])

    flag_cols = [c for c in analysis_df.columns if c.startswith('FLAG_')]
    # Use red_flags sheet data instead if flag_cols not in analysis
    # We'll handle this in the main function

    summary_df = pd.DataFrame(rows, columns=['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6'])
    return summary_df


def format_excel(writer, sheets):
    """Apply formatting to the Excel workbook."""
    workbook = writer.book

    # Define formats
    header_fmt = workbook.add_format({
        'bold': True, 'bg_color': '#4472C4', 'font_color': 'white',
        'border': 1, 'text_wrap': True, 'valign': 'vcenter'
    })
    pct_fmt = workbook.add_format({'num_format': '0.0%'})
    num_fmt = workbook.add_format({'num_format': '#,##0.0'})
    int_fmt = workbook.add_format({'num_format': '#,##0'})

    # Color formats for decision buckets (matching original script palette)
    bucket_fmts = {
        'GATES_CLEARED': workbook.add_format({'bg_color': '#92D050', 'font_color': '#000000'}),
        'SCREEN_PASSED_EXPENSIVE': workbook.add_format({'bg_color': '#FFC000', 'font_color': '#000000'}),
        'SCREEN_PASSED_FLAGS': workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700'}),
        'SCREEN_PASSED_VERIFY': workbook.add_format({'bg_color': '#BDD7EE', 'font_color': '#000000'}),
        'SCREEN_MARGINAL': workbook.add_format({'bg_color': '#D9D9D9', 'font_color': '#000000'}),
        'DATA_INCOMPLETE': workbook.add_format({'bg_color': '#EDEDED', 'font_color': '#808080'}),
        'SCREEN_FAILED': workbook.add_format({'bg_color': '#FF6B6B', 'font_color': '#000000'}),
        'CONTRARIAN_BET': workbook.add_format({'bg_color': '#7030A0', 'font_color': '#FFFFFF'}),
        'VALUE_TRAP': workbook.add_format({'bg_color': '#C00000', 'font_color': '#FFFFFF'}),
    }

    for sheet_name, df in sheets.items():
        ws = writer.sheets[sheet_name]

        # Format headers
        for col_idx, col in enumerate(df.columns):
            ws.write(0, col_idx, col, header_fmt)

        # Auto-fit column widths (approximate)
        for col_idx, col in enumerate(df.columns):
            max_len = max(len(str(col)), df[col].astype(str).str.len().max())
            ws.set_column(col_idx, col_idx, min(max_len + 2, 30))

        # Freeze first row and first column
        ws.freeze_panes(1, 1)

        # Add autofilter
        if sheet_name != 'Summary':
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)

        # Conditional formatting for Analysis sheet
        if sheet_name == 'Analysis' and 'Decision_Bucket' in df.columns:
            bucket_col = df.columns.get_loc('Decision_Bucket')
            for row_idx in range(len(df)):
                val = df.iloc[row_idx, bucket_col]
                fmt = bucket_fmts.get(val)
                if fmt:
                    ws.write(row_idx + 1, bucket_col, val, fmt)


def main():
    parser = argparse.ArgumentParser(description='Generate stock analysis from parquet files')
    parser.add_argument('--output', default=os.path.join(BASE_DIR, 'stock_analysis.xlsx'),
                        help='Output Excel file path')
    args = parser.parse_args()

    print("=" * 70)
    print("STOCK ANALYSIS FROM PARQUET FILES")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Load parquet files
    print("\n[1/10] Loading parquet files...")
    dfs = load_parquet_files()
    if len(dfs) < 6:
        print(f"ERROR: Expected 6 parquet files, found {len(dfs)}. Aborting.")
        sys.exit(1)

    # 2. Merge all data
    print("\n[2/10] Merging datasets...")
    merged = merge_all(dfs)

    # 3. Build individual sheets
    print("\n[3/10] Building Master sheet...")
    master_df = build_master_sheet(merged)
    print(f"  {len(master_df)} stocks")

    print("[4/10] Building Valuation sheet...")
    valuation_df = build_valuation_sheet(merged)

    print("[5/10] Building Quality sheet...")
    quality_df = build_quality_sheet(merged)

    print("[6/10] Building Cash Flow sheet...")
    cashflow_df = build_cashflow_sheet(merged)

    print("[7/10] Building Leverage sheet...")
    leverage_df = build_leverage_sheet(merged)

    print("[8/10] Building Growth sheet...")
    growth_df = build_growth_sheet(merged)

    print("[9/10] Building Shareholding, Neglected Firm, Dividends, Red Flags...")
    shareholding_df = build_shareholding_sheet(merged)
    neglected_df = build_neglected_firm_sheet(merged, quality_df, leverage_df,
                                              cashflow_df, growth_df, shareholding_df)
    dividends_df = build_dividends_sheet(merged)
    red_flags_df = build_red_flags_sheet(merged, quality_df, cashflow_df, leverage_df, valuation_df)

    print("[10/10] Building Analysis sheet and Summary...")
    analysis_df = build_analysis_sheet(merged, quality_df, valuation_df, leverage_df,
                                       growth_df, cashflow_df, red_flags_df, shareholding_df)
    summary_df = build_summary_sheet(analysis_df, master_df)

    # Create sort order by composite score (descending)
    sort_idx = analysis_df['Composite_Score'].astype(float).values.argsort()[::-1]

    # Apply sort to all sheets (they all share the same row indexing from merged)
    analysis_df = analysis_df.iloc[sort_idx].reset_index(drop=True)

    sheet_dfs = {
        'Master': master_df, 'Valuation': valuation_df, 'Quality': quality_df,
        'Cash_Flow': cashflow_df, 'Leverage': leverage_df, 'Growth': growth_df,
        'Shareholding': shareholding_df, 'Neglected_Firm': neglected_df,
        'Dividends': dividends_df, 'Red_Flags': red_flags_df,
    }
    for name, df in sheet_dfs.items():
        sheet_dfs[name] = df.iloc[sort_idx].reset_index(drop=True)

    # Write to Excel
    print(f"\nWriting to {args.output}...")
    all_sheets = {'Analysis': analysis_df, **sheet_dfs, 'Summary': summary_df}

    with pd.ExcelWriter(args.output, engine='xlsxwriter') as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Apply formatting
        format_excel(writer, all_sheets)

    file_size = os.path.getsize(args.output)
    print(f"\nDone! Output: {args.output} ({file_size / 1024 / 1024:.1f} MB)")

    # Print summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total stocks analyzed: {len(analysis_df)}")
    dc = analysis_df['Decision_Bucket'].value_counts()
    for bucket in ['GATES_CLEARED', 'SCREEN_PASSED_EXPENSIVE', 'SCREEN_PASSED_FLAGS',
                    'SCREEN_MARGINAL', 'SCREEN_FAILED', 'CONTRARIAN_BET', 'VALUE_TRAP']:
        cnt = dc.get(bucket, 0)
        print(f"  {bucket:25s}: {cnt:5d} ({cnt/len(analysis_df)*100:.1f}%)")

    cs = analysis_df['Composite_Score'].astype(float)
    print(f"\nComposite Score: mean={cs.mean():.1f}, median={cs.median():.1f}, "
          f"std={cs.std():.1f}, range=[{cs.min():.1f}, {cs.max():.1f}]")

    # Top 20
    print(f"\nTop 20 stocks by Composite Score:")
    top20 = analysis_df.head(20)[['NSE_Code', 'Composite_Score', 'Decision_Bucket',
                                   'Business_Quality_Score', 'Growth_Durability_Score',
                                   'Valuation_Comfort_Score']].to_string(index=False)
    print(top20)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
