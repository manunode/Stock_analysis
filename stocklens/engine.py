"""
Sheet-building engine: every build_*_sheet function and add_peer_ranking.
Depends on: rules (CONFIG, COLS, SCORING_BINS, ...), primitives, data (col, get_col).
"""

import logging
import numpy as np
import pandas as pd

from .rules import (
    COLS, CONFIG, SCORING_BINS, FINANCIAL_SECTORS_LOWER,
    STRUCTURAL_RED_FLAGS, PRICING_RED_FLAGS, RED_FLAG_DEFINITIONS,
    IDENTIFIER_COLS,
)
from .primitives import safe_div, safe_str_lower, vectorized_score, vectorized_string_build
from .data import col, get_col


def _id_cols(source) -> dict:
    """Build the standard identifier columns dict from a DataFrame or values source."""
    if isinstance(source, pd.DataFrame):
        return {
            'ISIN': source[COLS['ISIN_CODE']],
            'NSE_Code': source[COLS['NSE_CODE']],
            'BSE_Code': source[COLS['BSE_CODE']],
        }
    # Assume it's an array-like with .values already extracted
    return {
        'ISIN': source['ISIN'].values if hasattr(source, 'values') else source['ISIN'],
        'NSE_Code': source['NSE_Code'].values if hasattr(source, 'values') else source['NSE_Code'],
        'BSE_Code': source['BSE_Code'].values if hasattr(source, 'values') else source['BSE_Code'],
    }


# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL SHEET BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_master_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build the Master sheet with stock identification info."""
    is_fin = safe_str_lower(m[COLS['INDUSTRY_GROUP']]).isin(FINANCIAL_SECTORS_LOWER) | \
             safe_str_lower(m[COLS['INDUSTRY']]).isin(FINANCIAL_SECTORS_LOWER)
    return pd.DataFrame({
        **_id_cols(m),
        'Stock_Name': m[COLS['NAME']], 'Sector': m[COLS['INDUSTRY_GROUP']], 'Industry': m[COLS['INDUSTRY']],
        'Market_Cap': m[COLS['MARKET_CAP']], 'Current_Price': m[COLS['CURRENT_PRICE']],
        'Is_Financial_Sector': is_fin.map({True: 'Yes', False: 'No'}),
        'Face_Value': get_col(m, 'FACE_VALUE', 'BALANCE'),
    })


def build_valuation_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build valuation metrics sheet with industry-relative valuation."""
    pe, pbv = get_col(m, 'PE', 'RATIOS'), get_col(m, 'PBV', 'RATIOS')
    val_comfort = (vectorized_score(pd.Series(pe), SCORING_BINS['PE']) +
                   vectorized_score(pd.Series(pbv), SCORING_BINS['PBV']) +
                   vectorized_score(pd.Series(get_col(m, 'EVEBITDA', 'USER_RATIOS')), SCORING_BINS['EV_EBITDA']) +
                   vectorized_score(pd.Series(get_col(m, 'PEG_RATIO', 'USER_RATIOS')), SCORING_BINS['PEG']))
    val_band = np.where(pd.isna(val_comfort), 'N/A',
               np.where(val_comfort >= 70, 'CHEAP', np.where(val_comfort >= 50, 'FAIR',
               np.where(val_comfort >= 30, 'EXPENSIVE', 'VERY_EXPENSIVE'))))

    # Industry-relative valuation
    industry_pe = pd.Series(get_col(m, 'INDUSTRY_PE', 'RATIOS'), dtype=float)
    industry_pbv = pd.Series(get_col(m, 'INDUSTRY_PBV', 'RATIOS'), dtype=float)
    pe_s = pd.Series(pe, dtype=float)
    pbv_s = pd.Series(pbv, dtype=float)

    pe_vs_industry = pd.Series(np.where(
        (pe_s > 0) & (industry_pe > 0), safe_div(pe_s, industry_pe), np.nan), dtype=float)
    pbv_vs_industry = pd.Series(np.where(
        (pbv_s > 0) & (industry_pbv > 0), safe_div(pbv_s, industry_pbv), np.nan), dtype=float)

    avg_vs_industry = pd.Series(np.where(
        pd.notna(pe_vs_industry) & pd.notna(pbv_vs_industry),
        (pe_vs_industry + pbv_vs_industry) / 2.0,
        np.where(pd.notna(pe_vs_industry), pe_vs_industry, pbv_vs_industry)), dtype=float)

    industry_val_label = np.where(pd.isna(avg_vs_industry), 'N/A',
                         np.where(avg_vs_industry <= 0.5, 'DISCOUNT',
                         np.where(avg_vs_industry <= 1.0, 'FAIR',
                         np.where(avg_vs_industry <= 1.5, 'PREMIUM', 'VERY_EXPENSIVE'))))

    return pd.DataFrame({
        **_id_cols(m),
        'PE': pe, 'PBV': pbv, 'EV_EBITDA': get_col(m, 'EVEBITDA', 'USER_RATIOS'),
        'PEG': get_col(m, 'PEG_RATIO', 'USER_RATIOS'), 'Price_To_Sales': get_col(m, 'PRICE_TO_SALES', 'USER_RATIOS'),
        'Earnings_Yield': safe_div(100.0, pe), 'Valuation_Band': val_band, 'Valuation_Comfort_Score': val_comfort,
        'PE_Vs_Industry': np.round(pe_vs_industry, 2), 'PBV_Vs_Industry': np.round(pbv_vs_industry, 2),
        'Industry_Relative_Val': industry_val_label,
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
    npm_current = safe_div(pat * 100, sales)
    npm_ly = pd.Series(get_col(m, 'NPM_LAST_YEAR', 'ANNUAL'), dtype=float)
    npm_py = pd.Series(get_col(m, 'NPM_PRECEDING_YEAR', 'ANNUAL'), dtype=float)

    # Effective Tax Rate
    pat_ly = pd.Series(get_col(m, 'PAT_LAST_YEAR', 'ANNUAL'), dtype=float)
    pbt_ly = pd.Series(get_col(m, 'PBT_LAST_YEAR', 'ANNUAL'), dtype=float)
    pbt_py = pd.Series(get_col(m, 'PBT_PRECEDING_YEAR', 'ANNUAL'), dtype=float)
    pat_py = pd.Series(get_col(m, 'PAT_PRECEDING_YEAR', 'ANNUAL'), dtype=float)
    etr_ly = pd.Series(np.where((~pd.isna(pbt_ly)) & (pbt_ly > 1), 1.0 - safe_div(pat_ly, pbt_ly), np.nan), dtype=float)
    etr_py = pd.Series(np.where((~pd.isna(pbt_py)) & (pbt_py > 1), 1.0 - safe_div(pat_py, pbt_py), np.nan), dtype=float)
    etr_ly = etr_ly.clip(-1.0, 1.0)
    etr_py = etr_py.clip(-1.0, 1.0)

    return pd.DataFrame({
        **_id_cols(m),
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

    # TTM PAT from quarterly data with stub-period guard.
    # Guard: if any single quarter exceeds 60% of TTM total, the TTM figure is
    # unreliable (likely a stub period or one-off event). In that case we fall
    # back to annual PAT. This is deliberately conservative — seasonal businesses
    # (e.g., agriculture) may trigger this, but using annual PAT is safer than a
    # distorted TTM figure.
    np_q1 = pd.Series(get_col(m, 'NP_Q1', 'QUARTERLY'), dtype=float)
    np_q2 = pd.Series(get_col(m, 'NP_Q2', 'QUARTERLY'), dtype=float)
    np_q3 = pd.Series(get_col(m, 'NP_Q3', 'QUARTERLY'), dtype=float)
    np_q4 = pd.Series(get_col(m, 'NP_Q4', 'QUARTERLY'), dtype=float)
    pat_ttm_raw = np_q1 + np_q2 + np_q3 + np_q4
    max_quarter = np.maximum(np.maximum(np.abs(np_q1.fillna(0)), np.abs(np_q2.fillna(0))),
                             np.maximum(np.abs(np_q3.fillna(0)), np.abs(np_q4.fillna(0))))
    ttm_abs = np.abs(pat_ttm_raw).replace(0, np.nan)
    stub_risk = pd.notna(pat_ttm_raw) & (max_quarter > ttm_abs * 0.6)
    pat_ttm = pd.Series(np.where(stub_risk, np.nan, pat_ttm_raw), dtype=float)
    pat = pd.Series(np.where(pd.notna(pat_ttm) & (pat_ttm != 0), pat_ttm, pat_annual), dtype=float)

    cfo_pat = safe_div(cfo, pat)
    cfo_3yr = pd.Series(get_col(m, 'CFO_3Y_CUMULATIVE', 'CASHFLOW'), dtype=float)
    cfo_5yr = pd.Series(get_col(m, 'CFO_5Y_CUMULATIVE', 'CASHFLOW'), dtype=float)

    # 3yr CFO/PAT: average of individual year ratios
    pat_ly = pd.Series(get_col(m, 'PAT_LAST_YEAR', 'ANNUAL'), dtype=float)
    pat_py = pd.Series(get_col(m, 'PAT_PRECEDING_YEAR', 'ANNUAL'), dtype=float)
    # Infer year-3 CFO with bounds checking
    cfo_yr3_raw = cfo_3yr - cfo - cfo_py
    avg_recent_cfo = (np.abs(cfo) + np.abs(cfo_py)) / 2.0
    cfo_yr3_suspicious = (avg_recent_cfo > 0) & (np.abs(cfo_yr3_raw) > avg_recent_cfo * 5.0)
    cfo_yr3_inferred = pd.Series(np.where(cfo_yr3_suspicious, np.nan, cfo_yr3_raw), dtype=float)
    r1 = pd.Series(safe_div(cfo, pat_annual), dtype=float)
    r2 = pd.Series(safe_div(cfo_py, pat_ly), dtype=float)
    r3 = pd.Series(safe_div(cfo_yr3_inferred, pat_py), dtype=float)
    ratio_sum = r1.fillna(0) + r2.fillna(0) + r3.fillna(0)
    ratio_count = (~pd.isna(r1)).astype(int) + (~pd.isna(r2)).astype(int) + (~pd.isna(r3)).astype(int)
    pat_3yr_cum = pat_annual + pat_ly + pat_py
    cfo_pat_3yr_avg = pd.Series(
        np.where(ratio_count >= 2, ratio_sum / ratio_count, safe_div(cfo_3yr, pat_3yr_cum)), dtype=float)

    worst_yr = np.minimum(np.minimum(
        np.where(pd.isna(r1), 999, r1), np.where(pd.isna(r2), 999, r2)),
        np.where(pd.isna(r3), 999, r3))
    worst_yr = np.where(worst_yr == 999, np.nan, worst_yr)

    pos_cfo_count = (cfo > 0).astype(int) + (cfo_py > 0).astype(int) + (cfo_yr3_inferred > 0).astype(int)

    wc_growth = safe_div(wc - wc_py, np.abs(wc_py)) * 100
    rev_growth = safe_div(sales - sales_ly, np.abs(sales_ly)) * 100

    fcf = pd.Series(get_col(m, 'FCF_LAST_YEAR', 'CASHFLOW'), dtype=float)
    fcf_py = pd.Series(get_col(m, 'FCF_PRECEDING_YEAR', 'CASHFLOW'), dtype=float)

    cff = pd.Series(get_col(m, 'CFF_LAST_YEAR', 'CASHFLOW'), dtype=float)
    cff_py = pd.Series(get_col(m, 'CFF_PRECEDING_YEAR', 'CASHFLOW'), dtype=float)

    cfi = pd.Series(get_col(m, 'CFI_LAST_YEAR', 'CASHFLOW'), dtype=float)
    cfi_py = pd.Series(get_col(m, 'CFI_PRECEDING_YEAR', 'CASHFLOW'), dtype=float)
    cfi_3yr = pd.Series(get_col(m, 'CFI_3Y_CUMULATIVE', 'CASHFLOW'), dtype=float)
    cfi_5yr = pd.Series(get_col(m, 'CFI_5Y_CUMULATIVE', 'CASHFLOW'), dtype=float)

    cfi_cfo_ratio = safe_div(np.abs(cfi_3yr), cfo_3yr)

    return pd.DataFrame({
        **_id_cols(m),
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

    debt_py = pd.Series(get_col(m, 'DEBT_PY', 'BALANCE'), dtype=float)

    return pd.DataFrame({
        **_id_cols(m),
        'Debt_Equity': de, 'Interest_Coverage': ic, 'Debt_Trend': debt_trend,
        'Financial_Strength_Score': fin_strength, 'Current_Ratio': current_ratio,
        'Total_Debt': debt, 'Debt_3Yr_Back': debt_3yr, 'Debt_Preceding_Year': debt_py,
        'Total_Assets': get_col(m, 'TOTAL_ASSETS', 'BALANCE'),
        'Debt_Capacity': get_col(m, 'DEBT_CAPACITY', 'USER_RATIOS'),
        'Capex': capex, 'Depreciation': np.round(depreciation, 1), 'Capex_To_Depreciation': np.round(capex_to_dep, 2),
    })


def build_growth_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build growth metrics sheet — uses get_col with alias support for EBITDA."""
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

    opm_q_change = opm_q - opm_q_prev
    opm_yoy_change = opm_q - opm_q_yoy
    q_rev_yoy_growth = safe_div(sales_q - sales_q_yoy, np.abs(sales_q_yoy)) * 100
    q_op_yoy_growth = safe_div(op_q - op_q_yoy, np.abs(op_q_yoy)) * 100

    q_deteriorating = (
        (~pd.isna(opm_q_change)) & (~pd.isna(opm_yoy_change))
        & (opm_q_change < -2) & (opm_yoy_change < -3)
    )

    return pd.DataFrame({
        **_id_cols(m),
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
        **_id_cols(m),
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

    inst_score = np.where(inst <= 1, 40, np.where(inst <= 3, 30, np.where(inst <= 5, 20, np.where(inst <= 10, 10, 0))))
    mcap_score = np.where(pd.isna(mcap), 0, np.where(mcap <= 500, 30, np.where(mcap <= 2000, 20, np.where(mcap <= 5000, 10, 0))))
    sh_score = np.where(pd.isna(num_sh), 10, np.where(num_sh <= 5000, 20, np.where(num_sh <= 20000, 10, np.where(num_sh <= 50000, 5, 0))))
    neglect_score = inst_score + mcap_score + sh_score

    generic_candidate = np.where((neglect_score >= 50) & (roe_5 >= 10) & (de <= 1.0), 'Yes',
                        np.where(neglect_score >= 40, 'Maybe', 'No'))

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
        **_id_cols(m),
        'Generic_Stock_Candidate': generic_candidate, 'Neglect_Score': neglect_score,
        'Neglect_Reasons': reasons, 'Institutional_Holding': inst, 'ROE_5Yr_Avg': roe_5,
        'Debt_Equity': de, 'Market_Cap': mcap, 'Num_Shareholders': num_sh,
    })


def build_dividends_sheet(m: pd.DataFrame) -> pd.DataFrame:
    """Build dividends sheet."""
    return pd.DataFrame({
        **_id_cols(m),
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
    opm_ly = quality_df['OPM_Last_Year'].values.astype(float)
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

    # ── Enhanced POOR_CASH_CONVERSION ──
    flag_poor_cash = (
        ((~np.isnan(cfo_pat_3yr)) & (cfo_pat_3yr < CONFIG['CFO_PAT_LOW_THRESHOLD']) & (cfo_pat_3yr >= 0))
        | ((np.isnan(cfo_pat_3yr)) & (~np.isnan(cfo_pat)) & (cfo_pat < CONFIG['CFO_PAT_LOW_THRESHOLD']) & (cfo_pat >= 0))
        | ((~np.isnan(worst_yr_cfo_pat)) & (worst_yr_cfo_pat < CONFIG['CFO_PAT_WORST_YEAR_THRESHOLD']) & (worst_yr_cfo_pat >= 0)
           & (~np.isnan(cfo_pat_3yr)) & (cfo_pat_3yr < CONFIG['CFO_PAT_BORDERLINE_THRESHOLD']))
    )

    # ── Enhanced INCONSISTENT_CFO ──
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

    # ── NPM vs OPM Divergence ──
    npm_slope = safe_div(npm_current - npm_py, 2.0)
    opm_slope = safe_div(opm - opm_py, 2.0)
    npm_opm_gap = npm_slope - opm_slope
    flag_npm_opm_divergence = (
        (~np.isnan(npm_slope)) & (~np.isnan(opm_slope))
        & np.isfinite(npm_opm_gap) & (npm_opm_gap > CONFIG['NPM_OPM_GAP_THRESHOLD'])
    )

    # ── Effective Tax Rate flag ──
    etr_ly = quality_df['ETR_Last_Year'].values.astype(float) / 100.0
    etr_py = quality_df['ETR_Preceding_Year'].values.astype(float) / 100.0
    flag_low_effective_tax = (
        (~np.isnan(etr_ly)) & (~np.isnan(etr_py))
        & (etr_ly < CONFIG['ETR_LOW_THRESHOLD']) & (etr_py < CONFIG['ETR_LOW_THRESHOLD'])
        & (etr_ly >= 0) & (etr_py >= 0)
    )

    # ── Asset Milking flag ──
    capex_to_dep = leverage_df['Capex_To_Depreciation'].values.astype(float)
    depreciation = leverage_df['Depreciation'].values.astype(float)
    flag_asset_milking = (
        (~np.isnan(capex_to_dep)) & (~np.isnan(depreciation))
        & (depreciation > 5) & (capex_to_dep < 0.5) & (capex_to_dep >= 0)
    )

    # ── Debt spike detection ──
    debt_py = leverage_df['Debt_Preceding_Year'].values.astype(float)
    flag_debt_spike_1yr = (
        (~np.isnan(debt)) & (~np.isnan(debt_py)) & (debt_py > 0)
        & (debt > debt_py * 1.5) & ((debt - debt_py) > 50)
    )

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

    # Industry-relative valuation flag
    pe_vs_ind = valuation_df['PE_Vs_Industry'].values.astype(float)
    pbv_vs_ind = valuation_df['PBV_Vs_Industry'].values.astype(float)
    flag_expensive_vs_industry = (
        (~np.isnan(pe_vs_ind)) & (~np.isnan(pbv_vs_ind))
        & (pe_vs_ind > 2.0) & (pbv_vs_ind > 2.0)
    )

    pricing_flags = {
        'FLAG_HIGH_PE': (~np.isnan(pe)) & (pe > CONFIG['PE_HIGH_THRESHOLD']),
        'FLAG_NEGATIVE_PE': (~np.isnan(pe)) & (pe < 0),
        'FLAG_HIGH_EV_EBITDA': (~np.isnan(ev_ebitda)) & (ev_ebitda > CONFIG['EV_EBITDA_HIGH_THRESHOLD']),
        'FLAG_NEGATIVE_EBITDA': (~np.isnan(ev_ebitda)) & (ev_ebitda < 0),
        'FLAG_HIGH_PBV_ROE': (~np.isnan(pbv)) & (~np.isnan(roe)) & (roe > 0) & (pbv > roe / 2),
        'FLAG_EXPENSIVE_VS_INDUSTRY': flag_expensive_vs_industry,
    }
    all_flags = {**structural_flags, **pricing_flags}

    # ── Sector-based severity adjustments ──
    is_cap_intensive = (~np.isnan(asset_turnover)) & (asset_turnover < CONFIG['ASSET_TURNOVER_CAPITAL_INTENSIVE'])
    is_very_cap_intensive = (~np.isnan(asset_turnover)) & (asset_turnover < CONFIG['ASSET_TURNOVER_VERY_CAPITAL_INTENSIVE'])

    weight_overrides = {
        'FLAG_POOR_CASH_CONVERSION': np.where(is_cap_intensive & flag_poor_cash, 0.5, 2.0),
        'FLAG_NEGATIVE_CFO': np.where(is_very_cap_intensive & structural_flags['FLAG_NEGATIVE_CFO'], 1.0, 2.0),
        'FLAG_INCONSISTENT_CFO': np.where(is_cap_intensive & flag_inconsistent_cfo, 1.0, 2.0),
    }

    sector_adj = np.full(n, '', dtype=object)
    adj_text = [
        (is_cap_intensive & flag_poor_cash, "POOR_CASH_CONVERSION: CRITICAL→MINOR"),
        (is_very_cap_intensive & structural_flags['FLAG_NEGATIVE_CFO'], "NEGATIVE_CFO: CRITICAL→MAJOR"),
        (is_cap_intensive & flag_inconsistent_cfo, "INCONSISTENT_CFO: CRITICAL→MAJOR"),
    ]
    for cond, text in adj_text:
        sector_adj = np.where(cond, np.where(sector_adj == '', text, sector_adj + ' | ' + text), sector_adj)

    # Calculate severity
    critical_count, major_count, minor_count = np.zeros(n), np.zeros(n), np.zeros(n)
    total_count, quality_severity, pricing_severity = np.zeros(n), np.zeros(n), np.zeros(n)

    for fname, farr in structural_flags.items():
        defn = RED_FLAG_DEFINITIONS.get(fname.replace('FLAG_', ''), {})
        total_count += farr.astype(int)
        if fname in weight_overrides:
            weight = weight_overrides[fname]
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

    # Build flag name strings
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

    # ── Earnings Quality classification ──
    accruals = cashflow_df['Accruals'].values.astype(float)
    eq_issues = np.zeros(n, dtype=int)
    eq_issues += ((~np.isnan(cfo_pat_3yr)) & (cfo_pat_3yr < CONFIG['CFO_PAT_EARNINGS_QUALITY_THRESHOLD'])).astype(int)
    eq_issues += ((~np.isnan(accruals)) & (accruals > CONFIG['ACCRUALS_AGGRESSIVE_THRESHOLD'])).astype(int)
    eq_issues += structural_flags.get('FLAG_HIGH_OTHER_INCOME', np.zeros(n, dtype=bool)).astype(int)
    eq_issues += ((~np.isnan(pos_cfo_years)) & (pos_cfo_years < 2)).astype(int)
    eq_issues += structural_flags.get('FLAG_LOW_EFFECTIVE_TAX', np.zeros(n, dtype=bool)).astype(int)
    earnings_quality_label = np.where(eq_issues >= 3, 'Aggressive',
                             np.where(eq_issues >= 1, 'Mixed', 'Clean'))

    # ── Cyclic Peak Risk Detection ──
    rev_g1 = growth_df['Revenue_Growth_1Yr'].values.astype(float)
    rev_g3 = growth_df['Revenue_Growth_3Yr'].values.astype(float)
    opm_avg_3 = (opm + opm_ly + opm_py) / 3.0

    cyclic_signals = np.zeros(n, dtype=int)
    cyclic_signals += ((~np.isnan(opm)) & (~np.isnan(opm_avg_3)) & (opm > opm_avg_3 * 1.2)).astype(int)
    cyclic_signals += ((~np.isnan(rev_g1)) & (~np.isnan(rev_g3)) & (rev_g3 > 0) & (rev_g1 < rev_g3 * 0.5)).astype(int)
    cyclic_signals += ((~np.isnan(roce)) & (roce > 25) & (~np.isnan(rev_g1)) & (rev_g1 < 10)).astype(int)

    cyclic_peak_risk = np.where(cyclic_signals >= 2, 'HIGH',
                       np.where(cyclic_signals >= 1, 'MODERATE', 'LOW'))

    rf_data = {
        'ISIN': m[COLS['ISIN_CODE']].values, 'NSE_Code': m[COLS['NSE_CODE']].values,
        'BSE_Code': m[COLS['BSE_CODE']].values,
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
        if decision[i] in ('SCREEN_FAILED', 'SCREEN_MARGINAL') and pchg >= CONFIG['PROMOTER_BUY_THRESHOLD'] and quality_severity[i] < CONFIG['CONVICTION_SEVERITY_CAP']:
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
        **_id_cols(m),
        'Decision_Bucket': decision, 'MCAP': mcap, 'Conviction_Override': conviction_override,
        'SCREEN_ELIGIBLE': np.where(quality_severity >= CONFIG['SEVERITY_FAILED_THRESHOLD'], 'NO', 'YES'),
        'Investment_Thesis': thesis, 'Reject_Reason': reject_reason,
        'Composite_Score': np.round(composite, 1), 'Score_Band': score_band,
        'Quality_Risk': red_flags_df['Quality_Risk'].values, 'Quality_Severity': np.round(quality_severity, 1),
        'Critical_Flags': critical, 'Business_Quality_Score': np.round(bq, 1),
        'Growth_Durability_Score': np.round(gd, 1), 'Valuation_Comfort_Score': np.round(vc, 1),
        'Financial_Strength_Score': np.round(fs, 1), 'Cash_Flow_Score': np.round(cf_score.astype(float), 1),
    })


def add_peer_ranking(analysis_df: pd.DataFrame, quality_df: pd.DataFrame,
                     valuation_df: pd.DataFrame, leverage_df: pd.DataFrame,
                     cashflow_df: pd.DataFrame) -> pd.DataFrame:
    """Add percentile-based peer ranking WITHIN survivors only."""
    survivor_buckets = {'GATES_CLEARED', 'SCREEN_PASSED_EXPENSIVE', 'CONTRARIAN_BET'}
    is_survivor = analysis_df['Decision_Bucket'].isin(survivor_buckets)
    n = len(analysis_df)

    if is_survivor.sum() < 3:
        analysis_df['Peer_Rank'] = np.nan
        analysis_df['Peer_Percentile'] = np.nan
        analysis_df['Peer_Rank_Breakdown'] = ''
        return analysis_df

    roe = quality_df['ROE_Latest'].values.astype(float)
    roce = quality_df['ROCE_Latest'].values.astype(float)
    cfo_pat = cashflow_df['CFO_PAT_3Yr_Avg'].values.astype(float)
    de = leverage_df['Debt_Equity'].values.astype(float)
    pe_vs_ind = valuation_df['PE_Vs_Industry'].values.astype(float)

    survivor_idx = np.where(is_survivor)[0]
    survivor_data = pd.DataFrame({
        'idx': survivor_idx,
        'ROE': roe[survivor_idx],
        'ROCE': roce[survivor_idx],
        'CFO_PAT': cfo_pat[survivor_idx],
        'DE': de[survivor_idx],
        'PE_Vs_Ind': pe_vs_ind[survivor_idx],
    })

    roe_pctile = survivor_data['ROE'].rank(pct=True, na_option='bottom')
    roce_pctile = survivor_data['ROCE'].rank(pct=True, na_option='bottom')
    cfo_pctile = survivor_data['CFO_PAT'].rank(pct=True, na_option='bottom')
    de_pctile = 1.0 - survivor_data['DE'].rank(pct=True, na_option='top')
    val_pctile = 1.0 - survivor_data['PE_Vs_Ind'].rank(pct=True, na_option='top')

    peer_score = (0.25 * roe_pctile + 0.25 * roce_pctile + 0.25 * cfo_pctile +
                  0.10 * de_pctile + 0.15 * val_pctile)

    peer_rank = peer_score.rank(ascending=False, method='min').astype(int)

    breakdown = [
        f"ROE:{r*100:.0f}p ROCE:{rc*100:.0f}p CFO:{c*100:.0f}p Debt:{d*100:.0f}p ValRel:{v*100:.0f}p"
        for r, rc, c, d, v in zip(roe_pctile, roce_pctile, cfo_pctile, de_pctile, val_pctile)
    ]

    analysis_df['Peer_Rank'] = np.nan
    analysis_df['Peer_Percentile'] = np.nan
    analysis_df['Peer_Rank_Breakdown'] = ''

    analysis_df.loc[survivor_idx, 'Peer_Rank'] = peer_rank.values
    analysis_df.loc[survivor_idx, 'Peer_Percentile'] = np.round(peer_score.values * 100, 1)
    analysis_df.loc[survivor_idx, 'Peer_Rank_Breakdown'] = breakdown

    survivor_count = is_survivor.sum()
    logging.info(f"Peer ranking: {survivor_count} survivors ranked "
                 f"(GATES_CLEARED + SCREEN_PASSED_EXPENSIVE + CONTRARIAN_BET)")

    return analysis_df


def build_decision_audit_sheet(m: pd.DataFrame, analysis_df: pd.DataFrame,
                                quality_df: pd.DataFrame, valuation_df: pd.DataFrame,
                                leverage_df: pd.DataFrame, growth_df: pd.DataFrame,
                                cashflow_df: pd.DataFrame, red_flags_df: pd.DataFrame,
                                shareholding_df: pd.DataFrame) -> pd.DataFrame:
    """Build Decision_Audit sheet: per-stock trace of how each Decision_Bucket was reached."""
    n = len(analysis_df)

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
    industry_val = valuation_df['Industry_Relative_Val'].values

    score_breakdown = np.empty(n, dtype=object)
    severity_gate = np.empty(n, dtype=object)
    pricing_gate = np.empty(n, dtype=object)
    earnings_gate = np.empty(n, dtype=object)
    score_gate = np.empty(n, dtype=object)
    override_col = np.empty(n, dtype=object)
    flags_detail = np.empty(n, dtype=object)
    narrative = np.empty(n, dtype=object)

    for i in range(n):
        score_breakdown[i] = (
            f"Quality:{bq[i]:.0f}(35%) + Growth:{gd[i]:.0f}(25%) + "
            f"Valuation:{vc[i]:.0f}(20%) + Strength:{fs[i]:.0f}(10%) + "
            f"CashFlow:{cf[i]:.0f}(10%) = {composite[i]:.1f} [{band[i]}]"
        )

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

        # Gate 2: Pricing flags
        if gate_stopped:
            pricing_gate[i] = "Not reached (stopped at severity gate)"
        elif pf and b in ('A', 'B'):
            pricing_gate[i] = f"Flags [{pf}] + Band {b} -> SCREEN_PASSED_EXPENSIVE"
            gate_stopped = 'pricing'
        elif pf:
            pricing_gate[i] = f"Flags [{pf}] present but Band {b} not A/B -> continued"
        else:
            pricing_gate[i] = "No pricing flags -> PASSED"

        # Gate 3: Earnings quality
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

        # Gate 4: Score band
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

        iv = industry_val[i] if pd.notna(industry_val[i]) else 'N/A'
        if iv not in ('N/A', 'FAIR'):
            parts.append(f"Vs industry: {iv}.")

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
