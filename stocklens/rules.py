"""
Pure configuration: column mappings, thresholds, scoring bins, red-flag definitions.
No imports beyond numpy (for bin boundaries). No I/O, no pandas.
"""

import numpy as np

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
    "EXPENSIVE_VS_INDUSTRY": {"severity": "MINOR", "weight": 0.5, "meaning": "PE and PBV both >2x industry median—expensive relative to sector peers"},
}

RED_FLAG_DEFINITIONS = {**STRUCTURAL_RED_FLAGS, **PRICING_RED_FLAGS}
