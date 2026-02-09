"""
ETL script to load relevant stock analysis data from Excel into DuckDB.

Reads stock_analysis.xlsx (12 sheets, ~240 columns), filters to ~153 relevant
columns (removing duplicates, derivable values, verbose text, and metadata),
and persists them into 11 normalized DuckDB tables.

Usage:
    python load_to_duckdb.py
"""

import pandas as pd
import duckdb
import os

EXCEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_analysis.xlsx")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_analysis.duckdb")


def read_sheet(xlsx, sheet_name, columns=None):
    """Read a sheet from Excel, optionally selecting specific columns."""
    df = pd.read_excel(xlsx, sheet_name=sheet_name)
    if columns:
        # Only keep columns that actually exist in the sheet
        available = [c for c in columns if c in df.columns]
        df = df[available]
    return df


def load_data():
    """Load all relevant data from the Excel file into DataFrames."""
    print(f"Reading {EXCEL_PATH}...")
    xlsx = EXCEL_PATH

    # ── 1. stocks (from Master) ──────────────────────────────────────────
    # Core identity table. Drop availability meta-flags that are operational,
    # not analytical. Keep Is_Financial_Sector as it affects screening logic.
    stocks = read_sheet(xlsx, "Master", [
        "NSE_Code", "Stock_Name", "Sector", "Industry", "BSE_Code", "ISIN",
        "Market_Cap", "Data_Type", "Years_Available", "Latest_Year",
        "Fiscal_Year_End", "Is_Financial_Sector",
    ])

    # ── 2. analysis (from Analysis) ──────────────────────────────────────
    # Screening decisions. Drop Market_Cap/PE/Piotroski_Score (duplicated from
    # Master/Valuation/Quality), Generic_Stock_Candidate (from Neglected_Firm),
    # and Financial_Analysis_Gap (operational meta-flag).
    analysis = read_sheet(xlsx, "Analysis", [
        "NSE_Code", "Decision_Bucket", "SCREEN_ELIGIBLE", "Investment_Thesis",
        "Reject_Reason", "Composite_Score", "Sector_Relative_Adj",
        "Score_Band", "Quality_Risk", "Critical_Flags", "Major_Flags",
        "Primary_Concern", "Conviction_Override",
    ])

    # ── 3. valuation (from Valuation) ────────────────────────────────────
    # Drop PE_Annual (have PE/PE_TTM), Market_Cap_Annual (have Market_Cap in
    # stocks), PEG_TTM & PEG_Computed (redundant with PEG),
    # Distance_From_52W_High (derivable from LTP & Week52_High).
    valuation = read_sheet(xlsx, "Valuation", [
        "NSE_Code", "PE", "PE_TTM", "PBV", "EV_EBITDA", "PEG",
        "Price_To_Sales", "Earnings_Yield", "Valuation_Band",
        "Valuation_Comfort_Score", "Enterprise_Value", "LTP",
        "Week52_High", "Week52_Low", "Price_Position_52W",
        "Return_1Yr", "Returns_vs_Nifty50_Qtr", "Returns_vs_Sector_Qtr",
        "Return_vs_Nifty_1Yr",
    ])

    # ── 4. quality (from Quality) ────────────────────────────────────────
    # Drop EBITDA absolute (keep EBITDA_Margin), ROE_ROA_Gap (derivable from
    # ROE - ROA), Exceptional_Items_Latest (have Count), High_Other_Income_Years
    # (granular detail of Other_Income_Pct_PAT).
    quality = read_sheet(xlsx, "Quality", [
        "NSE_Code", "Business_Quality_Score", "Earnings_Quality",
        "Piotroski_Score", "Piotroski_Assessment",
        "ROE_Latest", "ROE_3Yr_Avg", "ROE_5Yr_Avg", "ROE_Trend",
        "ROCE_Latest", "ROCE_3Yr_Avg", "ROCE_5Yr_Avg", "ROCE_Trend",
        "ROA_Latest",
        "OPM_Latest", "OPM_TTM", "OPM_Trend",
        "NPM_Latest", "NPM_Trend",
        "EBITDA_Margin",
        "Leverage_Driven", "Exceptional_Items_Count",
        "Other_Income_Pct_PAT",
    ])

    # ── 5. cash_flow (from Cash_Flow) ────────────────────────────────────
    # Drop WC_Growth_Pct, Rev_Growth_Pct, WC_Rev_Growth_Ratio (growth-related
    # duplicates better captured in Growth table), CFI (capex outflow, less
    # actionable in screening).
    cash_flow = read_sheet(xlsx, "Cash_Flow", [
        "NSE_Code", "CFO_Latest", "PAT_Latest", "CFO_PAT_Latest",
        "CFO_PAT_3Yr_Avg", "Positive_CFO_Years", "CFO_Trend",
        "CFROA", "Accruals", "CEPS",
    ])

    # ── 6. leverage (from Leverage) ──────────────────────────────────────
    # Drop Equity_Capital, Working_Capital, WC_Turnover (granular balance sheet
    # items derivable from other fields), Capital_Intensity (categorical derived
    # from Asset_Turnover), Trade_Payables (absolute, have Payables_Days).
    leverage = read_sheet(xlsx, "Leverage", [
        "NSE_Code", "Debt_Equity", "LT_Debt_Equity", "Net_Debt_EBITDA",
        "Interest_Coverage", "Debt_Trend", "Financial_Strength_Score",
        "Current_Ratio", "Quick_Ratio", "Total_Debt", "Total_Equity",
        "Total_Assets", "Asset_Turnover",
        "Receivable_Days", "Recv_Days_Trend",
        "Inventory_Days", "Inv_Days_Trend",
        "Payables_Days", "Cash_Conversion_Cycle",
    ])

    # ── 7. growth (from Growth) ──────────────────────────────────────────
    # Drop Revenue_Growth_2Yr/3Yr (have CAGR_3Yr), Share_Dilution_Pct,
    # Dividend_Payout_NP, Retention_Ratio (belong in dividends/derivable),
    # Dividend_Per_Share (in Dividends table).
    growth = read_sheet(xlsx, "Growth", [
        "NSE_Code", "Revenue_Growth_TTM", "NP_Growth_TTM",
        "Revenue_Growth_Qtr_YoY", "NP_Growth_Qtr_YoY",
        "Revenue_Growth_1Yr", "Revenue_CAGR_3Yr",
        "NP_Growth", "EPS_Growth_3Yr",
        "Revenue_Volatility", "Profit_Consistency",
        "Profit_Growth_Consistency", "Growth_Durability_Score",
        "Total_Revenue", "Book_Value_Per_Share", "EPS", "ROIC",
    ])

    # ── 8. shareholding (from Shareholding) ──────────────────────────────
    # Drop Shareholding_Available (meta flag in Master),
    # Institutional_Holding_Overview (variant of Institutional_Holding),
    # Others_Holding (derivable: 100 - promoter - public - institutional),
    # Governance_Flags (mostly NaN), Inter_Se_Pct_Of_Disposals,
    # Market_Sell_Pct_MCap, Recent_Market_Buy_6m (very granular insider detail).
    shareholding = read_sheet(xlsx, "Shareholding", [
        "NSE_Code", "Promoter_Holding", "Promoter_Pledge", "Pledge_Risk",
        "Promoter_Change_1Yr",
        "FII_Holding", "FII_Change_1Yr",
        "MF_Holding", "MF_Change_1Yr",
        "Institutional_Holding", "Public_Holding",
        "Insider_Action", "Insider_Sentiment", "Insider_Context",
    ])

    # ── 9. dividends (from Dividends) ────────────────────────────────────
    # Drop Corporate_Actions_Available (meta, in Master),
    # Years_Since_Dividend (derivable from dates), Dividend_Paying_Years
    # (redundant with Count/Consistency), individual bonus/split/rights details
    # (keep only counts).
    dividends = read_sheet(xlsx, "Dividends", [
        "NSE_Code", "Dividend_Count", "Latest_Dividend_Date",
        "Latest_Dividend_Amount", "Latest_Dividend_Type",
        "Dividend_5Yr_Consistency", "Dividend_Trend",
        "Total_Dividends_Paid",
        "Bonus_Count", "Split_Count", "Rights_Count",
    ])

    # ── 10. neglected_firm (from Neglected_Firm) ─────────────────────────
    # Most columns duplicate other tables. Keep only the unique analytical
    # outputs: Neglect_Score, Neglect_Age_Warning, Generic_Stock_Candidate.
    neglected_firm = read_sheet(xlsx, "Neglected_Firm", [
        "NSE_Code", "Generic_Stock_Candidate", "Neglect_Score",
        "Neglect_Age_Warning",
    ])

    # ── 11. red_flags (from Red_Flags) ───────────────────────────────────
    # Drop Quality_Flags/Pricing_Flags/Red_Flags/Red_Flags_Explained (verbose
    # text versions of the structured FLAG_ columns), Sector_Warnings &
    # Sector_Adjustments_Made (mostly NaN), Sector_Adjustment_Needed (meta).
    red_flags = read_sheet(xlsx, "Red_Flags", [
        "NSE_Code", "Quality_Risk", "Quality_Severity",
        "Quality_Flag_Count", "Pricing_Flag_Count", "Red_Flag_Count",
        # Individual flags (boolean indicators)
        "FLAG_LOW_ROE", "FLAG_DECLINING_ROE",
        "FLAG_LOW_ROCE", "FLAG_DECLINING_ROCE",
        "FLAG_POOR_CASH_CONVERSION", "FLAG_NEGATIVE_CFO",
        "FLAG_INCONSISTENT_CFO", "FLAG_FREQUENT_EXCEPTIONALS",
        "FLAG_HIGH_OTHER_INCOME", "FLAG_RISING_RECEIVABLES",
        "FLAG_RISING_INVENTORY", "FLAG_MARGIN_COMPRESSION",
        "FLAG_RISING_DEBT", "FLAG_WC_DIVERGENCE",
        "FLAG_NPM_OPM_DIVERGENCE",
        "FLAG_HIGH_PE", "FLAG_NEGATIVE_PE",
        "FLAG_HIGH_EV_EBITDA", "FLAG_NEGATIVE_EBITDA",
        "FLAG_HIGH_PBV_ROE",
    ])

    return {
        "stocks": stocks,
        "analysis": analysis,
        "valuation": valuation,
        "quality": quality,
        "cash_flow": cash_flow,
        "leverage": leverage,
        "growth": growth,
        "shareholding": shareholding,
        "dividends": dividends,
        "neglected_firm": neglected_firm,
        "red_flags": red_flags,
    }


# ── DDL for all tables ───────────────────────────────────────────────────────

DDL = """
-- Core stock identity and metadata
CREATE TABLE IF NOT EXISTS stocks (
    nse_code            TEXT PRIMARY KEY,
    stock_name          TEXT,
    sector              TEXT,
    industry            TEXT,
    bse_code            INTEGER,
    isin                TEXT,
    market_cap          DOUBLE,
    data_type           TEXT,
    years_available     INTEGER,
    latest_year         TEXT,
    fiscal_year_end     TEXT,
    is_financial_sector BOOLEAN
);

-- Screening decisions and composite scores
CREATE TABLE IF NOT EXISTS analysis (
    nse_code            TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    decision_bucket     TEXT,
    screen_eligible     BOOLEAN,
    investment_thesis   TEXT,
    reject_reason       TEXT,
    composite_score     INTEGER,
    sector_relative_adj TEXT,
    score_band          TEXT,
    quality_risk        TEXT,
    critical_flags      TEXT,
    major_flags         TEXT,
    primary_concern     TEXT,
    conviction_override TEXT
);

-- Valuation multiples and price metrics
CREATE TABLE IF NOT EXISTS valuation (
    nse_code                TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    pe                      DOUBLE,
    pe_ttm                  DOUBLE,
    pbv                     DOUBLE,
    ev_ebitda               DOUBLE,
    peg                     DOUBLE,
    price_to_sales          DOUBLE,
    earnings_yield          DOUBLE,
    valuation_band          TEXT,
    valuation_comfort_score INTEGER,
    enterprise_value        DOUBLE,
    ltp                     DOUBLE,
    week52_high             DOUBLE,
    week52_low              DOUBLE,
    price_position_52w      DOUBLE,
    return_1yr              DOUBLE,
    returns_vs_nifty50_qtr  DOUBLE,
    returns_vs_sector_qtr   DOUBLE,
    return_vs_nifty_1yr     DOUBLE
);

-- Profitability and earnings quality metrics
CREATE TABLE IF NOT EXISTS quality (
    nse_code               TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    business_quality_score INTEGER,
    earnings_quality       TEXT,
    piotroski_score        INTEGER,
    piotroski_assessment   TEXT,
    roe_latest             DOUBLE,
    roe_3yr_avg            DOUBLE,
    roe_5yr_avg            DOUBLE,
    roe_trend              DOUBLE,
    roce_latest            DOUBLE,
    roce_3yr_avg           DOUBLE,
    roce_5yr_avg           DOUBLE,
    roce_trend             DOUBLE,
    roa_latest             DOUBLE,
    opm_latest             DOUBLE,
    opm_ttm                DOUBLE,
    opm_trend              DOUBLE,
    npm_latest             DOUBLE,
    npm_trend              DOUBLE,
    ebitda_margin          DOUBLE,
    leverage_driven        BOOLEAN,
    exceptional_items_count INTEGER,
    other_income_pct_pat   DOUBLE
);

-- Cash flow health indicators
CREATE TABLE IF NOT EXISTS cash_flow (
    nse_code          TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    cfo_latest        DOUBLE,
    pat_latest        DOUBLE,
    cfo_pat_latest    DOUBLE,
    cfo_pat_3yr_avg   DOUBLE,
    positive_cfo_years INTEGER,
    cfo_trend         DOUBLE,
    cfroa             DOUBLE,
    accruals          DOUBLE,
    ceps              DOUBLE
);

-- Debt, liquidity, and efficiency ratios
CREATE TABLE IF NOT EXISTS leverage (
    nse_code                TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    debt_equity             DOUBLE,
    lt_debt_equity          DOUBLE,
    net_debt_ebitda         DOUBLE,
    interest_coverage       DOUBLE,
    debt_trend              DOUBLE,
    financial_strength_score INTEGER,
    current_ratio           DOUBLE,
    quick_ratio             DOUBLE,
    total_debt              DOUBLE,
    total_equity            DOUBLE,
    total_assets            DOUBLE,
    asset_turnover          DOUBLE,
    receivable_days         DOUBLE,
    recv_days_trend         DOUBLE,
    inventory_days          DOUBLE,
    inv_days_trend          DOUBLE,
    payables_days           DOUBLE,
    cash_conversion_cycle   DOUBLE
);

-- Revenue and profit growth metrics
CREATE TABLE IF NOT EXISTS growth (
    nse_code                  TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    revenue_growth_ttm        DOUBLE,
    np_growth_ttm             DOUBLE,
    revenue_growth_qtr_yoy    DOUBLE,
    np_growth_qtr_yoy         DOUBLE,
    revenue_growth_1yr        DOUBLE,
    revenue_cagr_3yr          DOUBLE,
    np_growth                 DOUBLE,
    eps_growth_3yr            DOUBLE,
    revenue_volatility        DOUBLE,
    profit_consistency        INTEGER,
    profit_growth_consistency DOUBLE,
    growth_durability_score   INTEGER,
    total_revenue             DOUBLE,
    book_value_per_share      DOUBLE,
    eps                       DOUBLE,
    roic                      DOUBLE
);

-- Ownership structure and insider activity
CREATE TABLE IF NOT EXISTS shareholding (
    nse_code              TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    promoter_holding      DOUBLE,
    promoter_pledge       DOUBLE,
    pledge_risk           TEXT,
    promoter_change_1yr   DOUBLE,
    fii_holding           DOUBLE,
    fii_change_1yr        DOUBLE,
    mf_holding            DOUBLE,
    mf_change_1yr         DOUBLE,
    institutional_holding DOUBLE,
    public_holding        DOUBLE,
    insider_action        TEXT,
    insider_sentiment     TEXT,
    insider_context       TEXT
);

-- Dividend history and corporate actions summary
CREATE TABLE IF NOT EXISTS dividends (
    nse_code                TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    dividend_count          INTEGER,
    latest_dividend_date    TEXT,
    latest_dividend_amount  DOUBLE,
    latest_dividend_type    TEXT,
    dividend_5yr_consistency INTEGER,
    dividend_trend          TEXT,
    total_dividends_paid    DOUBLE,
    bonus_count             INTEGER,
    split_count             INTEGER,
    rights_count            INTEGER
);

-- Neglected/undiscovered firm assessment
CREATE TABLE IF NOT EXISTS neglected_firm (
    nse_code                 TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    generic_stock_candidate  BOOLEAN,
    neglect_score            TEXT,
    neglect_age_warning      TEXT
);

-- Risk flags (individual boolean indicators)
CREATE TABLE IF NOT EXISTS red_flags (
    nse_code                    TEXT PRIMARY KEY REFERENCES stocks(nse_code),
    quality_risk                TEXT,
    quality_severity            INTEGER,
    quality_flag_count          INTEGER,
    pricing_flag_count          INTEGER,
    red_flag_count              INTEGER,
    flag_low_roe                BOOLEAN,
    flag_declining_roe          BOOLEAN,
    flag_low_roce               BOOLEAN,
    flag_declining_roce         BOOLEAN,
    flag_poor_cash_conversion   BOOLEAN,
    flag_negative_cfo           BOOLEAN,
    flag_inconsistent_cfo       BOOLEAN,
    flag_frequent_exceptionals  BOOLEAN,
    flag_high_other_income      BOOLEAN,
    flag_rising_receivables     BOOLEAN,
    flag_rising_inventory       BOOLEAN,
    flag_margin_compression     BOOLEAN,
    flag_rising_debt            BOOLEAN,
    flag_wc_divergence          BOOLEAN,
    flag_npm_opm_divergence     BOOLEAN,
    flag_high_pe                BOOLEAN,
    flag_negative_pe            BOOLEAN,
    flag_high_ev_ebitda         BOOLEAN,
    flag_negative_ebitda        BOOLEAN,
    flag_high_pbv_roe           BOOLEAN
);

-- Sector level aggregated metrics (computed from stock-level data)
CREATE TABLE IF NOT EXISTS sector_summary (
    sector                      TEXT PRIMARY KEY,
    num_companies               INTEGER,
    avg_market_cap              DOUBLE,
    avg_pe                      DOUBLE,
    avg_peg                     DOUBLE,
    avg_pbv                     DOUBLE,
    avg_roe                     DOUBLE,
    avg_roce                    DOUBLE,
    avg_roa                     DOUBLE
);

-- Industry level aggregated metrics (computed from stock-level data)
CREATE TABLE IF NOT EXISTS industry_summary (
    industry                    TEXT PRIMARY KEY,
    num_companies               INTEGER,
    avg_market_cap              DOUBLE,
    avg_pe                      DOUBLE,
    avg_peg                     DOUBLE,
    avg_pbv                     DOUBLE,
    avg_roe                     DOUBLE,
    avg_roce                    DOUBLE,
    avg_roa                     DOUBLE
);
"""


def bool_convert(series, true_values=("YES", "Yes", "yes", "TRUE", "True", "true", "1")):
    """Convert a text series to boolean, handling common yes/no patterns."""
    return series.map(
        lambda v: True if str(v).strip() in true_values
        else (False if pd.notna(v) else None)
    )


def prepare_dataframes(data):
    """Clean and type-convert DataFrames before insertion."""

    # stocks
    df = data["stocks"]
    df["Is_Financial_Sector"] = bool_convert(df["Is_Financial_Sector"])
    df.columns = [c.lower() for c in df.columns]

    # analysis
    df = data["analysis"]
    df["SCREEN_ELIGIBLE"] = bool_convert(df["SCREEN_ELIGIBLE"])
    df.columns = [c.lower() for c in df.columns]

    # quality
    df = data["quality"]
    df["Leverage_Driven"] = bool_convert(df["Leverage_Driven"])
    df.columns = [c.lower() for c in df.columns]

    # neglected_firm
    df = data["neglected_firm"]
    df["Generic_Stock_Candidate"] = bool_convert(df["Generic_Stock_Candidate"])
    df.columns = [c.lower() for c in df.columns]

    # All other tables just need lowercase columns
    for name in ("valuation", "cash_flow", "leverage", "growth",
                 "shareholding", "dividends", "red_flags"):
        data[name].columns = [c.lower() for c in data[name].columns]

    # Convert FLAG_ integer columns to boolean in red_flags
    flag_cols = [c for c in data["red_flags"].columns if c.startswith("flag_")]
    for col in flag_cols:
        data["red_flags"][col] = data["red_flags"][col].map(
            lambda v: bool(v) if pd.notna(v) else None
        )

    return data


def create_database(data):
    """Create DuckDB database and load all tables."""
    # Remove existing DB to start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = duckdb.connect(DB_PATH)

    # Create schema
    con.execute(DDL)

    # Insert order matters due to foreign keys: stocks first, then the rest
    table_order = [
        "stocks", "analysis", "valuation", "quality", "cash_flow",
        "leverage", "growth", "shareholding", "dividends",
        "neglected_firm", "red_flags",
    ]

    for table_name in table_order:
        df = data[table_name]
        # Use DuckDB's efficient DataFrame ingestion
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        cols = con.execute(
            f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{table_name}'"
        ).fetchone()[0]
        print(f"  {table_name:20s} → {count:>5,} rows, {cols:>3} columns")

    # Compute and populate sector_summary from loaded tables
    con.execute("""
        INSERT INTO sector_summary
        SELECT
            s.sector,
            COUNT(*)                     AS num_companies,
            ROUND(AVG(s.market_cap), 2)  AS avg_market_cap,
            ROUND(AVG(v.pe), 2)          AS avg_pe,
            ROUND(AVG(v.peg), 2)         AS avg_peg,
            ROUND(AVG(v.pbv), 2)         AS avg_pbv,
            ROUND(AVG(q.roe_latest), 2)  AS avg_roe,
            ROUND(AVG(q.roce_latest), 2) AS avg_roce,
            ROUND(AVG(q.roa_latest), 2)  AS avg_roa
        FROM stocks s
        JOIN valuation v ON s.nse_code = v.nse_code
        JOIN quality q   ON s.nse_code = q.nse_code
        GROUP BY s.sector
        ORDER BY s.sector
    """)
    count = con.execute("SELECT COUNT(*) FROM sector_summary").fetchone()[0]
    cols = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'sector_summary'"
    ).fetchone()[0]
    print(f"  {'sector_summary':20s} → {count:>5,} rows, {cols:>3} columns  (computed)")

    # Compute and populate industry_summary from loaded tables
    con.execute("""
        INSERT INTO industry_summary
        SELECT
            s.industry,
            COUNT(*)                     AS num_companies,
            ROUND(AVG(s.market_cap), 2)  AS avg_market_cap,
            ROUND(AVG(v.pe), 2)          AS avg_pe,
            ROUND(AVG(v.peg), 2)         AS avg_peg,
            ROUND(AVG(v.pbv), 2)         AS avg_pbv,
            ROUND(AVG(q.roe_latest), 2)  AS avg_roe,
            ROUND(AVG(q.roce_latest), 2) AS avg_roce,
            ROUND(AVG(q.roa_latest), 2)  AS avg_roa
        FROM stocks s
        JOIN valuation v ON s.nse_code = v.nse_code
        JOIN quality q   ON s.nse_code = q.nse_code
        GROUP BY s.industry
        ORDER BY s.industry
    """)
    count = con.execute("SELECT COUNT(*) FROM industry_summary").fetchone()[0]
    cols = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'industry_summary'"
    ).fetchone()[0]
    print(f"  {'industry_summary':20s} → {count:>5,} rows, {cols:>3} columns  (computed)")

    # Create view: per-stock comparison against sector & industry averages
    con.execute("""
        CREATE OR REPLACE VIEW stock_vs_benchmarks AS
        SELECT
            s.nse_code,
            s.stock_name,
            s.sector,
            s.industry,
            s.market_cap,

            -- Stock metrics
            v.pe                        AS pe,
            v.peg                       AS peg,
            v.pbv                       AS pbv,
            q.roe_latest                AS roe,
            q.roce_latest               AS roce,
            q.roa_latest                AS roa,

            -- Sector averages
            sec.avg_pe                  AS sector_avg_pe,
            sec.avg_peg                 AS sector_avg_peg,
            sec.avg_pbv                 AS sector_avg_pbv,
            sec.avg_roe                 AS sector_avg_roe,
            sec.avg_roce               AS sector_avg_roce,
            sec.avg_roa                 AS sector_avg_roa,

            -- Industry averages
            ind.avg_pe                  AS industry_avg_pe,
            ind.avg_peg                 AS industry_avg_peg,
            ind.avg_pbv                 AS industry_avg_pbv,
            ind.avg_roe                 AS industry_avg_roe,
            ind.avg_roce               AS industry_avg_roce,
            ind.avg_roa                 AS industry_avg_roa,

            -- Difference vs sector (positive = stock is higher than sector avg)
            ROUND(v.pe  - sec.avg_pe, 2)           AS pe_vs_sector,
            ROUND(v.peg - sec.avg_peg, 2)          AS peg_vs_sector,
            ROUND(v.pbv - sec.avg_pbv, 2)          AS pbv_vs_sector,
            ROUND(q.roe_latest  - sec.avg_roe, 2)  AS roe_vs_sector,
            ROUND(q.roce_latest - sec.avg_roce, 2) AS roce_vs_sector,
            ROUND(q.roa_latest  - sec.avg_roa, 2)  AS roa_vs_sector,

            -- Difference vs industry (positive = stock is higher than industry avg)
            ROUND(v.pe  - ind.avg_pe, 2)           AS pe_vs_industry,
            ROUND(v.peg - ind.avg_peg, 2)          AS peg_vs_industry,
            ROUND(v.pbv - ind.avg_pbv, 2)          AS pbv_vs_industry,
            ROUND(q.roe_latest  - ind.avg_roe, 2)  AS roe_vs_industry,
            ROUND(q.roce_latest - ind.avg_roce, 2) AS roce_vs_industry,
            ROUND(q.roa_latest  - ind.avg_roa, 2)  AS roa_vs_industry

        FROM stocks s
        JOIN valuation v        ON s.nse_code = v.nse_code
        JOIN quality q          ON s.nse_code = q.nse_code
        JOIN sector_summary sec ON s.sector   = sec.sector
        JOIN industry_summary ind ON s.industry = ind.industry
    """)
    print(f"  {'stock_vs_benchmarks':20s} → view created (per-stock vs sector & industry averages)")

    return con


def verify_database(con):
    """Run verification queries and print summary."""
    print("\n── Verification ────────────────────────────────────────────")

    # Total columns across all tables
    total_cols = con.execute("""
        SELECT SUM(col_count) FROM (
            SELECT table_name, COUNT(*) as col_count
            FROM information_schema.columns
            WHERE table_schema = 'main'
            GROUP BY table_name
        )
    """).fetchone()[0]
    print(f"  Total columns across all tables: {total_cols}")

    # Sample query: top 5 stocks by composite score that passed screening
    print("\n  Top 5 stocks by composite score (screen-eligible):")
    rows = con.execute("""
        SELECT s.nse_code, s.stock_name, s.sector,
               a.composite_score, a.score_band, v.pe, q.roe_latest
        FROM stocks s
        JOIN analysis a ON s.nse_code = a.nse_code
        JOIN valuation v ON s.nse_code = v.nse_code
        JOIN quality q ON s.nse_code = q.nse_code
        WHERE a.screen_eligible = true
        ORDER BY a.composite_score DESC
        LIMIT 5
    """).fetchall()
    print(f"  {'NSE Code':<15} {'Name':<25} {'Sector':<30} {'Score':>5} {'Band':>5} {'PE':>7} {'ROE':>7}")
    print(f"  {'─'*15} {'─'*25} {'─'*30} {'─'*5} {'─'*5} {'─'*7} {'─'*7}")
    for r in rows:
        pe_str = f"{r[5]:.1f}" if r[5] is not None else "N/A"
        roe_str = f"{r[6]:.1f}" if r[6] is not None else "N/A"
        print(f"  {str(r[0]):<15} {str(r[1]):<25} {str(r[2]):<30} {r[3]:>5} {str(r[4]):>5} {pe_str:>7} {roe_str:>7}")

    # Sample query: sector distribution
    print("\n  Stocks per sector (top 10):")
    rows = con.execute("""
        SELECT sector, COUNT(*) as cnt
        FROM stocks
        GROUP BY sector
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
    for r in rows:
        print(f"    {str(r[0]):<45} {r[1]:>5}")

    # Sector summary sample
    print("\n  Sector summary (top 10 by number of companies):")
    rows = con.execute("""
        SELECT sector, num_companies, avg_market_cap,
               avg_pe, avg_peg, avg_pbv, avg_roe, avg_roce, avg_roa
        FROM sector_summary
        ORDER BY num_companies DESC
        LIMIT 10
    """).fetchall()
    fmt = "    {:<40s} {:>5} {:>10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}"
    print(fmt.format("Sector", "#Co", "Avg MCap", "PE", "PEG", "PBV", "ROE", "ROCE", "ROA"))
    print(fmt.format("─"*40, "─"*5, "─"*10, "─"*7, "─"*7, "─"*7, "─"*7, "─"*7, "─"*7))
    for r in rows:
        def f(v): return f"{v:.1f}" if v is not None else "N/A"
        print(fmt.format(
            str(r[0])[:40], r[1],
            f"{r[2]:,.0f}" if r[2] else "N/A",
            f(r[3]), f(r[4]), f(r[5]), f(r[6]), f(r[7]), f(r[8])
        ))

    # Industry summary sample
    print("\n  Industry summary (top 10 by number of companies):")
    rows = con.execute("""
        SELECT industry, num_companies, avg_market_cap,
               avg_pe, avg_peg, avg_pbv, avg_roe, avg_roce, avg_roa
        FROM industry_summary
        ORDER BY num_companies DESC
        LIMIT 10
    """).fetchall()
    print(fmt.format("Industry", "#Co", "Avg MCap", "PE", "PEG", "PBV", "ROE", "ROCE", "ROA"))
    print(fmt.format("─"*40, "─"*5, "─"*10, "─"*7, "─"*7, "─"*7, "─"*7, "─"*7, "─"*7))
    for r in rows:
        def f(v): return f"{v:.1f}" if v is not None else "N/A"
        print(fmt.format(
            str(r[0])[:40], r[1],
            f"{r[2]:,.0f}" if r[2] else "N/A",
            f(r[3]), f(r[4]), f(r[5]), f(r[6]), f(r[7]), f(r[8])
        ))

    # Stock vs benchmarks view sample
    print("\n  Stock vs Benchmarks (5 screen-eligible stocks — PE & ROE comparison):")
    rows = con.execute("""
        SELECT b.nse_code, b.stock_name,
               b.pe, b.sector_avg_pe, b.industry_avg_pe,
               b.roe, b.sector_avg_roe, b.industry_avg_roe
        FROM stock_vs_benchmarks b
        JOIN analysis a ON b.nse_code = a.nse_code
        WHERE a.screen_eligible = true
        ORDER BY a.composite_score DESC
        LIMIT 5
    """).fetchall()
    hdr = "    {:<12s} {:<20s} {:>7} {:>9} {:>9} {:>7} {:>9} {:>9}"
    print(hdr.format("NSE Code", "Name", "PE", "Sec Avg", "Ind Avg", "ROE", "Sec Avg", "Ind Avg"))
    print(hdr.format("─"*12, "─"*20, "─"*7, "─"*9, "─"*9, "─"*7, "─"*9, "─"*9))
    for r in rows:
        def f(v): return f"{v:.1f}" if v is not None else "N/A"
        print(hdr.format(str(r[0])[:12], str(r[1])[:20],
                         f(r[2]), f(r[3]), f(r[4]),
                         f(r[5]), f(r[6]), f(r[7])))

    # Red flag summary
    print("\n  Red flag distribution:")
    rows = con.execute("""
        SELECT quality_risk, COUNT(*) as cnt
        FROM red_flags
        GROUP BY quality_risk
        ORDER BY cnt DESC
    """).fetchall()
    for r in rows:
        print(f"    {str(r[0]):<15} {r[1]:>5}")


def main():
    print("=" * 65)
    print("  Stock Analysis → DuckDB ETL")
    print("=" * 65)

    # Step 1: Load from Excel
    data = load_data()

    # Step 2: Clean and convert types
    print("\nPreparing data...")
    data = prepare_dataframes(data)

    # Step 3: Create database and load
    print(f"\nCreating DuckDB database at {DB_PATH}...")
    con = create_database(data)

    # Step 4: Verify
    verify_database(con)

    con.close()
    print(f"\nDone. Database saved to {DB_PATH}")
    print(f"Database file size: {os.path.getsize(DB_PATH) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
