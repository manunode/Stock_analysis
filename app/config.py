"""Application configuration: color maps, thresholds, display settings."""

import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_analysis.duckdb")
APP_TITLE = "StockLens"

# ── Score Band colors ────────────────────────────────────────────────────────
BAND_COLORS = {
    "A": {"bg": "bg-emerald-500", "text": "text-white", "label": "Excellent"},
    "B": {"bg": "bg-blue-500", "text": "text-white", "label": "Good"},
    "C": {"bg": "bg-yellow-500", "text": "text-gray-900", "label": "Average"},
    "D": {"bg": "bg-orange-500", "text": "text-white", "label": "Below Avg"},
    "F": {"bg": "bg-red-500", "text": "text-white", "label": "Poor"},
}

# ── Decision Bucket colors ───────────────────────────────────────────────────
BUCKET_COLORS = {
    "GATES_CLEARED": {"bg": "bg-emerald-500", "text": "text-white", "border": "border-emerald-600"},
    "SCREEN_PASSED_VERIFY": {"bg": "bg-lime-500", "text": "text-gray-900", "border": "border-lime-600"},
    "SCREEN_PASSED_FLAGS": {"bg": "bg-amber-400", "text": "text-gray-900", "border": "border-amber-500"},
    "SCREEN_PASSED_EXPENSIVE": {"bg": "bg-orange-400", "text": "text-white", "border": "border-orange-500"},
    "SCREEN_MARGINAL": {"bg": "bg-yellow-400", "text": "text-gray-900", "border": "border-yellow-500"},
    "CONTRARIAN_BET": {"bg": "bg-indigo-500", "text": "text-white", "border": "border-indigo-600"},
    "VALUE_TRAP": {"bg": "bg-red-500", "text": "text-white", "border": "border-red-600"},
    "DATA_INCOMPLETE": {"bg": "bg-slate-400", "text": "text-white", "border": "border-slate-500"},
    "SCREEN_FAILED": {"bg": "bg-gray-400", "text": "text-gray-800", "border": "border-gray-500"},
}

# ── Quality Risk colors ──────────────────────────────────────────────────────
RISK_COLORS = {
    "LOW": {"bg": "bg-green-100", "text": "text-green-800", "dot": "bg-green-500"},
    "MEDIUM": {"bg": "bg-amber-100", "text": "text-amber-800", "dot": "bg-amber-500"},
    "HIGH": {"bg": "bg-red-100", "text": "text-red-800", "dot": "bg-red-500"},
}

# ── Valuation Band colors ───────────────────────────────────────────────────
VALUATION_COLORS = {
    "Cheap": {"bg": "bg-green-100", "text": "text-green-800"},
    "Fair": {"bg": "bg-blue-100", "text": "text-blue-800"},
    "Expensive": {"bg": "bg-orange-100", "text": "text-orange-800"},
    "Very_Expensive": {"bg": "bg-red-100", "text": "text-red-800"},
    "Unknown": {"bg": "bg-gray-100", "text": "text-gray-600"},
}

# ── Bucket display names ────────────────────────────────────────────────────
BUCKET_LABELS = {
    "GATES_CLEARED": "Gates Cleared",
    "SCREEN_PASSED_VERIFY": "Passed - Verify",
    "SCREEN_PASSED_FLAGS": "Passed - Flags",
    "SCREEN_PASSED_EXPENSIVE": "Passed - Expensive",
    "SCREEN_MARGINAL": "Marginal",
    "CONTRARIAN_BET": "Contrarian Bet",
    "VALUE_TRAP": "Value Trap",
    "DATA_INCOMPLETE": "Data Incomplete",
    "SCREEN_FAILED": "Screen Failed",
}

# ── Red flag display names ──────────────────────────────────────────────────
FLAG_LABELS = {
    "flag_low_roe": "Low ROE",
    "flag_declining_roe": "Declining ROE",
    "flag_low_roce": "Low ROCE",
    "flag_declining_roce": "Declining ROCE",
    "flag_poor_cash_conversion": "Poor Cash Conversion",
    "flag_negative_cfo": "Negative CFO",
    "flag_inconsistent_cfo": "Inconsistent CFO",
    "flag_frequent_exceptionals": "Frequent Exceptionals",
    "flag_high_other_income": "High Other Income",
    "flag_rising_receivables": "Rising Receivables",
    "flag_rising_inventory": "Rising Inventory",
    "flag_margin_compression": "Margin Compression",
    "flag_rising_debt": "Rising Debt",
    "flag_wc_divergence": "WC Divergence",
    "flag_npm_opm_divergence": "NPM/OPM Divergence",
    "flag_high_pe": "High PE",
    "flag_negative_pe": "Negative PE",
    "flag_high_ev_ebitda": "High EV/EBITDA",
    "flag_negative_ebitda": "Negative EBITDA",
    "flag_high_pbv_roe": "High PBV/ROE",
}

QUALITY_FLAGS = [
    "flag_low_roe", "flag_declining_roe", "flag_low_roce", "flag_declining_roce",
    "flag_poor_cash_conversion", "flag_negative_cfo", "flag_inconsistent_cfo",
    "flag_frequent_exceptionals", "flag_high_other_income", "flag_rising_receivables",
    "flag_rising_inventory", "flag_margin_compression", "flag_rising_debt",
    "flag_wc_divergence", "flag_npm_opm_divergence",
]

PRICING_FLAGS = [
    "flag_high_pe", "flag_negative_pe", "flag_high_ev_ebitda",
    "flag_negative_ebitda", "flag_high_pbv_roe",
]
