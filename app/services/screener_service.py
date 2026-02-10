"""Dynamic SQL query builder for the screener.

Uses a declarative filter registry to map UI parameters → SQL fragments,
keeping the builder predictable and auditable.
"""

from app.database import query_all, query_value

PAGE_SIZE = 50


# ── Filter Registry ──────────────────────────────────────────────────────────
# Each entry: param_name → (sql_column, filter_type)
# filter_type is one of: "in", ">=", "<=", "bool_true", "bool_zero"

IN_FILTERS: dict[str, str] = {
    "bucket":    "a.decision_bucket",
    "band":      "a.score_band",
    "risk":      "a.quality_risk",
    "valuation": "v.valuation_band",
    "sector":    "s.sector",
    "industry":  "s.industry",
}

RANGE_FILTERS: dict[str, tuple[str, str]] = {
    "min_score":    ("a.composite_score", ">="),
    "max_score":    ("a.composite_score", "<="),
    "min_mcap":     ("s.market_cap",      ">="),
    "max_mcap":     ("s.market_cap",      "<="),
    "min_pe":       ("v.pe",              ">="),
    "max_pe":       ("v.pe",              "<="),
    "min_roe":      ("q.roe_latest",      ">="),
    "min_roce":     ("q.roce_latest",     ">="),
    "max_de":       ("l.debt_equity",     "<="),
    "min_promoter": ("sh.promoter_holding",">="),
}

BOOL_FILTERS: dict[str, str] = {
    "screen_eligible": "a.screen_eligible = true",
    "no_red_flags":    "rf.red_flag_count = 0",
}

ALLOWED_SORTS: dict[str, str] = {
    "score":  "a.composite_score",
    "name":   "s.stock_name",
    "mcap":   "s.market_cap",
    "pe":     "v.pe",
    "roe":    "q.roe_latest",
    "roce":   "q.roce_latest",
    "return": "v.return_1yr",
    "de":     "l.debt_equity",
}

BASE_QUERY = """
    SELECT s.nse_code, s.stock_name, s.sector, s.industry, s.market_cap,
           a.decision_bucket, a.composite_score, a.score_band,
           a.quality_risk, a.screen_eligible,
           v.pe, v.pbv, v.peg, v.valuation_band, v.ltp,
           v.return_1yr, v.valuation_comfort_score,
           q.roe_latest, q.roce_latest, q.piotroski_score,
           q.business_quality_score,
           l.debt_equity,
           g.revenue_growth_ttm, g.eps,
           sh.promoter_holding,
           rf.red_flag_count
    FROM stocks s
    JOIN analysis a ON s.nse_code = a.nse_code
    JOIN valuation v ON s.nse_code = v.nse_code
    JOIN quality q ON s.nse_code = q.nse_code
    JOIN leverage l ON s.nse_code = l.nse_code
    JOIN growth g ON s.nse_code = g.nse_code
    JOIN shareholding sh ON s.nse_code = sh.nse_code
    JOIN red_flags rf ON s.nse_code = rf.nse_code
"""


def _normalize_list(val) -> list:
    """Ensure a filter value is always a list."""
    if isinstance(val, list):
        return val
    return [val]


def build_screener_query(filters: dict) -> tuple[str, list]:
    """Build a parameterized SQL query from screener filter parameters.

    All filter→SQL mappings come from the registries above.
    No ad-hoc string interpolation of user input.
    """
    conditions: list[str] = []
    params: list = []

    # IN filters: categorical multi-select
    for param, column in IN_FILTERS.items():
        values = filters.get(param)
        if not values:
            continue
        values = _normalize_list(values)
        placeholders = ", ".join("?" for _ in values)
        conditions.append(f"{column} IN ({placeholders})")
        params.extend(values)

    # Range filters: numeric comparisons
    for param, (column, op) in RANGE_FILTERS.items():
        val = filters.get(param)
        if val is None or val == "":
            continue
        try:
            params.append(float(val))
            conditions.append(f"{column} {op} ?")
        except (ValueError, TypeError):
            pass  # silently skip non-numeric input

    # Boolean toggles: fixed SQL fragments (no user input in the SQL)
    for param, sql_fragment in BOOL_FILTERS.items():
        if filters.get(param):
            conditions.append(sql_fragment)

    # Assemble WHERE clause
    sql = BASE_QUERY
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    # Sorting: whitelist-validated
    sort_key = filters.get("sort", "score")
    sort_column = ALLOWED_SORTS.get(sort_key, "a.composite_score")
    sort_direction = "ASC" if filters.get("dir") == "ASC" else "DESC"
    sql += f" ORDER BY {sort_column} {sort_direction} NULLS LAST"

    return sql, params


def get_screener_count(filters: dict) -> int:
    """Return total matching rows for current filters."""
    sql, params = build_screener_query(filters)
    count_sql = f"SELECT COUNT(*) FROM ({sql.split('ORDER BY')[0]}) _c"
    return query_value(count_sql, params) or 0


def get_screener_results(filters: dict, page: int = 1) -> list[dict]:
    """Run the screener query for a given page. Returns rows."""
    sql, params = build_screener_query(filters)
    offset = (max(1, page) - 1) * PAGE_SIZE
    sql += f" LIMIT {PAGE_SIZE} OFFSET {offset}"
    return query_all(sql, params)


def get_filter_options() -> dict:
    """Return all available filter option values."""
    return {
        "buckets": [r["decision_bucket"] for r in query_all(
            "SELECT DISTINCT decision_bucket FROM analysis ORDER BY decision_bucket"
        )],
        "bands": [r["score_band"] for r in query_all(
            "SELECT DISTINCT score_band FROM analysis WHERE score_band IS NOT NULL ORDER BY score_band"
        )],
        "risks": [r["quality_risk"] for r in query_all(
            "SELECT DISTINCT quality_risk FROM red_flags ORDER BY quality_risk"
        )],
        "valuation_bands": [r["valuation_band"] for r in query_all(
            "SELECT DISTINCT valuation_band FROM valuation WHERE valuation_band IS NOT NULL ORDER BY valuation_band"
        )],
        "sectors": [r["sector"] for r in query_all(
            "SELECT DISTINCT sector FROM stocks ORDER BY sector"
        )],
        "industries": [r["industry"] for r in query_all(
            "SELECT DISTINCT industry FROM stocks ORDER BY industry"
        )],
    }
