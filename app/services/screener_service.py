"""Dynamic SQL query builder for the screener."""

from app.database import query_all, query_value


def build_screener_query(filters: dict) -> tuple[str, list]:
    """Build a dynamic SQL query from screener filter parameters."""
    base = """
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
    conditions = []
    params = []

    # Decision bucket filter
    if filters.get("bucket"):
        buckets = filters["bucket"] if isinstance(filters["bucket"], list) else [filters["bucket"]]
        placeholders = ",".join(["?" for _ in buckets])
        conditions.append(f"a.decision_bucket IN ({placeholders})")
        params.extend(buckets)

    # Score band filter
    if filters.get("band"):
        bands = filters["band"] if isinstance(filters["band"], list) else [filters["band"]]
        placeholders = ",".join(["?" for _ in bands])
        conditions.append(f"a.score_band IN ({placeholders})")
        params.extend(bands)

    # Quality risk filter
    if filters.get("risk"):
        risks = filters["risk"] if isinstance(filters["risk"], list) else [filters["risk"]]
        placeholders = ",".join(["?" for _ in risks])
        conditions.append(f"a.quality_risk IN ({placeholders})")
        params.extend(risks)

    # Valuation band filter
    if filters.get("valuation"):
        vals = filters["valuation"] if isinstance(filters["valuation"], list) else [filters["valuation"]]
        placeholders = ",".join(["?" for _ in vals])
        conditions.append(f"v.valuation_band IN ({placeholders})")
        params.extend(vals)

    # Sector filter
    if filters.get("sector"):
        sectors = filters["sector"] if isinstance(filters["sector"], list) else [filters["sector"]]
        placeholders = ",".join(["?" for _ in sectors])
        conditions.append(f"s.sector IN ({placeholders})")
        params.extend(sectors)

    # Industry filter
    if filters.get("industry"):
        industries = filters["industry"] if isinstance(filters["industry"], list) else [filters["industry"]]
        placeholders = ",".join(["?" for _ in industries])
        conditions.append(f"s.industry IN ({placeholders})")
        params.extend(industries)

    # Numeric range filters
    range_filters = {
        "min_score": ("a.composite_score", ">="),
        "max_score": ("a.composite_score", "<="),
        "min_mcap": ("s.market_cap", ">="),
        "max_mcap": ("s.market_cap", "<="),
        "min_pe": ("v.pe", ">="),
        "max_pe": ("v.pe", "<="),
        "min_roe": ("q.roe_latest", ">="),
        "min_roce": ("q.roce_latest", ">="),
        "max_de": ("l.debt_equity", "<="),
        "min_promoter": ("sh.promoter_holding", ">="),
    }
    for key, (col, op) in range_filters.items():
        val = filters.get(key)
        if val is not None and val != "":
            try:
                conditions.append(f"{col} {op} ?")
                params.append(float(val))
            except ValueError:
                pass

    # Boolean toggles
    if filters.get("screen_eligible"):
        conditions.append("a.screen_eligible = true")
    if filters.get("no_red_flags"):
        conditions.append("rf.red_flag_count = 0")
    if filters.get("dividend_paying"):
        conditions.append("d.dividend_count > 0")

    if conditions:
        base += " WHERE " + " AND ".join(conditions)

    # Sorting
    sort_col = filters.get("sort", "a.composite_score")
    sort_dir = filters.get("dir", "DESC")
    allowed_sorts = {
        "score": "a.composite_score",
        "name": "s.stock_name",
        "mcap": "s.market_cap",
        "pe": "v.pe",
        "roe": "q.roe_latest",
        "roce": "q.roce_latest",
        "return": "v.return_1yr",
        "de": "l.debt_equity",
    }
    sort_column = allowed_sorts.get(sort_col, "a.composite_score")
    sort_direction = "ASC" if sort_dir == "ASC" else "DESC"
    base += f" ORDER BY {sort_column} {sort_direction} NULLS LAST"

    return base, params


def get_screener_results(filters: dict) -> list[dict]:
    """Run the screener query and return results."""
    sql, params = build_screener_query(filters)
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
