"""Peer and multi-stock comparison logic."""

from app.database import query_all


def get_comparison_data(nse_codes: list[str]) -> list[dict]:
    """Return full comparison data for a list of stock codes."""
    if not nse_codes:
        return []
    placeholders = ",".join(["?" for _ in nse_codes])
    return query_all(f"""
        SELECT
            s.nse_code, s.stock_name, s.sector, s.industry, s.market_cap,
            a.decision_bucket, a.composite_score, a.score_band, a.quality_risk,
            v.pe, v.pbv, v.peg, v.ev_ebitda, v.price_to_sales,
            v.earnings_yield, v.valuation_band, v.ltp,
            v.return_1yr, v.valuation_comfort_score,
            q.business_quality_score, q.piotroski_score, q.roe_latest,
            q.roce_latest, q.roa_latest, q.opm_latest, q.npm_latest,
            q.ebitda_margin, q.earnings_quality,
            cf.cfo_pat_latest, cf.positive_cfo_years, cf.cfroa,
            l.debt_equity, l.interest_coverage, l.current_ratio,
            l.financial_strength_score,
            g.revenue_growth_ttm, g.np_growth_ttm, g.eps_growth_3yr,
            g.growth_durability_score, g.eps, g.roic,
            sh.promoter_holding, sh.promoter_pledge, sh.institutional_holding,
            rf.red_flag_count, rf.quality_flag_count, rf.pricing_flag_count
        FROM stocks s
        LEFT JOIN analysis a ON s.nse_code = a.nse_code
        LEFT JOIN valuation v ON s.nse_code = v.nse_code
        LEFT JOIN quality q ON s.nse_code = q.nse_code
        LEFT JOIN cash_flow cf ON s.nse_code = cf.nse_code
        LEFT JOIN leverage l ON s.nse_code = l.nse_code
        LEFT JOIN growth g ON s.nse_code = g.nse_code
        LEFT JOIN shareholding sh ON s.nse_code = sh.nse_code
        LEFT JOIN red_flags rf ON s.nse_code = rf.nse_code
        WHERE s.nse_code IN ({placeholders})
        ORDER BY a.composite_score DESC
    """, nse_codes)
