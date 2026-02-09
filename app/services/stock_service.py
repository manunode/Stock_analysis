"""Assemble a complete stock profile by joining all tables."""

from app.database import query_one, query_all


def get_stock_profile(nse_code: str) -> dict | None:
    """Return the full stock profile as a single dict, or None if not found."""
    row = query_one("""
        SELECT
            s.nse_code, s.stock_name, s.sector, s.industry, s.bse_code, s.isin,
            s.market_cap, s.data_type, s.years_available, s.latest_year,
            s.fiscal_year_end, s.is_financial_sector,

            a.decision_bucket, a.screen_eligible, a.investment_thesis,
            a.reject_reason, a.composite_score, a.sector_relative_adj,
            a.score_band, a.quality_risk, a.critical_flags, a.major_flags,
            a.primary_concern, a.conviction_override,

            v.pe, v.pe_ttm, v.pbv, v.ev_ebitda, v.peg, v.price_to_sales,
            v.earnings_yield, v.valuation_band, v.valuation_comfort_score,
            v.enterprise_value, v.ltp, v.week52_high, v.week52_low,
            v.price_position_52w, v.return_1yr, v.returns_vs_nifty50_qtr,
            v.returns_vs_sector_qtr, v.return_vs_nifty_1yr,

            q.business_quality_score, q.earnings_quality, q.piotroski_score,
            q.piotroski_assessment, q.roe_latest, q.roe_3yr_avg, q.roe_5yr_avg,
            q.roe_trend, q.roce_latest, q.roce_3yr_avg, q.roce_5yr_avg,
            q.roce_trend, q.roa_latest, q.opm_latest, q.opm_ttm, q.opm_trend,
            q.npm_latest, q.npm_trend, q.ebitda_margin, q.leverage_driven,
            q.exceptional_items_count, q.other_income_pct_pat,

            cf.cfo_latest, cf.pat_latest, cf.cfo_pat_latest, cf.cfo_pat_3yr_avg,
            cf.positive_cfo_years, cf.cfo_trend, cf.cfroa, cf.accruals, cf.ceps,

            l.debt_equity, l.lt_debt_equity, l.net_debt_ebitda,
            l.interest_coverage, l.debt_trend, l.financial_strength_score,
            l.current_ratio, l.quick_ratio, l.total_debt, l.total_equity,
            l.total_assets, l.asset_turnover, l.receivable_days,
            l.recv_days_trend, l.inventory_days, l.inv_days_trend,
            l.payables_days, l.cash_conversion_cycle,

            g.revenue_growth_ttm, g.np_growth_ttm, g.revenue_growth_qtr_yoy,
            g.np_growth_qtr_yoy, g.revenue_growth_1yr, g.revenue_cagr_3yr,
            g.np_growth, g.eps_growth_3yr, g.revenue_volatility,
            g.profit_consistency, g.profit_growth_consistency,
            g.growth_durability_score, g.total_revenue,
            g.book_value_per_share, g.eps, g.roic,

            sh.promoter_holding, sh.promoter_pledge, sh.pledge_risk,
            sh.promoter_change_1yr, sh.fii_holding, sh.fii_change_1yr,
            sh.mf_holding, sh.mf_change_1yr, sh.institutional_holding,
            sh.public_holding, sh.insider_action, sh.insider_sentiment,
            sh.insider_context,

            d.dividend_count, d.latest_dividend_date, d.latest_dividend_amount,
            d.latest_dividend_type, d.dividend_5yr_consistency,
            d.dividend_trend, d.total_dividends_paid,
            d.bonus_count, d.split_count, d.rights_count,

            nf.generic_stock_candidate, nf.neglect_score, nf.neglect_age_warning,

            rf.quality_risk AS rf_quality_risk, rf.quality_severity,
            rf.quality_flag_count, rf.pricing_flag_count, rf.red_flag_count,
            rf.flag_low_roe, rf.flag_declining_roe,
            rf.flag_low_roce, rf.flag_declining_roce,
            rf.flag_poor_cash_conversion, rf.flag_negative_cfo,
            rf.flag_inconsistent_cfo, rf.flag_frequent_exceptionals,
            rf.flag_high_other_income, rf.flag_rising_receivables,
            rf.flag_rising_inventory, rf.flag_margin_compression,
            rf.flag_rising_debt, rf.flag_wc_divergence,
            rf.flag_npm_opm_divergence,
            rf.flag_high_pe, rf.flag_negative_pe,
            rf.flag_high_ev_ebitda, rf.flag_negative_ebitda,
            rf.flag_high_pbv_roe,

            sec.avg_pe AS sector_avg_pe, sec.avg_peg AS sector_avg_peg,
            sec.avg_pbv AS sector_avg_pbv, sec.avg_roe AS sector_avg_roe,
            sec.avg_roce AS sector_avg_roce, sec.avg_roa AS sector_avg_roa,
            sec.avg_market_cap AS sector_avg_market_cap,

            ind.avg_pe AS industry_avg_pe, ind.avg_peg AS industry_avg_peg,
            ind.avg_pbv AS industry_avg_pbv, ind.avg_roe AS industry_avg_roe,
            ind.avg_roce AS industry_avg_roce, ind.avg_roa AS industry_avg_roa,
            ind.avg_market_cap AS industry_avg_market_cap

        FROM stocks s
        LEFT JOIN analysis a ON s.nse_code = a.nse_code
        LEFT JOIN valuation v ON s.nse_code = v.nse_code
        LEFT JOIN quality q ON s.nse_code = q.nse_code
        LEFT JOIN cash_flow cf ON s.nse_code = cf.nse_code
        LEFT JOIN leverage l ON s.nse_code = l.nse_code
        LEFT JOIN growth g ON s.nse_code = g.nse_code
        LEFT JOIN shareholding sh ON s.nse_code = sh.nse_code
        LEFT JOIN dividends d ON s.nse_code = d.nse_code
        LEFT JOIN neglected_firm nf ON s.nse_code = nf.nse_code
        LEFT JOIN red_flags rf ON s.nse_code = rf.nse_code
        LEFT JOIN sector_summary sec ON s.sector = sec.sector
        LEFT JOIN industry_summary ind ON s.industry = ind.industry
        WHERE s.nse_code = ?
    """, [nse_code])
    return row


def get_peer_stocks(sector: str, nse_code: str, limit: int = 10) -> list[dict]:
    """Return top peer stocks in the same sector, excluding the current stock."""
    return query_all("""
        SELECT s.nse_code, s.stock_name, s.market_cap,
               a.composite_score, a.score_band, v.pe, v.pbv,
               q.roe_latest, q.roce_latest
        FROM stocks s
        JOIN analysis a ON s.nse_code = a.nse_code
        JOIN valuation v ON s.nse_code = v.nse_code
        JOIN quality q ON s.nse_code = q.nse_code
        WHERE s.sector = ? AND s.nse_code != ?
        ORDER BY a.composite_score DESC
        LIMIT ?
    """, [sector, nse_code, limit])
