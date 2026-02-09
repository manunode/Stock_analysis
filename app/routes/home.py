"""Home dashboard route."""

from fastapi import APIRouter, Request
from app.main import templates
from app.database import query_all, query_value

router = APIRouter(tags=["pages"])


@router.get("/")
def dashboard(request: Request):
    # Decision bucket distribution
    buckets = query_all("""
        SELECT decision_bucket, COUNT(*) as cnt
        FROM analysis
        GROUP BY decision_bucket
        ORDER BY cnt DESC
    """)

    # Score band distribution
    bands = query_all("""
        SELECT score_band, COUNT(*) as cnt
        FROM analysis
        WHERE score_band IS NOT NULL
        GROUP BY score_band
        ORDER BY score_band
    """)

    # Quality risk distribution
    risks = query_all("""
        SELECT quality_risk, COUNT(*) as cnt
        FROM red_flags
        GROUP BY quality_risk
        ORDER BY cnt DESC
    """)

    # Top opportunities (GATES_CLEARED, sorted by score)
    top_stocks = query_all("""
        SELECT s.nse_code, s.stock_name, s.sector, s.market_cap,
               a.composite_score, a.score_band, a.quality_risk,
               v.pe, v.valuation_band, v.return_1yr,
               q.roe_latest, q.roce_latest
        FROM stocks s
        JOIN analysis a ON s.nse_code = a.nse_code
        JOIN valuation v ON s.nse_code = v.nse_code
        JOIN quality q ON s.nse_code = q.nse_code
        WHERE a.decision_bucket = 'GATES_CLEARED'
        ORDER BY a.composite_score DESC
        LIMIT 25
    """)

    # Sector summary for heatmap
    sectors = query_all("""
        SELECT sector, num_companies, avg_market_cap, avg_roe, avg_pe
        FROM sector_summary
        ORDER BY num_companies DESC
    """)

    total_stocks = query_value("SELECT COUNT(*) FROM stocks")

    return templates.TemplateResponse("pages/home.html", {
        "request": request,
        "buckets": buckets,
        "bands": bands,
        "risks": risks,
        "top_stocks": top_stocks,
        "sectors": sectors,
        "total_stocks": total_stocks,
    })
