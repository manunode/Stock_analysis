"""Industry detail route."""

from fastapi import APIRouter, Request, HTTPException
from app.main import templates
from app.database import query_all, query_one

router = APIRouter(tags=["pages"])


@router.get("/industry/{industry_name}")
def industry_detail(request: Request, industry_name: str):
    industry = query_one("""
        SELECT * FROM industry_summary WHERE industry = ?
    """, [industry_name])
    if industry is None:
        raise HTTPException(status_code=404, detail=f"Industry '{industry_name}' not found")

    # Get parent sector(s)
    parent_sectors = query_all("""
        SELECT DISTINCT sector FROM stocks WHERE industry = ?
    """, [industry_name])

    stocks = query_all("""
        SELECT s.nse_code, s.stock_name, s.sector, s.market_cap,
               a.composite_score, a.score_band, a.decision_bucket, a.quality_risk,
               v.pe, v.valuation_band,
               q.roe_latest, q.roce_latest
        FROM stocks s
        JOIN analysis a ON s.nse_code = a.nse_code
        JOIN valuation v ON s.nse_code = v.nse_code
        JOIN quality q ON s.nse_code = q.nse_code
        WHERE s.industry = ?
        ORDER BY a.composite_score DESC
    """, [industry_name])

    return templates.TemplateResponse("pages/industry_detail.html", {
        "request": request,
        "industry": industry,
        "stocks": stocks,
        "parent_sectors": parent_sectors,
    })
