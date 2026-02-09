"""Sector overview and detail routes."""

from fastapi import APIRouter, Request, HTTPException
from app.main import templates
from app.database import query_all, query_one

router = APIRouter(tags=["pages"])


@router.get("/sectors")
def sector_list(request: Request):
    sectors = query_all("""
        SELECT sector, num_companies, avg_market_cap, avg_pe, avg_peg,
               avg_pbv, avg_roe, avg_roce, avg_roa
        FROM sector_summary
        ORDER BY num_companies DESC
    """)
    return templates.TemplateResponse("pages/sector_list.html", {
        "request": request,
        "sectors": sectors,
    })


@router.get("/sector/{sector_name}")
def sector_detail(request: Request, sector_name: str):
    sector = query_one("""
        SELECT * FROM sector_summary WHERE sector = ?
    """, [sector_name])
    if sector is None:
        raise HTTPException(status_code=404, detail=f"Sector '{sector_name}' not found")

    stocks = query_all("""
        SELECT s.nse_code, s.stock_name, s.industry, s.market_cap,
               a.composite_score, a.score_band, a.decision_bucket, a.quality_risk,
               v.pe, v.valuation_band,
               q.roe_latest, q.roce_latest
        FROM stocks s
        JOIN analysis a ON s.nse_code = a.nse_code
        JOIN valuation v ON s.nse_code = v.nse_code
        JOIN quality q ON s.nse_code = q.nse_code
        WHERE s.sector = ?
        ORDER BY a.composite_score DESC
    """, [sector_name])

    industries = query_all("""
        SELECT i.industry, i.num_companies, i.avg_market_cap, i.avg_pe,
               i.avg_roe, i.avg_roce
        FROM industry_summary i
        JOIN (SELECT DISTINCT industry FROM stocks WHERE sector = ?) si
            ON i.industry = si.industry
        ORDER BY i.num_companies DESC
    """, [sector_name])

    # Bucket distribution for this sector
    bucket_dist = query_all("""
        SELECT a.decision_bucket, COUNT(*) as cnt
        FROM analysis a JOIN stocks s ON a.nse_code = s.nse_code
        WHERE s.sector = ?
        GROUP BY a.decision_bucket
        ORDER BY cnt DESC
    """, [sector_name])

    return templates.TemplateResponse("pages/sector_detail.html", {
        "request": request,
        "sector": sector,
        "stocks": stocks,
        "industries": industries,
        "bucket_dist": bucket_dist,
    })
