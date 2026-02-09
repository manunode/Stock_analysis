"""JSON API endpoints for search and data access."""

from fastapi import APIRouter, Query
from app.database import query_all

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/search")
def search_stocks(q: str = Query("", min_length=1)):
    """Typeahead search for stocks by NSE code or name."""
    results = query_all("""
        SELECT s.nse_code, s.stock_name, s.sector, s.market_cap,
               a.composite_score, a.score_band
        FROM stocks s
        LEFT JOIN analysis a ON s.nse_code = a.nse_code
        WHERE s.nse_code ILIKE ? OR s.stock_name ILIKE ?
        ORDER BY s.market_cap DESC NULLS LAST
        LIMIT 12
    """, [f"{q}%", f"%{q}%"])
    return results
