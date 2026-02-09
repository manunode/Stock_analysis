"""Side-by-side stock comparison route."""

from fastapi import APIRouter, Request, Query
from app.main import templates
from app.services.comparison_service import get_comparison_data

router = APIRouter(tags=["pages"])


@router.get("/compare")
def compare(request: Request, stocks: str = Query("")):
    nse_codes = [c.strip().upper() for c in stocks.split(",") if c.strip()] if stocks else []
    nse_codes = nse_codes[:5]  # Max 5 stocks

    comparison = get_comparison_data(nse_codes) if nse_codes else []

    return templates.TemplateResponse("pages/compare.html", {
        "request": request,
        "comparison": comparison,
        "nse_codes": nse_codes,
    })
