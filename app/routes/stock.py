"""Single stock detail route."""

from fastapi import APIRouter, Request, HTTPException
from app.main import templates
from app.services.stock_service import get_stock_profile, get_peer_stocks

router = APIRouter(tags=["pages"])


@router.get("/stock/{nse_code}")
def stock_detail(request: Request, nse_code: str):
    stock = get_stock_profile(nse_code.upper())
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Stock {nse_code} not found")

    peers = get_peer_stocks(stock["sector"], stock["nse_code"], limit=8)

    return templates.TemplateResponse("pages/stock_detail.html", {
        "request": request,
        "s": stock,
        "peers": peers,
    })
