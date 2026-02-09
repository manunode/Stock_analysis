"""Stock screener route with dynamic filtering."""

from fastapi import APIRouter, Request
from app.main import templates
from app.services.screener_service import get_screener_results, get_filter_options

router = APIRouter(tags=["pages"])


@router.get("/screener")
def screener(request: Request):
    params = dict(request.query_params)

    # Convert multi-value params (filter out empty strings from form defaults)
    multi_keys = ["bucket", "band", "risk", "valuation", "sector", "industry"]
    filters = {}
    for key in multi_keys:
        values = [v for v in request.query_params.getlist(key) if v]
        if values:
            filters[key] = values

    # Single value params
    single_keys = [
        "min_score", "max_score", "min_mcap", "max_mcap",
        "min_pe", "max_pe", "min_roe", "min_roce", "max_de",
        "min_promoter", "sort", "dir",
    ]
    for key in single_keys:
        val = params.get(key)
        if val:
            filters[key] = val

    # Boolean toggles
    for key in ["screen_eligible", "no_red_flags", "dividend_paying"]:
        if params.get(key):
            filters[key] = True

    results = get_screener_results(filters)
    options = get_filter_options()

    # Check if HTMX partial request
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("partials/screener_table.html", {
            "request": request,
            "results": results,
            "filters": filters,
        })

    return templates.TemplateResponse("pages/screener.html", {
        "request": request,
        "results": results,
        "options": options,
        "filters": filters,
    })
