"""Methodology page route."""

from fastapi import APIRouter, Request
from app.main import templates

router = APIRouter(tags=["pages"])


@router.get("/methodology")
def methodology(request: Request):
    return templates.TemplateResponse("pages/methodology.html", {
        "request": request,
    })
