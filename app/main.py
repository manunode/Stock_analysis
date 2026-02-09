"""FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app import config

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title=config.APP_TITLE)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ── Jinja2 global context ────────────────────────────────────────────────────
def _fmt(value, decimals=1, fallback="--"):
    """Format a numeric value for display."""
    if value is None:
        return fallback
    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return fallback


def _pct(value, decimals=1, fallback="--"):
    """Format as percentage."""
    if value is None:
        return fallback
    try:
        return f"{float(value):,.{decimals}f}%"
    except (ValueError, TypeError):
        return fallback


def _sign_class(value, invert=False):
    """Return CSS class based on sign. invert=True means lower is better (e.g., PE)."""
    if value is None:
        return "text-gray-400"
    try:
        v = float(value)
    except (ValueError, TypeError):
        return "text-gray-400"
    if invert:
        v = -v
    if v > 0:
        return "text-green-600"
    elif v < 0:
        return "text-red-600"
    return "text-gray-500"


def _trend_arrow(value):
    """Return trend arrow character."""
    if value is None:
        return "--"
    try:
        v = float(value)
    except (ValueError, TypeError):
        return "--"
    if v > 0.5:
        return "↑"
    elif v < -0.5:
        return "↓"
    return "→"


templates.env.globals.update(
    APP_TITLE=config.APP_TITLE,
    BAND_COLORS=config.BAND_COLORS,
    BUCKET_COLORS=config.BUCKET_COLORS,
    BUCKET_LABELS=config.BUCKET_LABELS,
    RISK_COLORS=config.RISK_COLORS,
    VALUATION_COLORS=config.VALUATION_COLORS,
    FLAG_LABELS=config.FLAG_LABELS,
    QUALITY_FLAGS=config.QUALITY_FLAGS,
    PRICING_FLAGS=config.PRICING_FLAGS,
)

templates.env.filters["fmt"] = _fmt
templates.env.filters["pct"] = _pct
templates.env.filters["sign_class"] = _sign_class
templates.env.filters["trend_arrow"] = _trend_arrow


# ── Register routes ──────────────────────────────────────────────────────────
from app.routes import home, stock, screener, sector, industry, compare, api  # noqa: E402

app.include_router(home.router)
app.include_router(stock.router)
app.include_router(screener.router)
app.include_router(sector.router)
app.include_router(industry.router)
app.include_router(compare.router)
app.include_router(api.router)
