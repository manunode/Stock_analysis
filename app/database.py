"""DuckDB read-only connection and query helpers."""

import duckdb
from app.config import DB_PATH

_connection = None


def get_db():
    """Get a read-only DuckDB connection (singleton)."""
    global _connection
    if _connection is None:
        _connection = duckdb.connect(DB_PATH, read_only=True)
    return _connection


def query_one(sql: str, params: list | None = None) -> dict | None:
    """Execute SQL and return one row as a dict, or None."""
    con = get_db()
    result = con.execute(sql, params or []).fetchone()
    if result is None:
        return None
    columns = [desc[0] for desc in con.description]
    return dict(zip(columns, result))


def query_all(sql: str, params: list | None = None) -> list[dict]:
    """Execute SQL and return all rows as a list of dicts."""
    con = get_db()
    result = con.execute(sql, params or []).fetchall()
    columns = [desc[0] for desc in con.description]
    return [dict(zip(columns, row)) for row in result]


def query_value(sql: str, params: list | None = None):
    """Execute SQL and return a single scalar value."""
    con = get_db()
    result = con.execute(sql, params or []).fetchone()
    return result[0] if result else None
