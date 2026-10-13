"""Thin DuckDB access layer shared by analytics, tools, API and dashboard."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pharmapulse.config import get_settings


@lru_cache(maxsize=4)
def _connect(path: str) -> duckdb.DuckDBPyConnection:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run `make data` (or `pharmapulse build-data`) first."
        )
    return duckdb.connect(path, read_only=True)


def connection(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    return _connect(str(path or get_settings().db_path))


def query(
    sql: str, params: list[Any] | None = None, path: str | Path | None = None
) -> pd.DataFrame:
    """Run SQL and return a DataFrame. Uses a cursor so callers can share the connection."""
    cur = connection(path).cursor()
    try:
        return cur.execute(sql, params or []).df()
    finally:
        cur.close()
