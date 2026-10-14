"""Market share, growth and launch-uptake analytics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pharmapulse.db import query

GROUPINGS = {"region": "h.region", "specialty": "h.specialty", "none": None}


def _period_clause(start: str | None, end: str | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if start:
        clauses.append("r.week >= ?")
        params.append(start)
    if end:
        clauses.append("r.week <= ?")
        params.append(end)
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def latest_week() -> pd.Timestamp:
    return pd.Timestamp(query("SELECT MAX(week) AS w FROM weekly_rx")["w"].iloc[0])


def share(
    metric: str = "trx",
    by: str = "none",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Share of market by brand, optionally split by region or specialty."""
    if metric not in {"trx", "nrx"}:
        raise ValueError("metric must be 'trx' or 'nrx'")
    if by not in GROUPINGS:
        raise ValueError(f"by must be one of {list(GROUPINGS)}")
    col = GROUPINGS[by]
    where, params = _period_clause(start, end)
    group = f"{col} AS {by}, " if col else ""
    partition = f"PARTITION BY {col}" if col else ""
    sql = f"""
        SELECT {group} r.brand,
               SUM(r.{metric}) AS volume,
               SUM(r.{metric}) / SUM(SUM(r.{metric})) OVER ({partition}) AS share
        FROM weekly_rx r JOIN hcps h USING (hcp_id)
        WHERE 1=1 {where}
        GROUP BY {"1, 2" if col else "1"}
        ORDER BY {"1, share DESC" if col else "share DESC"}
    """
    df = query(sql, params)
    df["share"] = df["share"].round(4)
    return df


def share_trend(metric: str = "trx", freq: str = "month") -> pd.DataFrame:
    """Share by brand over time (``freq`` = week | month | quarter)."""
    if freq not in {"week", "month", "quarter"}:
        raise ValueError("freq must be week, month or quarter")
    sql = f"""
        WITH n AS (
            SELECT date_trunc('{freq}', week) AS period, COUNT(DISTINCT week) AS weeks
            FROM weekly_rx GROUP BY 1
        )
        SELECT date_trunc('{freq}', r.week) AS period, r.brand, SUM(r.{metric}) AS volume,
               ANY_VALUE(n.weeks) AS weeks
        FROM weekly_rx r JOIN n ON date_trunc('{freq}', r.week) = n.period
        GROUP BY 1, 2 ORDER BY 1, 2
    """
    df = query(sql)
    # months hold 4 or 5 weeks; report a per-week rate so periods are comparable
    df["avg_weekly_volume"] = (df["volume"] / df["weeks"]).round(1)
    df["share"] = (df["volume"] / df.groupby("period")["volume"].transform("sum")).round(4)
    return df


def growth(metric: str = "trx", weeks: int = 13) -> pd.DataFrame:
    """Latest ``weeks`` vs prior ``weeks`` and vs same period last year, per brand."""
    if metric not in {"trx", "nrx"}:  # interpolated into SQL below
        raise ValueError("metric must be 'trx' or 'nrx'")
    lw = latest_week()
    cur_start = lw - pd.Timedelta(weeks=weeks - 1)
    prev_start = cur_start - pd.Timedelta(weeks=weeks)
    ly_start, ly_end = cur_start - pd.Timedelta(weeks=52), lw - pd.Timedelta(weeks=52)
    sql = f"""
        SELECT brand,
          SUM(CASE WHEN week BETWEEN ? AND ? THEN {metric} END) AS current,
          SUM(CASE WHEN week >= ? AND week < ? THEN {metric} END) AS prior_period,
          SUM(CASE WHEN week BETWEEN ? AND ? THEN {metric} END) AS same_period_last_year
        FROM weekly_rx GROUP BY 1 ORDER BY current DESC
    """
    df = query(sql, [cur_start, lw, prev_start, cur_start, ly_start, ly_end]).fillna(0)
    df["pop_growth"] = (df["current"] / df["prior_period"].where(df["prior_period"] > 0) - 1).round(
        4
    )
    df["yoy_growth"] = (
        df["current"] / df["same_period_last_year"].where(df["same_period_last_year"] > 0) - 1
    ).round(4)
    df["current_share"] = (df["current"] / df["current"].sum()).round(4)
    df.attrs["period"] = {"start": cur_start.date().isoformat(), "end": lw.date().isoformat()}
    return df


def launch_uptake(brand: str = "Glucora", metric: str = "nrx", weeks: int = 52) -> pd.DataFrame:
    """Weekly share since a brand's first prescription (launch curve)."""
    if metric not in {"trx", "nrx"}:  # interpolated into SQL below
        raise ValueError("metric must be 'trx' or 'nrx'")
    sql = f"""
        WITH first AS (SELECT MIN(week) AS launch FROM weekly_rx WHERE brand = ?),
        wk AS (
            SELECT week, SUM({metric}) AS market,
                   SUM(CASE WHEN brand = ? THEN {metric} ELSE 0 END) AS brand_volume
            FROM weekly_rx GROUP BY 1
        )
        SELECT CAST(datediff('week', first.launch, wk.week) AS INTEGER) AS weeks_since_launch,
               wk.week, wk.brand_volume, wk.brand_volume / wk.market AS share
        FROM wk, first
        WHERE wk.week >= first.launch
        ORDER BY 1 LIMIT ?
    """
    df = query(sql, [brand, brand, weeks])
    df["share"] = df["share"].round(4)
    return df
