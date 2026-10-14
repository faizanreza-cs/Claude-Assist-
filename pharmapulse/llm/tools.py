"""Tools the agent can call. Each wraps a tested analytics function.

Every tool returns JSON with ``rows`` (records) and optional ``note`` / ``meta`` so the
model can cite exact figures, and the same payload can be checked by the eval harness.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import pandas as pd

from pharmapulse.analytics import (
    forecasting,
    hcp_segmentation,
    market_share,
    patient_journey,
    promo_response,
)
from pharmapulse.config import get_settings
from pharmapulse.db import connection
from pharmapulse.llm.sql_guard import validate

BRANDS = ["Glyvance", "Sitaglin", "Dapaflo", "Glucora"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
SPECIALTIES = ["Endocrinology", "Primary Care", "Internal Medicine", "Nurse Practitioner"]

_brand = {"type": "string", "enum": BRANDS}
_metric = {
    "type": "string",
    "enum": ["trx", "nrx"],
    "description": "trx = total prescriptions, nrx = new prescriptions",
}


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.date.astype(str)
    return json.loads(out.to_json(orient="records", date_format="iso"))


def _ok(rows: pd.DataFrame | list[dict[str, Any]], note: str | None = None, **meta: Any) -> dict:
    payload: dict[str, Any] = {"rows": rows if isinstance(rows, list) else _records(rows)}
    if note:
        payload["note"] = note
    if meta:
        payload["meta"] = meta
    return payload


# --------------------------------------------------------------------------- implementations


def t_market_share(metric: str = "trx", by: str = "none", start: str | None = None,
                   end: str | None = None) -> dict:  # fmt: skip
    if not start and not end:
        lw = market_share.latest_week()
        start = (lw - pd.Timedelta(weeks=12)).date().isoformat()
        end = lw.date().isoformat()
    df = market_share.share(metric, by, start, end)
    return _ok(
        df, note=f"{metric.upper()} share of market, {start} to {end}.", start=start, end=end
    )


def t_brand_growth(metric: str = "trx", weeks: int = 13) -> dict:
    df = market_share.growth(metric, weeks)
    return _ok(df, note=f"Latest {weeks} weeks vs prior {weeks} weeks (pop) and vs last year (yoy).",
               **df.attrs["period"])  # fmt: skip


def t_forecast_brand(brand: str, horizon_weeks: int = 13, metric: str = "trx",
                     model: str = "auto") -> dict:  # fmt: skip
    bt = forecasting.backtest(brand, metric, horizon=13)
    if model == "auto":  # the same 13-week backtest the note and table report
        model = str(bt.sort_values("wape").iloc[0]["model"])
    fc = forecasting.forecast(brand, horizon_weeks, metric, model)
    d = fc.to_dict()
    weeks = d.pop("weeks")
    return _ok(
        weeks,
        note=(f"{d['model']} model selected by 13-week holdout WAPE. Intervals are 80%. "
              f"Total forecast {d['total_forecast']} vs last-13-week actual {d['last_13w_actual']}."),
        summary=d,
        backtest=_records(bt),
    )  # fmt: skip


def t_launch_uptake(brand: str = "Glucora", metric: str = "nrx", weeks: int = 52) -> dict:
    df = market_share.launch_uptake(brand, metric, weeks)
    if df.empty:
        return _ok(df, note=f"No prescriptions found for {brand}.")
    latest = df.iloc[-1]
    return _ok(df.iloc[::4] if len(df) > 20 else df,
               note=(f"{metric.upper()} share by weeks since launch (every 4th week shown). "
                     f"Latest: week {int(latest['weeks_since_launch'])} since launch, share "
                     f"{float(latest['share']):.4f}."),
               latest_share=float(latest["share"]))  # fmt: skip


def t_patient_persistence(by: str = "brand", gap_days: int = 60) -> dict:
    df = patient_journey.persistence(by=by, gap_days=gap_days)
    return _ok(df, note=f"Kaplan-Meier persistence on index therapy; discontinuation = gap > "
                        f"{gap_days} days or switch.")  # fmt: skip


def t_patient_adherence(window_days: int = 365) -> dict:
    df = patient_journey.adherence(window_days)
    return _ok(df, note=f"PDC over {window_days} days from index; adherent = PDC >= 0.8.")


def t_switching_matrix(gap_days: int = 60) -> dict:
    return _ok(patient_journey.switching_matrix(gap_days),
               note="Switches out of index brand within the persistence gap window.")  # fmt: skip


def t_next_best_targets(
    n: int = 10, region: str | None = None, specialty: str | None = None
) -> dict:
    df = hcp_segmentation.next_best_targets(max(1, min(n, 50)), region, specialty)
    return _ok(df, note="Score = Glyvance headroom TRx (last 52w) discounted by existing call "
                        "intensity.")  # fmt: skip


def t_hcp_segments() -> dict:
    return _ok(hcp_segmentation.segment_summary(),
               note="Value (deciles 6-10) x loyalty (share >= national share) segmentation.")  # fmt: skip


def t_decile_summary() -> dict:
    return _ok(hcp_segmentation.decile_summary(),
               note="Volume deciles on last-52-week market TRx; decile 10 = top 10% of volume.")  # fmt: skip


def t_promo_response(decay: float = 0.5) -> dict:
    r = promo_response.fit(decay).to_dict()
    return _ok([r], note=str(r["caveat"]))


def t_run_sql(sql: str) -> dict:
    s = get_settings()
    safe = validate(sql, s.sql_row_limit)
    cur = connection().cursor()
    timer = threading.Timer(10.0, cur.interrupt)
    timer.start()
    try:
        df = cur.execute(safe).df()
    finally:
        timer.cancel()
        cur.close()
    return _ok(df, executed_sql=safe, row_count=len(df))


# --------------------------------------------------------------------------- registry

TOOLS: dict[str, tuple[Callable[..., dict], dict[str, Any]]] = {
    "market_share": (t_market_share, {
        "description": "Share of market by brand. Defaults to the latest 13 weeks. Use `by` to "
                       "split by region or specialty, or start/end (YYYY-MM-DD) for a custom period.",
        "input_schema": {"type": "object", "properties": {
            "metric": _metric,
            "by": {"type": "string", "enum": ["none", "region", "specialty"]},
            "start": {"type": "string", "description": "inclusive start week, YYYY-MM-DD"},
            "end": {"type": "string", "description": "inclusive end week, YYYY-MM-DD"},
        }},
    }),
    "brand_growth": (t_brand_growth, {
        "description": "Volume, period-over-period and year-over-year growth and current share "
                       "for every brand over the latest N weeks.",
        "input_schema": {"type": "object", "properties": {
            "metric": _metric,
            "weeks": {"type": "integer", "minimum": 4, "maximum": 52},
        }},
    }),
    "forecast_brand": (t_forecast_brand, {
        "description": "Forecast weekly TRx or NRx for a brand with 80% intervals. Auto-selects "
                       "the best of seasonal-naive / ETS / gradient boosting by holdout WAPE and "
                       "returns the backtest table.",
        "input_schema": {"type": "object", "properties": {
            "brand": _brand,
            "horizon_weeks": {"type": "integer", "minimum": 1, "maximum": 52},
            "metric": _metric,
            "model": {"type": "string", "enum": ["auto", "seasonal_naive", "ets", "gbm"]},
        }, "required": ["brand"]},
    }),
    "launch_uptake": (t_launch_uptake, {
        "description": "Launch curve: a brand's weekly share by weeks since its first Rx.",
        "input_schema": {"type": "object", "properties": {
            "brand": _brand, "metric": _metric,
            "weeks": {"type": "integer", "minimum": 4, "maximum": 104},
        }},
    }),
    "patient_persistence": (t_patient_persistence, {
        "description": "Kaplan-Meier persistence at 90/180/365 days and median time on therapy, "
                       "from pharmacy claims, grouped by brand, age band or region.",
        "input_schema": {"type": "object", "properties": {
            "by": {"type": "string", "enum": ["brand", "age_band", "region"]},
            "gap_days": {"type": "integer", "minimum": 15, "maximum": 180},
        }},
    }),
    "patient_adherence": (t_patient_adherence, {
        "description": "Adherence by index brand: mean proportion of days covered (PDC) and "
                       "% of patients with PDC >= 0.8.",
        "input_schema": {"type": "object", "properties": {
            "window_days": {"type": "integer", "minimum": 90, "maximum": 365},
        }},
    }),
    "switching_matrix": (t_switching_matrix, {
        "description": "Patient switching flows between brands (from -> to, counts and % of "
                       "index patients).",
        "input_schema": {"type": "object", "properties": {
            "gap_days": {"type": "integer", "minimum": 15, "maximum": 180},
        }},
    }),
    "next_best_targets": (t_next_best_targets, {
        "description": "Rank prescribers (HCPs) for our brand Glyvance by growth headroom, "
                       "optionally filtered by region and specialty.",
        "input_schema": {"type": "object", "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 50},
            "region": {"type": "string", "enum": REGIONS},
            "specialty": {"type": "string", "enum": SPECIALTIES},
        }},
    }),
    "hcp_segments": (t_hcp_segments, {
        "description": "HCP value x loyalty segments for Glyvance with counts, volume, median "
                       "share and call intensity.",
        "input_schema": {"type": "object", "properties": {}},
    }),
    "decile_summary": (t_decile_summary, {
        "description": "HCP volume deciles: HCP counts, market and Glyvance TRx, share and calls "
                       "per HCP by decile.",
        "input_schema": {"type": "object", "properties": {}},
    }),
    "promo_response": (t_promo_response, {
        "description": "Estimated response of Glyvance NRx to rep calls (adstocked elasticity, "
                       "95% CI, incremental NRx per 100 calls). Observational.",
        "input_schema": {"type": "object", "properties": {
            "decay": {"type": "number", "minimum": 0, "maximum": 0.9},
        }},
    }),
    "run_sql": (t_run_sql, {
        "description": "Run one read-only DuckDB SELECT for questions no other tool covers. "
                       "Tables: brands(brand, molecule_class, manufacturer, is_our_brand, "
                       "list_price_per_30d); regions(region, population_index); hcps(hcp_id, "
                       "specialty, region, potential_trx_per_week); weekly_rx(hcp_id, week, brand, "
                       "trx, nrx); promo(hcp_id, week, brand, calls, samples); formulary(region, "
                       "brand, year, quarter, tier 1=preferred..3=restricted); patients(patient_id, "
                       "age_band, region, index_date, index_brand); claims(claim_id, patient_id, "
                       "brand, fill_date, days_supply). Rows are capped.",
        "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}},
                         "required": ["sql"]},
    }),
}  # fmt: skip


def tool_definitions() -> list[dict[str, Any]]:
    return [{"name": name, **spec} for name, (_, spec) in TOOLS.items()]


def execute(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run a tool; returns (json_string, is_error). Errors are returned to the model, not raised."""
    if name not in TOOLS:
        return json.dumps({"error": f"unknown tool {name!r}"}), True
    fn, _ = TOOLS[name]
    try:
        return json.dumps(fn(**args), default=str), False
    except Exception as e:  # duckdb errors, interrupts, etc.
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
