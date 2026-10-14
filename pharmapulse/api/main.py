"""REST API for PharmaPulse. Run: ``uvicorn pharmapulse.api.main:app --reload``."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from pharmapulse import __version__
from pharmapulse.config import get_settings
from pharmapulse.llm import tools
from pharmapulse.llm.agent import Agent
from pharmapulse.llm.provider import get_provider

app = FastAPI(
    title="PharmaPulse API",
    version=__version__,
    description="Commercial analytics and an LLM copilot for a (synthetic) type 2 diabetes market.",
)


class Brand(StrEnum):
    glyvance = "Glyvance"
    sitaglin = "Sitaglin"
    dapaflo = "Dapaflo"
    glucora = "Glucora"


class Metric(StrEnum):
    trx = "trx"
    nrx = "nrx"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def _run(name: str, **kwargs: Any) -> dict[str, Any]:
    import json

    out, is_error = tools.execute(name, {k: v for k, v in kwargs.items() if v is not None})
    payload = json.loads(out)
    if is_error:
        raise HTTPException(status_code=400, detail=payload["error"])
    return payload


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "db": str(s.db_path),
        "provider": get_provider().name,
    }


@app.get("/share")
def share(
    metric: Metric = Metric.trx,
    by: str = Query("none", pattern="^(none|region|specialty)$"),
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    return _run("market_share", metric=metric.value, by=by, start=start, end=end)


@app.get("/growth")
def growth(metric: Metric = Metric.trx, weeks: int = Query(13, ge=4, le=52)) -> dict[str, Any]:
    return _run("brand_growth", metric=metric.value, weeks=weeks)


@app.get("/forecast/{brand}")
def forecast(
    brand: Brand,
    horizon: int = Query(13, ge=1, le=52),
    metric: Metric = Metric.trx,
    model: str = Query("auto", pattern="^(auto|seasonal_naive|ets|gbm)$"),
) -> dict[str, Any]:
    return _run(
        "forecast_brand", brand=brand.value, horizon_weeks=horizon, metric=metric.value, model=model
    )


@app.get("/persistence")
def persistence(
    by: str = Query("brand", pattern="^(brand|age_band|region)$"),
    gap_days: int = Query(60, ge=15, le=180),
) -> dict[str, Any]:
    return _run("patient_persistence", by=by, gap_days=gap_days)


@app.get("/targets")
def targets(
    n: int = Query(25, ge=1, le=50), region: str | None = None, specialty: str | None = None
) -> dict[str, Any]:
    return _run("next_best_targets", n=n, region=region, specialty=specialty)


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    # the provider comes from server config (PHARMAPULSE_PROVIDER), never from the caller
    return Agent(get_provider()).ask(req.question).to_dict()
