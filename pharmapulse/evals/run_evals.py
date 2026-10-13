"""Evaluate the agent on golden business questions.

Metrics
-------
* **tool_accuracy**   – the agent called at least one of the expected tools
* **answer_accuracy** – every ground-truth fact (number or label) appears in the answer
* **error_rate**      – questions where a tool returned an error
* latency and token usage per question

Run ``python -m pharmapulse.evals.run_evals`` (mock provider by default in CI) or with
``--provider anthropic`` to measure the real model. Results are written to
``reports/eval_results.json`` and a markdown summary to ``reports/eval_summary.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pharmapulse.analytics import (
    forecasting,
    hcp_segmentation,
    market_share,
    patient_journey,
    promo_response,
)
from pharmapulse.db import query
from pharmapulse.llm.agent import Agent
from pharmapulse.llm.provider import get_provider

GOLDEN = Path(__file__).with_name("golden_questions.yaml")
Fact = float | int | str


# --------------------------------------------------------------------------- ground truth


def _latest_period() -> tuple[str, str]:
    lw = market_share.latest_week()
    return (lw - pd.Timedelta(weeks=12)).date().isoformat(), lw.date().isoformat()


def _share(brand: str, metric: str = "trx", by: str = "none", key: str | None = None) -> float:
    start, end = _latest_period()
    df = market_share.share(metric, by, start, end)
    if key is not None:
        df = df[df[by] == key]
    return float(df.loc[df["brand"] == brand, "share"].iloc[0])


def _persist(by: str, key: str) -> float:
    df = patient_journey.persistence(by=by)
    return float(df.loc[df[by] == key, "persistent_365d"].iloc[0])


def _growth(brand: str) -> float:
    g = market_share.growth()
    return float(g.loc[g["brand"] == brand, "yoy_growth"].iloc[0])


def _switch_top(brand: str) -> list[Fact]:
    m = patient_journey.switching_matrix()
    row = m[m["from_brand"] == brand].iloc[0]
    return [str(row["to_brand"]), int(row["patients"])]


TRUTH: dict[str, Callable[[], list[Fact]]] = {
    "share_latest_glyvance": lambda: ["Glyvance", _share("Glyvance")],
    "share_latest_glucora_nrx": lambda: [_share("Glucora", "nrx")],
    "share_southeast_glyvance": lambda: [_share("Glyvance", by="region", key="Southeast")],
    "share_endo_glyvance": lambda: [_share("Glyvance", by="specialty", key="Endocrinology")],
    "growth_glyvance_yoy": lambda: [_growth("Glyvance")],
    "growth_sitaglin_yoy": lambda: [_growth("Sitaglin")],
    "forecast_glyvance_13": lambda: [
        forecasting.forecast("Glyvance", 13).to_dict()["total_forecast"]
    ],
    "forecast_glucora_8": lambda: [forecasting.forecast("Glucora", 8).to_dict()["total_forecast"]],
    "forecast_glyvance_nrx_13": lambda: [
        forecasting.forecast("Glyvance", 13, "nrx").to_dict()["total_forecast"]
    ],
    "launch_glucora_latest": lambda: [
        float(market_share.launch_uptake("Glucora", "nrx", 52)["share"].iloc[-1])
    ],
    "persistence_glyvance_365": lambda: [_persist("brand", "Glyvance")],
    "persistence_young_365": lambda: [_persist("age_band", "18-44")],
    "persistence_sitaglin_365": lambda: [_persist("brand", "Sitaglin")],
    "adherence_glyvance": lambda: [
        float(v)
        for v in patient_journey.adherence()
        .set_index("brand")
        .loc["Glyvance", ["mean_pdc", "pct_adherent"]]
    ],
    "switch_glyvance_top": lambda: _switch_top("Glyvance"),
    "switch_sitaglin_top": lambda: _switch_top("Sitaglin")[:1],
    "target_top_hcp": lambda: [str(hcp_segmentation.next_best_targets(1)["hcp_id"].iloc[0])],
    "target_top_midwest": lambda: [
        str(hcp_segmentation.next_best_targets(1, region="Midwest")["hcp_id"].iloc[0])
    ],
    "segment_growth_targets": lambda: [
        int(
            hcp_segmentation.segment_summary()
            .set_index("segment")
            .loc["Growth targets - invest", "hcps"]
        )
    ],
    "decile10_share": lambda: [
        float(hcp_segmentation.decile_summary().set_index("decile").loc[10, "our_share"])
    ],
    "promo_per100": lambda: [round(promo_response.fit().incremental_nrx_per_100_calls, 2)],
    "sql_endo_count": lambda: [
        int(query("SELECT COUNT(*) n FROM hcps WHERE specialty='Endocrinology'")["n"].iloc[0])
    ],
    "sql_tier1_count": lambda: [
        int(
            query("SELECT COUNT(*) n FROM formulary WHERE brand='Glyvance' AND tier=1")["n"].iloc[0]
        )
    ],
    "sql_glucora_patients": lambda: [
        int(query("SELECT COUNT(*) n FROM patients WHERE index_brand='Glucora'")["n"].iloc[0])
    ],
}


# --------------------------------------------------------------------------- grading

_NUM = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([kKmM%]?)")


def extract_numbers(text: str) -> list[float]:
    out = []
    for raw, suffix in _NUM.findall(text):
        try:
            n = float(raw.replace(",", ""))
        except ValueError:
            continue
        if suffix in "kK" and suffix:
            n *= 1_000
        elif suffix in "mM" and suffix:
            n *= 1_000_000
        out.append(n)
    return out


def fact_present(fact: Fact, answer: str) -> bool:
    if isinstance(fact, str):
        return fact.lower() in answer.lower()
    nums = extract_numbers(answer)
    v = float(fact)
    if abs(v) <= 1.5 and not float(v).is_integer():  # a rate / share
        return any(abs(n - v) <= 0.0015 or abs(n - 100 * v) <= 0.15 for n in nums)
    tol = max(abs(v) * 0.005, 0.6)  # counts and volumes: 0.5% (allows "62.7K")
    return any(abs(n - v) <= tol for n in nums)


@dataclass
class EvalRow:
    id: str
    question: str
    expected_tools: list[str]
    tools_used: list[str]
    tool_ok: bool
    facts: list[Fact]
    facts_found: list[bool]
    answer_ok: bool
    tool_error: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    answer: str


def run(provider_name: str = "mock", only: list[str] | None = None) -> list[EvalRow]:
    cases = yaml.safe_load(GOLDEN.read_text())
    if only:
        cases = [c for c in cases if c["id"] in only]
    agent = Agent(get_provider(provider_name))
    rows = []
    for case in cases:
        res = agent.ask(case["question"])
        facts = TRUTH[case["truth"]]()
        found = [fact_present(f, res.answer) for f in facts]
        rows.append(
            EvalRow(
                id=case["id"],
                question=case["question"],
                expected_tools=case["expected_tools"],
                tools_used=res.tools_used,
                tool_ok=bool(set(case["expected_tools"]) & set(res.tools_used)),
                facts=facts,
                facts_found=found,
                answer_ok=all(found),
                tool_error=any(s.is_error for s in res.steps),
                latency_ms=res.latency_ms,
                input_tokens=res.usage.get("input_tokens", 0),
                output_tokens=res.usage.get("output_tokens", 0),
                answer=res.answer,
            )
        )
        mark = "PASS" if rows[-1].tool_ok and rows[-1].answer_ok else "FAIL"
        print(f"[{mark}] {case['id']:<20} tools={res.tools_used} facts={found}", flush=True)
    return rows


def summarise(rows: list[EvalRow], provider: str) -> dict[str, Any]:
    n = len(rows)
    return {
        "provider": provider,
        "questions": n,
        "tool_accuracy": round(sum(r.tool_ok for r in rows) / n, 4),
        "answer_accuracy": round(sum(r.answer_ok for r in rows) / n, 4),
        "pass_rate": round(sum(r.tool_ok and r.answer_ok for r in rows) / n, 4),
        "tool_error_rate": round(sum(r.tool_error for r in rows) / n, 4),
        "median_latency_ms": int(statistics.median(r.latency_ms for r in rows)),
        "total_input_tokens": sum(r.input_tokens for r in rows),
        "total_output_tokens": sum(r.output_tokens for r in rows),
    }


def write_reports(rows: list[EvalRow], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = summary["provider"]
    (out_dir / f"eval_results_{tag}.json").write_text(
        json.dumps({"summary": summary, "rows": [r.__dict__ for r in rows]}, indent=2, default=str)
    )
    lines = [
        f"# Eval summary ({tag})",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
        "| id | tool ok | answer ok | tools used |",
        "|---|---|---|---|",
        *[
            f"| {r.id} | {'✅' if r.tool_ok else '❌'} | {'✅' if r.answer_ok else '❌'} "
            f"| {', '.join(r.tools_used)} |"
            for r in rows
        ],
    ]
    (out_dir / f"eval_summary_{tag}.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--provider", default="mock", help="mock | anthropic | openai")
    p.add_argument("--only", nargs="*", help="run only these case ids")
    p.add_argument("--out", default="reports")
    p.add_argument("--min-pass-rate", type=float, default=0.0, help="exit 1 below this")
    a = p.parse_args(argv)
    rows = run(a.provider, a.only)
    summary = summarise(rows, a.provider)
    write_reports(rows, summary, Path(a.out))
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass_rate"] >= a.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
