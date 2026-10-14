"""Provider-agnostic LLM interface.

The agent speaks one message format (Anthropic-style content blocks) and every provider
translates to and from it. That keeps the agent loop identical whether it runs on the
Anthropic API (default), OpenAI, or the deterministic ``MockProvider`` used in tests and CI.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pharmapulse.config import Settings, get_settings


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Turn:
    """One assistant response, normalised across providers."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "refusal" | "max_tokens" | ...
    # Exactly what to append to history as the assistant message content. For Anthropic this
    # is the raw response content (thinking blocks must be echoed back unchanged).
    content: list[Any]
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn: ...


# --------------------------------------------------------------------------- Anthropic


def _require_model(s: Settings, provider: str) -> str:
    if not s.model:
        raise ValueError(
            f"PHARMAPULSE_MODEL is not set; set it to the {provider} model id to use, "
            "or use PHARMAPULSE_PROVIDER=mock"
        )
    return s.model


class AnthropicProvider:
    """Anthropic Messages API via the official ``anthropic`` SDK."""

    name = "anthropic"

    def __init__(self, settings: Settings | None = None) -> None:
        import anthropic

        s = settings or get_settings()
        self.model = _require_model(s, "Anthropic")
        self.effort = s.effort
        self.fallbacks = s.fallbacks
        self.client = anthropic.Anthropic()

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": messages,
            "tools": tools,
            "output_config": {"effort": self.effort},
        }
        if self.fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
            resp = self.client.beta.messages.create(**kwargs)
        else:
            resp = self.client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        if resp.stop_reason == "refusal":
            text = text or "The model declined to answer this request."
        return Turn(
            text=text,
            tool_calls=calls,
            stop_reason=str(resp.stop_reason),
            content=list(resp.content),
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )


# --------------------------------------------------------------------------- OpenAI


def _block(b: Any, key: str, default: Any = None) -> Any:
    return b.get(key, default) if isinstance(b, dict) else getattr(b, key, default)


class OpenAIProvider:
    """Optional OpenAI backend (``pip install pharmapulse[openai]``)."""

    name = "openai"

    def __init__(self, settings: Settings | None = None) -> None:
        from openai import OpenAI

        s = settings or get_settings()
        self.model = os.environ.get("PHARMAPULSE_OPENAI_MODEL") or _require_model(s, "OpenAI")
        self.client = OpenAI()

    @staticmethod
    def _convert(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
                continue
            if m["role"] == "assistant":
                text = "".join(
                    _block(b, "text", "") for b in content if _block(b, "type") == "text"
                )
                calls = [
                    {
                        "id": _block(b, "id"),
                        "type": "function",
                        "function": {
                            "name": _block(b, "name"),
                            "arguments": json.dumps(_block(b, "input")),
                        },
                    }
                    for b in content
                    if _block(b, "type") == "tool_use"
                ]
                msg: dict[str, Any] = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            else:
                for b in content:
                    if _block(b, "type") == "tool_result":
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": b["tool_use_id"],
                                "content": b["content"],
                            }
                        )
                    elif _block(b, "type") == "text":
                        out.append({"role": "user", "content": b["text"]})
        return out

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        oa_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        resp = self.client.chat.completions.create(
            model=self.model, messages=self._convert(system, messages), tools=oa_tools
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, input=json.loads(c.function.arguments or "{}"))
            for c in (msg.tool_calls or [])
        ]
        content: list[Any] = []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        content += [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls
        ]
        return Turn(
            text=msg.content or "",
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            content=content,
        )


# --------------------------------------------------------------------------- Mock


@dataclass
class _Route:
    pattern: str
    tool: str
    args: dict[str, Any]


# Keyword router: maps a question to the tool a competent analyst would reach for.
# Used for offline demos, unit tests and CI evals of the *tooling* (not of model quality).
_ROUTES: list[_Route] = [
    _Route(r"forecast|next \d+ weeks|predict|projection", "forecast_brand", {}),
    _Route(r"launch|uptake|since launch", "launch_uptake", {}),
    _Route(r"persisten|discontinu|drop.?off|stay on", "patient_persistence", {}),
    _Route(r"adheren|\bpdc\b|proportion of days", "patient_adherence", {}),
    _Route(r"switch", "switching_matrix", {}),
    _Route(r"target|which hcps|call plan|next best", "next_best_targets", {}),
    _Route(r"segment", "hcp_segments", {}),
    _Route(r"decile", "decile_summary", {}),
    _Route(r"promo|rep call|calls? (drive|work|impact)|roi|elasticit", "promo_response", {}),
    _Route(r"grow|growth|yoy|year over year|trend", "brand_growth", {}),
    _Route(r"share|market", "market_share", {}),
]
_BRANDS = ["Glyvance", "Sitaglin", "Dapaflo", "Glucora"]
_REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
_SPECIALTIES = ["Endocrinology", "Primary Care", "Internal Medicine", "Nurse Practitioner"]


# A few canned SQL templates so the mock can exercise the run_sql path end-to-end.
_SQL_TEMPLATES: list[tuple[str, str]] = [
    (r"how many (\w[\w ]*?) hcps",
     "SELECT COUNT(*) AS hcps FROM hcps WHERE lower(specialty) = '{0}'"),
    (r"tier 1",
     "SELECT COUNT(*) AS region_quarters FROM formulary WHERE brand = '{brand}' AND tier = 1"),
    (r"patients started on",
     "SELECT COUNT(*) AS patients FROM patients WHERE index_brand = '{brand}'"),
]  # fmt: skip


def mock_route(question: str) -> tuple[str, dict[str, Any]]:
    """Deterministically choose a tool + arguments for a question."""
    q = question.lower()
    if "sql" in q:
        brand_ = next((b for b in _BRANDS if b.lower() in q), "Glyvance")
        for pattern, template in _SQL_TEMPLATES:
            m = re.search(pattern, q)
            if m:
                return "run_sql", {"sql": template.format(*m.groups(), brand=brand_)}
    brand = next((b for b in _BRANDS if b.lower() in q), None)
    region = next((r for r in _REGIONS if r.lower() in q), None)
    specialty = next((s for s in _SPECIALTIES if s.lower() in q), None)
    by = "region" if re.search(r"by region|each region|per region", q) else None
    by = by or ("specialty" if re.search(r"by specialty|each specialty|per specialty", q) else None)
    weeks = re.search(r"(\d+)\s*weeks?", q)
    metric = "nrx" if re.search(r"\bnrx\b|new (rx|prescription|patient)", q) else "trx"

    for r in _ROUTES:
        if re.search(r.pattern, q):
            args: dict[str, Any] = dict(r.args)
            if r.tool == "forecast_brand":
                args |= {
                    "brand": brand or "Glyvance",
                    "horizon_weeks": int(weeks.group(1)) if weeks else 13,
                    "metric": metric,
                }
            elif r.tool == "launch_uptake":
                args |= {"brand": brand or "Glucora"}
            elif r.tool == "patient_persistence":
                args |= {
                    "by": "age_band" if "age" in q else ("region" if by == "region" else "brand")
                }
            elif r.tool == "next_best_targets":
                args |= {"n": 10}
                if region:
                    args["region"] = region
                if specialty:
                    args["specialty"] = specialty
            elif r.tool in {"market_share", "brand_growth"}:
                args |= {"metric": metric}
                if r.tool == "market_share" and by:
                    args["by"] = by
            return r.tool, args
    return "run_sql", {
        "sql": "SELECT brand, SUM(trx) AS trx FROM weekly_rx GROUP BY 1 ORDER BY 2 DESC"
    }


class MockProvider:
    """Deterministic stand-in for an LLM: routes by keywords, then summarises tool output."""

    name = "mock"
    model = "mock-router-v1"

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            tool, args = mock_route(last["content"])
            call = ToolCall(id=f"mock_{len(messages)}", name=tool, input=args)
            content = [{"type": "tool_use", "id": call.id, "name": tool, "input": args}]
            return Turn(text="", tool_calls=[call], stop_reason="tool_use", content=content)
        # after tool results: produce a grounded summary of the first result
        result = next(
            b["content"]
            for b in last["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        )
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            payload = {"result": result}
        text = "Here is what the data shows (mock mode, no LLM call):\n\n" + _summarise(payload)
        return Turn(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            content=[{"type": "text", "text": text}],
        )


def _summarise(payload: dict[str, Any]) -> str:
    if "error" in payload:
        return f"The tool returned an error: {payload['error']}"
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        keys = list(rows[0].keys())
        head = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
        body = "\n".join("| " + " | ".join(str(r.get(k)) for k in keys) + " |" for r in rows[:10])
        note = f"\n\n{payload['note']}" if payload.get("note") else ""
        return head + body + note
    return "```json\n" + json.dumps(payload, indent=2, default=str)[:2000] + "\n```"


# --------------------------------------------------------------------------- factory


def _has_anthropic_credentials() -> bool:
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_PROFILE")
    )


def get_provider(name: str | None = None, settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    name = (name or s.provider).lower()
    if name == "auto":
        name = "anthropic" if _has_anthropic_credentials() and s.model else "mock"
    if name == "anthropic":
        return AnthropicProvider(s)
    if name == "openai":
        return OpenAIProvider(s)
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider {name!r} (anthropic | openai | mock | auto)")
