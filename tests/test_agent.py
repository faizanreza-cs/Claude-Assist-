import json

from pharmapulse.llm import tools
from pharmapulse.llm.agent import Agent
from pharmapulse.llm.provider import MockProvider, ToolCall, Turn, mock_route


def test_tool_definitions_are_valid_schemas():
    defs = tools.tool_definitions()
    names = {d["name"] for d in defs}
    assert {"market_share", "forecast_brand", "run_sql"} <= names
    for d in defs:
        assert d["input_schema"]["type"] == "object"
        assert len(d["description"]) > 20


def test_every_tool_executes_with_defaults():
    defaults = {
        "forecast_brand": {"brand": "Glyvance", "horizon_weeks": 4},
        "run_sql": {"sql": "SELECT COUNT(*) n FROM hcps"},
    }
    for name in tools.TOOLS:
        out, is_error = tools.execute(name, defaults.get(name, {}))
        assert not is_error, (name, out)
        assert "rows" in json.loads(out)


def test_tool_errors_are_returned_not_raised():
    out, is_error = tools.execute("run_sql", {"sql": "DROP TABLE hcps"})
    assert is_error and "UnsafeSQLError" in json.loads(out)["error"]
    out, is_error = tools.execute("nope", {})
    assert is_error


def test_mock_routes():
    assert mock_route("Forecast Glucora TRx for 6 weeks") == (
        "forecast_brand",
        {"brand": "Glucora", "horizon_weeks": 6, "metric": "trx"},
    )
    assert mock_route("Show share by region")[1]["by"] == "region"
    assert mock_route("which brand do switchers go to")[0] == "switching_matrix"


def test_agent_end_to_end_with_mock():
    res = Agent(MockProvider()).ask("What is the market share by brand?")
    assert res.tools_used == ["market_share"]
    assert "Glyvance" in res.answer
    assert res.stop_reason == "end_turn"


class ScriptedProvider:
    """Calls two tools in parallel, then answers. Checks the loop's message bookkeeping."""

    name, model = "scripted", "scripted"

    def __init__(self):
        self.seen = []

    def complete(self, system, messages, tools_):
        self.seen.append(messages[-1])
        if len(messages) == 1:
            calls = [
                ToolCall("a", "brand_growth", {}),
                ToolCall("b", "run_sql", {"sql": "DROP TABLE x"}),
            ]
            content = [
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls
            ]
            return Turn("", calls, "tool_use", content)
        return Turn("done", [], "end_turn", [{"type": "text", "text": "done"}])


def test_parallel_tool_results_returned_in_one_message():
    p = ScriptedProvider()
    res = Agent(p).ask("q")
    results_msg = p.seen[-1]
    assert results_msg["role"] == "user"
    assert [b["tool_use_id"] for b in results_msg["content"]] == ["a", "b"]
    assert results_msg["content"][1]["is_error"] is True
    assert res.answer == "done" and len(res.steps) == 2


class LoopingProvider:
    name, model = "loop", "loop"

    def complete(self, system, messages, tools_):
        c = ToolCall(f"id{len(messages)}", "hcp_segments", {})
        return Turn(
            "", [c], "tool_use", [{"type": "tool_use", "id": c.id, "name": c.name, "input": {}}]
        )


def test_agent_stops_at_step_limit():
    res = Agent(LoopingProvider(), max_steps=3).ask("q")
    assert res.stop_reason == "max_steps" and len(res.steps) == 3


def test_openai_message_conversion():
    from pharmapulse.llm.provider import OpenAIProvider

    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {"a": 1}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}],
        },
    ]
    out = OpenAIProvider._convert("sys", msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


class _FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.kwargs = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="Checking share."),
                SimpleNamespace(
                    type="tool_use", id="tu_1", name="market_share", input={"by": "region"}
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=1200, output_tokens=80),
        )


def test_anthropic_provider_request_and_parsing():
    from types import SimpleNamespace

    from pharmapulse.llm.provider import AnthropicProvider

    p = object.__new__(AnthropicProvider)  # skip client construction (no API key in CI)
    p.model, p.effort = "claude-opus-5-5", "medium"
    fake = _FakeMessages()
    p.client = SimpleNamespace(beta=SimpleNamespace(messages=fake), messages=fake)
    turn = p.complete("sys", [{"role": "user", "content": "q"}], tools.tool_definitions())

    assert fake.kwargs["output_config"] == {"effort": "medium"}
    assert fake.kwargs["fallbacks"] == "default"
    assert fake.kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert turn.tool_calls[0].name == "market_share" and turn.tool_calls[0].input == {
        "by": "region"
    }
    assert turn.text == "Checking share."
    assert len(turn.content) == 3  # thinking block kept so it can be echoed back unchanged
    assert turn.usage == {"input_tokens": 1200, "output_tokens": 80}
