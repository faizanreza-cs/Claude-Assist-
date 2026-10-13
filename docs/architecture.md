# Architecture

```
          ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
question →│  FastAPI /ask │     │ Streamlit UI │     │  CLI `ask`     │
          └──────┬───────┘     └──────┬───────┘     └──────┬────────┘
                 └────────────┬───────┴────────────────────┘
                              ▼
                     llm/agent.py  (tool-use loop, trace, step limit)
                              │
             ┌────────────────┼────────────────────┐
             ▼                ▼                    ▼
   llm/provider.py      llm/tools.py          evals/run_evals.py
   Claude │ OpenAI │    12 tools, JSON        golden questions +
   Mock                 outputs w/ notes      computed ground truth
                              │
             ┌────────────────┴───────────────┐
             ▼                                ▼
      analytics/* (pure functions,     llm/sql_guard.py
      tested, cached where slow)       (sqlglot validation)
             └───────────────┬────────────────┘
                             ▼
                    db.py → DuckDB (read-only)
                             ▲
                    data/synth.py + schema.sql
```

## Agent loop

1. `messages = [user question]`; call the provider with the system prompt and tool catalogue.
2. On `tool_use`, append the assistant content **unchanged**. With Claude this includes
   thinking blocks, which must be echoed back as-is.
3. Execute every requested tool. Return **all** results in one user message, using
   `is_error: true` for failures so the model can recover.
4. Repeat until the model answers without tools, refuses, or hits `max_steps` (default 8).

The returned `AgentResult` includes each step's tool, arguments, output and latency, plus
token usage. The API returns it, the UI shows it in an expander, and evals grade it.

## Provider abstraction

`LLMProvider.complete(system, messages, tools) -> Turn` normalises text, tool calls, stop
reason, history content and usage. History is stored in Anthropic content-block format, and
`OpenAIProvider._convert` translates it to chat-completions messages. `MockProvider` routes
questions with keyword rules and summarises the first tool result, so the full stack runs
offline.

Claude specifics: `output_config.effort` is configurable (default `medium`). Server-side
refusal fallback (`fallbacks: "default"`) is enabled for models that support it, and a
`refusal` stop reason is handled before any content is read.

## Performance notes

- DuckDB runs in-process and read-only, with a shared connection and a cursor per query.
- Patient episodes (a Python pass over ~120k claims) are cached per gap setting, and the
  dashboard caches every heavy call with `st.cache_data`.
- The full test suite runs in about 25 s against a small seeded database built per session.
