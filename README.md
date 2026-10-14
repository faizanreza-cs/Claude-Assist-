<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/logo-dark.svg">
  <img src="docs/img/logo.svg" alt="PharmaPulse" height="100">
</picture>

<p><em>An AI analytics copilot for pharma brand teams.</em></p>

[![CI](https://github.com/faizanreza-cs/pharmapulse/actions/workflows/ci.yml/badge.svg)](https://github.com/faizanreza-cs/pharmapulse/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2)](https://www.linkedin.com/in/YOUR-LINKEDIN-ID)
[![X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/YOUR_X_HANDLE)

</div>

## ⚡ Overview

PharmaPulse answers the questions a brand director actually asks: *"Are we gaining share?
What will TRx look like next quarter? Why are patients dropping off? Which prescribers should
reps call next? Is the field force paying for itself?"* It answers them two ways: through a
dashboard and REST API backed by tested analytics, and through an **LLM agent** that calls
those same analytics as tools and replies with figures traced to their source.

![Overview](docs/img/overview.png)

> **All data is synthetic.** A seeded generator builds a realistic US type 2 diabetes market
> (4 brands, 2,000 HCPs, 12,000 patients, 3 years of weekly prescriptions, promotion, payer
> access and pharmacy claims). No real patients, prescribers or sales figures are used.

## What it does

| Business question | Method | Where |
|---|---|---|
| Share, growth and launch tracking | TRx/NRx share of market by region and specialty, PoP and YoY growth, competitor launch curve | `analytics/market_share.py` |
| Demand forecasting | Seasonal-naive, Holt-Winters ETS and gradient boosting, auto-selected by 13-week holdout WAPE, with 80% intervals | `analytics/forecasting.py` |
| Patient persistence | Kaplan-Meier time on therapy (60-day allowable gap, switches as events, right-censoring) | `analytics/patient_journey.py` |
| Adherence | PDC over 12 months with overlapping supply shifted forward; adherent = PDC ≥ 0.8 | `analytics/patient_journey.py` |
| Switching | Source → destination brand flows | `analytics/patient_journey.py` |
| HCP targeting | Volume deciling, value × loyalty segmentation, K-means behavioural clusters, next-best-target headroom score | `analytics/hcp_segmentation.py` |
| Promotional response | HCP-month panel regression on adstocked calls with month fixed effects; incremental NRx per 100 calls, with its observational caveat | `analytics/promo_response.py` |
| Ad-hoc questions | LLM agent with 12 tools, including guarded read-only SQL | `llm/` |

## The AI layer

```mermaid
flowchart LR
    U[Brand team question] --> A[Agent loop<br/>llm/agent.py]
    A -->|system prompt + tool catalogue| P{LLMProvider}
    P --> C[Anthropic API<br/>official SDK]
    P --> O[OpenAI<br/>optional]
    P --> M[Mock router<br/>offline / CI]
    A -->|tool calls| T[Tools<br/>llm/tools.py]
    T --> AN[Analytics modules<br/>forecasting · share · persistence · targeting · promo]
    T --> G[SQL guard<br/>sqlglot: SELECT-only, allow-list, LIMIT, timeout]
    AN --> DB[(DuckDB<br/>synthetic star schema)]
    G --> DB
    A --> R[Answer + full trace<br/>tools, args, outputs, latency, tokens]
    R --> API[FastAPI /ask] & UI[Streamlit chat]
```

The design choices that matter in production:

- **Every number comes from a tool.** The model never computes or recalls figures. Tool
  outputs are JSON with the exact rows, plus notes on the period, definitions and caveats.
- **Plain, testable tool-use loop** (no framework). Parallel tool calls come back in a single
  message, tool errors return to the model as `is_error` results instead of crashing, and a
  step limit bounds runaway loops. Each piece has a unit test with scripted providers.
- **Provider-agnostic.** The Anthropic API is the default (via the official `anthropic` SDK, with effort
  control and server-side refusal fallback). OpenAI is a drop-in extra, and a deterministic
  mock router lets CI exercise the whole pipeline with no API key.
- **Guardrails on model-written SQL.** Queries are parsed with `sqlglot`: one `SELECT` only,
  table allow-list, no file/table functions, an enforced `LIMIT`, and a 10-second interrupt.
- **Evals as a first-class artifact.** 25 golden business questions, each with ground truth
  *computed from the analytics layer at eval time*, graded on tool selection and on whether
  every key figure appears in the answer (numeric tolerance, %/K/M aware).

## Results

**Forecast backtest** (13-week holdout, WAPE, lower is better)

| Brand | Seasonal naive | ETS | Gradient boosting |
|---|---|---|---|
| Glyvance (ours) | 9.0% | **4.0%** | 9.1% |
| Sitaglin | 5.6% | 3.6% | **2.6%** |
| Dapaflo | 4.0% | **1.3%** | 3.2% |
| Glucora (2024 launch) | 6.2% | **3.5%** | 8.0% |
| **Mean** | 6.2% | **3.1%** | 5.7% |

**Agent eval**: `make eval` runs the 25 golden questions offline with the mock router:
100% tool selection and 100% answer accuracy. That number checks the *plumbing*
(routing → tools → grounded answer), not model quality. `make eval-llm` runs the same
suite against the Anthropic API and writes `reports/eval_summary_anthropic.md` with accuracy, latency
and token usage.

## Screenshots

| Forecast with intervals and backtest | Patient persistence (Kaplan-Meier) |
|---|---|
| ![Forecast](docs/img/forecast.png) | ![Patients](docs/img/patients.png) |
| **HCP targeting and promo response** | **Ask PharmaPulse (agent with trace)** |
| ![HCP](docs/img/hcp.png) | ![Ask](docs/img/ask.png) |

## Quickstart

```bash
git clone https://github.com/faizanreza-cs/pharmapulse.git pharmapulse && cd pharmapulse
make setup        # uv venv + editable install with app and dev extras
make data         # build data/pharmapulse.duckdb (~10 s)
make app          # dashboard on http://localhost:8501
make api          # REST API on http://localhost:8000/docs
```

Ask the agent from the command line:

```bash
export ANTHROPIC_API_KEY=...          # optional; without it the offline mock router is used
export PHARMAPULSE_MODEL=...          # the Anthropic model id to use
pharmapulse ask "Which brand has the best 12-month persistence, and how does Glyvance compare?" --trace
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Forecast Glyvance TRx for the next 13 weeks"}'
```

Docker: `docker compose up` starts the API (`:8000`) and the dashboard (`:8501`).

| Setting | Default | Purpose |
|---|---|---|
| `PHARMAPULSE_PROVIDER` | `auto` | `anthropic`, `openai`, `mock`; `auto` uses the Anthropic API when credentials and a model id are set |
| `PHARMAPULSE_MODEL` | none | model id for the chosen provider (required unless `mock`) |
| `PHARMAPULSE_EFFORT` | `medium` | Anthropic effort level (`low` … `max`) |
| `PHARMAPULSE_FALLBACKS` | `0` | `1` enables Anthropic server-side refusal fallback |
| `PHARMAPULSE_DB` | `data/pharmapulse.duckdb` | database path |

## Engineering

- `make test`: 63 pytest tests, offline. They cover generator determinism, hand-checked PDC
  and persistence fixtures, share invariants, forecast intervals, SQL-guard attack cases,
  agent-loop bookkeeping, OpenAI message conversion and the API.
- Executed notebooks: [market EDA](notebooks/01_market_eda.ipynb) ·
  [forecast backtest and promo response](notebooks/02_forecast_backtest.ipynb).

More detail: [architecture](docs/architecture.md) · [methodology and definitions](docs/methodology.md) ·
[model card and limitations](docs/model_card.md).

## License

MIT. Built by [Faizan Reza](https://github.com/faizanreza-cs).
