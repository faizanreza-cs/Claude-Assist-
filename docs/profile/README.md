<!--
  Profile README for github.com/faizanreza-cs.
  To publish: create a public repo named exactly `faizanreza-cs` and put this file at its root.
  Fill in the TODO links before publishing.
-->

# Hi, I'm Faizan Reza

**AI Engineer · Pharma Commercial Analytics**: I build LLM agents and analytics that help
brand teams forecast demand, understand patient behaviour and decide where to spend.

📍 Delhi, India · 🎓 Computer Science Engineering graduate · 💼 Open to AI engineering and pharma analytics roles

[![Email](https://img.shields.io/badge/email-faizanreza01%40gmail.com-blue)](mailto:faizanreza01@gmail.com)
<!-- TODO: [![LinkedIn](https://img.shields.io/badge/LinkedIn-profile-0A66C2)](https://www.linkedin.com/in/YOUR-HANDLE) -->

---

### 🔭 Featured project: [PharmaPulse](https://github.com/faizanreza-cs/Claude-Assist-)

An AI analytics copilot for a pharma brand team, built end to end on a seeded synthetic
type 2 diabetes market (4 brands · 2,000 HCPs · 12,000 patients · 3 years of weekly Rx).

- **LLM agent** (Claude by default, provider-agnostic) that answers brand questions by calling
  12 analytics tools. Every figure is traced to a tool output; model-written SQL runs through
  a `sqlglot` read-only guard.
- **Forecasting**: seasonal-naive / Holt-Winters / gradient boosting, auto-selected by holdout
  WAPE (mean **3.1%** for the selected ETS model), with prediction intervals.
- **Patient analytics**: Kaplan-Meier persistence, PDC adherence, switching flows from claims.
- **HCP targeting**: volume deciles, value × loyalty segments, next-best-target scoring,
  adstocked promotional response.
- **Evals and engineering**: 25-question golden eval harness, 62 tests, FastAPI, Streamlit,
  Docker, GitHub Actions.

<img src="https://raw.githubusercontent.com/faizanreza-cs/Claude-Assist-/main/docs/img/overview.png" width="720" alt="PharmaPulse dashboard"/>

---

### 🧰 Toolkit

| Area | Tools |
|---|---|
| AI engineering | LLM tool use and agents, evals, prompt design, RAG, Anthropic SDK, OpenAI SDK |
| Analytics and ML | Python, pandas, statsmodels, scikit-learn, lifelines, time-series forecasting, survival analysis |
| Data | SQL, DuckDB, data modelling (star schemas), sqlglot |
| Apps and delivery | FastAPI, Streamlit, Plotly, Docker, GitHub Actions, pytest, ruff, mypy |
| Pharma domain | TRx/NRx and syndicated Rx data, market share and launch analytics, patient persistence and adherence (PDC), HCP deciling and targeting, promotional response, payer and formulary access |

### 🌱 Currently building

- Drug-safety signal detection on FDA adverse event data (FAERS) with a label-grounded RAG assistant
- Clinical-trial enrollment forecasting from ClinicalTrials.gov

---

<sub>Open to roles in AI engineering, LLM applications and pharma / life-sciences analytics. Reach me at faizanreza01@gmail.com.</sub>
