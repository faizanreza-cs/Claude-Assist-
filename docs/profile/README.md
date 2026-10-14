<!--
  Profile README for github.com/faizanreza-cs.
  To publish: create a public repo named exactly `faizanreza-cs` and put this file at its root.
  Fill in the TODO links before publishing.
-->

# Hi, I'm Faizan Reza

**AI Engineer · Consumer Health, CPG & Pharma Analytics · LLM Agents**: I build analytics and
AI that teams can put into production: retail media and marketing measurement, recommendation
engines and review NLP for consumer brands, analytics for pharma and healthcare teams, and
governed LLM agents and automations.

📍 Delhi, India · 🎓 Computer Science Engineering graduate · 💼 Available for freelance projects and full-time AI engineering roles

[![Email](https://img.shields.io/badge/email-faizanreza01%40gmail.com-blue)](mailto:faizanreza01@gmail.com)
<!-- TODO: [![LinkedIn](https://img.shields.io/badge/LinkedIn-profile-0A66C2)](https://www.linkedin.com/in/YOUR-HANDLE) -->

---

### 🔭 Projects

| Project | What it does | Highlights |
|---|---|---|
| [**retail-media-lab**](https://github.com/faizanreza-cs/retail-media-lab) | Amazon Ads analytics for a supplements brand: AMC-style SQL, attribution, a geo holdout, a marketing mix model and a budget optimizer | Every method scored against a simulator's truth (geo-test iROAS 2.03 vs true 1.90; budget plan reaches 89% of the best possible) |
| [**shelfsense**](https://github.com/faizanreza-cs/shelfsense) | Consumer insights for a health and nutrition store: recommender, review aspect sentiment, monitoring | Learned re-ranker with business rules, LLM labels distilled into a cheap classifier, an alert that caught a reformulated product in its first month |
| [**agentkit-enterprise**](https://github.com/faizanreza-cs/agentkit-enterprise) | Starter kit for AI agents companies can approve for production | Role-based tool policy, human approval with separation of duties, PII redaction, hash-chained audit log, budgets, governance eval |
| [**docflow-agent**](https://github.com/faizanreza-cs/docflow-agent) | Invoices, receipts and contracts → validated structured data | LLM structured outputs, totals/date/duplicate checks, human review queue, CSV / Sheets / webhook export |
| [**ragdesk**](https://github.com/faizanreza-cs/ragdesk) | Answers from company docs with citations, refuses to guess | Hybrid BM25 + vector retrieval, citation guardrails, eval suite, website widget, Slack bot |
| [**pharma-mcp**](https://github.com/faizanreza-cs/pharma-mcp) | MCP server for FDA labels, FAERS, recalls and ClinicalTrials.gov | PRR/ROR signal detection with CIs, cached and rate-limited, tested through a real MCP client |
| [**PharmaPulse**](https://github.com/faizanreza-cs/pharmapulse) | AI analytics copilot for a pharma brand team | Forecasting (3.1% WAPE), Kaplan-Meier persistence, HCP targeting, 12-tool agent with SQL guardrails |

Every project ships with tests, CI and Docker; the agent projects also include evals and an offline demo mode.

<img src="https://raw.githubusercontent.com/faizanreza-cs/retail-media-lab/main/docs/img/attribution_vs_truth.png" width="420" alt="retail-media-lab: attribution methods against the true causal share"/> <img src="https://raw.githubusercontent.com/faizanreza-cs/shelfsense/main/docs/img/monitoring.png" width="420" alt="shelfsense: monitoring flags a reformulated product"/>

### 🤝 What I can build for you

- **Retail media and marketing analytics**: Amazon Ads and AMC-style analysis, attribution, incrementality tests, marketing mix models, budget optimization
- **Consumer insights**: recommendation engines, review and survey NLP with LLMs, model monitoring
- **Document automation**: invoices, receipts, contracts, forms, lab reports → your ERP, CRM or spreadsheets
- **Knowledge assistants**: cited answers over policies, product docs and support content, on your site or in Slack
- **Internal agents**: support desk, finance ops and IT automation, with the controls your security team asks for
- **MCP servers and integrations**: connect AI assistants to your databases, APIs and SaaS tools
- **Pharma and healthcare analytics**: forecasting, patient journey, HCP targeting, pharmacovigilance

---

### 🧰 Toolkit

| Area | Tools |
|---|---|
| AI engineering | LLM agents and tool use, MCP servers, RAG, structured outputs, evals, guardrails, Anthropic SDK, OpenAI SDK, Azure OpenAI |
| Analytics and ML | Python, pandas, scikit-learn, statsmodels, recommender systems, NLP, causal inference (difference-in-differences, synthetic control), marketing mix modelling, forecasting, survival analysis |
| Data | SQL, Snowflake, DuckDB, data modelling (star schemas), sqlglot |
| BI and platforms | Power BI (DAX), Dataiku, Microsoft Copilot Studio |
| Apps and delivery | FastAPI, Streamlit, Plotly, Docker, GitHub Actions, pytest, ruff, mypy |
| Consumer health and CPG | Retail media (Amazon Ads, AMC-style data), e-commerce, consumer insights from reviews, model monitoring |
| Pharma domain | TRx/NRx and syndicated Rx data, market share and launch analytics, patient persistence and adherence (PDC), HCP deciling and targeting, promotional response, payer and formulary access |

### 🌱 Currently building

- Bayesian pharmacovigilance signals (EBGM) and label-grounded safety Q&A on top of pharma-mcp
- Clinical-trial enrollment forecasting from ClinicalTrials.gov

---

<sub>Open to freelance projects and roles in AI engineering, consumer health and CPG analytics, AI automation and pharma / life-sciences analytics. Reach me at faizanreza01@gmail.com.</sub>
