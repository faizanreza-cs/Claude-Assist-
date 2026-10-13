# Model card and limitations

## Intended use

A portfolio and reference implementation of commercial analytics plus an LLM copilot for a
pharma brand team. It is not a validated system for real commercial or medical decisions.

## Data

Fully synthetic and seeded (`SynthConfig(seed=42)`). Behavioural effects are built into the
generator (see `methodology.md`). The analytics recover those effects, which demonstrates
method correctness; it says nothing about real-world effect sizes.

## Forecasting

- A single-origin 13-week holdout. Rolling-origin evaluation is on the roadmap.
- Intervals assume roughly normal residuals and widen with √step. Not calibrated for shocks
  such as loss of exclusivity, supply disruption or formulary changes.

## Promotional response

Observational association with HCP-potential and month controls. Rep targeting is confounded
with prescribing potential, so the estimate should not be read as causal ROI.

## LLM agent

- Answers are grounded in tool outputs, and the system prompt forbids estimating figures.
  Wording, emphasis and interpretation can still be wrong. The trace is always returned so
  users can check the source.
- `run_sql` is restricted to read-only, allow-listed, row-limited queries with a timeout.
- The mock eval (100%) measures pipeline correctness only. Model quality must be measured with
  `make eval-claude`. Golden questions use fixed phrasing, so add paraphrases before drawing
  conclusions about robustness.
- No PHI is present. With real patient data, apply de-identification, access control, audit
  logging, and a zero-data-retention agreement with the LLM provider.
