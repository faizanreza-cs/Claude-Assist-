# Methodology and definitions

## Data model (synthetic)

| Table | Grain | Notes |
|---|---|---|
| `weekly_rx` | HCP × brand × week | TRx (total) and NRx (new) prescriptions, the shape of a syndicated prescription feed |
| `promo` | HCP × week | rep calls and samples for our brand |
| `formulary` | region × brand × quarter | payer tier: 1 preferred, 2 non-preferred, 3 restricted / prior auth |
| `claims` | fill | patient pharmacy fills with days supply |
| `hcps`, `patients`, `brands`, `regions` | master data | specialty, region, potential, age band, index brand |

Behaviour built into the generator, so the analytics have something real to find:
- market growth of ~6% a year, winter seasonality and a holiday dip
- Glyvance (our GLP-1) gaining share; Sitaglin (DPP-4) eroding; Glucora launching at week 78
  with a logistic uptake curve
- formulary tier shifts brand choice (preferred +, restricted −); Glyvance wins a preferred
  contract in the Southeast from 2024
- adstocked rep calls lift Glyvance choice at the HCP level
- discontinuation hazard by brand and age band, refill lateness, and switching paths

## Market metrics

- **Share of market** = brand volume / total volume of the four brands, over the period.
- **Growth**: latest *N* weeks vs the prior *N* weeks (PoP) and vs the same weeks a year ago (YoY).
- **Launch uptake**: brand share by weeks since its first prescription.
- **Average weekly volume by month**: monthly totals divided by the number of weeks in each
  month, so 4- and 5-week months compare fairly.

## Forecasting

- Weekly national series per brand, zero-filled.
- Models: seasonal naive (last year × recent YoY growth; falls back to a 4-week mean without
  two years of history), Holt-Winters ETS (additive damped trend + 52-week seasonality;
  trend-only for short launch series), gradient boosting on lags (1, 2, 4, 52), rolling means,
  week-of-year and a time index, forecast recursively.
- Selection: lowest WAPE on a 13-week holdout. Metrics reported: WAPE, MAPE, bias.
- Intervals: 80%, from the holdout residual standard deviation × √step. Simple, explainable,
  and good enough for planning ranges.

## Patient analytics

- **Persistence**: first episode on the index brand. The episode ends at a gap > 60 days
  between the end of supply and the next fill (overlapping supply carried forward), or at a
  switch. Patients still on therapy are censored. Kaplan-Meier via `lifelines`.
- **PDC**: days covered / 365 from index, counting only fills of the index brand and shifting
  early refills forward. Only patients with a full 365-day window are included.
  Adherent = PDC ≥ 0.8.
- **Switching**: the first different brand filled inside the gap window.

## HCP targeting

- **Deciles** are built on cumulative market volume: decile 10 holds the HCPs who write the
  top ~10% of category TRx. Non-writers are decile 0.
- **Segments**: value (deciles 6–10) × loyalty (our share ≥ national share) gives four
  actionable groups: protect, invest, maintain, digital-only. K-means clusters on
  (log volume, share, log calls) are provided alongside for exploration.
- **Next best target** score = (market TRx − our TRx) / √(1 + calls/12). This ranks headroom
  and discounts HCPs who are already heavily called on (diminishing returns).

## Promotional response

`log1p(NRx) ~ log1p(adstock(calls, λ=0.5)) + log1p(HCP potential) + month fixed effects`
on an HCP-month panel. The marginal effect is averaged over active HCP-months and scaled by
1/(1−λ) for the lifetime of a call's adstock, then reported per 100 calls. The estimate is
**observational**: targeting is confounded with potential. It is useful for direction and
relative sizing. A causal answer needs a call-plan experiment or a quasi-experimental design.
