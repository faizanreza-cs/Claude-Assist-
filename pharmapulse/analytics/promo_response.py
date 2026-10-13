"""Promotional response: how much do rep calls move our brand's NRx?

A HCP-month panel regression with adstock (carry-over) on calls:

    log1p(NRx_it) = b * log1p(adstock(calls)_it) + c * log1p(potential_i) + month_t + e_it

``b`` is the call elasticity. We also translate it into the business number brand
teams ask for: incremental NRx per 100 additional calls at current call levels.
This is observational, so it's an association with confounding by targeting.
The docs and the agent's answers say so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pharmapulse.db import query

OUR_BRAND = "Glyvance"


def _panel() -> pd.DataFrame:
    sql = """
        WITH m AS (SELECT DISTINCT date_trunc('month', week) AS month FROM weekly_rx),
        grid AS (SELECT h.hcp_id, h.potential_trx_per_week, m.month FROM hcps h, m),
        rx AS (
            SELECT hcp_id, date_trunc('month', week) AS month, SUM(nrx) AS nrx
            FROM weekly_rx WHERE brand = ? GROUP BY 1, 2
        ),
        pr AS (
            SELECT hcp_id, date_trunc('month', week) AS month, SUM(calls) AS calls
            FROM promo WHERE brand = ? GROUP BY 1, 2
        )
        SELECT g.hcp_id, g.month, g.potential_trx_per_week AS potential,
               COALESCE(rx.nrx, 0) AS nrx, COALESCE(pr.calls, 0) AS calls
        FROM grid g LEFT JOIN rx USING (hcp_id, month) LEFT JOIN pr USING (hcp_id, month)
        ORDER BY g.hcp_id, g.month
    """
    return query(sql, [OUR_BRAND, OUR_BRAND])


def adstock(calls: pd.Series, decay: float) -> pd.Series:
    out, carry = [], 0.0
    for c in calls:
        carry = c + decay * carry
        out.append(carry)
    return pd.Series(out, index=calls.index)


@dataclass
class PromoResponse:
    brand: str
    decay: float
    elasticity: float
    ci_low: float
    ci_high: float
    incremental_nrx_per_100_calls: float
    r_squared: float
    observations: int
    caveat: str = (
        "Observational estimate: reps target high-potential HCPs, so this is an association "
        "adjusted for HCP potential and month, not a randomized causal effect."
    )

    def to_dict(self) -> dict[str, object]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def fit(decay: float = 0.5) -> PromoResponse:
    df = _panel()
    df["ad_calls"] = df.groupby("hcp_id", group_keys=False)["calls"].apply(
        lambda s: adstock(s, decay)
    )
    df["y"] = np.log1p(df["nrx"])
    df["x"] = np.log1p(df["ad_calls"])
    df["lp"] = np.log1p(df["potential"])
    df["month"] = df["month"].astype(str)
    model = smf.ols("y ~ x + lp + C(month)", data=df).fit()
    b = float(model.params["x"])
    lo, hi = (float(v) for v in model.conf_int().loc["x"])

    # marginal effect: d nrx / d calls at observed levels, averaged over HCP-months w/ calls
    active = df[df["ad_calls"] > 0]
    dy_dx = b * (1 + active["nrx"]) / (1 + active["ad_calls"])
    # an extra call today contributes 1/(1-decay) adstock units over its lifetime
    per_call = float(dy_dx.mean()) / (1 - decay)
    return PromoResponse(
        brand=OUR_BRAND,
        decay=decay,
        elasticity=b,
        ci_low=lo,
        ci_high=hi,
        incremental_nrx_per_100_calls=per_call * 100,
        r_squared=float(model.rsquared),
        observations=int(model.nobs),
    )
