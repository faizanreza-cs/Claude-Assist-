"""Patient-level analytics from pharmacy claims: persistence, adherence, switching.

Definitions follow common pharma / RWE conventions:

* **Persistence** – time from index fill until a gap longer than ``gap_days`` (default 60)
  between the end of one fill's supply and the next fill, or a switch to another brand.
  Patients still on therapy at data end are right-censored. Estimated with Kaplan-Meier.
* **PDC** (proportion of days covered) – share of days in a fixed window from index
  with drug on hand, overlapping supply shifted forward. ``PDC >= 0.8`` = adherent.
* **Switching** – next different brand filled within the gap window after the last fill.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from pharmapulse.db import query


def load_claims() -> pd.DataFrame:
    sql = """
        SELECT c.patient_id, c.brand, c.fill_date, c.days_supply, p.age_band, p.region
        FROM claims c JOIN patients p USING (patient_id)
        ORDER BY c.patient_id, c.fill_date
    """
    df = query(sql)
    df["fill_date"] = pd.to_datetime(df["fill_date"])
    return df


def therapy_episodes(claims: pd.DataFrame, gap_days: int = 60) -> pd.DataFrame:
    """One row per patient's *first* episode on their index brand.

    Returns ``duration_days`` and ``event`` (1 = discontinued or switched, 0 = censored).
    """
    data_end = claims["fill_date"].max()  # last observed day; later days were never seen
    rows = []
    for pid, g in claims.groupby("patient_id", sort=False):
        brand = g["brand"].iloc[0]
        start = g["fill_date"].iloc[0]
        supply_end = start + pd.Timedelta(days=int(g["days_supply"].iloc[0]))
        event, switched_to = 0, None
        for fill, ds, b in zip(
            g["fill_date"].iloc[1:], g["days_supply"].iloc[1:], g["brand"].iloc[1:], strict=True
        ):
            gap = (fill - supply_end).days
            if gap > gap_days:
                event = 1
                break
            if b != brand:
                event, switched_to = 1, b
                break
            supply_end = max(supply_end, fill) + pd.Timedelta(days=int(ds))
        else:
            # no more fills: discontinued if the gap to data end exceeds the allowance
            if (data_end - supply_end).days > gap_days:
                event = 1
        end = supply_end if event else min(supply_end + pd.Timedelta(days=gap_days), data_end)
        rows.append(
            {
                "patient_id": pid,
                "brand": brand,
                "age_band": g["age_band"].iloc[0],
                "region": g["region"].iloc[0],
                "index_date": start,
                "duration_days": max((end - start).days, 1),
                "event": event,
                "switched_to": switched_to,
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def _all_episodes(gap_days: int) -> pd.DataFrame:
    return therapy_episodes(load_claims(), gap_days)


def episodes(brand: str | None = None, gap_days: int = 60) -> pd.DataFrame:
    """Cached first-episode table, optionally filtered to one index brand."""
    ep = _all_episodes(gap_days)
    return (ep[ep["brand"] == brand] if brand else ep).copy()


def persistence(
    brand: str | None = None,
    by: str = "brand",
    gap_days: int = 60,
    timepoints: tuple[int, ...] = (90, 180, 365),
) -> pd.DataFrame:
    """Kaplan-Meier persistence at ``timepoints`` days, grouped by ``by``."""
    if by not in {"brand", "age_band", "region"}:
        raise ValueError("by must be brand, age_band or region")
    ep = episodes(brand, gap_days)
    rows = []
    for key, g in ep.groupby(by):
        km = KaplanMeierFitter().fit(g["duration_days"], g["event"])
        row: dict[str, object] = {by: key, "patients": len(g)}
        for t in timepoints:
            row[f"persistent_{t}d"] = round(float(km.predict(t)), 4)
        median = km.median_survival_time_
        row["median_days"] = None if np.isinf(median) else round(float(median), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"persistent_{timepoints[-1]}d", ascending=False)


def km_curve(brand: str, gap_days: int = 60, max_days: int = 540) -> pd.DataFrame:
    ep = episodes(brand, gap_days)
    km = KaplanMeierFitter().fit(ep["duration_days"], ep["event"])
    grid = np.arange(0, max_days + 1, 15)
    return pd.DataFrame({"day": grid, "persistent": km.predict(grid).to_numpy().round(4)})


def pdc(claims: pd.DataFrame, window_days: int = 365) -> pd.DataFrame:
    """Proportion of days covered over ``window_days`` from each patient's index fill.

    Only patients with a full observation window are included.
    """
    data_end = claims["fill_date"].max()
    out = []
    for pid, g in claims.groupby("patient_id", sort=False):
        start = g["fill_date"].iloc[0]
        if start + pd.Timedelta(days=window_days) > data_end:
            continue
        covered = np.zeros(window_days, dtype=bool)
        cursor = 0  # next uncovered day, so overlapping supply is shifted forward
        for fill, ds in zip(g["fill_date"], g["days_supply"], strict=True):
            day = max((fill - start).days, cursor)
            if day >= window_days:
                break
            end = min(day + int(ds), window_days)
            covered[day:end] = True
            cursor = end
        out.append({"patient_id": pid, "brand": g["brand"].iloc[0], "pdc": covered.mean()})
    return pd.DataFrame(out)


@lru_cache(maxsize=4)
def adherence(window_days: int = 365) -> pd.DataFrame:
    """Mean PDC and % adherent (PDC >= 0.8) by index brand."""
    claims = load_claims()
    first_brand = claims.groupby("patient_id")["brand"].transform("first")
    on_index = claims[claims["brand"] == first_brand]
    p = pdc(on_index, window_days)
    res = (
        p.groupby("brand")
        .agg(
            patients=("pdc", "size"),
            mean_pdc=("pdc", "mean"),
            pct_adherent=("pdc", lambda s: (s >= 0.8).mean()),
        )
        .reset_index()
        .sort_values("mean_pdc", ascending=False)
    )
    res[["mean_pdc", "pct_adherent"]] = res[["mean_pdc", "pct_adherent"]].round(4)
    return res


def switching_matrix(gap_days: int = 60) -> pd.DataFrame:
    """Counts of switches from index brand to the next brand (source -> destination)."""
    ep = episodes(None, gap_days)
    sw = ep.dropna(subset=["switched_to"])
    m = sw.groupby(["brand", "switched_to"]).size().reset_index(name="patients")
    totals = ep.groupby("brand").size()
    m["pct_of_index_patients"] = (m["patients"] / m["brand"].map(totals)).round(4)
    return m.rename(columns={"brand": "from_brand", "switched_to": "to_brand"}).sort_values(
        "patients", ascending=False
    )
