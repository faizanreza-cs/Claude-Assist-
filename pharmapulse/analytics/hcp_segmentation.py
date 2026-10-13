"""HCP targeting: deciling, behavioural segmentation and next-best-target scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pharmapulse.db import query

OUR_BRAND = "Glyvance"


def hcp_profile(weeks: int = 52) -> pd.DataFrame:
    """Per-HCP market volume, our-brand volume/share, and promotional intensity."""
    sql = """
        WITH lw AS (SELECT MAX(week) AS w FROM weekly_rx),
        rx AS (
            SELECT hcp_id, SUM(trx) AS market_trx,
                   SUM(CASE WHEN brand = ? THEN trx ELSE 0 END) AS our_trx,
                   SUM(CASE WHEN brand = ? THEN nrx ELSE 0 END) AS our_nrx
            FROM weekly_rx, lw
            WHERE week > lw.w - INTERVAL (?) WEEK
            GROUP BY 1
        ),
        pr AS (
            SELECT hcp_id, SUM(calls) AS calls, SUM(samples) AS samples
            FROM promo, lw WHERE week > lw.w - INTERVAL (?) WEEK GROUP BY 1
        )
        SELECT h.hcp_id, h.specialty, h.region,
               COALESCE(rx.market_trx, 0) AS market_trx,
               COALESCE(rx.our_trx, 0) AS our_trx,
               COALESCE(rx.our_nrx, 0) AS our_nrx,
               COALESCE(pr.calls, 0) AS calls,
               COALESCE(pr.samples, 0) AS samples
        FROM hcps h LEFT JOIN rx USING (hcp_id) LEFT JOIN pr USING (hcp_id)
    """
    df = query(sql, [OUR_BRAND, OUR_BRAND, weeks, weeks])
    df["our_share"] = (df["our_trx"] / df["market_trx"].where(df["market_trx"] > 0)).fillna(0)
    return df


def add_deciles(df: pd.DataFrame, col: str = "market_trx") -> pd.DataFrame:
    """Standard pharma deciling: decile 10 = HCPs writing the top 10% of volume.

    HCPs are sorted by volume and assigned to deciles by *cumulative volume*, so each
    decile holds ~10% of the market (not 10% of HCPs). Zero writers are decile 0.
    """
    df = df.sort_values(col, ascending=True).copy()
    total = df[col].sum()
    cum = df[col].cumsum() / total if total else df[col] * 0
    df["decile"] = np.where(df[col] > 0, np.ceil(cum * 10).clip(1, 10), 0).astype(int)
    return df.sort_values(col, ascending=False)


def decile_summary(weeks: int = 52) -> pd.DataFrame:
    df = add_deciles(hcp_profile(weeks))
    out = (
        df.groupby("decile")
        .agg(
            hcps=("hcp_id", "size"),
            market_trx=("market_trx", "sum"),
            our_trx=("our_trx", "sum"),
            calls=("calls", "sum"),
        )
        .reset_index()
        .sort_values("decile", ascending=False)
    )
    out["our_share"] = (out["our_trx"] / out["market_trx"].where(out["market_trx"] > 0)).round(4)
    out["calls_per_hcp"] = (out["calls"] / out["hcps"]).round(2)
    return out


SEGMENTS = {
    (True, True): "Loyalists - protect",
    (True, False): "Growth targets - invest",
    (False, True): "Niche advocates - maintain",
    (False, False): "Low priority - digital only",
}


def segment(weeks: int = 52) -> pd.DataFrame:
    """Value x loyalty segmentation used for call planning.

    * value  = HCP sits in deciles 6-10, i.e. the HCPs writing the top half of market volume
    * loyalty = our share at the HCP is at or above our national share

    Plus a K-means behavioural cluster (volume, share, call intensity) for analysts who
    want a data-driven view next to the rule-based one.
    """
    df = add_deciles(hcp_profile(weeks))
    df = df[df["market_trx"] > 0].copy()
    national_share = df["our_trx"].sum() / df["market_trx"].sum()
    df["high_value"] = df["decile"] >= 6
    df["loyal"] = df["our_share"] >= national_share
    df["segment"] = [SEGMENTS[(v, s)] for v, s in zip(df["high_value"], df["loyal"], strict=True)]
    X = np.column_stack([np.log1p(df["market_trx"]), df["our_share"], np.log1p(df["calls"])])
    df["cluster"] = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(
        StandardScaler().fit_transform(X)
    )
    return df


def segment_summary(weeks: int = 52) -> pd.DataFrame:
    df = segment(weeks)
    out = (
        df.groupby("segment")
        .agg(
            hcps=("hcp_id", "size"),
            market_trx=("market_trx", "sum"),
            median_share=("our_share", "median"),
            calls_per_hcp=("calls", "mean"),
        )
        .reset_index()
        .sort_values("market_trx", ascending=False)
    )
    return out.round({"median_share": 4, "calls_per_hcp": 2})


def next_best_targets(
    n: int = 25,
    region: str | None = None,
    specialty: str | None = None,
    weeks: int = 52,
) -> pd.DataFrame:
    """Rank HCPs by *headroom*: market volume we don't yet win, discounted if already
    heavily called on (diminishing returns). Score = market_trx * (1 - share) / sqrt(1 + calls/12).
    """
    df = hcp_profile(weeks)
    if region:
        df = df[df["region"] == region]
    if specialty:
        df = df[df["specialty"] == specialty]
    df = df.assign(
        headroom_trx=df["market_trx"] - df["our_trx"],
        score=(df["market_trx"] - df["our_trx"]) / np.sqrt(1 + df["calls"] / 12),
    )
    cols = [
        "hcp_id",
        "specialty",
        "region",
        "market_trx",
        "our_trx",
        "our_share",
        "calls",
        "headroom_trx",
        "score",
    ]
    out = df.sort_values("score", ascending=False).head(n)[cols]
    return out.round({"our_share": 4, "score": 1})
