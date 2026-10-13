"""Reproducible synthetic pharma commercial data.

The generator models a type-2 diabetes market with three competing brands and a
mid-period launch. Everything is seeded so tests and evals are deterministic.

Tables produced (all pandas DataFrames, persisted to DuckDB by ``build_database``):

* ``brands``        – brand master (class, launch week, list price)
* ``regions``       – sales regions
* ``hcps``          – prescribers with specialty, region, potential
* ``weekly_rx``     – HCP x brand x week NRx / TRx (the "IQVIA-style" feed)
* ``promo``         – HCP x week rep calls and samples
* ``formulary``     – region x brand x quarter payer access tier
* ``patients``      – patient master (age band, region, index brand)
* ``claims``        – pharmacy fills (patient, brand, fill date, days supply)

The data is synthetic. No real patients, prescribers or sales figures are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

START_DATE = pd.Timestamp("2023-01-02")  # a Monday


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 42
    n_weeks: int = 156  # 3 years
    n_hcps: int = 2_000
    n_patients: int = 12_000
    launch_week: int = 78  # Glucora launches half-way through


BRANDS = pd.DataFrame(
    {
        "brand": ["Glyvance", "Sitaglin", "Dapaflo", "Glucora"],
        "molecule_class": ["GLP-1 RA", "DPP-4i", "SGLT2i", "GLP-1 RA"],
        "manufacturer": ["Our Co", "Competitor A", "Competitor B", "Competitor C"],
        "is_our_brand": [True, False, False, False],
        "list_price_per_30d": [935.0, 540.0, 610.0, 990.0],
    }
)

REGIONS = pd.DataFrame(
    {
        "region": ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
        "population_index": [1.10, 1.25, 0.95, 0.85, 1.00],
    }
)

SPECIALTIES = ["Endocrinology", "Primary Care", "Internal Medicine", "Nurse Practitioner"]
SPECIALTY_WEIGHTS = [0.12, 0.48, 0.28, 0.12]
SPECIALTY_VOLUME = {
    "Endocrinology": 3.2,
    "Primary Care": 1.0,
    "Internal Medicine": 1.3,
    "Nurse Practitioner": 0.7,
}
# relative preference of each specialty for each brand
SPECIALTY_AFFINITY = {
    "Endocrinology": [1.5, 0.6, 1.0, 1.4],
    "Primary Care": [0.9, 1.3, 1.0, 0.8],
    "Internal Medicine": [1.0, 1.1, 1.1, 1.0],
    "Nurse Practitioner": [0.8, 1.4, 0.9, 0.7],
}
BASE_SHARE = np.array([0.30, 0.38, 0.32, 0.0])
AGE_BANDS = ["18-44", "45-64", "65+"]


def _weeks(cfg: SynthConfig) -> pd.DatetimeIndex:
    return pd.date_range(START_DATE, periods=cfg.n_weeks, freq="W-MON")


def _make_hcps(rng: np.random.Generator, cfg: SynthConfig) -> pd.DataFrame:
    specialty = rng.choice(SPECIALTIES, size=cfg.n_hcps, p=SPECIALTY_WEIGHTS)
    region = rng.choice(
        REGIONS["region"],
        size=cfg.n_hcps,
        p=REGIONS["population_index"] / REGIONS["population_index"].sum(),
    )
    volume = np.array([SPECIALTY_VOLUME[s] for s in specialty])
    potential = rng.lognormal(mean=1.6, sigma=0.7, size=cfg.n_hcps) * volume
    return pd.DataFrame(
        {
            "hcp_id": [f"H{i:05d}" for i in range(cfg.n_hcps)],
            "specialty": specialty,
            "region": region,
            "potential_trx_per_week": potential.round(3),
        }
    )


def _make_formulary(rng: np.random.Generator, cfg: SynthConfig) -> pd.DataFrame:
    quarters = sorted({(d.year, d.quarter) for d in _weeks(cfg)})
    rows = []
    for region in REGIONS["region"]:
        for b_idx, brand in enumerate(BRANDS["brand"]):
            tier = int(rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))
            for year, q in quarters:
                if brand == "Glucora" and (year, q) < (2024, 3):
                    continue
                # occasional access change
                if rng.random() < 0.08:
                    tier = int(np.clip(tier + rng.choice([-1, 1]), 1, 3))
                if b_idx == 0 and region == "Southeast" and (year, q) >= (2024, 1):
                    tier = 1  # we won a preferred contract
                rows.append(
                    {
                        "region": region,
                        "brand": brand,
                        "year": year,
                        "quarter": q,
                        "tier": tier,
                    }
                )
    return pd.DataFrame(rows)


def _make_promo(rng: np.random.Generator, hcps: pd.DataFrame, cfg: SynthConfig) -> pd.DataFrame:
    weeks = _weeks(cfg)
    # reps target higher-potential HCPs more
    rank = hcps["potential_trx_per_week"].rank(pct=True).to_numpy()
    call_rate = 0.05 + 0.55 * rank**2
    calls = rng.poisson(np.outer(call_rate, np.ones(cfg.n_weeks)))
    # quarterly campaign bursts
    burst = np.array([1.6 if (w % 13) < 3 else 1.0 for w in range(cfg.n_weeks)])
    calls = rng.poisson(calls * burst)
    samples = rng.binomial(calls * 4, 0.6)
    df = pd.DataFrame(
        {
            "hcp_id": np.repeat(hcps["hcp_id"].to_numpy(), cfg.n_weeks),
            "week": np.tile(weeks, cfg.n_hcps),
            "brand": "Glyvance",
            "calls": calls.ravel(),
            "samples": samples.ravel(),
        }
    )
    return df[df["calls"] > 0].reset_index(drop=True)


def _adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = np.zeros(x.shape[0])
    for t in range(x.shape[1]):
        carry = x[:, t] + decay * carry
        out[:, t] = carry
    return out


def _make_weekly_rx(
    rng: np.random.Generator,
    hcps: pd.DataFrame,
    promo: pd.DataFrame,
    formulary: pd.DataFrame,
    cfg: SynthConfig,
) -> pd.DataFrame:
    weeks = _weeks(cfg)
    n_h, n_w, n_b = cfg.n_hcps, cfg.n_weeks, len(BRANDS)
    t = np.arange(n_w)

    # market grows ~6%/yr with a winter seasonality and holiday dip
    market = 1.0 + 0.06 * t / 52 + 0.05 * np.cos(2 * np.pi * (t - 2) / 52)
    holiday = np.isin([d.isocalendar().week for d in weeks], [52, 1]).astype(float)
    market = market * (1 - 0.12 * holiday)

    # brand preference over time (logits)
    logits = np.log(np.maximum(BASE_SHARE, 1e-9))[None, :].repeat(n_w, axis=0)
    logits[:, 0] += 0.25 * np.clip((t - 20) / 80, 0, 1)  # Glyvance gains share
    logits[:, 1] -= 0.30 * t / n_w  # DPP-4 erodes
    launch = cfg.launch_week
    uptake = np.where(t >= launch, 1 / (1 + np.exp(-(t - launch - 18) / 7)), 0.0)
    logits[:, 3] = np.where(t >= launch, np.log(0.22 * uptake + 1e-6), -50.0)

    # formulary effect by region/brand/quarter
    tier_effect = {1: 0.25, 2: 0.0, 3: -0.35}
    q_of_week = [(d.year, d.quarter) for d in weeks]
    tier_lookup = {(r.region, r.brand, r.year, r.quarter): r.tier for r in formulary.itertuples()}

    # promo adstock (Glyvance only)
    promo_w = promo.pivot_table(
        index="hcp_id", columns="week", values="calls", aggfunc="sum", fill_value=0
    ).reindex(index=hcps["hcp_id"], columns=weeks, fill_value=0)
    ad = _adstock(promo_w.to_numpy(dtype=float), decay=0.6)
    promo_lift = 0.18 * np.log1p(ad)

    spec = hcps["specialty"].to_numpy()
    affinity = np.array([SPECIALTY_AFFINITY[s] for s in spec])  # n_h x n_b
    region = hcps["region"].to_numpy()
    potential = hcps["potential_trx_per_week"].to_numpy()

    trx = np.zeros((n_h, n_w, n_b))
    for w in range(n_w):
        lg = logits[w][None, :] + np.log(affinity)
        for b_idx, brand in enumerate(BRANDS["brand"]):
            tiers = np.array([tier_lookup.get((r, brand, *q_of_week[w]), 2) for r in region])
            lg[:, b_idx] += np.vectorize(tier_effect.get)(tiers)
        lg[:, 0] += promo_lift[:, w]
        p = np.exp(lg - lg.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        lam = potential[:, None] * market[w] * p
        trx[:, w, :] = rng.poisson(lam)

    # NRx ~ new starts: a share of TRx, higher for newly launched brand
    nrx_ratio = np.array([0.28, 0.22, 0.25, 0.55])
    nrx = rng.binomial(trx.astype(int), nrx_ratio[None, None, :])

    idx = pd.MultiIndex.from_product(
        [hcps["hcp_id"], weeks, BRANDS["brand"]], names=["hcp_id", "week", "brand"]
    )
    df = pd.DataFrame({"trx": trx.ravel().astype(int), "nrx": nrx.ravel().astype(int)}, index=idx)
    df = df[df["trx"] > 0].reset_index()
    return df


def _make_patients_and_claims(
    rng: np.random.Generator, cfg: SynthConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = cfg.n_patients
    weeks = _weeks(cfg)
    end = weeks[-1]
    region = rng.choice(REGIONS["region"], size=n)
    age_band = rng.choice(AGE_BANDS, size=n, p=[0.18, 0.47, 0.35])
    start_offsets = rng.integers(0, cfg.n_weeks * 7 - 60, size=n)
    index_date = START_DATE + pd.to_timedelta(start_offsets, unit="D")
    launch_date = START_DATE + pd.Timedelta(weeks=cfg.launch_week)

    probs_pre = np.array([0.32, 0.36, 0.32, 0.0])
    probs_post = np.array([0.30, 0.27, 0.28, 0.15])
    index_brand = np.array(
        [
            rng.choice(BRANDS["brand"], p=probs_post if d >= launch_date else probs_pre)
            for d in index_date
        ]
    )

    # weekly hazard of discontinuation per brand (GLP-1s have GI side effects early on)
    base_gap = {"Glyvance": 0.10, "Sitaglin": 0.14, "Dapaflo": 0.12, "Glucora": 0.09}
    late_prob = {"Glyvance": 0.18, "Sitaglin": 0.30, "Dapaflo": 0.24, "Glucora": 0.15}
    age_mult = {"18-44": 1.35, "45-64": 1.0, "65+": 0.85}
    switch_to = {
        "Glyvance": ["Glucora", "Dapaflo"],
        "Sitaglin": ["Glyvance", "Dapaflo", "Glucora"],
        "Dapaflo": ["Glyvance", "Glucora"],
        "Glucora": ["Glyvance"],
    }

    claim_rows: list[tuple[str, str, pd.Timestamp, int]] = []
    for i in range(n):
        pid = f"P{i:06d}"
        brand = str(index_brand[i])
        date = index_date[i]
        stop_p = 0.045 * age_mult[str(age_band[i])] * (1.0 + (brand == "Sitaglin") * 0.35)
        while date <= end:
            days_supply = 90 if rng.random() < 0.2 else 30
            claim_rows.append((pid, brand, date, days_supply))
            if rng.random() < stop_p:
                # discontinue, sometimes switch
                if rng.random() < 0.35 and brand in switch_to:
                    gap = int(rng.integers(5, 40))
                    brand = str(rng.choice(switch_to[brand]))
                    if brand == "Glucora" and date < launch_date:
                        break
                    date = date + pd.Timedelta(days=days_supply + gap)
                    continue
                break
            gap = 0
            if rng.random() < late_prob[brand]:
                gap = int(rng.exponential(base_gap[brand] * 60))
            date = date + pd.Timedelta(days=days_supply + gap)

    claims = pd.DataFrame(claim_rows, columns=["patient_id", "brand", "fill_date", "days_supply"])
    claims.insert(0, "claim_id", [f"C{i:07d}" for i in range(len(claims))])
    patients = pd.DataFrame(
        {
            "patient_id": [f"P{i:06d}" for i in range(n)],
            "age_band": age_band,
            "region": region,
            "index_date": index_date,
            "index_brand": index_brand,
        }
    )
    return patients, claims


def generate(cfg: SynthConfig | None = None) -> dict[str, pd.DataFrame]:
    """Generate every table. Deterministic for a given config."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    hcps = _make_hcps(rng, cfg)
    formulary = _make_formulary(rng, cfg)
    promo = _make_promo(rng, hcps, cfg)
    weekly_rx = _make_weekly_rx(rng, hcps, promo, formulary, cfg)
    patients, claims = _make_patients_and_claims(rng, cfg)
    return {
        "brands": BRANDS.copy(),
        "regions": REGIONS.copy(),
        "hcps": hcps,
        "formulary": formulary,
        "promo": promo,
        "weekly_rx": weekly_rx,
        "patients": patients,
        "claims": claims,
    }


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def build_database(path: str | Path, cfg: SynthConfig | None = None) -> Path:
    """Generate the data and write it to a DuckDB file using ``schema.sql``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    tables = generate(cfg)
    con = duckdb.connect(str(path))
    try:
        con.execute(SCHEMA_PATH.read_text())
        for name, df in tables.items():
            con.register("_df", df)
            cols = ", ".join(df.columns)
            con.execute(f"INSERT INTO {name} ({cols}) SELECT {cols} FROM _df")
            con.unregister("_df")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return path
