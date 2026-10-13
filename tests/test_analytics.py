import numpy as np
import pandas as pd
import pytest

from pharmapulse.analytics import (
    forecasting,
    hcp_segmentation,
    market_share,
    patient_journey,
    promo_response,
)

# ---------------------------------------------------------------- forecasting


def test_metrics_on_known_values():
    a, p = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert forecasting.wape(a, p) == pytest.approx(30 / 300)
    assert forecasting.mape(a, p) == pytest.approx((0.1 + 0.1) / 2)
    assert forecasting.bias(a, p) == pytest.approx(-10 / 300)


@pytest.mark.parametrize("model", forecasting.MODELS)
def test_each_model_forecasts_horizon(model):
    fc = forecasting.forecast("Glyvance", horizon=8, model=model)
    assert len(fc.forecast) == 8
    assert (fc.forecast["lower"] <= fc.forecast["yhat"]).all()
    assert (fc.forecast["yhat"] <= fc.forecast["upper"]).all()
    assert (fc.forecast["yhat"] >= 0).all()


def test_backtest_beats_naive_mean_and_is_reasonable():
    bt = forecasting.backtest("Dapaflo")
    assert set(bt["model"]) == set(forecasting.MODELS)
    assert bt["wape"].min() < 0.15


def test_auto_model_picks_lowest_wape():
    bt = forecasting.backtest("Sitaglin")
    assert forecasting.forecast("Sitaglin", 13).model == bt.sort_values("wape").iloc[0]["model"]


def test_forecast_rejects_bad_horizon():
    with pytest.raises(ValueError):
        forecasting.forecast("Glyvance", horizon=0)


# ---------------------------------------------------------------- market share


@pytest.mark.parametrize("by", ["none", "region", "specialty"])
def test_shares_sum_to_one(by):
    df = market_share.share(by=by)
    sums = df.groupby(by)["share"].sum() if by != "none" else pd.Series([df["share"].sum()])
    assert np.allclose(sums, 1.0, atol=1e-3)


def test_share_trend_sums_to_one_each_period():
    df = market_share.share_trend(freq="quarter")
    assert np.allclose(df.groupby("period")["share"].sum(), 1.0, atol=1e-3)


def test_growth_has_all_brands_and_period():
    g = market_share.growth()
    assert set(g["brand"]) == {"Glyvance", "Sitaglin", "Dapaflo", "Glucora"}
    assert g.attrs["period"]["end"] > g.attrs["period"]["start"]


def test_launch_uptake_starts_at_zero_weeks():
    df = market_share.launch_uptake("Glucora", weeks=10)
    assert df["weeks_since_launch"].iloc[0] == 0
    assert df["share"].between(0, 1).all()


# ---------------------------------------------------------------- patient journey


def _claims(rows):
    df = pd.DataFrame(rows, columns=["patient_id", "brand", "fill_date", "days_supply"])
    df["fill_date"] = pd.to_datetime(df["fill_date"])
    df["age_band"], df["region"] = "45-64", "West"
    return df


def test_pdc_hand_checked():
    # 30-day fills on day 0, 30, 90 with a 100-day window -> covered 0-60 and 90-100 = 70/100
    c = _claims(
        [
            ("P1", "A", "2024-01-01", 30),
            ("P1", "A", "2024-01-31", 30),
            ("P1", "A", "2024-03-31", 30),
            ("P2", "A", "2024-12-31", 30),  # extends data end so P1 has a full window
        ]
    )
    out = patient_journey.pdc(c, window_days=100).set_index("patient_id")
    assert out.loc["P1", "pdc"] == pytest.approx(0.70)


def test_pdc_shifts_overlapping_supply_forward():
    # early refill on day 20 of a 30-day fill: overlap pushed forward, not double counted
    c = _claims(
        [
            ("P1", "A", "2024-01-01", 30),
            ("P1", "A", "2024-01-21", 30),
            ("P2", "A", "2024-12-31", 30),
        ]
    )
    out = patient_journey.pdc(c, window_days=100).set_index("patient_id")
    assert out.loc["P1", "pdc"] == pytest.approx(0.60)


def test_episode_gap_rule_and_switch():
    c = _claims(
        [
            ("P1", "A", "2024-01-01", 30),
            ("P1", "A", "2024-01-31", 30),
            ("P1", "A", "2024-06-01", 30),  # gap > 60 days after supply end (Mar 1) -> event
            ("P2", "A", "2024-01-01", 30),
            ("P2", "B", "2024-02-05", 30),  # switch
            ("P3", "A", "2024-12-01", 30),  # still on therapy at data end -> censored
        ]
    )
    ep = patient_journey.therapy_episodes(c, gap_days=60).set_index("patient_id")
    assert ep.loc["P1", "event"] == 1 and ep.loc["P1", "duration_days"] == 60
    assert ep.loc["P2", "event"] == 1 and ep.loc["P2", "switched_to"] == "B"
    assert ep.loc["P3", "event"] == 0


def test_persistence_is_monotone_and_bounded():
    df = patient_journey.persistence()
    assert (df["persistent_90d"] >= df["persistent_180d"]).all()
    assert (df["persistent_180d"] >= df["persistent_365d"]).all()
    assert df["persistent_365d"].between(0, 1).all()


def test_dpp4_persists_worse_than_glp1():
    # the generator gives Sitaglin a higher discontinuation hazard
    df = patient_journey.persistence().set_index("brand")
    assert df.loc["Sitaglin", "persistent_365d"] < df.loc["Glyvance", "persistent_365d"]


def test_switching_matrix_excludes_self_switches():
    m = patient_journey.switching_matrix()
    assert (m["from_brand"] != m["to_brand"]).all()


# ---------------------------------------------------------------- HCP targeting


def test_deciles_split_volume_evenly():
    df = hcp_segmentation.add_deciles(hcp_segmentation.hcp_profile())
    vol = df[df["decile"] > 0].groupby("decile")["market_trx"].sum()
    assert len(vol) == 10
    assert vol.max() / vol.sum() < 0.2  # no decile holds more than ~2x its 10% share


def test_segments_cover_all_writers():
    seg = hcp_segmentation.segment()
    assert seg["segment"].notna().all()
    assert seg["segment"].nunique() == 4


def test_next_best_targets_filters_and_sorts():
    t = hcp_segmentation.next_best_targets(5, region="West")
    assert (t["region"] == "West").all()
    assert t["score"].is_monotonic_decreasing


# ---------------------------------------------------------------- promo response


def test_adstock():
    s = promo_response.adstock(pd.Series([1.0, 0.0, 0.0]), decay=0.5)
    assert s.tolist() == [1.0, 0.5, 0.25]


def test_promo_response_positive_effect():
    r = promo_response.fit()
    assert r.elasticity > 0 and r.ci_low > 0
    assert r.incremental_nrx_per_100_calls > 0
