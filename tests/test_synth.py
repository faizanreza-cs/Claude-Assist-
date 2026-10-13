import pandas as pd

from pharmapulse.data.synth import SynthConfig, generate
from pharmapulse.db import query

CFG = SynthConfig(seed=3, n_hcps=50, n_patients=200)


def test_generation_is_deterministic():
    a, b = generate(CFG), generate(CFG)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_changes_data():
    a = generate(CFG)["weekly_rx"]
    b = generate(SynthConfig(seed=4, n_hcps=50, n_patients=200))["weekly_rx"]
    assert not a.equals(b)


def test_nrx_never_exceeds_trx():
    rx = generate(CFG)["weekly_rx"]
    assert (rx["nrx"] <= rx["trx"]).all()
    assert (rx["trx"] > 0).all()


def test_launch_brand_has_no_rx_before_launch():
    tables = generate(CFG)
    rx = tables["weekly_rx"]
    launch = pd.Timestamp("2023-01-02") + pd.Timedelta(weeks=CFG.launch_week)
    assert rx[(rx["brand"] == "Glucora") & (rx["week"] < launch)].empty
    assert not tables["claims"][
        (tables["claims"]["brand"] == "Glucora") & (tables["claims"]["fill_date"] < launch)
    ].shape[0]


def test_database_loaded_with_all_tables(small_db):
    for table in [
        "brands",
        "regions",
        "hcps",
        "weekly_rx",
        "promo",
        "formulary",
        "patients",
        "claims",
    ]:
        assert query(f"SELECT COUNT(*) AS n FROM {table}")["n"].iloc[0] > 0
    weeks = query("SELECT COUNT(DISTINCT week) AS n FROM weekly_rx")["n"].iloc[0]
    assert weeks == 156
