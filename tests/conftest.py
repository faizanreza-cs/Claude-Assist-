"""Build a small, seeded database once per test session."""

from __future__ import annotations

import os

import pytest

from pharmapulse.data.synth import SynthConfig, build_database

SMALL = SynthConfig(seed=7, n_hcps=300, n_patients=2_000)


@pytest.fixture(scope="session", autouse=True)
def small_db(tmp_path_factory: pytest.TempPathFactory):
    path = build_database(tmp_path_factory.mktemp("db") / "test.duckdb", SMALL)
    old = os.environ.get("PHARMAPULSE_DB")
    os.environ["PHARMAPULSE_DB"] = str(path)
    os.environ["PHARMAPULSE_PROVIDER"] = "mock"
    from pharmapulse.analytics import patient_journey

    patient_journey._all_episodes.cache_clear()
    patient_journey.adherence.cache_clear()
    yield path
    if old is None:
        os.environ.pop("PHARMAPULSE_DB", None)
    else:
        os.environ["PHARMAPULSE_DB"] = old
