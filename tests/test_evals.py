from pharmapulse.evals.run_evals import extract_numbers, fact_present, run, summarise


def test_number_extraction_handles_formats():
    assert extract_numbers("62.7K TRx, 1,234 HCPs, 37.5%") == [62700.0, 1234.0, 37.5]


def test_fact_matching():
    assert fact_present(0.3750, "Glyvance holds 37.5% share")
    assert fact_present(0.3750, "share 0.375")
    assert not fact_present(0.3750, "share 35.0%")
    assert fact_present(62723.8, "about 62.7K TRx")
    assert fact_present(221, "221 HCPs")
    assert fact_present("H00560", "top target h00560")


def test_mock_eval_subset_passes():
    rows = run("mock", only=["share_latest", "persistence_brand", "sql_hcps"])
    s = summarise(rows, "mock")
    assert s["pass_rate"] == 1.0
