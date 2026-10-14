import pytest

from pharmapulse.llm.sql_guard import UnsafeSQLError, validate


def test_select_gets_limit():
    out = validate("SELECT brand, SUM(trx) FROM weekly_rx GROUP BY 1", row_limit=50)
    assert "LIMIT 50" in out


def test_existing_small_limit_kept():
    assert "LIMIT 5" in validate("SELECT * FROM hcps LIMIT 5", row_limit=50)


def test_percent_limit_still_capped():
    assert validate("SELECT * FROM claims LIMIT 10 PERCENT", row_limit=50).endswith("LIMIT 50")


def test_large_limit_capped():
    out = validate("SELECT * FROM hcps LIMIT 100000", row_limit=50)
    assert out.endswith("LIMIT 50")


def test_cte_and_join_allowed():
    sql = """WITH t AS (SELECT hcp_id, SUM(trx) v FROM weekly_rx GROUP BY 1)
             SELECT h.specialty, AVG(t.v) FROM t JOIN hcps h USING (hcp_id) GROUP BY 1"""
    validate(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE hcps",
        "DELETE FROM hcps",
        "INSERT INTO hcps VALUES ('x','y','z',1)",
        "UPDATE hcps SET region = 'x'",
        "CREATE TABLE x AS SELECT 1",
        "SELECT 1; DROP TABLE hcps",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM secrets",
        "COPY hcps TO '/tmp/x.csv'",
        "ATTACH 'other.db'",
        "PRAGMA database_list",
        "SELECT getenv('HOME')",
    ],
)
def test_unsafe_sql_rejected(sql):
    with pytest.raises(UnsafeSQLError):
        validate(sql)


@pytest.fixture
def secret(tmp_path):
    f = tmp_path / "secret.csv"
    f.write_text("token\nTOPSECRET\n")
    return f.as_posix()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM brands, LATERAL read_ndjson('{f}')",
        "SELECT * FROM brands, LATERAL json_execute_serialized_sql("
        "json_serialize_sql('SELECT * FROM read_text(''{f}'')'))",
        'SELECT * FROM (WITH "{f}" AS (SELECT 1) SELECT 1) t, "{f}"',
        'WITH a AS (SELECT * FROM "{f}"), "{f}" AS (SELECT 1) SELECT * FROM a',
    ],
)
def test_file_reads_blocked_end_to_end(secret, sql):
    from pharmapulse.llm import tools

    out, is_error = tools.execute("run_sql", {"sql": sql.format(f=secret)})
    assert is_error and "TOPSECRET" not in out


def test_tool_args_cannot_inject_sql():
    from pharmapulse.llm import tools

    for name in ("brand_growth", "launch_uptake"):
        out, is_error = tools.execute(name, {"metric": "CAST(1 AS INT)"})
        assert is_error and "metric must be" in out
