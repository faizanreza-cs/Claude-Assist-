"""Guardrails for model-written SQL.

The agent may run ad-hoc SQL, but only:
* a single read-only ``SELECT`` (CTEs allowed),
* against an allow-listed set of tables,
* with no file/extension/pragma functions (``read_csv``, ``read_parquet`` ...),
* with an enforced row ``LIMIT``.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = frozenset(
    {"brands", "regions", "hcps", "weekly_rx", "promo", "formulary", "patients", "claims"}
)
BLOCKED_FUNCTIONS = frozenset(
    {
        "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
        "read_text", "read_blob", "glob", "query_table", "query", "getenv",
        "duckdb_settings", "duckdb_extensions", "pragma_database_list",
    }
)  # fmt: skip


class UnsafeSQLError(ValueError):
    pass


def validate(sql: str, row_limit: int = 200) -> str:
    """Return a safe, LIMIT-ed version of ``sql`` or raise ``UnsafeSQLError``."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQLError("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.Union | exp.Intersect | exp.Except):
        raise UnsafeSQLError(f"only SELECT queries are allowed (got {tree.key.upper()})")
    for node in tree.walk():
        if isinstance(node, exp.Insert | exp.Update | exp.Delete | exp.Drop | exp.Create
                      | exp.Alter | exp.Command | exp.Copy):  # fmt: skip
            raise UnsafeSQLError("write / DDL operations are not allowed")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if isinstance(table.this, exp.Func) or not name:
            raise UnsafeSQLError("table functions are not allowed")
        if name not in ALLOWED_TABLES and name not in cte_names:
            raise UnsafeSQLError(f"table {table.name!r} is not allowed")
        if table.args.get("db") or table.args.get("catalog"):
            raise UnsafeSQLError("schema-qualified tables are not allowed")

    for func in tree.find_all(exp.Func):
        fname = (func.sql_name() if not isinstance(func, exp.Anonymous) else func.name).lower()
        if fname in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(f"function {fname!r} is not allowed")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(row_limit)
    else:
        try:
            n = int(limit.expression.name)
        except (AttributeError, ValueError):
            n = row_limit + 1
        if n > row_limit:
            tree = tree.limit(row_limit)
    return tree.sql(dialect="duckdb")
