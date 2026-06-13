"""Persistent smoke test for tools.py.

Runs the real tool functions against the known fixture sample_data/sales.csv.
No model, no network, no API key — just proves the tools behave and that the
intentionally-messy fixture is intact. Run it with:

    python test_tools.py

Exits 0 and prints PASS lines if everything checks out; raises (non-zero exit)
on the first failed assertion.
"""

import tools

CSV = "sample_data/sales.csv"


def _count(sql: str) -> int:
    """Run a single-number query and return it as an int."""
    # query_csv returns a text table: "header\nvalue\n(1 rows)". The value is
    # the second line.
    out = tools.query_csv(CSV, sql)
    return int(out.splitlines()[1])


def test_list_files_finds_csv():
    assert "sales.csv" in tools.list_files("sample_data").splitlines()


def test_read_file_returns_header():
    first_line = tools.read_file(CSV).splitlines()[0]
    assert first_line.startswith("order_id,order_date,customer_name")


def test_row_count_is_40():
    assert _count(f"SELECT count(*) FROM '{CSV}'") == 40


def test_messy_categories_present():
    # 4 Electronics + 4 Furniture spelling/casing variants = 8 distinct strings.
    # If this drops to 2, the messy fixture was accidentally cleaned.
    assert _count(f"SELECT count(DISTINCT product_category) FROM '{CSV}'") == 8


def test_duplicate_order_ids():
    out = tools.query_csv(
        CSV,
        f"SELECT order_id FROM '{CSV}' GROUP BY order_id HAVING count(*) > 1 "
        "ORDER BY order_id",
    )
    assert "1014" in out and "1020" in out


def test_dispatch_routes_to_query_csv():
    out = tools.dispatch("query_csv", {"file_path": CSV, "sql": f"SELECT 7 FROM '{CSV}' LIMIT 1"})
    assert "7" in out


def test_bad_sql_raises():
    # The agent loop relies on this: a bad call raises, the loop catches it and
    # feeds the error back so the model can self-correct.
    raised = False
    try:
        tools.query_csv(CSV, "SELECT * FROM no_such_table")
    except Exception:
        raised = True
    assert raised, "expected a bad SQL query to raise"


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} checks passed.")


if __name__ == "__main__":
    main()
