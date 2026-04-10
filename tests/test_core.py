"""Unit tests for pure functions that don't require Ollama or external services."""

import numpy as np
import pandas as pd
import pytest

from data_cleaning import parse_size_generic, clean_text
from duckdb_connect import extract_sql, basic_sql_safety_check


# =============================================================
# parse_size_generic
# =============================================================
class TestParseSizeGeneric:
    def test_ml(self):
        qty, unit = parse_size_generic("60 ml")
        assert qty == 60.0
        assert unit == "ml"

    def test_oz_converts_to_ml(self):
        qty, unit = parse_size_generic("2 oz")
        assert unit == "ml"
        assert round(qty, 2) == 59.15

    def test_fl_oz(self):
        qty, unit = parse_size_generic("1.7 fl oz")
        assert unit == "ml"
        assert qty > 0

    def test_grams(self):
        qty, unit = parse_size_generic("50 g")
        assert qty == 50.0
        assert unit == "g"

    def test_multi_unit_picks_ml(self):
        qty, unit = parse_size_generic("2 oz/ 60 mL")
        assert unit == "ml"

    def test_empty_string(self):
        qty, unit = parse_size_generic("")
        assert np.isnan(qty)
        assert unit is None

    def test_none_input(self):
        qty, unit = parse_size_generic(None)
        assert np.isnan(qty)
        assert unit is None

    def test_no_unit(self):
        qty, unit = parse_size_generic("just text")
        assert np.isnan(qty)
        assert unit is None

    def test_pcs(self):
        qty, unit = parse_size_generic("10 pcs")
        assert qty == 10.0
        assert unit == "count"


# =============================================================
# extract_sql
# =============================================================
class TestExtractSQL:
    def test_plain_select(self):
        assert extract_sql("SELECT * FROM sephora") == "SELECT * FROM sephora"

    def test_fenced_sql_block(self):
        raw = "```sql\nSELECT brand_name FROM sephora;\n```"
        assert extract_sql(raw) == "SELECT brand_name FROM sephora"

    def test_fenced_generic_block(self):
        raw = "```\nSELECT count(*) FROM sephora\n```"
        assert extract_sql(raw) == "SELECT count(*) FROM sephora"

    def test_with_cte(self):
        raw = "WITH cte AS (SELECT * FROM sephora) SELECT * FROM cte"
        result = extract_sql(raw)
        assert result.startswith("WITH")

    def test_empty_input(self):
        assert extract_sql("") is None
        assert extract_sql(None) is None

    def test_no_sql(self):
        assert extract_sql("I don't know the answer") is None

    def test_strips_trailing_semicolon(self):
        result = extract_sql("SELECT 1 FROM sephora;")
        assert not result.endswith(";")


# =============================================================
# basic_sql_safety_check
# =============================================================
class TestSQLSafetyCheck:
    def test_valid_select(self):
        ok, err = basic_sql_safety_check("SELECT * FROM sephora")
        assert ok is True
        assert err is None

    def test_valid_with(self):
        ok, err = basic_sql_safety_check("WITH cte AS (SELECT 1 FROM sephora) SELECT * FROM cte")
        assert ok is True

    def test_rejects_delete(self):
        ok, err = basic_sql_safety_check("DELETE FROM sephora")
        assert ok is False

    def test_rejects_drop(self):
        ok, err = basic_sql_safety_check("DROP TABLE sephora")
        assert ok is False

    def test_rejects_insert(self):
        ok, err = basic_sql_safety_check("INSERT INTO sephora VALUES (1)")
        assert ok is False

    def test_rejects_empty(self):
        ok, err = basic_sql_safety_check("")
        assert ok is False

    def test_rejects_missing_table(self):
        ok, err = basic_sql_safety_check("SELECT * FROM other_table")
        assert ok is False
        assert "sephora" in err.lower()

    def test_rejects_multiple_statements(self):
        ok, err = basic_sql_safety_check("SELECT 1 FROM sephora; DROP TABLE sephora")
        assert ok is False


# =============================================================
# clean_text (smoke test with minimal DataFrame)
# =============================================================
class TestCleanText:
    def _make_df(self, n=25):
        """Create a minimal DataFrame that passes clean_text requirements."""
        ratings = [1.0, 2.0, 3.0, 4.0, 5.0] * (n // 5)
        return pd.DataFrame({
            "product_id": ["P1"] * n,
            "product_name": ["Test Product"] * n,
            "brand_id": [1] * n,
            "brand_name": ["TestBrand"] * n,
            "rating": ratings[:n],
            "loves_count": [100] * n,
            "price_usd": [10.0 + i * 0.5 for i in range(n)],
            "primary_category": ["Skincare"] * n,
            "secondary_category": ["Moisturizers"] * n,
            "tertiary_category": ["Face Cream"] * n,
            "author_id": [f"A{i}" for i in range(n)],
            "review_text": [f"Unique review text number {i} about this product" for i in range(n)],
            "review_title": ["Good"] * n,
            "submission_time": ["2023-01-01"] * n,
            "size": ["50 ml"] * n,
            "sale_price_usd": [None] * n,
        })

    def test_returns_dataframe(self):
        df = self._make_df()
        result = clean_text(df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_expected_columns(self):
        df = self._make_df()
        result = clean_text(df)
        for col in ["review_text_clean", "rating_bucket", "final_price_usd", "size_qty", "size_unit", "price_per_100"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_rating_bucket_values(self):
        df = self._make_df()
        result = clean_text(df)
        valid_buckets = {"negative", "neutral", "positive"}
        actual = set(result["rating_bucket"].dropna().unique())
        assert actual.issubset(valid_buckets)
