"""Tests for dbx_toolkit.io_utils.

Only functions that work with a local SparkSession (no Delta Lake or Unity
Catalog required) are exercised here.  Functions that depend on Delta /
Unity Catalog are represented by stub tests marked as skipped.
"""

from __future__ import annotations

import os
import uuid

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from dbx_toolkit.io_utils import (
    list_tables,
    read_csv_with_schema,
    read_table_safe,
    table_exists,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_view(prefix: str = "test_view") -> str:
    """Return a unique temp-view name to avoid cross-test collisions."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 1. table_exists
# ---------------------------------------------------------------------------

class TestTableExists:
    def test_returns_true_for_existing_temp_view(self, spark: SparkSession) -> None:
        view_name = _unique_view("te_exists")
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
        df.createOrReplaceTempView(view_name)

        assert table_exists(spark, view_name) is True

    def test_returns_false_for_nonexistent_table(self, spark: SparkSession) -> None:
        assert table_exists(spark, "this_table_does_not_exist_xyz") is False

    def test_returns_false_after_view_is_dropped(self, spark: SparkSession) -> None:
        view_name = _unique_view("te_dropped")
        df = spark.createDataFrame([(1,)], ["id"])
        df.createOrReplaceTempView(view_name)
        assert table_exists(spark, view_name) is True

        spark.catalog.dropTempView(view_name)
        assert table_exists(spark, view_name) is False


# ---------------------------------------------------------------------------
# 2. read_table_safe — basic reads
# ---------------------------------------------------------------------------

class TestReadTableSafe:
    @pytest.fixture(autouse=True)
    def _register_view(self, spark: SparkSession) -> None:
        """Register a temp view used by every test in this class."""
        self.view_name = _unique_view("rts")
        data = [(1, "alice", 30), (2, "bob", 25), (3, "carol", 35)]
        schema = StructType([
            StructField("id", IntegerType(), nullable=False),
            StructField("name", StringType(), nullable=True),
            StructField("age", IntegerType(), nullable=True),
        ])
        df = spark.createDataFrame(data, schema)
        df.createOrReplaceTempView(self.view_name)

    def test_returns_none_for_nonexistent_table(self, spark: SparkSession) -> None:
        result = read_table_safe(spark, "nonexistent_table_abc123")
        assert result is None

    def test_returns_all_rows_by_default(self, spark: SparkSession) -> None:
        df = read_table_safe(spark, self.view_name)
        assert df is not None
        assert df.count() == 3

    def test_column_selection(self, spark: SparkSession) -> None:
        df = read_table_safe(spark, self.view_name, columns=["id", "name"])
        assert df is not None
        assert df.columns == ["id", "name"]
        assert df.count() == 3

    def test_filter_expr(self, spark: SparkSession) -> None:
        df = read_table_safe(spark, self.view_name, filter_expr="age > 25")
        assert df is not None
        assert df.count() == 2  # alice (30) and carol (35)

    def test_limit(self, spark: SparkSession) -> None:
        df = read_table_safe(spark, self.view_name, limit=2)
        assert df is not None
        assert df.count() == 2

    def test_columns_and_filter_and_limit_combined(self, spark: SparkSession) -> None:
        df = read_table_safe(
            spark,
            self.view_name,
            columns=["id", "age"],
            filter_expr="age >= 30",
            limit=1,
        )
        assert df is not None
        assert df.columns == ["id", "age"]
        assert df.count() == 1

    def test_limit_zero_returns_empty_dataframe(self, spark: SparkSession) -> None:
        df = read_table_safe(spark, self.view_name, limit=0)
        assert df is not None
        assert df.count() == 0


# ---------------------------------------------------------------------------
# 3. read_table_safe — input validation
# ---------------------------------------------------------------------------

class TestReadTableSafeValidation:
    @pytest.fixture(autouse=True)
    def _register_view(self, spark: SparkSession) -> None:
        self.view_name = _unique_view("rts_val")
        df = spark.createDataFrame([(1, "x")], ["id", "label"])
        df.createOrReplaceTempView(self.view_name)

    def test_raises_value_error_for_nonexistent_column(self, spark: SparkSession) -> None:
        with pytest.raises(ValueError, match="nonexistent_col"):
            read_table_safe(spark, self.view_name, columns=["id", "nonexistent_col"])

    def test_raises_value_error_for_all_columns_missing(self, spark: SparkSession) -> None:
        with pytest.raises(ValueError):
            read_table_safe(spark, self.view_name, columns=["foo", "bar"])

    def test_raises_value_error_for_negative_limit(self, spark: SparkSession) -> None:
        with pytest.raises(ValueError, match="limit"):
            read_table_safe(spark, self.view_name, limit=-1)


# ---------------------------------------------------------------------------
# 4. read_csv_with_schema
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires DBFS or Volume - not available on serverless")
class TestReadCsvWithSchema:
    CSV_PATH = "/tmp/test_io_utils_sample.csv"

    @pytest.fixture(autouse=True)
    def _write_csv(self) -> None:
        """Write a small CSV to /tmp before each test."""
        content = "id,name,score\n1,alice,95\n2,bob,80\n3,carol,88\n"
        with open(self.CSV_PATH, "w") as fh:
            fh.write(content)

    def test_reads_csv_with_infer_schema(self, spark: SparkSession) -> None:
        df = read_csv_with_schema(spark, self.CSV_PATH)
        assert df.count() == 3
        assert set(df.columns) == {"id", "name", "score"}

    def test_reads_csv_with_explicit_schema(self, spark: SparkSession) -> None:
        schema = StructType([
            StructField("id", IntegerType(), nullable=True),
            StructField("name", StringType(), nullable=True),
            StructField("score", IntegerType(), nullable=True),
        ])
        df = read_csv_with_schema(spark, self.CSV_PATH, schema=schema)
        assert df.count() == 3
        assert df.schema == schema

    def test_reads_csv_correct_values(self, spark: SparkSession) -> None:
        df = read_csv_with_schema(spark, self.CSV_PATH)
        rows = {row["name"]: row["score"] for row in df.collect()}
        assert rows["alice"] == 95
        assert rows["bob"] == 80
        assert rows["carol"] == 88

    def test_reads_csv_with_custom_delimiter(self, spark: SparkSession) -> None:
        pipe_path = "/tmp/test_io_utils_pipe.csv"
        with open(pipe_path, "w") as fh:
            fh.write("id|city\n1|London\n2|Paris\n")
        df = read_csv_with_schema(spark, pipe_path, delimiter="|")
        assert df.count() == 2
        assert "city" in df.columns

    def test_reads_csv_null_values(self, spark: SparkSession) -> None:
        null_csv_path = "/tmp/test_io_utils_null.csv"
        with open(null_csv_path, "w") as fh:
            fh.write("id,name\n1,alice\n2,NA\n3,\n")
        df = read_csv_with_schema(spark, null_csv_path, null_values=["NA", ""])
        rows = {row["id"]: row["name"] for row in df.collect()}
        # "NA" and empty string should be treated as null
        assert rows[2] is None
        assert rows[3] is None

    def test_reads_csv_no_header(self, spark: SparkSession) -> None:
        no_header_path = "/tmp/test_io_utils_noheader.csv"
        with open(no_header_path, "w") as fh:
            fh.write("1,alice\n2,bob\n")
        df = read_csv_with_schema(spark, no_header_path, header=False)
        assert df.count() == 2
        # Without header, Spark uses _c0, _c1, ...
        assert "_c0" in df.columns


# ---------------------------------------------------------------------------
# 5. list_tables
# ---------------------------------------------------------------------------

class TestListTables:
    def test_temp_view_appears_in_listing(self, spark: SparkSession) -> None:
        view_name = _unique_view("lt_present")
        df = spark.createDataFrame([(1,)], ["id"])
        df.createOrReplaceTempView(view_name)

        results = list_tables(spark)
        names = [r["name"] for r in results]
        assert view_name in names

    def test_multiple_views_appear_in_listing(self, spark: SparkSession) -> None:
        view_a = _unique_view("lt_multi_a")
        view_b = _unique_view("lt_multi_b")
        base_df = spark.createDataFrame([(1,)], ["id"])
        base_df.createOrReplaceTempView(view_a)
        base_df.createOrReplaceTempView(view_b)

        results = list_tables(spark)
        names = [r["name"] for r in results]
        assert view_a in names
        assert view_b in names

    def test_pattern_filter_matches_only_matching_tables(self, spark: SparkSession) -> None:
        prefix = f"lt_pat_{uuid.uuid4().hex[:6]}"
        view_match = f"{prefix}_sales"
        view_no_match = _unique_view("lt_other")
        base_df = spark.createDataFrame([(1,)], ["id"])
        base_df.createOrReplaceTempView(view_match)
        base_df.createOrReplaceTempView(view_no_match)

        results = list_tables(spark, pattern=f"{prefix}_*")
        names = [r["name"] for r in results]
        assert view_match in names
        assert view_no_match not in names

    def test_result_has_expected_keys(self, spark: SparkSession) -> None:
        view_name = _unique_view("lt_keys")
        spark.createDataFrame([(1,)], ["id"]).createOrReplaceTempView(view_name)

        results = list_tables(spark)
        match = next((r for r in results if r["name"] == view_name), None)
        assert match is not None
        assert set(match.keys()) == {"catalog", "database", "name", "table_type", "is_temporary"}

    def test_temp_view_is_temporary_flag(self, spark: SparkSession) -> None:
        view_name = _unique_view("lt_tmp_flag")
        spark.createDataFrame([(1,)], ["id"]).createOrReplaceTempView(view_name)

        results = list_tables(spark)
        match = next((r for r in results if r["name"] == view_name), None)
        assert match is not None
        assert match["is_temporary"] == "True"


# ---------------------------------------------------------------------------
# 6. Stub tests for Delta / Unity Catalog functions
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires Delta Lake / Unity Catalog")
def test_write_table_stub() -> None:
    """write_table writes a DataFrame to a Delta table and optionally optimizes."""
    pass


@pytest.mark.skip(reason="Requires Delta Lake / Unity Catalog")
def test_merge_into_stub() -> None:
    """merge_into performs a Delta MERGE (upsert) on a target table."""
    pass


@pytest.mark.skip(reason="Requires Delta Lake / Unity Catalog")
def test_scd_type2_stub() -> None:
    """scd_type2 implements a Slowly Changing Dimension Type 2 merge pattern."""
    pass


@pytest.mark.skip(reason="Requires Delta Lake / Unity Catalog")
def test_backup_table_stub() -> None:
    """backup_table creates a DEEP CLONE of a Delta table with a timestamp suffix."""
    pass


@pytest.mark.skip(reason="Requires Delta Lake / Unity Catalog")
def test_get_table_info_stub() -> None:
    """get_table_info returns metadata (row count, size, partitions) for a Delta table."""
    pass


@pytest.mark.skip(reason="Requires Delta Lake / Unity Catalog")
def test_create_table_if_not_exists_stub() -> None:
    """create_table_if_not_exists creates a Delta table only when it is absent."""
    pass
