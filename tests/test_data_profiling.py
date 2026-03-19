"""
Tests for dbx_toolkit.data_profiling module.

Uses the session-scoped `spark` fixture defined in conftest.py.
"""

import math

import pytest
from pyspark.sql import types as T
from pyspark.sql import functions as F

from dbx_toolkit.data_profiling import (
    profile_table,
    null_report,
    cardinality_report,
    value_counts,
    numeric_summary,
    correlation_matrix,
    detect_outliers_iqr,
    compare_dataframes,
    schema_diff,
    data_quality_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col_map(rows, key_col, val_col):
    """Return {key: val} dict from a list of collected Row objects."""
    return {row[key_col]: row[val_col] for row in rows}


# ===========================================================================
# 1. profile_table
# ===========================================================================

class TestProfileTable:

    def test_mixed_types(self, spark):
        """Profile a DataFrame with string, int, and double columns."""
        df = spark.createDataFrame(
            [("alice", 30, 1.5), ("bob", 25, 2.0), ("carol", 35, 3.5)],
            ["name", "age", "score"],
        )
        result = profile_table(df)
        rows = result.collect()

        assert result.count() == 3
        col_names = [r["column_name"] for r in rows]
        assert set(col_names) == {"name", "age", "score"}

        # Verify expected output columns exist
        expected_cols = {
            "column_name", "data_type", "non_null_count", "null_count",
            "null_pct", "distinct_count", "min", "max", "mean", "stddev",
            "sample_values",
        }
        assert expected_cols == set(result.columns)

        by_col = {r["column_name"]: r for r in rows}

        # age column checks
        age = by_col["age"]
        assert age["non_null_count"] == 3
        assert age["null_count"] == 0
        assert age["null_pct"] == 0.0
        assert age["distinct_count"] == 3
        assert age["mean"] is not None
        assert age["stddev"] is not None

        # name column: mean and stddev should be None (not numeric)
        name = by_col["name"]
        assert name["mean"] is None
        assert name["stddev"] is None

    def test_map_array_struct_columns_do_not_error(self, spark):
        """MAP/ARRAY/STRUCT columns must not raise errors (min/max bug regression)."""
        schema = T.StructType([
            T.StructField("id",    T.IntegerType(), True),
            T.StructField("tags",  T.ArrayType(T.StringType()), True),
            T.StructField("props", T.MapType(T.StringType(), T.StringType()), True),
            T.StructField("info",  T.StructType([
                T.StructField("x", T.IntegerType(), True),
            ]), True),
        ])
        data = [
            (1, ["a", "b"], {"k": "v"}, (10,)),
            (2, ["c"],      {"m": "n"}, (20,)),
        ]
        df = spark.createDataFrame(data, schema)

        # Should not raise any exception
        result = profile_table(df)
        assert result.count() == 4

        by_col = {r["column_name"]: r for r in result.collect()}

        # Non-orderable types must have None for min/max
        for col_name in ("tags", "props", "info"):
            assert by_col[col_name]["min"] is None
            assert by_col[col_name]["max"] is None

        # Orderable column (id) should have min/max populated
        assert by_col["id"]["min"] is not None
        assert by_col["id"]["max"] is not None

    def test_with_nulls(self, spark):
        """Null counts and percentages are computed correctly."""
        df = spark.createDataFrame(
            [("alice", 30), ("bob", None), (None, None)],
            ["name", "age"],
        )
        result = profile_table(df)
        by_col = {r["column_name"]: r for r in result.collect()}

        assert by_col["name"]["null_count"] == 1
        assert by_col["name"]["null_pct"] == pytest.approx(33.3333, abs=0.01)
        assert by_col["age"]["null_count"] == 2
        assert by_col["age"]["null_pct"] == pytest.approx(66.6667, abs=0.01)

    def test_empty_dataframe(self, spark):
        """Empty DataFrame returns correctly-structured empty result."""
        df = spark.createDataFrame([], T.StructType([
            T.StructField("id",   T.IntegerType(), True),
            T.StructField("name", T.StringType(),  True),
        ]))
        result = profile_table(df)

        assert result.count() == 0
        expected_cols = {
            "column_name", "data_type", "non_null_count", "null_count",
            "null_pct", "distinct_count", "min", "max", "mean", "stddev",
            "sample_values",
        }
        assert set(result.columns) == expected_cols

    def test_sampling_parameter(self, spark):
        """sample_size parameter limits the profiled rows without erroring."""
        data = [(i, f"name_{i}") for i in range(200)]
        df = spark.createDataFrame(data, ["id", "name"])

        # sample_size smaller than total row count — should still return one row per column
        result = profile_table(df, sample_size=50)
        assert result.count() == 2  # two columns → two profile rows

        # sample_size >= total rows — full DataFrame used
        result_full = profile_table(df, sample_size=10_000)
        assert result_full.count() == 2


# ===========================================================================
# 2. null_report
# ===========================================================================

class TestNullReport:

    def test_with_nulls(self, spark):
        """Columns with nulls are reported with correct counts and percentages."""
        df = spark.createDataFrame(
            [(1, "a", 1.0), (2, None, None), (3, None, None)],
            ["id", "name", "score"],
        )
        result = null_report(df)
        rows = result.collect()
        by_col = _col_map(rows, "column_name", "null_count")

        assert by_col["id"] == 0
        assert by_col["name"] == 2
        assert by_col["score"] == 2

        pct_map = _col_map(rows, "column_name", "null_pct")
        assert pct_map["name"] == pytest.approx(66.6667, abs=0.01)

        # Result is sorted by null_pct descending: id last
        col_order = [r["column_name"] for r in rows]
        assert col_order[-1] == "id"

    def test_no_nulls(self, spark):
        """All null_counts are 0 when there are no nulls."""
        df = spark.createDataFrame(
            [(1, "a"), (2, "b")],
            ["id", "name"],
        )
        result = null_report(df)
        for row in result.collect():
            assert row["null_count"] == 0
            assert row["null_pct"] == 0.0

    def test_empty_dataframe(self, spark):
        """Empty DataFrame returns empty result with correct schema."""
        df = spark.createDataFrame([], T.StructType([
            T.StructField("id",   T.IntegerType(), True),
            T.StructField("name", T.StringType(),  True),
        ]))
        result = null_report(df)
        assert result.count() == 0
        assert set(result.columns) == {"column_name", "null_count", "null_pct"}


# ===========================================================================
# 3. cardinality_report
# ===========================================================================

class TestCardinalityReport:

    def test_unique_column(self, spark):
        """A column with all-unique values has cardinality_ratio == 1.0."""
        df = spark.createDataFrame(
            [(1,), (2,), (3,), (4,)],
            ["id"],
        )
        result = cardinality_report(df)
        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["distinct_count"] == 4
        assert rows[0]["cardinality_ratio"] == pytest.approx(1.0)

    def test_constant_column(self, spark):
        """A constant column has distinct_count == 1 and cardinality_ratio near 0."""
        df = spark.createDataFrame(
            [("X",), ("X",), ("X",), ("X",)],
            ["category"],
        )
        result = cardinality_report(df)
        rows = result.collect()
        assert rows[0]["distinct_count"] == 1
        assert rows[0]["cardinality_ratio"] == pytest.approx(0.25)

    def test_sorted_descending(self, spark):
        """Result is sorted by cardinality_ratio descending."""
        df = spark.createDataFrame(
            [(1, "A"), (2, "A"), (3, "B"), (4, "A")],
            ["id", "category"],
        )
        result = cardinality_report(df)
        ratios = [r["cardinality_ratio"] for r in result.collect()]
        assert ratios == sorted(ratios, reverse=True)

    def test_empty_dataframe(self, spark):
        """Empty DataFrame returns empty result with correct schema."""
        df = spark.createDataFrame([], T.StructType([
            T.StructField("id", T.IntegerType(), True),
        ]))
        result = cardinality_report(df)
        assert result.count() == 0
        assert set(result.columns) == {"column_name", "distinct_count", "cardinality_ratio"}


# ===========================================================================
# 4. value_counts
# ===========================================================================

class TestValueCounts:

    def test_basic_counts(self, spark):
        """Counts and percentages are computed correctly."""
        df = spark.createDataFrame(
            [("A",), ("A",), ("B",), ("C",), ("A",)],
            ["cat"],
        )
        result = value_counts(df, "cat")
        rows = result.collect()
        by_val = {r["value"]: r for r in rows}

        assert by_val["A"]["count"] == 3
        assert by_val["B"]["count"] == 1
        assert by_val["C"]["count"] == 1
        assert by_val["A"]["pct"] == pytest.approx(60.0, abs=0.01)

        # Result is sorted by count descending
        counts = [r["count"] for r in rows]
        assert counts == sorted(counts, reverse=True)

    def test_top_n_limit(self, spark):
        """top_n limits the number of rows returned."""
        data = [(str(i),) for i in range(50)]
        df = spark.createDataFrame(data, ["val"])
        result = value_counts(df, "val", top_n=5)
        assert result.count() == 5

    def test_invalid_column_raises(self, spark):
        """Requesting a non-existent column raises ValueError."""
        df = spark.createDataFrame([(1,), (2,)], ["id"])
        with pytest.raises(ValueError, match="nonexistent"):
            value_counts(df, "nonexistent")

    def test_empty_dataframe(self, spark):
        """Empty DataFrame returns empty result with correct schema."""
        df = spark.createDataFrame([], T.StructType([
            T.StructField("cat", T.StringType(), True),
        ]))
        result = value_counts(df, "cat")
        assert result.count() == 0
        assert set(result.columns) == {"value", "count", "pct"}


# ===========================================================================
# 5. numeric_summary
# ===========================================================================

class TestNumericSummary:

    def test_all_numeric_columns(self, spark):
        """Summary statistics computed for all numeric columns automatically."""
        df = spark.createDataFrame(
            [(1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0)],
            ["a", "b"],
        )
        result = numeric_summary(df)
        rows = result.collect()

        assert result.count() == 2
        expected_cols = {
            "column_name", "mean", "stddev", "min", "p25", "p50", "p75",
            "max", "skewness", "kurtosis",
        }
        assert set(result.columns) == expected_cols

        by_col = {r["column_name"]: r for r in rows}
        assert by_col["a"]["mean"] == pytest.approx(2.5)
        assert by_col["a"]["min"] == pytest.approx(1.0)
        assert by_col["a"]["max"] == pytest.approx(4.0)

    def test_specific_column_selection(self, spark):
        """Only the requested columns are summarised."""
        df = spark.createDataFrame(
            [(1, 10, 100), (2, 20, 200), (3, 30, 300)],
            ["a", "b", "c"],
        )
        result = numeric_summary(df, columns=["b"])
        rows = result.collect()
        assert result.count() == 1
        assert rows[0]["column_name"] == "b"

    def test_no_numeric_columns_returns_empty(self, spark):
        """DataFrame with only string columns returns empty result."""
        df = spark.createDataFrame(
            [("x",), ("y",)],
            ["name"],
        )
        result = numeric_summary(df)
        assert result.count() == 0
        assert "column_name" in result.columns

    def test_empty_dataframe(self, spark):
        """Empty DataFrame returns empty result with correct schema."""
        df = spark.createDataFrame([], T.StructType([
            T.StructField("val", T.DoubleType(), True),
        ]))
        result = numeric_summary(df)
        assert result.count() == 0
        expected_cols = {
            "column_name", "mean", "stddev", "min", "p25", "p50", "p75",
            "max", "skewness", "kurtosis",
        }
        assert set(result.columns) == expected_cols


# ===========================================================================
# 6. correlation_matrix
# ===========================================================================

class TestCorrelationMatrix:

    def test_correlated_columns(self, spark):
        """Perfectly correlated columns return correlation of 1.0."""
        df = spark.createDataFrame(
            [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)],
            ["x", "y"],
        )
        result = correlation_matrix(df)
        rows = result.collect()

        assert result.count() == 1
        assert rows[0]["column_a"] == "x"
        assert rows[0]["column_b"] == "y"
        assert rows[0]["pearson_correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_columns_cast_to_double(self, spark):
        """Integer columns are cast to double internally — no type errors."""
        df = spark.createDataFrame(
            [(i, i * 3) for i in range(1, 10)],
            ["int_a", "int_b"],
        )
        result = correlation_matrix(df, columns=["int_a", "int_b"])
        assert result.count() == 1
        corr = result.collect()[0]["pearson_correlation"]
        assert corr is not None
        assert not math.isnan(corr)

    def test_single_column_returns_empty(self, spark):
        """A single column produces an empty result (need at least 2 columns)."""
        df = spark.createDataFrame(
            [(1.0,), (2.0,), (3.0,)],
            ["x"],
        )
        result = correlation_matrix(df, columns=["x"])
        assert result.count() == 0
        assert set(result.columns) == {"column_a", "column_b", "pearson_correlation"}

    def test_all_null_column_returns_none_not_nan(self, spark):
        """When one column is all-null the correlation is stored as None, not NaN."""
        df = spark.createDataFrame(
            [(1.0, None), (2.0, None), (3.0, None)],
            T.StructType([
                T.StructField("x", T.DoubleType(), True),
                T.StructField("y", T.DoubleType(), True),
            ]),
        )
        result = correlation_matrix(df)
        assert result.count() == 1
        corr = result.collect()[0]["pearson_correlation"]
        assert corr is None


# ===========================================================================
# 7. detect_outliers_iqr
# ===========================================================================

class TestDetectOutliersIqr:

    def test_clear_outlier_flagged(self, spark):
        """A value far outside the IQR fences is flagged as an outlier."""
        # Values 1-9 are tightly clustered; 1000 is a clear outlier
        data = [(float(i),) for i in range(1, 10)] + [(1000.0,)]
        df = spark.createDataFrame(data, ["val"])
        result = detect_outliers_iqr(df, "val")

        outliers = result.filter(F.col("is_outlier") == True).collect()
        assert len(outliers) == 1
        assert outliers[0]["val"] == pytest.approx(1000.0)

    def test_no_outliers(self, spark):
        """Uniform data produces no outlier flags."""
        data = [(float(i),) for i in range(1, 11)]
        df = spark.createDataFrame(data, ["val"])
        result = detect_outliers_iqr(df, "val")

        outlier_count = result.filter(F.col("is_outlier") == True).count()
        assert outlier_count == 0

        # Output schema extends original with three new columns
        assert "iqr_lower_bound" in result.columns
        assert "iqr_upper_bound" in result.columns
        assert "is_outlier" in result.columns

    def test_null_values_flagged_false(self, spark):
        """Null values in the target column are given is_outlier = False."""
        df = spark.createDataFrame(
            [(1.0,), (2.0,), (None,), (3.0,)],
            T.StructType([T.StructField("val", T.DoubleType(), True)]),
        )
        result = detect_outliers_iqr(df, "val")
        null_rows = result.filter(F.col("val").isNull()).collect()

        assert len(null_rows) == 1
        assert null_rows[0]["is_outlier"] == False

    def test_invalid_column_raises(self, spark):
        """Non-existent column raises ValueError."""
        df = spark.createDataFrame([(1.0,)], ["val"])
        with pytest.raises(ValueError, match="missing"):
            detect_outliers_iqr(df, "missing")

    def test_empty_dataframe(self, spark):
        """Empty DataFrame returns correct schema with no rows."""
        df = spark.createDataFrame([], T.StructType([
            T.StructField("val", T.DoubleType(), True),
        ]))
        result = detect_outliers_iqr(df, "val")
        assert result.count() == 0
        assert "iqr_lower_bound" in result.columns
        assert "iqr_upper_bound" in result.columns
        assert "is_outlier" in result.columns


# ===========================================================================
# 8. compare_dataframes
# ===========================================================================

class TestCompareDataframes:

    def _make_df(self, spark, rows):
        return spark.createDataFrame(rows, ["id", "name", "value"])

    def test_identical_dataframes_no_diffs(self, spark):
        """Identical DataFrames produce an empty result."""
        data = [(1, "alice", 10), (2, "bob", 20)]
        df1 = self._make_df(spark, data)
        df2 = self._make_df(spark, data)
        result = compare_dataframes(df1, df2, key_columns=["id"])
        assert result.count() == 0

    def test_added_row_in_df2(self, spark):
        """A row present only in df2 is labelled only_in_df2."""
        df1 = self._make_df(spark, [(1, "alice", 10)])
        df2 = self._make_df(spark, [(1, "alice", 10), (2, "bob", 20)])
        result = compare_dataframes(df1, df2, key_columns=["id"])

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["diff_type"] == "only_in_df2"
        assert rows[0]["id"] == 2

    def test_removed_row_in_df1(self, spark):
        """A row present only in df1 is labelled only_in_df1."""
        df1 = self._make_df(spark, [(1, "alice", 10), (2, "bob", 20)])
        df2 = self._make_df(spark, [(1, "alice", 10)])
        result = compare_dataframes(df1, df2, key_columns=["id"])

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["diff_type"] == "only_in_df1"
        assert rows[0]["id"] == 2

    def test_changed_row(self, spark):
        """A row whose value columns differ is labelled changed; both versions appear."""
        df1 = self._make_df(spark, [(1, "alice", 10), (2, "bob", 20)])
        df2 = self._make_df(spark, [(1, "alice", 10), (2, "bob", 99)])
        result = compare_dataframes(df1, df2, key_columns=["id"])

        changed = result.filter(F.col("diff_type") == "changed").collect()
        # Both the df1 and df2 versions of the changed row must be present
        assert len(changed) == 2
        sources = {r["source"] for r in changed}
        assert sources == {"df1", "df2"}
        values = {r["value"] for r in changed}
        assert values == {20, 99}

    def test_empty_key_columns_raises(self, spark):
        """Passing an empty key_columns list raises ValueError."""
        df = self._make_df(spark, [(1, "a", 1)])
        with pytest.raises(ValueError, match="key_columns"):
            compare_dataframes(df, df, key_columns=[])

    def test_missing_key_column_raises(self, spark):
        """A key column missing from a DataFrame raises ValueError."""
        df = self._make_df(spark, [(1, "a", 1)])
        with pytest.raises(ValueError):
            compare_dataframes(df, df, key_columns=["nonexistent"])


# ===========================================================================
# 9. schema_diff
# ===========================================================================

class TestSchemaDiff:

    def test_added_column(self, spark):
        """A column present in df2 but not df1 is marked as added."""
        df1 = spark.createDataFrame([(1,)], ["id"])
        df2 = spark.createDataFrame([(1, "x")], ["id", "name"])
        result = schema_diff(df1, df2)

        rows = {r["column_name"]: r for r in result.collect()}
        assert rows["name"]["status"] == "added"
        assert rows["name"]["type_in_df1"] is None
        assert rows["name"]["type_in_df2"] is not None

    def test_removed_column(self, spark):
        """A column present in df1 but not df2 is marked as removed."""
        df1 = spark.createDataFrame([(1, "x")], ["id", "name"])
        df2 = spark.createDataFrame([(1,)], ["id"])
        result = schema_diff(df1, df2)

        rows = {r["column_name"]: r for r in result.collect()}
        assert rows["name"]["status"] == "removed"
        assert rows["name"]["type_in_df1"] is not None
        assert rows["name"]["type_in_df2"] is None

    def test_type_changed_column(self, spark):
        """A column whose type differs between df1 and df2 is marked as type_changed."""
        schema1 = T.StructType([
            T.StructField("id",  T.IntegerType(), True),
            T.StructField("val", T.IntegerType(), True),
        ])
        schema2 = T.StructType([
            T.StructField("id",  T.IntegerType(), True),
            T.StructField("val", T.DoubleType(),  True),
        ])
        df1 = spark.createDataFrame([(1, 10)], schema1)
        df2 = spark.createDataFrame([(1, 10.0)], schema2)
        result = schema_diff(df1, df2)

        rows = {r["column_name"]: r for r in result.collect()}
        assert rows["val"]["status"] == "type_changed"
        assert rows["id"]["status"] == "unchanged"

    def test_unchanged_column(self, spark):
        """A column with the same name and type in both DataFrames is unchanged."""
        df1 = spark.createDataFrame([(1,)], ["id"])
        df2 = spark.createDataFrame([(2,)], ["id"])
        result = schema_diff(df1, df2)

        rows = {r["column_name"]: r for r in result.collect()}
        assert rows["id"]["status"] == "unchanged"

    def test_result_columns(self, spark):
        """Result always contains the four expected columns."""
        df1 = spark.createDataFrame([(1,)], ["id"])
        df2 = spark.createDataFrame([(1,)], ["id"])
        result = schema_diff(df1, df2)
        assert set(result.columns) == {"column_name", "status", "type_in_df1", "type_in_df2"}


# ===========================================================================
# 10. data_quality_score
# ===========================================================================

class TestDataQualityScore:

    def test_perfect_data(self, spark):
        """Data with no nulls and varied values should score high."""
        df = spark.createDataFrame(
            [(i, f"name_{i}", float(i)) for i in range(1, 11)],
            ["id", "name", "score"],
        )
        result = data_quality_score(df)
        row = result.collect()[0]

        assert row["total_rows"] == 10
        assert row["total_columns"] == 3
        assert row["completeness_score"] == pytest.approx(100.0)
        # All columns have cardinality_ratio == 1.0 > 0.01 -> uniqueness == 100
        assert row["uniqueness_score"] == pytest.approx(100.0)
        # Both numeric cols (id, score) have non-zero stddev -> validity == 100
        assert row["validity_score"] == pytest.approx(100.0)
        assert row["overall_score"] == pytest.approx(100.0)

    def test_data_with_nulls_lowers_completeness(self, spark):
        """Nulls reduce the completeness score below 100."""
        df = spark.createDataFrame(
            [(1, "a"), (2, None), (3, None)],
            ["id", "name"],
        )
        result = data_quality_score(df)
        row = result.collect()[0]

        assert row["completeness_score"] < 100.0
        assert row["overall_score"] < 100.0

    def test_constant_column_lowers_validity(self, spark):
        """A numeric column with zero stddev lowers the validity score."""
        df = spark.createDataFrame(
            [(1, 5), (2, 5), (3, 5), (4, 5)],
            ["id", "constant_val"],
        )
        result = data_quality_score(df)
        row = result.collect()[0]

        # id has non-zero stddev, constant_val has zero stddev → 50% valid
        assert row["validity_score"] == pytest.approx(50.0)

    def test_empty_dataframe(self, spark):
        """Empty DataFrame returns all-zero scores."""
        df = spark.createDataFrame([], T.StructType([
            T.StructField("id", T.IntegerType(), True),
        ]))
        result = data_quality_score(df)
        row = result.collect()[0]

        assert row["total_rows"] == 0
        assert row["completeness_score"] == 0.0
        assert row["uniqueness_score"] == 0.0
        assert row["validity_score"] == 0.0
        assert row["overall_score"] == 0.0

    def test_result_schema(self, spark):
        """Result always has the six expected columns."""
        df = spark.createDataFrame([(1, "a")], ["id", "name"])
        result = data_quality_score(df)
        expected_cols = {
            "total_rows", "total_columns", "completeness_score",
            "uniqueness_score", "validity_score", "overall_score",
        }
        assert set(result.columns) == expected_cols
        assert result.count() == 1
