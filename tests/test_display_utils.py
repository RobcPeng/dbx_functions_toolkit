"""Tests for dbx_toolkit.display_utils."""

import pytest
from pyspark.sql import Row
from pyspark.sql import types as T

from dbx_toolkit.display_utils import (
    add_row_numbers,
    compare_side_by_side,
    crosstab_pct,
    format_number_columns,
    format_pct_columns,
    histogram_data,
    peek,
    pivot_summary,
    show_df,
    summary_table,
    top_n_by_group,
)


# ---------------------------------------------------------------------------
# 1. show_df
# ---------------------------------------------------------------------------

class TestShowDf:
    def test_returns_same_dataframe(self, spark):
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])
        result = show_df(df)
        assert result is df

    def test_returns_same_dataframe_with_options(self, spark):
        df = spark.createDataFrame([(1, "hello world")], ["id", "text"])
        result = show_df(df, n=1, truncate=True, vertical=True)
        assert result is df

    def test_does_not_mutate_schema(self, spark):
        df = spark.createDataFrame([(10,), (20,)], ["value"])
        result = show_df(df)
        assert result.columns == ["value"]
        assert result.count() == 2


# ---------------------------------------------------------------------------
# 2. peek
# ---------------------------------------------------------------------------

class TestPeek:
    def test_returns_same_dataframe(self, spark):
        df = spark.createDataFrame([(1, "x"), (2, "y"), (3, "z")], ["id", "label"])
        result = peek(df)
        assert result is df

    def test_returns_same_dataframe_custom_n(self, spark):
        df = spark.createDataFrame([(i,) for i in range(10)], ["num"])
        result = peek(df, n=3)
        assert result is df

    def test_does_not_lose_rows(self, spark):
        df = spark.createDataFrame([(i, str(i)) for i in range(50)], ["id", "val"])
        result = peek(df, n=5)
        assert result.count() == 50


# ---------------------------------------------------------------------------
# 3. summary_table
# ---------------------------------------------------------------------------

class TestSummaryTable:
    def test_row_count_equals_number_of_input_columns(self, spark):
        df = spark.createDataFrame(
            [(1, "alice", 3.14), (2, "bob", 2.71)],
            ["id", "name", "score"],
        )
        result = summary_table(df)
        assert result.count() == len(df.columns)  # 3

    def test_has_column_name_column(self, spark):
        df = spark.createDataFrame([(1, "a")], ["my_id", "my_label"])
        result = summary_table(df)
        assert "column" in result.columns

    def test_column_names_match_input(self, spark):
        df = spark.createDataFrame([(1, "x", 0.5)], ["alpha", "beta", "gamma"])
        result = summary_table(df)
        returned_names = {row["column"] for row in result.collect()}
        assert returned_names == {"alpha", "beta", "gamma"}

    def test_output_schema_fields(self, spark):
        df = spark.createDataFrame([(1,)], ["val"])
        result = summary_table(df)
        expected_cols = {
            "column", "data_type", "nullable",
            "null_count", "null_pct", "distinct_count", "sample_values",
        }
        assert set(result.columns) == expected_cols

    def test_null_count_computed(self, spark):
        schema = T.StructType([
            T.StructField("id", T.IntegerType(), True),
            T.StructField("val", T.StringType(), True),
        ])
        df = spark.createDataFrame([(1, None), (2, "hello"), (3, None)], schema=schema)
        result = summary_table(df)
        val_row = result.filter(result["column"] == "val").collect()[0]
        assert val_row["null_count"] == 2

    def test_single_column_dataframe(self, spark):
        df = spark.createDataFrame([(1,), (2,), (3,)], ["only_col"])
        result = summary_table(df)
        assert result.count() == 1
        assert result.collect()[0]["column"] == "only_col"


# ---------------------------------------------------------------------------
# 4. compare_side_by_side
# ---------------------------------------------------------------------------

class TestCompareSideBySide:
    def test_returns_dataframe(self, spark):
        df1 = spark.createDataFrame([(1, "a")], ["id", "name"])
        df2 = spark.createDataFrame([(1, "a")], ["id", "name"])
        result = compare_side_by_side(df1, df2)
        assert hasattr(result, "columns")

    def test_has_column_field(self, spark):
        df1 = spark.createDataFrame([(1, "a")], ["id", "name"])
        df2 = spark.createDataFrame([(1, "a")], ["id", "name"])
        result = compare_side_by_side(df1, df2)
        assert "column" in result.columns

    def test_row_count_is_union_of_columns(self, spark):
        df1 = spark.createDataFrame([(1,)], ["a"])
        df2 = spark.createDataFrame([(1,)], ["b"])
        result = compare_side_by_side(df1, df2)
        # Union of {"a"} and {"b"} = 2 distinct columns
        assert result.count() == 2

    def test_identical_schemas_all_match(self, spark):
        df1 = spark.createDataFrame([(1, "x")], ["id", "label"])
        df2 = spark.createDataFrame([(2, "y")], ["id", "label"])
        result = compare_side_by_side(df1, df2)
        rows = result.collect()
        assert all(row["types_match"] for row in rows)

    def test_column_only_in_df1_flagged(self, spark):
        df1 = spark.createDataFrame([(1, "a")], ["id", "extra"])
        df2 = spark.createDataFrame([(1,)], ["id"])
        result = compare_side_by_side(df1, df2, name1="left", name2="right")
        extra_row = result.filter(result["column"] == "extra").collect()[0]
        assert extra_row["in_left"] is True
        assert extra_row["in_right"] is False

    def test_custom_names_used_in_columns(self, spark):
        df1 = spark.createDataFrame([(1,)], ["id"])
        df2 = spark.createDataFrame([(1,)], ["id"])
        result = compare_side_by_side(df1, df2, name1="before", name2="after")
        assert "in_before" in result.columns
        assert "in_after" in result.columns
        assert "type_before" in result.columns
        assert "type_after" in result.columns

    def test_type_mismatch_detected(self, spark):
        df1 = spark.createDataFrame(
            [(1,)],
            T.StructType([T.StructField("val", T.IntegerType(), True)]),
        )
        df2 = spark.createDataFrame(
            [("hello",)],
            T.StructType([T.StructField("val", T.StringType(), True)]),
        )
        result = compare_side_by_side(df1, df2)
        val_row = result.filter(result["column"] == "val").collect()[0]
        assert val_row["types_match"] is False


# ---------------------------------------------------------------------------
# 5. format_number_columns
# ---------------------------------------------------------------------------

class TestFormatNumberColumns:
    def test_rounds_to_specified_decimals(self, spark):
        df = spark.createDataFrame([(1.23456,)], ["price"])
        result = format_number_columns(df, columns=["price"], decimals=2)
        value = result.collect()[0]["price"]
        assert round(value, 2) == 1.23

    def test_auto_detects_numeric_columns(self, spark):
        df = spark.createDataFrame(
            [(3.14159, "text")],
            ["amount", "label"],
        )
        result = format_number_columns(df, decimals=2)
        row = result.collect()[0]
        assert round(row["amount"], 2) == 3.14
        assert row["label"] == "text"

    def test_non_numeric_columns_unchanged(self, spark):
        df = spark.createDataFrame([(1.5, "keep_me")], ["num", "str_col"])
        result = format_number_columns(df, columns=["num"], decimals=0)
        assert result.collect()[0]["str_col"] == "keep_me"

    def test_preserves_row_count(self, spark):
        df = spark.createDataFrame([(i * 1.111,) for i in range(5)], ["val"])
        result = format_number_columns(df, decimals=1)
        assert result.count() == 5

    def test_decimals_zero(self, spark):
        df = spark.createDataFrame([(7.89,)], ["x"])
        result = format_number_columns(df, columns=["x"], decimals=0)
        # round to 0 decimals gives 8.0 as a double
        assert result.collect()[0]["x"] == 8.0

    def test_explicit_subset_of_columns(self, spark):
        df = spark.createDataFrame([(1.999, 2.999)], ["a", "b"])
        result = format_number_columns(df, columns=["a"], decimals=1)
        row = result.collect()[0]
        assert round(row["a"], 1) == 2.0
        # "b" is not in columns list, should remain unrounded
        assert row["b"] == 2.999


# ---------------------------------------------------------------------------
# 6. format_pct_columns
# ---------------------------------------------------------------------------

class TestFormatPctColumns:
    def test_formats_as_percentage_string(self, spark):
        df = spark.createDataFrame([(0.123,)], ["rate"])
        result = format_pct_columns(df, columns=["rate"], decimals=1)
        value = result.collect()[0]["rate"]
        assert value == "12.3%"

    def test_null_remains_null(self, spark):
        schema = T.StructType([T.StructField("rate", T.DoubleType(), True)])
        df = spark.createDataFrame([(None,)], schema=schema)
        result = format_pct_columns(df, columns=["rate"], decimals=1)
        assert result.collect()[0]["rate"] is None

    def test_output_column_is_string_type(self, spark):
        df = spark.createDataFrame([(0.5,)], ["pct"])
        result = format_pct_columns(df, columns=["pct"], decimals=1)
        dtype = dict(result.dtypes)["pct"]
        assert dtype == "string"

    def test_zero_formats_correctly(self, spark):
        df = spark.createDataFrame([(0.0,)], ["pct"])
        result = format_pct_columns(df, columns=["pct"], decimals=1)
        assert result.collect()[0]["pct"] == "0.0%"

    def test_one_formats_correctly(self, spark):
        df = spark.createDataFrame([(1.0,)], ["pct"])
        result = format_pct_columns(df, columns=["pct"], decimals=1)
        assert result.collect()[0]["pct"] == "100.0%"

    def test_multiple_columns_formatted(self, spark):
        df = spark.createDataFrame([(0.25, 0.75)], ["conv", "churn"])
        result = format_pct_columns(df, columns=["conv", "churn"], decimals=1)
        row = result.collect()[0]
        assert row["conv"] == "25.0%"
        assert row["churn"] == "75.0%"

    def test_decimals_parameter_respected(self, spark):
        df = spark.createDataFrame([(0.12345,)], ["rate"])
        result = format_pct_columns(df, columns=["rate"], decimals=2)
        assert result.collect()[0]["rate"] == "12.35%"


# ---------------------------------------------------------------------------
# 7. add_row_numbers
# ---------------------------------------------------------------------------

class TestAddRowNumbers:
    def test_adds_row_num_column(self, spark):
        df = spark.createDataFrame([(1,), (2,), (3,)], ["val"])
        result = add_row_numbers(df)
        assert "row_num" in result.columns

    def test_row_num_is_first_column(self, spark):
        df = spark.createDataFrame([(1, "a")], ["id", "name"])
        result = add_row_numbers(df)
        assert result.columns[0] == "row_num"

    def test_original_columns_preserved(self, spark):
        df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])
        result = add_row_numbers(df)
        assert "id" in result.columns
        assert "name" in result.columns

    def test_row_count_unchanged(self, spark):
        df = spark.createDataFrame([(i,) for i in range(7)], ["val"])
        result = add_row_numbers(df)
        assert result.count() == 7

    def test_ids_are_monotonically_increasing(self, spark):
        df = spark.createDataFrame([(i,) for i in range(5)], ["val"])
        result = add_row_numbers(df)
        ids = [row["row_num"] for row in result.orderBy("row_num").collect()]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)  # all unique

    def test_custom_column_name(self, spark):
        df = spark.createDataFrame([(1,), (2,)], ["x"])
        result = add_row_numbers(df, col_name="my_id")
        assert "my_id" in result.columns
        assert result.columns[0] == "my_id"


# ---------------------------------------------------------------------------
# 8. pivot_summary
# ---------------------------------------------------------------------------

class TestPivotSummary:
    def test_simple_groupby_sum(self, spark):
        df = spark.createDataFrame(
            [("A", 10), ("A", 20), ("B", 5)],
            ["region", "revenue"],
        )
        result = pivot_summary(df, group_col="region", value_col="revenue", agg_func="sum")
        rows = {row["region"]: row["sum_revenue"] for row in result.collect()}
        assert rows["A"] == 30
        assert rows["B"] == 5

    def test_groupby_count(self, spark):
        df = spark.createDataFrame(
            [("X", 1), ("X", 2), ("Y", 3)],
            ["cat", "val"],
        )
        result = pivot_summary(df, group_col="cat", value_col="val", agg_func="count")
        rows = {row["cat"]: row["count_val"] for row in result.collect()}
        assert rows["X"] == 2
        assert rows["Y"] == 1

    def test_groupby_avg(self, spark):
        df = spark.createDataFrame(
            [("A", 10.0), ("A", 20.0), ("B", 5.0)],
            ["region", "revenue"],
        )
        result = pivot_summary(df, group_col="region", value_col="revenue", agg_func="avg")
        rows = {row["region"]: row["avg_revenue"] for row in result.collect()}
        assert rows["A"] == 15.0

    def test_pivot_col_creates_wide_format(self, spark):
        df = spark.createDataFrame(
            [("A", "P1", 10), ("A", "P2", 20), ("B", "P1", 5)],
            ["region", "product", "revenue"],
        )
        result = pivot_summary(
            df,
            group_col="region",
            value_col="revenue",
            agg_func="sum",
            pivot_col="product",
        )
        # Should have region + one column per product
        assert "region" in result.columns
        assert "P1" in result.columns
        assert "P2" in result.columns

    def test_invalid_agg_func_raises(self, spark):
        df = spark.createDataFrame([(1, 2)], ["a", "b"])
        with pytest.raises(ValueError, match="agg_func"):
            pivot_summary(df, group_col="a", value_col="b", agg_func="median")

    def test_result_ordered_by_group_col(self, spark):
        df = spark.createDataFrame(
            [("C", 1), ("A", 2), ("B", 3)],
            ["grp", "val"],
        )
        result = pivot_summary(df, group_col="grp", value_col="val", agg_func="sum")
        groups = [row["grp"] for row in result.collect()]
        assert groups == sorted(groups)


# ---------------------------------------------------------------------------
# 9. histogram_data
# ---------------------------------------------------------------------------

class TestHistogramData:
    def test_returns_correct_columns(self, spark):
        df = spark.createDataFrame([(float(i),) for i in range(100)], ["amount"])
        result = histogram_data(df, column="amount", n_bins=10)
        assert set(result.columns) == {"bin_index", "bin_start", "bin_end", "count"}

    def test_correct_number_of_bins(self, spark):
        df = spark.createDataFrame([(float(i),) for i in range(50)], ["val"])
        result = histogram_data(df, column="val", n_bins=5)
        assert result.count() == 5

    def test_total_count_equals_input_rows(self, spark):
        n_rows = 30
        df = spark.createDataFrame([(float(i),) for i in range(n_rows)], ["x"])
        result = histogram_data(df, column="x", n_bins=6)
        total = result.agg({"count": "sum"}).collect()[0][0]
        assert total == n_rows

    def test_ordered_by_bin_index(self, spark):
        df = spark.createDataFrame([(float(i),) for i in range(20)], ["val"])
        result = histogram_data(df, column="val", n_bins=4)
        indices = [row["bin_index"] for row in result.collect()]
        assert indices == sorted(indices)

    def test_bin_start_less_than_bin_end(self, spark):
        df = spark.createDataFrame([(float(i),) for i in range(10)], ["val"])
        result = histogram_data(df, column="val", n_bins=3)
        for row in result.collect():
            assert row["bin_start"] < row["bin_end"]

    def test_single_distinct_value_returns_one_bin(self, spark):
        df = spark.createDataFrame([(5.0,), (5.0,), (5.0,)], ["val"])
        result = histogram_data(df, column="val", n_bins=10)
        assert result.count() == 1
        assert result.collect()[0]["count"] == 3

    def test_nulls_are_excluded_from_counts(self, spark):
        schema = T.StructType([T.StructField("val", T.DoubleType(), True)])
        df = spark.createDataFrame([(1.0,), (2.0,), (None,), (3.0,)], schema=schema)
        result = histogram_data(df, column="val", n_bins=3)
        total = result.agg({"count": "sum"}).collect()[0][0]
        assert total == 3  # null excluded


# ---------------------------------------------------------------------------
# 10. top_n_by_group
# ---------------------------------------------------------------------------

class TestTopNByGroup:
    def test_returns_at_most_n_rows_per_group(self, spark):
        df = spark.createDataFrame(
            [("A", i) for i in range(10)] + [("B", i) for i in range(10)],
            ["cat", "score"],
        )
        result = top_n_by_group(df, group_col="cat", order_col="score", n=3)
        for cat in ["A", "B"]:
            count = result.filter(result["cat"] == cat).count()
            assert count == 3

    def test_adds_rank_column(self, spark):
        df = spark.createDataFrame(
            [("X", 5), ("X", 3), ("X", 7)],
            ["grp", "val"],
        )
        result = top_n_by_group(df, group_col="grp", order_col="val", n=3)
        assert "rank" in result.columns

    def test_rank_starts_at_one(self, spark):
        df = spark.createDataFrame(
            [("G", 10), ("G", 20), ("G", 30)],
            ["grp", "val"],
        )
        result = top_n_by_group(df, group_col="grp", order_col="val", n=3)
        ranks = {row["rank"] for row in result.collect()}
        assert 1 in ranks

    def test_descending_returns_highest_values(self, spark):
        df = spark.createDataFrame(
            [("A", v) for v in [1, 2, 3, 4, 5]],
            ["grp", "score"],
        )
        result = top_n_by_group(
            df, group_col="grp", order_col="score", n=2, ascending=False
        )
        scores = {row["score"] for row in result.collect()}
        assert 5 in scores
        assert 4 in scores
        assert 1 not in scores

    def test_ascending_returns_lowest_values(self, spark):
        df = spark.createDataFrame(
            [("A", v) for v in [1, 2, 3, 4, 5]],
            ["grp", "score"],
        )
        result = top_n_by_group(
            df, group_col="grp", order_col="score", n=2, ascending=True
        )
        scores = {row["score"] for row in result.collect()}
        assert 1 in scores
        assert 2 in scores
        assert 5 not in scores

    def test_multiple_groups_independent_ranking(self, spark):
        df = spark.createDataFrame(
            [("A", 100), ("A", 50), ("B", 200), ("B", 10)],
            ["grp", "val"],
        )
        result = top_n_by_group(df, group_col="grp", order_col="val", n=1)
        top_values = {row["grp"]: row["val"] for row in result.collect()}
        assert top_values["A"] == 100
        assert top_values["B"] == 200

    def test_n_larger_than_group_size_returns_all(self, spark):
        df = spark.createDataFrame(
            [("A", 1), ("A", 2)],
            ["grp", "val"],
        )
        result = top_n_by_group(df, group_col="grp", order_col="val", n=100)
        assert result.count() == 2


# ---------------------------------------------------------------------------
# 11. crosstab_pct
# ---------------------------------------------------------------------------

class TestCrosstabPct:
    def test_returns_dataframe(self, spark):
        df = spark.createDataFrame(
            [("A", "X"), ("A", "Y"), ("B", "X")],
            ["col1", "col2"],
        )
        result = crosstab_pct(df, col1="col1", col2="col2")
        assert hasattr(result, "columns")

    def test_first_column_is_label(self, spark):
        df = spark.createDataFrame(
            [("cat", "dog"), ("cat", "bird"), ("cat", "dog")],
            ["animal", "friend"],
        )
        result = crosstab_pct(df, col1="animal", col2="friend")
        # First column follows PySpark crosstab convention: <col1>_<col2>
        assert result.columns[0] == "animal_friend"

    def test_values_sum_to_100(self, spark):
        df = spark.createDataFrame(
            [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")],
            ["r", "c"],
        )
        result = crosstab_pct(df, col1="r", col2="c")
        label_col = result.columns[0]
        value_cols = result.columns[1:]
        from pyspark.sql import functions as F
        total = result.agg(
            sum(F.sum(F.col(col)) for col in value_cols).alias("total")
        ).collect()[0]["total"]
        assert abs(total - 100.0) < 0.01

    def test_numeric_columns_are_percentages(self, spark):
        # With 4 equal rows in a 2x2 grid each cell should be 25%
        df = spark.createDataFrame(
            [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")],
            ["r", "c"],
        )
        result = crosstab_pct(df, col1="r", col2="c")
        label_col = result.columns[0]
        value_cols = result.columns[1:]
        for row in result.collect():
            for col in value_cols:
                assert abs(row[col] - 25.0) < 0.01

    def test_distinct_col2_values_become_columns(self, spark):
        df = spark.createDataFrame(
            [("A", "P"), ("A", "Q"), ("B", "R")],
            ["row_dim", "col_dim"],
        )
        result = crosstab_pct(df, col1="row_dim", col2="col_dim")
        # P, Q, R should all be columns (excluding the label column)
        value_cols = set(result.columns[1:])
        assert {"P", "Q", "R"}.issubset(value_cols)
