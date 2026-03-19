"""
test_data_cleansing.py
======================
Pytest tests for dbx_toolkit.data_cleansing module.

All tests use the session-scoped ``spark`` fixture defined in conftest.py.
"""

import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from dbx_toolkit.data_cleansing import (
    cap_outliers,
    clean_column_names,
    deduplicate,
    drop_constant_columns,
    drop_null_columns,
    enforce_types,
    fill_nulls_by_type,
    flag_invalid_rows,
    normalize_values,
    remove_outliers,
    split_valid_invalid,
    standardize_strings,
    validate_schema,
)


# ---------------------------------------------------------------------------
# 1. clean_column_names
# ---------------------------------------------------------------------------

class TestCleanColumnNames:
    def test_spaces_replaced_with_underscores(self, spark):
        df = spark.createDataFrame([(1,)], ["First Name"])
        result = clean_column_names(df)
        assert result.columns == ["first_name"]

    def test_special_chars_replaced_with_underscores(self, spark):
        df = spark.createDataFrame([(1,)], ["Sales@2024!"])
        result = clean_column_names(df)
        assert result.columns == ["sales_2024"]

    def test_consecutive_specials_collapse_to_single_underscore(self, spark):
        df = spark.createDataFrame([(1,)], ["  hello   world  "])
        result = clean_column_names(df)
        assert result.columns == ["hello_world"]

    def test_uppercase_lowercased(self, spark):
        df = spark.createDataFrame([(1,)], ["LastName"])
        result = clean_column_names(df)
        assert result.columns == ["lastname"]

    @pytest.mark.skip(reason="Spark Connect does not support duplicate column names in createDataFrame")
    def test_duplicate_names_disambiguated(self, spark):
        df = spark.createDataFrame([(1, 2)], ["First Name", "first name"])
        result = clean_column_names(df)
        assert len(result.columns) == 2

    @pytest.mark.skip(reason="Spark Connect does not support duplicate column names in createDataFrame")
    def test_three_duplicate_names_disambiguated(self, spark):
        df = spark.createDataFrame([(1, 2, 3)], ["col!", "col!", "col!"])
        result = clean_column_names(df)
        assert len(result.columns) == 3

    def test_empty_after_clean_gets_col_placeholder(self, spark):
        # A name of purely special characters strips to nothing → "col"
        df = spark.createDataFrame([(1,)], ["___"])
        result = clean_column_names(df)
        assert result.columns == ["col"]

    def test_already_clean_names_unchanged(self, spark):
        df = spark.createDataFrame([(1, 2)], ["age", "salary"])
        result = clean_column_names(df)
        assert result.columns == ["age", "salary"]

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            clean_column_names(None)


# ---------------------------------------------------------------------------
# 2. drop_null_columns
# ---------------------------------------------------------------------------

class TestDropNullColumns:
    def test_all_null_column_dropped(self, spark):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("all_null", StringType()),
        ])
        df = spark.createDataFrame([(1, None), (2, None)], schema)
        result = drop_null_columns(df)
        assert "all_null" not in result.columns
        assert "id" in result.columns

    def test_partial_null_column_kept_at_default_threshold(self, spark):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("partial", StringType()),
        ])
        df = spark.createDataFrame([(1, "a"), (2, None)], schema)
        result = drop_null_columns(df)
        assert "partial" in result.columns

    def test_threshold_parameter_drops_at_50_pct(self, spark):
        # 2 out of 4 rows null = 50% → meets threshold of 0.5, so dropped
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("half_null", StringType()),
        ])
        df = spark.createDataFrame(
            [(1, "a"), (2, None), (3, None), (4, "b")], schema
        )
        result = drop_null_columns(df, threshold=0.5)
        assert "half_null" not in result.columns

    def test_threshold_keeps_column_below_threshold(self, spark):
        # 1 out of 4 rows null = 25% → below threshold of 0.5, so kept
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("low_null", StringType()),
        ])
        df = spark.createDataFrame(
            [(1, "a"), (2, None), (3, "c"), (4, "d")], schema
        )
        result = drop_null_columns(df, threshold=0.5)
        assert "low_null" in result.columns

    def test_invalid_threshold_raises(self, spark):
        df = spark.createDataFrame([(1,)], ["id"])
        with pytest.raises(ValueError, match="threshold"):
            drop_null_columns(df, threshold=0.0)

    def test_threshold_above_one_raises(self, spark):
        df = spark.createDataFrame([(1,)], ["id"])
        with pytest.raises(ValueError, match="threshold"):
            drop_null_columns(df, threshold=1.5)

    def test_empty_dataframe_returned_unchanged(self, spark):
        schema = StructType([StructField("id", IntegerType())])
        df = spark.createDataFrame([], schema)
        result = drop_null_columns(df)
        assert result.columns == ["id"]


# ---------------------------------------------------------------------------
# 3. drop_constant_columns
# ---------------------------------------------------------------------------

class TestDropConstantColumns:
    def test_single_value_column_dropped(self, spark):
        df = spark.createDataFrame(
            [(1, "x"), (2, "x"), (3, "x")],
            ["id", "constant_col"],
        )
        result = drop_constant_columns(df)
        assert "constant_col" not in result.columns
        assert "id" in result.columns

    def test_diverse_column_kept(self, spark):
        df = spark.createDataFrame(
            [(1, "a"), (2, "b"), (3, "c")],
            ["id", "diverse_col"],
        )
        result = drop_constant_columns(df)
        assert "diverse_col" in result.columns

    def test_all_null_column_dropped(self, spark):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("all_null", StringType()),
        ])
        df = spark.createDataFrame([(1, None), (2, None)], schema)
        result = drop_constant_columns(df)
        assert "all_null" not in result.columns

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            drop_constant_columns(None)


# ---------------------------------------------------------------------------
# 4. fill_nulls_by_type
# ---------------------------------------------------------------------------

class TestFillNullsByType:
    def test_numeric_nulls_filled_with_zero_by_default(self, spark):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("score", DoubleType()),
        ])
        df = spark.createDataFrame([(1, None), (2, 5.0)], schema)
        result = fill_nulls_by_type(df)
        rows = {r["id"]: r["score"] for r in result.collect()}
        assert rows[1] == 0.0
        assert rows[2] == 5.0

    def test_string_nulls_filled_with_unknown_by_default(self, spark):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ])
        df = spark.createDataFrame([(1, None), (2, "Alice")], schema)
        result = fill_nulls_by_type(df)
        rows = {r["id"]: r["name"] for r in result.collect()}
        assert rows[1] == "unknown"
        assert rows[2] == "Alice"

    def test_custom_numeric_fill_value(self, spark):
        schema = StructType([StructField("val", IntegerType())])
        df = spark.createDataFrame([(None,), (10,)], schema)
        result = fill_nulls_by_type(df, numeric_fill=-1)
        vals = [r["val"] for r in result.collect()]
        assert -1 in vals
        assert 10 in vals

    def test_custom_string_fill_value(self, spark):
        schema = StructType([StructField("label", StringType())])
        df = spark.createDataFrame([(None,), ("ok",)], schema)
        result = fill_nulls_by_type(df, string_fill="N/A")
        labels = [r["label"] for r in result.collect()]
        assert "N/A" in labels

    def test_non_null_values_unchanged(self, spark):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("val", IntegerType()),
        ])
        df = spark.createDataFrame([(1, 42)], schema)
        result = fill_nulls_by_type(df)
        assert result.collect()[0]["val"] == 42

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            fill_nulls_by_type(None)


# ---------------------------------------------------------------------------
# 5. deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_simple_dedup_removes_exact_duplicates(self, spark):
        df = spark.createDataFrame(
            [(1, "Alice"), (1, "Alice"), (2, "Bob")],
            ["id", "name"],
        )
        result = deduplicate(df)
        assert result.count() == 2

    def test_subset_dedup_on_single_column(self, spark):
        df = spark.createDataFrame(
            [(1, "Alice"), (1, "Alicia"), (2, "Bob")],
            ["id", "name"],
        )
        result = deduplicate(df, subset=["id"])
        assert result.count() == 2

    def test_ordered_dedup_keep_first(self, spark):
        # With order_by="score" ascending, keep="first" keeps lowest score per id
        df = spark.createDataFrame(
            [(1, 10), (1, 20), (2, 5)],
            ["id", "score"],
        )
        result = deduplicate(df, subset=["id"], order_by=["score"], keep="first")
        rows = {r["id"]: r["score"] for r in result.collect()}
        assert rows[1] == 10
        assert rows[2] == 5

    @pytest.mark.skip(reason="Spark Connect SortOrder limitation")
    def test_ordered_dedup_keep_last(self, spark):
        # With order_by="score" ascending, keep="last" keeps highest score per id
        df = spark.createDataFrame(
            [(1, 10), (1, 20), (2, 5)],
            ["id", "score"],
        )
        result = deduplicate(df, subset=["id"], order_by=["score"], keep="last")
        rows = {r["id"]: r["score"] for r in result.collect()}
        assert rows[1] == 20

    def test_descending_order_prefix(self, spark):
        # "-score" means descending; keep="first" keeps the highest score
        df = spark.createDataFrame(
            [(1, 10), (1, 99), (2, 5)],
            ["id", "score"],
        )
        result = deduplicate(df, subset=["id"], order_by=["-score"], keep="first")
        rows = {r["id"]: r["score"] for r in result.collect()}
        assert rows[1] == 99

    def test_invalid_keep_raises(self, spark):
        df = spark.createDataFrame([(1,)], ["id"])
        with pytest.raises(ValueError, match='keep must be'):
            deduplicate(df, keep="middle")

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            deduplicate(None)


# ---------------------------------------------------------------------------
# 6. standardize_strings
# ---------------------------------------------------------------------------

class TestStandardizeStrings:
    def test_leading_trailing_whitespace_trimmed(self, spark):
        df = spark.createDataFrame([("  Alice  ",)], ["name"])
        result = standardize_strings(df)
        assert result.collect()[0]["name"] == "alice"

    def test_multiple_spaces_collapsed(self, spark):
        df = spark.createDataFrame([("hello   world",)], ["name"])
        result = standardize_strings(df)
        assert result.collect()[0]["name"] == "hello world"

    def test_uppercased_string_lowercased(self, spark):
        df = spark.createDataFrame([("NEW YORK",)], ["city"])
        result = standardize_strings(df)
        assert result.collect()[0]["city"] == "new york"

    def test_specific_columns_parameter_only_affects_those_columns(self, spark):
        df = spark.createDataFrame(
            [("  Alice  ", "  LONDON  ")],
            ["name", "city"],
        )
        # Only standardize "name"; "city" should be unchanged
        result = standardize_strings(df, columns=["name"])
        row = result.collect()[0]
        assert row["name"] == "alice"
        assert row["city"] == "  LONDON  "

    def test_all_string_columns_processed_when_no_columns_param(self, spark):
        df = spark.createDataFrame(
            [("  Alice  ", "  BOB  ")],
            ["first", "last"],
        )
        result = standardize_strings(df)
        row = result.collect()[0]
        assert row["first"] == "alice"
        assert row["last"] == "bob"

    def test_missing_column_raises(self, spark):
        df = spark.createDataFrame([("Alice",)], ["name"])
        with pytest.raises(ValueError, match="not found in DataFrame"):
            standardize_strings(df, columns=["nonexistent"])

    def test_null_values_unchanged(self, spark):
        schema = StructType([StructField("name", StringType())])
        df = spark.createDataFrame([(None,)], schema)
        result = standardize_strings(df)
        assert result.collect()[0]["name"] is None

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            standardize_strings(None)


# ---------------------------------------------------------------------------
# 7. remove_outliers
# ---------------------------------------------------------------------------

class TestRemoveOutliers:
    def _make_df(self, spark, values):
        schema = StructType([StructField("val", DoubleType())])
        return spark.createDataFrame([(v,) for v in values], schema)

    def test_iqr_removes_extreme_low_value(self, spark):
        # Normal range 1-10, extreme outlier at -1000
        df = self._make_df(spark, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, -1000.0])
        result = remove_outliers(df, column="val", method="iqr")
        vals = [r["val"] for r in result.collect()]
        assert -1000.0 not in vals

    def test_iqr_removes_extreme_high_value(self, spark):
        df = self._make_df(spark, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 9999.0])
        result = remove_outliers(df, column="val", method="iqr")
        vals = [r["val"] for r in result.collect()]
        assert 9999.0 not in vals

    def test_iqr_keeps_normal_values(self, spark):
        df = self._make_df(spark, [1.0, 2.0, 3.0, 4.0, 5.0])
        result = remove_outliers(df, column="val", method="iqr")
        assert result.count() == 5

    def test_zscore_removes_extreme_outlier(self, spark):
        # 50 values near 5.0, then one extreme outlier — ensures z-score >> 3
        normal = [5.0] * 50
        df = self._make_df(spark, normal + [10000.0])
        result = remove_outliers(df, column="val", method="zscore", z_threshold=3.0)
        vals = [r["val"] for r in result.collect()]
        assert 10000.0 not in vals

    def test_zscore_keeps_inliers(self, spark):
        df = self._make_df(spark, [1.0, 2.0, 3.0, 4.0, 5.0])
        result = remove_outliers(df, column="val", method="zscore", z_threshold=3.0)
        assert result.count() == 5

    def test_nulls_are_retained_not_treated_as_outliers(self, spark):
        schema = StructType([StructField("val", DoubleType())])
        df = spark.createDataFrame([(1.0,), (None,), (2.0,)], schema)
        result = remove_outliers(df, column="val", method="iqr")
        assert result.count() == 3

    def test_invalid_method_raises(self, spark):
        df = self._make_df(spark, [1.0])
        with pytest.raises(ValueError, match='method must be'):
            remove_outliers(df, column="val", method="mad")

    def test_missing_column_raises(self, spark):
        df = self._make_df(spark, [1.0])
        with pytest.raises(ValueError, match="not found in DataFrame"):
            remove_outliers(df, column="nonexistent", method="iqr")


# ---------------------------------------------------------------------------
# 8. cap_outliers
# ---------------------------------------------------------------------------

class TestCapOutliers:
    def _make_df(self, spark, values):
        schema = StructType([StructField("val", DoubleType())])
        return spark.createDataFrame([(v,) for v in values], schema)

    def test_iqr_extreme_high_value_capped(self, spark):
        df = self._make_df(spark, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 9999.0])
        result = cap_outliers(df, column="val", method="iqr")
        vals = [r["val"] for r in result.collect()]
        assert 9999.0 not in vals
        # Row still exists (11 rows in, 11 rows out)
        assert result.count() == 11

    def test_iqr_extreme_low_value_capped(self, spark):
        df = self._make_df(spark, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, -9999.0])
        result = cap_outliers(df, column="val", method="iqr")
        vals = [r["val"] for r in result.collect()]
        assert -9999.0 not in vals
        assert result.count() == 11

    def test_normal_values_not_changed(self, spark):
        df = self._make_df(spark, [1.0, 2.0, 3.0, 4.0, 5.0])
        result = cap_outliers(df, column="val", method="iqr")
        original_vals = sorted([r["val"] for r in df.collect()])
        result_vals = sorted([r["val"] for r in result.collect()])
        assert original_vals == result_vals

    def test_percentile_method_caps_extremes(self, spark):
        values = [float(i) for i in range(1, 101)]  # 1..100
        df = self._make_df(spark, values)
        result = cap_outliers(
            df, column="val", method="percentile", lower_pct=0.05, upper_pct=0.95
        )
        vals = [r["val"] for r in result.collect()]
        assert max(vals) <= 95.0
        assert min(vals) >= 5.0

    def test_null_values_left_unchanged(self, spark):
        schema = StructType([StructField("val", DoubleType())])
        df = spark.createDataFrame([(1.0,), (None,), (9999.0,)], schema)
        result = cap_outliers(df, column="val", method="iqr")
        null_rows = [r for r in result.collect() if r["val"] is None]
        assert len(null_rows) == 1

    def test_invalid_method_raises(self, spark):
        df = self._make_df(spark, [1.0])
        with pytest.raises(ValueError, match='method must be'):
            cap_outliers(df, column="val", method="zscore")

    def test_missing_column_raises(self, spark):
        df = self._make_df(spark, [1.0])
        with pytest.raises(ValueError, match="not found in DataFrame"):
            cap_outliers(df, column="nonexistent")


# ---------------------------------------------------------------------------
# 9. validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_missing_columns_reported(self, spark):
        df = spark.createDataFrame([(1,)], ["id"])
        report = validate_schema(df, {"id": "integer", "name": "string"})
        assert "name" in report["missing_cols"]

    def test_extra_columns_reported(self, spark):
        df = spark.createDataFrame([(1, "Alice")], ["id", "name"])
        report = validate_schema(df, {"id": "integer"})
        assert "name" in report["extra_cols"]

    def test_type_mismatch_reported(self, spark):
        schema = StructType([StructField("age", StringType())])
        df = spark.createDataFrame([("30",)], schema)
        report = validate_schema(df, {"age": "integer"})
        assert "age" in report["type_mismatches"]
        assert report["type_mismatches"]["age"]["expected"] == "integer"
        assert report["type_mismatches"]["age"]["actual"] == "string"

    def test_perfect_match_returns_empty_report(self, spark):
        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ])
        df = spark.createDataFrame([(1, "Alice")], schema)
        report = validate_schema(df, {"id": "integer", "name": "string"})
        assert report["missing_cols"] == []
        assert report["extra_cols"] == []
        assert report["type_mismatches"] == {}

    def test_int_alias_normalised(self, spark):
        # Spark reports "integer"; expected schema uses "int" alias — should not mismatch
        schema = StructType([StructField("id", IntegerType())])
        df = spark.createDataFrame([(1,)], schema)
        report = validate_schema(df, {"id": "int"})
        assert "id" not in report["type_mismatches"]

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            validate_schema(None, {"id": "integer"})


# ---------------------------------------------------------------------------
# 10. enforce_types
# ---------------------------------------------------------------------------

class TestEnforceTypes:
    def test_successful_string_to_integer_cast(self, spark):
        df = spark.createDataFrame([("42",)], ["age"])
        result = enforce_types(df, {"age": "integer"})
        row = result.collect()[0]
        assert row["age"] == 42
        assert isinstance(row["age"], int)

    def test_invalid_cast_becomes_null(self, spark):
        df = spark.createDataFrame([("not_a_number",)], ["age"])
        result = enforce_types(df, {"age": "integer"})
        assert result.collect()[0]["age"] is None

    def test_column_not_in_type_map_left_unchanged(self, spark):
        df = spark.createDataFrame([(1, "Alice")], ["id", "name"])
        result = enforce_types(df, {"id": "double"})
        # "name" column still present and of string type
        assert dict(result.dtypes)["name"] == "string"

    def test_column_in_type_map_but_absent_from_df_silently_ignored(self, spark):
        df = spark.createDataFrame([(1,)], ["id"])
        # Should not raise even though "missing_col" is not in df
        result = enforce_types(df, {"missing_col": "integer"})
        assert result.columns == ["id"]

    def test_multiple_columns_cast(self, spark):
        df = spark.createDataFrame([("1", "3.14")], ["int_col", "float_col"])
        result = enforce_types(df, {"int_col": "integer", "float_col": "double"})
        dtypes = dict(result.dtypes)
        assert dtypes["int_col"] in ("int", "integer")
        assert dtypes["float_col"] == "double"

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            enforce_types(None, {"id": "integer"})


# ---------------------------------------------------------------------------
# 11. flag_invalid_rows
# ---------------------------------------------------------------------------

class TestFlagInvalidRows:
    def test_valid_row_flagged_true(self, spark):
        schema = StructType([StructField("age", IntegerType())])
        df = spark.createDataFrame([(25,)], schema)
        result = flag_invalid_rows(df, {"age": "col('age').between(0, 120)"})
        assert result.collect()[0]["is_valid"] is True

    def test_invalid_row_flagged_false(self, spark):
        schema = StructType([StructField("age", IntegerType())])
        df = spark.createDataFrame([(-5,)], schema)
        result = flag_invalid_rows(df, {"age": "col('age').between(0, 120)"})
        assert result.collect()[0]["is_valid"] is False

    def test_multiple_rules_all_must_pass(self, spark):
        schema = StructType([
            StructField("age", IntegerType()),
            StructField("salary", DoubleType()),
        ])
        df = spark.createDataFrame(
            [(25, 50000.0), (25, -1.0), (200, 50000.0)],
            schema,
        )
        rules = {
            "age": "col('age').between(0, 120)",
            "salary": "col('salary') > 0",
        }
        result = flag_invalid_rows(df, rules)
        rows = {(r["age"], r["salary"]): r["is_valid"] for r in result.collect()}
        assert rows[(25, 50000.0)] is True
        assert rows[(25, -1.0)] is False
        assert rows[(200, 50000.0)] is False

    def test_null_in_rule_column_treated_as_invalid(self, spark):
        schema = StructType([StructField("age", IntegerType())])
        df = spark.createDataFrame([(None,)], schema)
        result = flag_invalid_rows(df, {"age": "col('age').between(0, 120)"})
        assert result.collect()[0]["is_valid"] is False

    def test_empty_rules_raises(self, spark):
        df = spark.createDataFrame([(1,)], ["id"])
        with pytest.raises(ValueError, match="rules must not be empty"):
            flag_invalid_rows(df, {})

    def test_invalid_expression_raises(self, spark):
        df = spark.createDataFrame([(1,)], ["id"])
        with pytest.raises(ValueError, match="Could not evaluate rule"):
            flag_invalid_rows(df, {"bad": "this is not valid python !@#"})

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            flag_invalid_rows(None, {"id": "col('id') > 0"})


# ---------------------------------------------------------------------------
# 12. split_valid_invalid
# ---------------------------------------------------------------------------

class TestSplitValidInvalid:
    def test_correct_split_into_two_dataframes(self, spark):
        schema = StructType([StructField("age", IntegerType())])
        df = spark.createDataFrame([(25,), (150,), (30,), (-1,)], schema)
        rules = {"age": "col('age').between(0, 120)"}
        valid_df, invalid_df = split_valid_invalid(df, rules)
        assert valid_df.count() == 2
        assert invalid_df.count() == 2

    def test_is_valid_column_not_present_in_output(self, spark):
        schema = StructType([StructField("age", IntegerType())])
        df = spark.createDataFrame([(25,), (150,)], schema)
        rules = {"age": "col('age').between(0, 120)"}
        valid_df, invalid_df = split_valid_invalid(df, rules)
        assert "is_valid" not in valid_df.columns
        assert "is_valid" not in invalid_df.columns

    def test_valid_df_contains_only_passing_rows(self, spark):
        schema = StructType([StructField("val", IntegerType())])
        df = spark.createDataFrame([(1,), (5,), (100,)], schema)
        rules = {"val": "col('val') < 10"}
        valid_df, invalid_df = split_valid_invalid(df, rules)
        valid_vals = [r["val"] for r in valid_df.collect()]
        assert 100 not in valid_vals
        invalid_vals = [r["val"] for r in invalid_df.collect()]
        assert 100 in invalid_vals

    def test_all_valid_returns_empty_invalid(self, spark):
        schema = StructType([StructField("val", IntegerType())])
        df = spark.createDataFrame([(1,), (2,), (3,)], schema)
        rules = {"val": "col('val') > 0"}
        valid_df, invalid_df = split_valid_invalid(df, rules)
        assert valid_df.count() == 3
        assert invalid_df.count() == 0

    def test_all_invalid_returns_empty_valid(self, spark):
        schema = StructType([StructField("val", IntegerType())])
        df = spark.createDataFrame([(-1,), (-2,), (-3,)], schema)
        rules = {"val": "col('val') > 0"}
        valid_df, invalid_df = split_valid_invalid(df, rules)
        assert valid_df.count() == 0
        assert invalid_df.count() == 3

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            split_valid_invalid(None, {"id": "col('id') > 0"})


# ---------------------------------------------------------------------------
# 13. normalize_values
# ---------------------------------------------------------------------------

class TestNormalizeValues:
    def test_mapping_applied_to_matching_values(self, spark):
        df = spark.createDataFrame([("Y",), ("N",), ("YES",)], ["flag"])
        mapping = {"Y": "yes", "N": "no", "YES": "yes"}
        result = normalize_values(df, column="flag", mapping=mapping)
        vals = [r["flag"] for r in result.collect()]
        assert "Y" not in vals
        assert "N" not in vals
        assert vals.count("yes") == 2
        assert vals.count("no") == 1

    def test_unmapped_values_unchanged(self, spark):
        df = spark.createDataFrame([("Y",), ("maybe",)], ["flag"])
        mapping = {"Y": "yes"}
        result = normalize_values(df, column="flag", mapping=mapping)
        vals = [r["flag"] for r in result.collect()]
        assert "maybe" in vals

    def test_null_values_unchanged(self, spark):
        schema = StructType([StructField("flag", StringType())])
        df = spark.createDataFrame([(None,), ("Y",)], schema)
        mapping = {"Y": "yes"}
        result = normalize_values(df, column="flag", mapping=mapping)
        null_count = sum(1 for r in result.collect() if r["flag"] is None)
        assert null_count == 1

    def test_gender_mapping(self, spark):
        df = spark.createDataFrame([("M",), ("F",), ("m",), ("f",), ("other",)], ["gender"])
        mapping = {"M": "male", "F": "female", "m": "male", "f": "female"}
        result = normalize_values(df, column="gender", mapping=mapping)
        rows = [r["gender"] for r in result.collect()]
        assert rows.count("male") == 2
        assert rows.count("female") == 2
        assert "other" in rows

    def test_missing_column_raises(self, spark):
        df = spark.createDataFrame([("Y",)], ["flag"])
        with pytest.raises(ValueError, match="not found in DataFrame"):
            normalize_values(df, column="nonexistent", mapping={"Y": "yes"})

    def test_empty_mapping_raises(self, spark):
        df = spark.createDataFrame([("Y",)], ["flag"])
        with pytest.raises(ValueError, match="mapping must not be empty"):
            normalize_values(df, column="flag", mapping={})

    def test_none_df_raises(self, spark):
        with pytest.raises(ValueError, match="df must not be None"):
            normalize_values(None, column="flag", mapping={"Y": "yes"})
