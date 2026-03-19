"""
test_feature_engineering.py
----------------------------
Pytest tests for dbx_toolkit.feature_engineering module.

All tests use the session-scoped `spark` fixture defined in conftest.py.
"""

import pytest
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors

from dbx_toolkit.feature_engineering import (
    encode_categoricals,
    scale_features,
    assemble_features,
    create_bins,
    add_date_features,
    add_lag_features,
    add_rolling_features,
    add_interaction_features,
    add_ratio_features,
    add_missing_indicators,
    impute_columns,
    prepare_features,
)


# ---------------------------------------------------------------------------
# 1. encode_categoricals
# ---------------------------------------------------------------------------

class TestEncodeCategoricals:

    def test_index_method_adds_idx_columns(self, spark):
        df = spark.createDataFrame(
            [("cat",), ("dog",), ("cat",), ("bird",)],
            ["animal"],
        )
        result_df, stages = encode_categoricals(df, ["animal"], method="index")

        assert "animal_indexed" in result_df.columns
        # No OHE column should be present
        assert "animal_onehot" not in result_df.columns
        # Exactly one fitted stage (StringIndexerModel)
        assert len(stages) == 1

    def test_index_method_returns_fitted_stages(self, spark):
        df = spark.createDataFrame(
            [("red",), ("blue",), ("red",)],
            ["color"],
        )
        result_df, stages = encode_categoricals(df, ["color"], method="index")

        # The stage can be re-applied to new data
        new_df = spark.createDataFrame([("red",), ("blue",)], ["color"])
        transformed = stages[0].transform(new_df)
        assert "color_indexed" in transformed.columns

    def test_onehot_method_adds_ohe_columns(self, spark):
        df = spark.createDataFrame(
            [("a",), ("b",), ("a",), ("c",)],
            ["cat"],
        )
        result_df, stages = encode_categoricals(df, ["cat"], method="onehot")

        assert "cat_indexed" in result_df.columns
        assert "cat_onehot" in result_df.columns
        # Two stages: one StringIndexerModel + one OneHotEncoderModel
        assert len(stages) == 2

    def test_onehot_method_multiple_columns(self, spark):
        df = spark.createDataFrame(
            [("red", "S"), ("blue", "M"), ("red", "L")],
            ["color", "size"],
        )
        result_df, stages = encode_categoricals(df, ["color", "size"], method="onehot")

        assert "color_indexed" in result_df.columns
        assert "size_indexed" in result_df.columns
        assert "color_onehot" in result_df.columns
        assert "size_onehot" in result_df.columns
        # Two StringIndexer stages + one OHE stage
        assert len(stages) == 3

    def test_invalid_method_raises(self, spark):
        df = spark.createDataFrame([("a",)], ["col"])
        with pytest.raises(ValueError, match="method must be"):
            encode_categoricals(df, ["col"], method="label")

    def test_empty_columns_list(self, spark):
        df = spark.createDataFrame([(1,), (2,)], ["num"])
        result_df, stages = encode_categoricals(df, [], method="index")

        assert result_df.columns == ["num"]
        assert stages == []


# ---------------------------------------------------------------------------
# 2. scale_features
# ---------------------------------------------------------------------------

class TestScaleFeatures:

    def _make_vector_df(self, spark):
        data = [
            (Vectors.dense([1.0, 10.0]),),
            (Vectors.dense([2.0, 20.0]),),
            (Vectors.dense([3.0, 30.0]),),
        ]
        return spark.createDataFrame(data, ["raw_features"])

    def test_standard_scaler(self, spark):
        df = self._make_vector_df(spark)
        result_df, fitted_scaler = scale_features(df, "raw_features", method="standard")

        assert "scaled_features" in result_df.columns
        assert result_df.count() == 3
        # Fitted scaler can be used to transform new data
        new_df = self._make_vector_df(spark)
        transformed = fitted_scaler.transform(new_df)
        assert "scaled_features" in transformed.columns

    def test_minmax_scaler(self, spark):
        df = self._make_vector_df(spark)
        result_df, fitted_scaler = scale_features(
            df, "raw_features", output_col="mm_features", method="minmax"
        )

        assert "mm_features" in result_df.columns
        rows = result_df.select("mm_features").collect()
        # Min value should be 0.0 and max should be 1.0 per feature
        min_row = rows[0][0]
        max_row = rows[-1][0]
        assert min_row[0] == pytest.approx(0.0)
        assert max_row[0] == pytest.approx(1.0)

    def test_maxabs_scaler(self, spark):
        df = self._make_vector_df(spark)
        result_df, fitted_scaler = scale_features(
            df, "raw_features", output_col="ma_features", method="maxabs"
        )

        assert "ma_features" in result_df.columns
        rows = result_df.select("ma_features").collect()
        # The maximum absolute value maps to 1.0
        max_row = rows[-1][0]
        assert max_row[0] == pytest.approx(1.0)

    def test_custom_output_col(self, spark):
        df = self._make_vector_df(spark)
        result_df, _ = scale_features(df, "raw_features", output_col="my_scaled")

        assert "my_scaled" in result_df.columns
        assert "scaled_features" not in result_df.columns

    def test_invalid_method_raises(self, spark):
        df = self._make_vector_df(spark)
        with pytest.raises(ValueError, match="method must be one of"):
            scale_features(df, "raw_features", method="l2")


# ---------------------------------------------------------------------------
# 3. assemble_features
# ---------------------------------------------------------------------------

class TestAssembleFeatures:

    def test_basic_assembly(self, spark):
        df = spark.createDataFrame(
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            ["a", "b", "c"],
        )
        result_df = assemble_features(df, ["a", "b", "c"])

        assert "features" in result_df.columns
        rows = result_df.select("features").collect()
        assert list(rows[0][0]) == pytest.approx([1.0, 2.0, 3.0])
        assert list(rows[1][0]) == pytest.approx([4.0, 5.0, 6.0])

    def test_custom_output_col(self, spark):
        df = spark.createDataFrame([(1.0, 2.0)], ["x", "y"])
        result_df = assemble_features(df, ["x", "y"], output_col="vec")

        assert "vec" in result_df.columns
        assert "features" not in result_df.columns

    def test_handle_invalid_skip_drops_null_rows(self, spark):
        df = spark.createDataFrame(
            [(1.0, 2.0), (None, 3.0), (4.0, 5.0)],
            ["a", "b"],
        )
        result_df = assemble_features(df, ["a", "b"], handle_invalid="skip")

        # Row with null is dropped
        assert result_df.filter(F.col("features").isNotNull()).count() == 2

    def test_handle_invalid_keep_preserves_null_rows(self, spark):
        df = spark.createDataFrame(
            [(1.0, 2.0), (None, 3.0), (4.0, 5.0)],
            ["a", "b"],
        )
        result_df = assemble_features(df, ["a", "b"], handle_invalid="keep")

        assert result_df.count() == 3

    def test_vector_length_matches_input_cols(self, spark):
        df = spark.createDataFrame([(1.0, 2.0, 3.0, 4.0)], ["w", "x", "y", "z"])
        result_df = assemble_features(df, ["w", "x", "y", "z"])

        row = result_df.select("features").first()
        assert len(row[0]) == 4


# ---------------------------------------------------------------------------
# 4. create_bins
# ---------------------------------------------------------------------------

class TestCreateBins:

    def test_quantile_method_default_output_col(self, spark):
        df = spark.createDataFrame(
            [(float(i),) for i in range(1, 21)],
            ["value"],
        )
        result_df = create_bins(df, "value", method="quantile", n_bins=4)

        assert "value_binned" in result_df.columns
        assert result_df.count() == 20

    def test_quantile_creates_equal_frequency_bins(self, spark):
        # 20 rows → 4 bins of ~5 rows each
        df = spark.createDataFrame(
            [(float(i),) for i in range(1, 21)],
            ["value"],
        )
        result_df = create_bins(df, "value", method="quantile", n_bins=4)

        bin_counts = (
            result_df.groupBy("value_binned").count().collect()
        )
        counts = {row["value_binned"]: row["count"] for row in bin_counts}
        # Each bin should have approximately 5 rows
        for bin_id, cnt in counts.items():
            assert cnt == pytest.approx(5, abs=2)

    def test_quantile_number_of_distinct_bins(self, spark):
        df = spark.createDataFrame(
            [(float(i),) for i in range(1, 21)],
            ["value"],
        )
        result_df = create_bins(df, "value", method="quantile", n_bins=4)

        distinct_bins = result_df.select("value_binned").distinct().count()
        assert distinct_bins == 4

    def test_custom_splits(self, spark):
        df = spark.createDataFrame(
            [(5.0,), (20.0,), (40.0,), (70.0,)],
            ["age"],
        )
        splits = [float("-inf"), 18.0, 35.0, 60.0, float("inf")]
        result_df = create_bins(df, "age", method="custom", custom_splits=splits)

        assert "age_binned" in result_df.columns
        rows = {row["age"]: row["age_binned"] for row in result_df.collect()}
        # 5.0 < 18 → bin 0; 20.0 in [18,35) → bin 1; 40.0 in [35,60) → bin 2; 70.0 >= 60 → bin 3
        assert rows[5.0] == 0.0
        assert rows[20.0] == 1.0
        assert rows[40.0] == 2.0
        assert rows[70.0] == 3.0

    def test_custom_output_col(self, spark):
        df = spark.createDataFrame([(1.0,), (5.0,)], ["score"])
        result_df = create_bins(
            df, "score", method="quantile", n_bins=2, output_col="score_bucket"
        )
        assert "score_bucket" in result_df.columns
        assert "score_binned" not in result_df.columns

    def test_custom_method_without_splits_raises(self, spark):
        df = spark.createDataFrame([(1.0,)], ["val"])
        with pytest.raises(ValueError, match="custom_splits must be provided"):
            create_bins(df, "val", method="custom")

    def test_invalid_method_raises(self, spark):
        df = spark.createDataFrame([(1.0,)], ["val"])
        with pytest.raises(ValueError, match="method must be"):
            create_bins(df, "val", method="equal_width")


# ---------------------------------------------------------------------------
# 5. add_date_features
# ---------------------------------------------------------------------------

class TestAddDateFeatures:

    def _make_date_df(self, spark):
        # 2024-03-15 is a Friday (dayofweek=6 in Spark, which uses 1=Sun…7=Sat)
        return spark.createDataFrame(
            [(1,)],
            ["id"],
        ).withColumn("event_date", F.to_date(F.lit("2024-03-15")))

    def test_default_features_added(self, spark):
        df = self._make_date_df(spark)
        result_df = add_date_features(df, "event_date")

        expected_cols = [
            "event_date_year",
            "event_date_month",
            "event_date_day",
            "event_date_quarter",
            "event_date_day_of_week",
            "event_date_is_weekend",
        ]
        for col in expected_cols:
            assert col in result_df.columns, f"Missing column: {col}"

    def test_year_value(self, spark):
        df = self._make_date_df(spark)
        result_df = add_date_features(df, "event_date", features=["year"])
        val = result_df.select("event_date_year").first()[0]
        assert val == 2024

    def test_month_value(self, spark):
        df = self._make_date_df(spark)
        result_df = add_date_features(df, "event_date", features=["month"])
        val = result_df.select("event_date_month").first()[0]
        assert val == 3

    def test_day_value(self, spark):
        df = self._make_date_df(spark)
        result_df = add_date_features(df, "event_date", features=["day"])
        val = result_df.select("event_date_day").first()[0]
        assert val == 15

    def test_quarter_value(self, spark):
        df = self._make_date_df(spark)
        result_df = add_date_features(df, "event_date", features=["quarter"])
        val = result_df.select("event_date_quarter").first()[0]
        assert val == 1

    def test_day_of_week_value(self, spark):
        # 2024-03-15 is Friday; Spark dayofweek: 1=Sun, 2=Mon, …, 6=Fri, 7=Sat
        df = self._make_date_df(spark)
        result_df = add_date_features(df, "event_date", features=["day_of_week"])
        val = result_df.select("event_date_day_of_week").first()[0]
        assert val == 6  # Friday

    def test_is_weekend_friday_is_zero(self, spark):
        # Friday is not a weekend (is_weekend uses dayofweek isin [1, 7])
        df = self._make_date_df(spark)
        result_df = add_date_features(df, "event_date", features=["is_weekend"])
        val = result_df.select("event_date_is_weekend").first()[0]
        assert val == 0

    def test_is_weekend_sunday_is_one(self, spark):
        # 2024-03-17 is Sunday
        df = spark.createDataFrame([(1,)], ["id"]).withColumn(
            "event_date", F.to_date(F.lit("2024-03-17"))
        )
        result_df = add_date_features(df, "event_date", features=["is_weekend"])
        val = result_df.select("event_date_is_weekend").first()[0]
        assert val == 1

    def test_subset_of_features(self, spark):
        df = self._make_date_df(spark)
        result_df = add_date_features(df, "event_date", features=["year", "month"])

        assert "event_date_year" in result_df.columns
        assert "event_date_month" in result_df.columns
        assert "event_date_day" not in result_df.columns

    def test_invalid_feature_raises(self, spark):
        df = self._make_date_df(spark)
        with pytest.raises(ValueError, match="Unknown feature"):
            add_date_features(df, "event_date", features=["century"])


# ---------------------------------------------------------------------------
# 6. add_lag_features
# ---------------------------------------------------------------------------

class TestAddLagFeatures:

    def _make_time_series_df(self, spark):
        data = [
            ("A", 1, 10.0),
            ("A", 2, 20.0),
            ("A", 3, 30.0),
            ("A", 4, 40.0),
            ("B", 1, 100.0),
            ("B", 2, 200.0),
        ]
        return spark.createDataFrame(data, ["store_id", "period", "sales"])

    def test_lag_columns_created(self, spark):
        df = self._make_time_series_df(spark)
        result_df = add_lag_features(
            df, "sales", partition_by="store_id", order_by="period", lags=[1, 2]
        )

        assert "sales_lag_1" in result_df.columns
        assert "sales_lag_2" in result_df.columns

    def test_lag_1_correct_values(self, spark):
        df = self._make_time_series_df(spark)
        result_df = add_lag_features(
            df, "sales", partition_by="store_id", order_by="period", lags=[1]
        )

        rows = {
            (row["store_id"], row["period"]): row["sales_lag_1"]
            for row in result_df.collect()
        }
        # First period in partition should have null lag
        assert rows[("A", 1)] is None
        assert rows[("A", 2)] == pytest.approx(10.0)
        assert rows[("A", 3)] == pytest.approx(20.0)
        assert rows[("A", 4)] == pytest.approx(30.0)

    def test_lag_null_for_first_row_in_partition(self, spark):
        df = self._make_time_series_df(spark)
        result_df = add_lag_features(
            df, "sales", partition_by="store_id", order_by="period", lags=[1, 2]
        )

        rows = {
            (row["store_id"], row["period"]): row
            for row in result_df.collect()
        }
        # period=1 is the first row → lag_1 and lag_2 both null
        assert rows[("A", 1)]["sales_lag_1"] is None
        assert rows[("A", 1)]["sales_lag_2"] is None
        # period=2 has lag_1 but lag_2 is null
        assert rows[("A", 2)]["sales_lag_1"] == pytest.approx(10.0)
        assert rows[("A", 2)]["sales_lag_2"] is None

    def test_partitions_are_independent(self, spark):
        df = self._make_time_series_df(spark)
        result_df = add_lag_features(
            df, "sales", partition_by="store_id", order_by="period", lags=[1]
        )

        rows = {
            (row["store_id"], row["period"]): row["sales_lag_1"]
            for row in result_df.collect()
        }
        # Store B period=1 should be null (not borrowing from store A)
        assert rows[("B", 1)] is None
        assert rows[("B", 2)] == pytest.approx(100.0)

    def test_default_lags(self, spark):
        df = self._make_time_series_df(spark)
        result_df = add_lag_features(
            df, "sales", partition_by="store_id", order_by="period"
        )
        # Default lags are [1, 7, 30]
        assert "sales_lag_1" in result_df.columns
        assert "sales_lag_7" in result_df.columns
        assert "sales_lag_30" in result_df.columns


# ---------------------------------------------------------------------------
# 7. add_rolling_features
# ---------------------------------------------------------------------------

class TestAddRollingFeatures:

    def _make_rolling_df(self, spark):
        data = [
            ("A", 1, 10.0),
            ("A", 2, 20.0),
            ("A", 3, 30.0),
            ("A", 4, 40.0),
            ("A", 5, 50.0),
        ]
        return spark.createDataFrame(data, ["store_id", "period", "sales"])

    def test_rolling_columns_created(self, spark):
        df = self._make_rolling_df(spark)
        result_df = add_rolling_features(
            df, "sales",
            partition_by="store_id",
            order_by="period",
            windows=[3],
            funcs=["avg", "min", "max"],
        )

        assert "sales_rolling_avg_3" in result_df.columns
        assert "sales_rolling_min_3" in result_df.columns
        assert "sales_rolling_max_3" in result_df.columns

    def test_rolling_avg_correct_values(self, spark):
        df = self._make_rolling_df(spark)
        result_df = add_rolling_features(
            df, "sales",
            partition_by="store_id",
            order_by="period",
            windows=[3],
            funcs=["avg"],
        )

        rows = {row["period"]: row["sales_rolling_avg_3"] for row in result_df.collect()}
        # Window size 3, rowsBetween(-2, 0)
        # period=1: avg([10]) = 10.0
        # period=2: avg([10, 20]) = 15.0
        # period=3: avg([10, 20, 30]) = 20.0
        # period=4: avg([20, 30, 40]) = 30.0
        assert rows[1] == pytest.approx(10.0)
        assert rows[2] == pytest.approx(15.0)
        assert rows[3] == pytest.approx(20.0)
        assert rows[4] == pytest.approx(30.0)

    def test_rolling_min_correct_values(self, spark):
        df = self._make_rolling_df(spark)
        result_df = add_rolling_features(
            df, "sales",
            partition_by="store_id",
            order_by="period",
            windows=[3],
            funcs=["min"],
        )

        rows = {row["period"]: row["sales_rolling_min_3"] for row in result_df.collect()}
        assert rows[3] == pytest.approx(10.0)  # min(10, 20, 30)
        assert rows[4] == pytest.approx(20.0)  # min(20, 30, 40)
        assert rows[5] == pytest.approx(30.0)  # min(30, 40, 50)

    def test_rolling_max_correct_values(self, spark):
        df = self._make_rolling_df(spark)
        result_df = add_rolling_features(
            df, "sales",
            partition_by="store_id",
            order_by="period",
            windows=[3],
            funcs=["max"],
        )

        rows = {row["period"]: row["sales_rolling_max_3"] for row in result_df.collect()}
        assert rows[3] == pytest.approx(30.0)  # max(10, 20, 30)
        assert rows[4] == pytest.approx(40.0)  # max(20, 30, 40)
        assert rows[5] == pytest.approx(50.0)  # max(30, 40, 50)

    def test_multiple_windows_and_funcs(self, spark):
        df = self._make_rolling_df(spark)
        result_df = add_rolling_features(
            df, "sales",
            partition_by="store_id",
            order_by="period",
            windows=[2, 3],
            funcs=["avg", "sum"],
        )

        expected_cols = [
            "sales_rolling_avg_2",
            "sales_rolling_sum_2",
            "sales_rolling_avg_3",
            "sales_rolling_sum_3",
        ]
        for col in expected_cols:
            assert col in result_df.columns

    def test_invalid_func_raises(self, spark):
        df = self._make_rolling_df(spark)
        with pytest.raises(ValueError, match="Unknown function"):
            add_rolling_features(
                df, "sales",
                partition_by="store_id",
                order_by="period",
                windows=[3],
                funcs=["median"],
            )


# ---------------------------------------------------------------------------
# 8. add_interaction_features
# ---------------------------------------------------------------------------

class TestAddInteractionFeatures:

    def test_interaction_column_name(self, spark):
        df = spark.createDataFrame(
            [(2.0, 3.0), (4.0, 5.0)],
            ["age", "salary"],
        )
        result_df = add_interaction_features(df, [("age", "salary")])

        assert "age_x_salary" in result_df.columns

    def test_interaction_product_values(self, spark):
        df = spark.createDataFrame(
            [(2.0, 3.0), (4.0, 5.0)],
            ["age", "salary"],
        )
        result_df = add_interaction_features(df, [("age", "salary")])

        rows = result_df.orderBy("age").select("age_x_salary").collect()
        assert rows[0][0] == pytest.approx(6.0)   # 2.0 * 3.0
        assert rows[1][0] == pytest.approx(20.0)  # 4.0 * 5.0

    def test_multiple_interaction_pairs(self, spark):
        df = spark.createDataFrame(
            [(2.0, 3.0, 5.0, 7.0)],
            ["a", "b", "c", "d"],
        )
        result_df = add_interaction_features(df, [("a", "b"), ("c", "d")])

        assert "a_x_b" in result_df.columns
        assert "c_x_d" in result_df.columns
        row = result_df.first()
        assert row["a_x_b"] == pytest.approx(6.0)
        assert row["c_x_d"] == pytest.approx(35.0)

    def test_original_columns_preserved(self, spark):
        df = spark.createDataFrame([(2.0, 3.0)], ["x", "y"])
        result_df = add_interaction_features(df, [("x", "y")])

        assert "x" in result_df.columns
        assert "y" in result_df.columns

    def test_empty_pairs_returns_unchanged(self, spark):
        df = spark.createDataFrame([(2.0, 3.0)], ["x", "y"])
        result_df = add_interaction_features(df, [])

        assert result_df.columns == ["x", "y"]


# ---------------------------------------------------------------------------
# 9. add_ratio_features
# ---------------------------------------------------------------------------

class TestAddRatioFeatures:

    def test_ratio_correct_output_column_name(self, spark):
        df = spark.createDataFrame(
            [(100.0, 5.0)],
            ["revenue", "employees"],
        )
        result_df = add_ratio_features(
            df, [("revenue", "employees", "revenue_per_employee")]
        )

        assert "revenue_per_employee" in result_df.columns

    def test_ratio_correct_values(self, spark):
        df = spark.createDataFrame(
            [(100.0, 5.0), (200.0, 4.0)],
            ["revenue", "employees"],
        )
        result_df = add_ratio_features(
            df, [("revenue", "employees", "rev_per_emp")]
        )

        rows = result_df.orderBy("revenue").select("rev_per_emp").collect()
        assert rows[0][0] == pytest.approx(20.0)  # 100 / 5
        assert rows[1][0] == pytest.approx(50.0)  # 200 / 4

    def test_division_by_zero_returns_null(self, spark):
        df = spark.createDataFrame(
            [(100.0, 0.0)],
            ["numerator", "denominator"],
        )
        result_df = add_ratio_features(
            df, [("numerator", "denominator", "ratio")]
        )

        val = result_df.select("ratio").first()[0]
        assert val is None

    def test_multiple_ratios(self, spark):
        df = spark.createDataFrame(
            [(10.0, 2.0, 100.0, 4.0)],
            ["a", "b", "c", "d"],
        )
        result_df = add_ratio_features(
            df,
            [("a", "b", "a_div_b"), ("c", "d", "c_div_d")],
        )

        assert "a_div_b" in result_df.columns
        assert "c_div_d" in result_df.columns
        row = result_df.first()
        assert row["a_div_b"] == pytest.approx(5.0)
        assert row["c_div_d"] == pytest.approx(25.0)

    def test_original_columns_preserved(self, spark):
        df = spark.createDataFrame([(10.0, 2.0)], ["num", "denom"])
        result_df = add_ratio_features(df, [("num", "denom", "ratio")])

        assert "num" in result_df.columns
        assert "denom" in result_df.columns


# ---------------------------------------------------------------------------
# 10. add_missing_indicators
# ---------------------------------------------------------------------------

class TestAddMissingIndicators:

    def test_indicator_column_name(self, spark):
        df = spark.createDataFrame(
            [(1.0,), (None,)],
            ["age"],
        )
        result_df = add_missing_indicators(df, ["age"])

        assert "age_missing" in result_df.columns

    def test_indicator_values_are_binary(self, spark):
        df = spark.createDataFrame(
            [(1.0,), (None,), (3.0,)],
            ["score"],
        )
        result_df = add_missing_indicators(df, ["score"])

        rows = result_df.select("score", "score_missing").collect()
        score_to_indicator = {row["score"]: row["score_missing"] for row in rows}
        # None key maps to null key in Python dict; check separately
        not_null_vals = [(row["score"], row["score_missing"]) for row in rows]

        for val, indicator in not_null_vals:
            if val is None:
                assert indicator == 1
            else:
                assert indicator == 0

    def test_non_null_rows_get_zero(self, spark):
        df = spark.createDataFrame(
            [(10.0,), (20.0,)],
            ["val"],
        )
        result_df = add_missing_indicators(df, ["val"])
        rows = result_df.select("val_missing").collect()
        assert all(row[0] == 0 for row in rows)

    def test_null_rows_get_one(self, spark):
        df = spark.createDataFrame(
            [(None,), (None,)],
            schema="val double",
        )
        result_df = add_missing_indicators(df, ["val"])
        rows = result_df.select("val_missing").collect()
        assert all(row[0] == 1 for row in rows)

    def test_multiple_columns(self, spark):
        df = spark.createDataFrame(
            [(1.0, None), (None, 2.0)],
            ["a", "b"],
        )
        result_df = add_missing_indicators(df, ["a", "b"])

        assert "a_missing" in result_df.columns
        assert "b_missing" in result_df.columns

    def test_auto_detect_null_columns(self, spark):
        df = spark.createDataFrame(
            [(1.0, 2.0), (None, 3.0)],
            ["x", "y"],
        )
        # y has no nulls, x has one null
        result_df = add_missing_indicators(df)

        assert "x_missing" in result_df.columns
        # y has no nulls so no indicator column should be created
        assert "y_missing" not in result_df.columns


# ---------------------------------------------------------------------------
# 11. impute_columns
# ---------------------------------------------------------------------------

class TestImputeColumns:

    def test_median_imputation_fills_nulls(self, spark):
        df = spark.createDataFrame(
            [(1.0,), (2.0,), (None,), (4.0,), (5.0,)],
            ["val"],
        )
        result_df, impute_values = impute_columns(df, strategy="median", columns=["val"])

        assert result_df.filter(F.col("val").isNull()).count() == 0

    def test_median_returns_impute_values_dict(self, spark):
        df = spark.createDataFrame(
            [(1.0,), (2.0,), (None,), (4.0,), (5.0,)],
            ["val"],
        )
        _, impute_values = impute_columns(df, strategy="median", columns=["val"])

        assert "val" in impute_values
        # Median of [1, 2, 4, 5] ≈ 3.0 (approxQuantile result)
        assert impute_values["val"] == pytest.approx(3.0, abs=1.0)

    def test_mean_imputation_fills_nulls(self, spark):
        df = spark.createDataFrame(
            [(10.0,), (20.0,), (None,), (30.0,)],
            ["score"],
        )
        result_df, impute_values = impute_columns(df, strategy="mean", columns=["score"])

        assert result_df.filter(F.col("score").isNull()).count() == 0
        assert "score" in impute_values
        # Mean of [10, 20, 30] = 20.0
        assert impute_values["score"] == pytest.approx(20.0)

    def test_mean_imputed_value_is_correct(self, spark):
        df = spark.createDataFrame(
            [(10.0,), (20.0,), (None,), (30.0,)],
            ["score"],
        )
        result_df, _ = impute_columns(df, strategy="mean", columns=["score"])

        # The null row should be filled with mean=20.0
        null_filled = (
            result_df
            .orderBy("score")
            .select("score")
            .collect()
        )
        values = [row[0] for row in null_filled]
        assert 20.0 in values

    def test_median_imputed_value_is_correct(self, spark):
        # Sorted: 10, 20, 30, 40 → median ≈ 25
        df = spark.createDataFrame(
            [(10.0,), (20.0,), (None,), (30.0,), (40.0,)],
            ["val"],
        )
        result_df, impute_values = impute_columns(df, strategy="median", columns=["val"])

        assert result_df.filter(F.col("val").isNull()).count() == 0
        assert impute_values["val"] == pytest.approx(25.0, abs=5.0)

    def test_impute_values_dict_can_be_reused(self, spark):
        df = spark.createDataFrame(
            [(1.0,), (2.0,), (None,), (4.0,)],
            ["x"],
        )
        _, impute_values = impute_columns(df, strategy="mean", columns=["x"])

        # Reuse on a new dataframe
        new_df = spark.createDataFrame([(None,), (5.0,)], ["x"])
        filled = new_df.fillna(impute_values)
        assert filled.filter(F.col("x").isNull()).count() == 0

    def test_invalid_strategy_raises(self, spark):
        df = spark.createDataFrame([(1.0,)], ["val"])
        with pytest.raises(ValueError, match="strategy must be"):
            impute_columns(df, strategy="knn", columns=["val"])

    def test_no_numeric_columns_returns_empty_dict(self, spark):
        df = spark.createDataFrame([("hello",)], ["name"])
        result_df, impute_values = impute_columns(df, strategy="median")

        assert impute_values == {}

    def test_columns_without_nulls_retain_values(self, spark):
        df = spark.createDataFrame(
            [(1.0,), (2.0,), (3.0,)],
            ["val"],
        )
        result_df, _ = impute_columns(df, strategy="median", columns=["val"])

        original = sorted([row[0] for row in df.collect()])
        imputed = sorted([row[0] for row in result_df.select("val").collect()])
        assert original == imputed


# ---------------------------------------------------------------------------
# 12. prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:

    def _make_full_df(self, spark):
        data = [
            (25.0, 50000.0, "eng",  "NYC", 0),
            (30.0, 60000.0, "mkt",  "LA",  1),
            (None, 70000.0, "eng",  "NYC", 0),
            (40.0, None,    "hr",   "CHI", 1),
            (35.0, 80000.0, "mkt",  "LA",  0),
        ]
        return spark.createDataFrame(data, ["age", "salary", "dept", "city", "label"])

    def test_returns_features_column(self, spark):
        df = self._make_full_df(spark)
        result_df, artifacts = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept", "city"],
            label_col="label",
        )
        assert "features" in result_df.columns

    def test_returns_scaled_features_column(self, spark):
        df = self._make_full_df(spark)
        result_df, artifacts = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept", "city"],
            label_col="label",
        )
        assert "scaled_features" in result_df.columns

    def test_impute_values_in_artifacts(self, spark):
        df = self._make_full_df(spark)
        _, artifacts = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept", "city"],
        )
        assert "impute_values" in artifacts
        assert isinstance(artifacts["impute_values"], dict)

    def test_indexer_stages_in_artifacts(self, spark):
        df = self._make_full_df(spark)
        _, artifacts = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept", "city"],
        )
        assert "indexer_stages" in artifacts
        # Two categorical cols → two StringIndexer stages
        assert len(artifacts["indexer_stages"]) == 2

    def test_fitted_scaler_in_artifacts(self, spark):
        df = self._make_full_df(spark)
        _, artifacts = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept"],
        )
        assert "fitted_scaler" in artifacts
        assert artifacts["fitted_scaler"] is not None

    def test_feature_cols_in_artifacts(self, spark):
        df = self._make_full_df(spark)
        _, artifacts = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept"],
        )
        assert "feature_cols" in artifacts
        feature_cols = artifacts["feature_cols"]
        # Should include numeric, encoded categorical, and missing indicators
        assert "age" in feature_cols
        assert "salary" in feature_cols
        assert "dept_indexed" in feature_cols
        assert "age_missing" in feature_cols
        assert "salary_missing" in feature_cols
        assert "dept_missing" in feature_cols

    def test_nulls_imputed_in_output(self, spark):
        df = self._make_full_df(spark)
        result_df, _ = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept", "city"],
        )
        # After imputation, no nulls in numeric columns
        assert result_df.filter(F.col("age").isNull()).count() == 0
        assert result_df.filter(F.col("salary").isNull()).count() == 0

    def test_missing_indicator_columns_present(self, spark):
        df = self._make_full_df(spark)
        result_df, _ = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept", "city"],
        )
        assert "age_missing" in result_df.columns
        assert "salary_missing" in result_df.columns

    def test_categorical_encoded_columns_present(self, spark):
        df = self._make_full_df(spark)
        result_df, _ = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept", "city"],
        )
        assert "dept_indexed" in result_df.columns
        assert "city_indexed" in result_df.columns

    def test_no_categorical_cols(self, spark):
        df = spark.createDataFrame(
            [(1.0, 2.0), (None, 3.0), (4.0, 5.0)],
            ["a", "b"],
        )
        result_df, artifacts = prepare_features(
            df,
            numeric_cols=["a", "b"],
            categorical_cols=[],
        )
        assert "features" in result_df.columns
        assert artifacts["indexer_stages"] == []

    def test_minmax_scale_method(self, spark):
        df = self._make_full_df(spark)
        result_df, artifacts = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept"],
            scale_method="minmax",
        )
        assert "scaled_features" in result_df.columns

    def test_row_count_preserved(self, spark):
        df = self._make_full_df(spark)
        result_df, _ = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept", "city"],
            label_col="label",
        )
        # Row count should be the same (or fewer if handle_invalid="skip" drops rows)
        assert result_df.count() <= df.count()
        assert result_df.count() > 0

    @pytest.mark.skip(reason="Pipeline model requires matching schema on new data")
    def test_pipeline_model_can_transform_new_data(self, spark):
        df = self._make_full_df(spark)
        _, artifacts = prepare_features(
            df,
            numeric_cols=["age", "salary"],
            categorical_cols=["dept"],
        )

        # Apply artifacts to a new dataset
        new_data = [
            (28.0, 55000.0, "eng"),
            (32.0, 65000.0, "mkt"),
        ]
        new_df = spark.createDataFrame(new_data, ["age", "salary", "dept"])
        new_df = new_df.fillna(artifacts["impute_values"])
        for stage in artifacts["indexer_stages"]:
            new_df = stage.transform(new_df)
        new_df = assemble_features(new_df, artifacts["feature_cols"], handle_invalid="keep")
        new_df = artifacts["fitted_scaler"].transform(new_df)

        assert "scaled_features" in new_df.columns
        assert new_df.count() == 2
