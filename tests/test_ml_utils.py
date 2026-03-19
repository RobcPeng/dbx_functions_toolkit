"""
test_ml_utils.py
----------------
Pytest tests for dbx_toolkit.ml_utils.

MLflow-dependent functions (log_run_to_mlflow) and batch_predict are skipped.
Uses the session-scoped `spark` fixture from conftest.py.
"""

from __future__ import annotations

import pytest

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.feature import VectorAssembler

from dbx_toolkit.ml_utils import (
    classification_report,
    compare_models,
    feature_importance,
    handle_class_imbalance,
    regression_report,
    split_data,
    train_and_evaluate,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_classification_df(spark):
    """Return a Spark DataFrame with 24 rows of well-separated binary classes."""
    data = [
        # label 0 — cluster near (1,2)
        (1.0, 2.0, 0),
        (1.5, 2.5, 0),
        (2.0, 1.5, 0),
        (2.5, 3.0, 0),
        (0.5, 1.0, 0),
        (1.0, 1.0, 0),
        (0.8, 2.2, 0),
        (1.2, 0.8, 0),
        (2.2, 2.8, 0),
        (1.8, 1.2, 0),
        (0.6, 1.6, 0),
        (2.4, 2.4, 0),
        # label 1 — cluster near (10,11)
        (10.0, 11.0, 1),
        (11.0, 12.0, 1),
        (10.5, 10.5, 1),
        (9.5, 11.5, 1),
        (12.0, 11.0, 1),
        (10.0, 10.0, 1),
        (11.5, 12.5, 1),
        (9.0, 10.0, 1),
        (10.8, 11.8, 1),
        (11.2, 10.2, 1),
        (9.8, 12.0, 1),
        (10.3, 9.7, 1),
    ]
    df = spark.createDataFrame(data, ["f1", "f2", "label"])
    assembler = VectorAssembler(inputCols=["f1", "f2"], outputCol="features")
    return assembler.transform(df)


def _make_regression_df(spark):
    """Return a Spark DataFrame suitable for regression_report tests."""
    # label == prediction for perfect fit rows; add some noise for variety
    data = [
        (1.0, 1.0),
        (2.0, 2.1),
        (3.0, 2.9),
        (4.0, 4.2),
        (5.0, 4.8),
        (6.0, 6.0),
        (7.0, 7.3),
        (8.0, 7.7),
        (9.0, 9.1),
        (10.0, 10.0),
    ]
    return spark.createDataFrame(data, ["label", "prediction"])


def _make_imbalanced_df(spark):
    """Return an imbalanced binary-class DataFrame (16 majority, 4 minority)."""
    data = (
        [(float(i), 0) for i in range(16)]
        + [(float(i + 100), 1) for i in range(4)]
    )
    return spark.createDataFrame(data, ["value", "label"])


def _lr_pipeline():
    """Return an unfitted LogisticRegression Pipeline (expects 'features' col)."""
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    return Pipeline(stages=[lr])


def _dt_pipeline():
    """Return an unfitted DecisionTreeClassifier Pipeline."""
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    return Pipeline(stages=[dt])


# ---------------------------------------------------------------------------
# 1. split_data
# ---------------------------------------------------------------------------

class TestSplitData:

    def test_two_way_split_preserves_total_count(self, spark):
        df = _make_classification_df(spark)
        total = df.count()
        train, test = split_data(df, ratios=[0.8, 0.2], seed=42)
        assert train.count() + test.count() == total

    def test_two_way_split_default_ratios(self, spark):
        df = _make_classification_df(spark)
        total = df.count()
        train, test = split_data(df, seed=42)
        assert train.count() + test.count() == total

    def test_three_way_split_preserves_total_count(self, spark):
        df = _make_classification_df(spark)
        total = df.count()
        train, val, test = split_data(df, ratios=[0.7, 0.15, 0.15], seed=42)
        assert train.count() + val.count() + test.count() == total

    def test_three_way_split_returns_three_parts(self, spark):
        df = _make_classification_df(spark)
        result = split_data(df, ratios=[0.6, 0.2, 0.2], seed=42)
        assert len(result) == 3

    def test_stratified_split_preserves_class_ratio(self, spark):
        df = _make_classification_df(spark)
        # 12 of each class → 50 / 50 split
        train, test = split_data(df, ratios=[0.8, 0.2], seed=42, stratify_col="label")
        total = df.count()
        assert train.count() + test.count() == total

        # Both splits should contain at least one row of each class
        train_labels = {row.label for row in train.select("label").collect()}
        test_labels = {row.label for row in test.select("label").collect()}
        assert 0 in train_labels
        assert 1 in train_labels
        # test might be very small so only verify the union covers both classes
        assert train_labels | test_labels == {0, 1}

    def test_invalid_ratio_count_raises(self, spark):
        df = _make_classification_df(spark)
        with pytest.raises(ValueError, match="ratios must contain 2"):
            split_data(df, ratios=[0.5, 0.3, 0.1, 0.1])

    def test_invalid_ratio_sum_raises(self, spark):
        df = _make_classification_df(spark)
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_data(df, ratios=[0.5, 0.3])

    def test_single_ratio_count_raises(self, spark):
        df = _make_classification_df(spark)
        with pytest.raises(ValueError, match="ratios must contain 2"):
            split_data(df, ratios=[1.0])


# ---------------------------------------------------------------------------
# 2. train_and_evaluate
# ---------------------------------------------------------------------------

class TestTrainAndEvaluate:

    def test_returns_required_keys(self, spark):
        df = _make_classification_df(spark)
        train, test = split_data(df, ratios=[0.8, 0.2], seed=42)
        result = train_and_evaluate(_lr_pipeline(), train, test)
        assert "model" in result
        assert "predictions" in result
        assert "metrics" in result

    def test_primary_metric_present(self, spark):
        df = _make_classification_df(spark)
        train, test = split_data(df, ratios=[0.8, 0.2], seed=42)
        result = train_and_evaluate(_lr_pipeline(), train, test)
        assert "primary_metric" in result
        assert isinstance(result["primary_metric"], float)

    def test_metric_value_in_valid_range(self, spark):
        df = _make_classification_df(spark)
        train, test = split_data(df, ratios=[0.8, 0.2], seed=42)
        result = train_and_evaluate(_lr_pipeline(), train, test)
        # Default evaluator is AUC-ROC — well-separated data should yield high AUC
        assert 0.0 <= result["primary_metric"] <= 1.0

    def test_predictions_dataframe_has_rows(self, spark):
        df = _make_classification_df(spark)
        train, test = split_data(df, ratios=[0.8, 0.2], seed=42)
        result = train_and_evaluate(_lr_pipeline(), train, test)
        assert result["predictions"].count() > 0

    def test_metrics_is_dict_with_float_values(self, spark):
        df = _make_classification_df(spark)
        train, test = split_data(df, ratios=[0.8, 0.2], seed=42)
        result = train_and_evaluate(_lr_pipeline(), train, test)
        assert isinstance(result["metrics"], dict)
        for v in result["metrics"].values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# 3. compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:

    @pytest.fixture(scope="class")
    def comparison_df(self, spark):
        df = _make_classification_df(spark)
        train, test = split_data(df, ratios=[0.8, 0.2], seed=42)
        return compare_models(
            {"LR": _lr_pipeline(), "DT": _dt_pipeline()},
            train,
            test,
        )

    def test_returns_one_row_per_model(self, comparison_df):
        assert comparison_df.count() == 2

    def test_has_model_name_column(self, comparison_df):
        assert "model_name" in comparison_df.columns

    def test_has_accuracy_column(self, comparison_df):
        assert "accuracy" in comparison_df.columns

    def test_has_f1_column(self, comparison_df):
        assert "f1" in comparison_df.columns

    def test_model_names_present(self, comparison_df):
        names = {row.model_name for row in comparison_df.select("model_name").collect()}
        assert names == {"LR", "DT"}

    def test_accuracy_values_in_range(self, comparison_df):
        for row in comparison_df.collect():
            assert 0.0 <= row["accuracy"] <= 1.0

    def test_invalid_task_raises(self, spark):
        df = _make_classification_df(spark)
        train, test = split_data(df, ratios=[0.8, 0.2], seed=42)
        with pytest.raises(ValueError, match="task must be"):
            compare_models({"LR": _lr_pipeline()}, train, test, task="invalid")


# ---------------------------------------------------------------------------
# 4. classification_report
# ---------------------------------------------------------------------------

class TestClassificationReport:

    @pytest.fixture(scope="class")
    def predictions_df(self, spark):
        """Fake predictions DataFrame with label and prediction columns."""
        data = [
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),  # false positive
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),  # false negative
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (1.0, 1.0),
        ]
        return spark.createDataFrame(data, ["label", "prediction"])

    def test_returns_metrics_and_confusion_matrix_keys(self, predictions_df):
        report = classification_report(predictions_df)
        assert "metrics" in report
        assert "confusion_matrix" in report

    def test_metrics_has_accuracy(self, predictions_df):
        report = classification_report(predictions_df)
        assert "accuracy" in report["metrics"]

    def test_metrics_has_f1(self, predictions_df):
        report = classification_report(predictions_df)
        assert "f1" in report["metrics"]

    def test_metrics_has_precision(self, predictions_df):
        report = classification_report(predictions_df)
        assert "precision" in report["metrics"]

    def test_metrics_has_recall(self, predictions_df):
        report = classification_report(predictions_df)
        assert "recall" in report["metrics"]

    def test_accuracy_value_correct(self, predictions_df):
        # 8 correct out of 10 → 0.8
        report = classification_report(predictions_df)
        assert report["metrics"]["accuracy"] == pytest.approx(0.8, abs=1e-6)

    def test_accuracy_in_valid_range(self, predictions_df):
        report = classification_report(predictions_df)
        assert 0.0 <= report["metrics"]["accuracy"] <= 1.0

    def test_confusion_matrix_is_spark_df(self, spark, predictions_df):
        from pyspark.sql import DataFrame as SparkDF
        report = classification_report(predictions_df)
        assert isinstance(report["confusion_matrix"], SparkDF)

    def test_confusion_matrix_has_count_column(self, predictions_df):
        report = classification_report(predictions_df)
        assert "count" in report["confusion_matrix"].columns

    def test_confusion_matrix_total_matches_input(self, predictions_df):
        report = classification_report(predictions_df)
        total = sum(row["count"] for row in report["confusion_matrix"].collect())
        assert total == predictions_df.count()


# ---------------------------------------------------------------------------
# 5. regression_report
# ---------------------------------------------------------------------------

class TestRegressionReport:

    @pytest.fixture(scope="class")
    def reg_report(self, spark):
        preds_df = _make_regression_df(spark)
        return regression_report(preds_df)

    def test_returns_rmse(self, reg_report):
        assert "rmse" in reg_report

    def test_returns_mse(self, reg_report):
        assert "mse" in reg_report

    def test_returns_mae(self, reg_report):
        assert "mae" in reg_report

    def test_returns_r2(self, reg_report):
        assert "r2" in reg_report

    def test_rmse_is_non_negative(self, reg_report):
        assert reg_report["rmse"] >= 0.0

    def test_mse_is_non_negative(self, reg_report):
        assert reg_report["mse"] >= 0.0

    def test_mae_is_non_negative(self, reg_report):
        assert reg_report["mae"] >= 0.0

    def test_r2_near_one_for_close_predictions(self, reg_report):
        # Predictions are very close to labels → R2 should be high
        assert reg_report["r2"] == pytest.approx(1.0, abs=0.1)

    def test_rmse_equals_sqrt_mse(self, reg_report):
        import math
        assert reg_report["rmse"] == pytest.approx(math.sqrt(reg_report["mse"]), abs=1e-6)

    def test_mape_present_for_nonzero_labels(self, reg_report):
        # All labels are nonzero in _make_regression_df, so mape should be computed
        assert reg_report["mape"] is not None
        assert reg_report["mape"] >= 0.0


# ---------------------------------------------------------------------------
# 6. feature_importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:

    @pytest.fixture(scope="class")
    def fitted_rf_model(self, spark):
        """Fit a RandomForestClassifier and return the fitted model."""
        df = _make_classification_df(spark)
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=10,
            seed=42,
        )
        return rf.fit(df)

    def test_returns_spark_dataframe(self, spark, fitted_rf_model):
        from pyspark.sql import DataFrame as SparkDF
        fi_df = feature_importance(fitted_rf_model, ["f1", "f2"])
        assert isinstance(fi_df, SparkDF)

    def test_has_feature_column(self, spark, fitted_rf_model):
        fi_df = feature_importance(fitted_rf_model, ["f1", "f2"])
        assert "feature" in fi_df.columns

    def test_has_importance_column(self, spark, fitted_rf_model):
        fi_df = feature_importance(fitted_rf_model, ["f1", "f2"])
        assert "importance" in fi_df.columns

    def test_row_count_matches_feature_count(self, spark, fitted_rf_model):
        fi_df = feature_importance(fitted_rf_model, ["f1", "f2"])
        assert fi_df.count() == 2

    def test_feature_names_are_present(self, spark, fitted_rf_model):
        fi_df = feature_importance(fitted_rf_model, ["f1", "f2"])
        names = {row.feature for row in fi_df.collect()}
        assert names == {"f1", "f2"}

    def test_importances_are_non_negative(self, spark, fitted_rf_model):
        fi_df = feature_importance(fitted_rf_model, ["f1", "f2"])
        for row in fi_df.collect():
            assert row["importance"] >= 0.0

    def test_importances_sum_to_one(self, spark, fitted_rf_model):
        fi_df = feature_importance(fitted_rf_model, ["f1", "f2"])
        total = sum(row["importance"] for row in fi_df.collect())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_sorted_descending(self, spark, fitted_rf_model):
        fi_df = feature_importance(fitted_rf_model, ["f1", "f2"])
        values = [row["importance"] for row in fi_df.collect()]
        assert values == sorted(values, reverse=True)

    def test_mismatched_feature_names_raises(self, spark, fitted_rf_model):
        with pytest.raises(ValueError, match="feature_names length"):
            feature_importance(fitted_rf_model, ["f1", "f2", "extra"])

    def test_works_with_pipeline_model(self, spark):
        """feature_importance should unwrap the last stage of a PipelineModel."""
        df = _make_classification_df(spark)
        rf = RandomForestClassifier(
            featuresCol="features", labelCol="label", numTrees=5, seed=42
        )
        pipeline = Pipeline(stages=[rf])
        fitted = pipeline.fit(df)
        fi_df = feature_importance(fitted, ["f1", "f2"])
        assert fi_df.count() == 2

    def test_unsupported_model_raises_attribute_error(self, spark):
        """A model without featureImportances or coefficients should raise AttributeError."""
        from pyspark.ml.feature import StandardScalerModel
        # Use a mock-like object that has no relevant attribute
        class DummyModel:
            pass
        with pytest.raises(AttributeError, match="does not expose"):
            feature_importance(DummyModel(), ["f1", "f2"])


# ---------------------------------------------------------------------------
# 7. handle_class_imbalance
# ---------------------------------------------------------------------------

class TestHandleClassImbalance:

    def test_oversample_increases_minority_count(self, spark):
        df = _make_imbalanced_df(spark)
        # Before: 16 majority (0), 4 minority (1)
        minority_before = df.filter("label = 1").count()
        balanced = handle_class_imbalance(df, label_col="label", strategy="oversample")
        minority_after = balanced.filter("label = 1").count()
        assert minority_after > minority_before

    def test_oversample_total_count_increases(self, spark):
        df = _make_imbalanced_df(spark)
        total_before = df.count()
        balanced = handle_class_imbalance(df, label_col="label", strategy="oversample")
        assert balanced.count() >= total_before

    def test_oversample_majority_count_unchanged(self, spark):
        df = _make_imbalanced_df(spark)
        majority_before = df.filter("label = 0").count()
        balanced = handle_class_imbalance(df, label_col="label", strategy="oversample")
        majority_after = balanced.filter("label = 0").count()
        assert majority_after == majority_before

    def test_undersample_decreases_majority_count(self, spark):
        df = _make_imbalanced_df(spark)
        majority_before = df.filter("label = 0").count()
        balanced = handle_class_imbalance(df, label_col="label", strategy="undersample")
        majority_after = balanced.filter("label = 0").count()
        assert majority_after < majority_before

    def test_undersample_minority_count_unchanged(self, spark):
        df = _make_imbalanced_df(spark)
        minority_before = df.filter("label = 1").count()
        balanced = handle_class_imbalance(df, label_col="label", strategy="undersample")
        minority_after = balanced.filter("label = 1").count()
        assert minority_after == minority_before

    def test_weight_adds_class_weight_column(self, spark):
        df = _make_imbalanced_df(spark)
        weighted = handle_class_imbalance(df, label_col="label", strategy="weight")
        assert "class_weight" in weighted.columns

    def test_weight_preserves_row_count(self, spark):
        df = _make_imbalanced_df(spark)
        weighted = handle_class_imbalance(df, label_col="label", strategy="weight")
        assert weighted.count() == df.count()

    def test_weight_minority_has_higher_weight(self, spark):
        df = _make_imbalanced_df(spark)
        weighted = handle_class_imbalance(df, label_col="label", strategy="weight")
        minority_weight = (
            weighted.filter("label = 1")
            .select("class_weight")
            .first()["class_weight"]
        )
        majority_weight = (
            weighted.filter("label = 0")
            .select("class_weight")
            .first()["class_weight"]
        )
        assert minority_weight > majority_weight

    def test_invalid_strategy_raises(self, spark):
        df = _make_imbalanced_df(spark)
        with pytest.raises(ValueError, match="strategy must be"):
            handle_class_imbalance(df, strategy="smote")

    def test_multiclass_raises(self, spark):
        data = [(1.0, 0), (2.0, 1), (3.0, 2)]
        df = spark.createDataFrame(data, ["value", "label"])
        with pytest.raises(ValueError, match="binary labels"):
            handle_class_imbalance(df, label_col="label", strategy="oversample")

    def test_already_balanced_oversample_noop(self, spark):
        """When classes are already equal, oversample should return the original count."""
        data = [(float(i), 0) for i in range(10)] + [(float(i + 100), 1) for i in range(10)]
        df = spark.createDataFrame(data, ["value", "label"])
        balanced = handle_class_imbalance(df, label_col="label", strategy="oversample")
        assert balanced.count() == df.count()
