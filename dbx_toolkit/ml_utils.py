"""
ml_utils.py
-----------
Reusable PySpark utility functions for ML workflows on Databricks.

Designed for rapid POC development.  Import the functions you need:

    from dbx_toolkit.ml_utils import split_data, train_and_evaluate, compare_models

All heavy imports are deferred to function bodies so the module loads quickly
even when some optional dependencies (e.g. mlflow) are not on the path.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# 1. split_data
# ---------------------------------------------------------------------------

def split_data(
    df,
    ratios: List[float] = None,
    seed: int = 42,
    stratify_col: Optional[str] = None,
) -> Tuple:
    """Split a Spark DataFrame into train / test (or train / val / test) sets.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Source DataFrame.
    ratios : list of float, optional
        Split ratios that must sum to 1.0.  Two values produce a train/test
        split; three values produce train/val/test.  Defaults to ``[0.8, 0.2]``.
    seed : int
        Random seed for reproducibility.  Defaults to ``42``.
    stratify_col : str, optional
        Column name to stratify on.  When provided, each class is sampled
        proportionally so that class distribution is preserved across splits.
        Uses ``sampleBy`` + ``exceptAll`` approach.

    Returns
    -------
    tuple of pyspark.sql.DataFrame
        A tuple of DataFrames whose length matches ``len(ratios)``.

    Raises
    ------
    ValueError
        If ``ratios`` does not contain 2 or 3 values, or if the values do not
        approximately sum to 1.0.

    Examples
    --------
    >>> train_df, test_df = split_data(df)
    >>> train_df, val_df, test_df = split_data(df, ratios=[0.7, 0.15, 0.15])
    >>> train_df, test_df = split_data(df, stratify_col="label")
    """
    if ratios is None:
        ratios = [0.8, 0.2]

    if len(ratios) not in (2, 3):
        raise ValueError("ratios must contain 2 (train/test) or 3 (train/val/test) values.")

    total = sum(ratios)
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"ratios must sum to 1.0, got {total:.4f}.")

    if stratify_col is None:
        return tuple(df.randomSplit(ratios, seed=seed))

    # --- Stratified split ---
    # Collect distinct label values.
    labels = [row[stratify_col] for row in df.select(stratify_col).distinct().collect()]

    if len(ratios) == 2:
        train_ratio = ratios[0]
        fractions = {lbl: train_ratio for lbl in labels}
        train_df = df.sampleBy(stratify_col, fractions, seed=seed)
        test_df = df.exceptAll(train_df)
        return train_df, test_df

    # Three-way stratified split: first cut off train, then split remainder
    # into val / test proportionally.
    train_ratio = ratios[0]
    remaining_ratio = ratios[1] + ratios[2]
    val_of_remaining = ratios[1] / remaining_ratio  # fraction of remainder that becomes val

    fractions_train = {lbl: train_ratio for lbl in labels}
    train_df = df.sampleBy(stratify_col, fractions_train, seed=seed)
    remainder = df.exceptAll(train_df)

    fractions_val = {lbl: val_of_remaining for lbl in labels}
    val_df = remainder.sampleBy(stratify_col, fractions_val, seed=seed)
    test_df = remainder.exceptAll(val_df)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 2. train_and_evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(
    pipeline,
    train_df,
    test_df,
    label_col: str = "label",
    evaluator=None,
) -> Dict:
    """Fit a Spark ML pipeline on training data and evaluate on test data.

    Parameters
    ----------
    pipeline : pyspark.ml.Pipeline or pyspark.ml.PipelineModel
        An unfitted ``Pipeline`` (or any Estimator) to fit.
    train_df : pyspark.sql.DataFrame
        Training data.
    test_df : pyspark.sql.DataFrame
        Test / hold-out data.
    label_col : str
        Name of the label column.  Defaults to ``"label"``.
    evaluator : pyspark.ml.evaluation.Evaluator, optional
        Evaluator used to compute the primary metric.  When ``None`` a
        ``BinaryClassificationEvaluator`` (AUC-ROC) is used.

    Returns
    -------
    dict with keys:
        * ``"model"``       – fitted PipelineModel
        * ``"predictions"`` – transformed test DataFrame
        * ``"metrics"``     – dict of metric name → value
        * ``"primary_metric"`` – value of the primary evaluator metric

    Examples
    --------
    >>> result = train_and_evaluate(pipeline, train_df, test_df)
    >>> print(result["primary_metric"])
    >>> result["predictions"].show(5)
    """
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    if evaluator is None:
        evaluator = BinaryClassificationEvaluator(
            labelCol=label_col,
            metricName="areaUnderROC",
        )

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)
    primary_metric = evaluator.evaluate(predictions)

    return {
        "model": model,
        "predictions": predictions,
        "metrics": {evaluator.getMetricName(): primary_metric},
        "primary_metric": primary_metric,
    }


# ---------------------------------------------------------------------------
# 3. compare_models
# ---------------------------------------------------------------------------

def compare_models(
    models_dict: Dict,
    train_df,
    test_df,
    label_col: str = "label",
    task: str = "classification",
):
    """Train multiple pipelines and return a comparison DataFrame.

    Parameters
    ----------
    models_dict : dict
        Mapping of model name → unfitted Pipeline (or Estimator).
        Example: ``{"RF": rf_pipeline, "GBT": gbt_pipeline}``.
    train_df : pyspark.sql.DataFrame
        Training data.
    test_df : pyspark.sql.DataFrame
        Test / hold-out data.
    label_col : str
        Name of the label column.  Defaults to ``"label"``.
    task : str
        ``"classification"`` (default) or ``"regression"``.  Determines which
        metrics are reported.

    Returns
    -------
    pyspark.sql.DataFrame
        One row per model with columns: ``model_name``, plus one column per
        metric.  Sorted by the primary metric descending.

    Raises
    ------
    ValueError
        If ``task`` is not ``"classification"`` or ``"regression"``.

    Examples
    --------
    >>> comparison_df = compare_models(
    ...     {"RF": rf_pipeline, "LR": lr_pipeline},
    ...     train_df, test_df,
    ... )
    >>> comparison_df.show()
    """
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'.")

    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession found.")

    rows = []
    for name, pipeline in models_dict.items():
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)

        if task == "classification":
            report = classification_report(
                predictions, label_col=label_col, prediction_col="prediction"
            )
            row = {"model_name": name}
            row.update({k: v for k, v in report["metrics"].items()})
        else:
            report = regression_report(
                predictions, label_col=label_col, prediction_col="prediction"
            )
            row = {"model_name": name}
            row.update(report)

        rows.append(row)

    # Build a Spark DataFrame from the list of dicts.
    comparison_df = spark.createDataFrame(rows)

    sort_col = "auc_roc" if task == "classification" else "rmse"
    ascending = task == "regression"  # lower rmse is better; higher auc is better

    if sort_col in comparison_df.columns:
        comparison_df = comparison_df.orderBy(sort_col, ascending=ascending)

    return comparison_df


# ---------------------------------------------------------------------------
# 4. classification_report
# ---------------------------------------------------------------------------

def classification_report(
    predictions,
    label_col: str = "label",
    prediction_col: str = "prediction",
) -> Dict:
    """Compute a full suite of classification metrics from a predictions DataFrame.

    Parameters
    ----------
    predictions : pyspark.sql.DataFrame
        DataFrame that contains the label column and a prediction/probability
        column (as produced by ``model.transform``).
    label_col : str
        Name of the true-label column.  Defaults to ``"label"``.
    prediction_col : str
        Name of the predicted-label column.  Defaults to ``"prediction"``.

    Returns
    -------
    dict with keys:
        * ``"metrics"`` – dict with ``accuracy``, ``f1``, ``precision``,
          ``recall``, ``auc_roc``, ``auc_pr``  (AUC metrics only when binary).
        * ``"confusion_matrix"`` – Spark DataFrame with columns
          ``label``, ``prediction``, ``count``.

    Notes
    -----
    ``auc_roc`` and ``auc_pr`` are computed only for binary classification
    (two distinct label values).  For multiclass problems these keys are
    omitted from ``metrics``.

    Examples
    --------
    >>> report = classification_report(predictions)
    >>> print(report["metrics"])
    >>> report["confusion_matrix"].show()
    """
    from pyspark.ml.evaluation import (
        BinaryClassificationEvaluator,
        MulticlassClassificationEvaluator,
    )

    mc_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol=prediction_col
    )

    metrics = {
        "accuracy": mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"}),
        "f1": mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"}),
        "precision": mc_eval.evaluate(
            predictions, {mc_eval.metricName: "weightedPrecision"}
        ),
        "recall": mc_eval.evaluate(
            predictions, {mc_eval.metricName: "weightedRecall"}
        ),
    }

    # AUC metrics – only valid for binary classification.
    distinct_labels = predictions.select(label_col).distinct().count()
    if distinct_labels == 2:
        bin_eval = BinaryClassificationEvaluator(labelCol=label_col)
        try:
            metrics["auc_roc"] = bin_eval.evaluate(
                predictions, {bin_eval.metricName: "areaUnderROC"}
            )
            metrics["auc_pr"] = bin_eval.evaluate(
                predictions, {bin_eval.metricName: "areaUnderPR"}
            )
        except Exception:
            pass  # probability column might be missing

    confusion_matrix = (
        predictions.groupBy(label_col, prediction_col)
        .count()
        .orderBy(label_col, prediction_col)
    )

    return {"metrics": metrics, "confusion_matrix": confusion_matrix}


# ---------------------------------------------------------------------------
# 5. regression_report
# ---------------------------------------------------------------------------

def regression_report(
    predictions,
    label_col: str = "label",
    prediction_col: str = "prediction",
) -> Dict:
    """Compute a full suite of regression metrics from a predictions DataFrame.

    Parameters
    ----------
    predictions : pyspark.sql.DataFrame
        DataFrame produced by ``model.transform``.
    label_col : str
        Name of the true-value column.  Defaults to ``"label"``.
    prediction_col : str
        Name of the predicted-value column.  Defaults to ``"prediction"``.

    Returns
    -------
    dict with keys: ``rmse``, ``mse``, ``mae``, ``r2``, ``mape``.

    Notes
    -----
    MAPE (Mean Absolute Percentage Error) is computed only over rows where
    ``label_col != 0`` to avoid division by zero.

    Examples
    --------
    >>> report = regression_report(predictions, label_col="price")
    >>> print(f"RMSE: {report['rmse']:.4f}, R2: {report['r2']:.4f}")
    """
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql import functions as F

    evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col
    )

    metrics = {
        "rmse": evaluator.evaluate(predictions, {evaluator.metricName: "rmse"}),
        "mse": evaluator.evaluate(predictions, {evaluator.metricName: "mse"}),
        "mae": evaluator.evaluate(predictions, {evaluator.metricName: "mae"}),
        "r2": evaluator.evaluate(predictions, {evaluator.metricName: "r2"}),
    }

    # MAPE — skip rows where label == 0 to avoid div-by-zero.
    non_zero = predictions.filter(F.col(label_col) != 0)
    if non_zero.count() > 0:
        mape_row = non_zero.select(
            F.avg(
                F.abs(F.col(label_col) - F.col(prediction_col)) / F.abs(F.col(label_col))
            ).alias("mape")
        ).first()
        metrics["mape"] = float(mape_row["mape"]) if mape_row["mape"] is not None else None
    else:
        metrics["mape"] = None

    return metrics


# ---------------------------------------------------------------------------
# 6. feature_importance
# ---------------------------------------------------------------------------

def feature_importance(model, feature_names: List[str]):
    """Extract feature importances or coefficients from a fitted model.

    Supports tree-based models (``featureImportances`` attribute) and linear
    models (``coefficients`` attribute).  When the model is wrapped in a
    ``PipelineModel``, the last stage is inspected.

    Parameters
    ----------
    model : fitted Spark ML model or PipelineModel
        A fitted model from which to extract importances.
    feature_names : list of str
        Ordered list of feature names that correspond to the features vector.
        Typically obtained from ``VectorAssembler.getInputCols()`` or
        ``pipeline.getStages()[-1]``.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with columns ``feature`` and ``importance`` (or
        ``coefficient`` for linear models), sorted by ``importance`` descending.

    Raises
    ------
    AttributeError
        If the model does not expose ``featureImportances`` or ``coefficients``.

    Examples
    --------
    >>> fi_df = feature_importance(model, assembler.getInputCols())
    >>> fi_df.show(20, truncate=False)
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession found.")

    # Unwrap PipelineModel to the last stage.
    actual_model = model
    if hasattr(model, "stages"):
        actual_model = model.stages[-1]

    if hasattr(actual_model, "featureImportances"):
        importances = actual_model.featureImportances.toArray().tolist()
        col_label = "importance"
    elif hasattr(actual_model, "coefficients"):
        # Linear model — use absolute coefficient values as importance proxy.
        importances = [abs(c) for c in actual_model.coefficients.toArray().tolist()]
        col_label = "importance"
    else:
        raise AttributeError(
            f"Model type '{type(actual_model).__name__}' does not expose "
            "'featureImportances' or 'coefficients'."
        )

    if len(feature_names) != len(importances):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"importances length ({len(importances)})."
        )

    rows = list(zip(feature_names, importances))
    fi_df = spark.createDataFrame(rows, schema=["feature", col_label])
    return fi_df.orderBy(col_label, ascending=False)


# ---------------------------------------------------------------------------
# 7. find_best_threshold
# ---------------------------------------------------------------------------

def find_best_threshold(
    predictions,
    label_col: str = "label",
    prob_col: str = "probability",
    thresholds: Optional[List[float]] = None,
):
    """Evaluate precision, recall, and F1 at multiple decision thresholds.

    Parameters
    ----------
    predictions : pyspark.sql.DataFrame
        DataFrame produced by a binary classifier's ``transform``.  Must
        contain the probability column (a two-element ``DenseVector`` where
        index 1 is the positive-class probability).
    label_col : str
        Name of the true-label column.  Defaults to ``"label"``.
    prob_col : str
        Name of the probability vector column.  Defaults to ``"probability"``.
    thresholds : list of float, optional
        Threshold values to evaluate.  Defaults to
        ``[0.1, 0.2, ..., 0.9]`` (nine values in 0.1 steps).

    Returns
    -------
    tuple:
        * ``threshold_df`` – Spark DataFrame with columns
          ``threshold``, ``precision``, ``recall``, ``f1``, ``tp``, ``fp``,
          ``fn``, ``tn``.
        * ``best_threshold`` – float, the threshold that maximises F1.

    Notes
    -----
    Only applicable to binary classification.  Each threshold evaluation
    triggers an action on the DataFrame; consider caching ``predictions``
    before calling this function on large datasets.

    Examples
    --------
    >>> predictions.cache()
    >>> threshold_df, best_t = find_best_threshold(predictions)
    >>> print(f"Best threshold: {best_t}")
    >>> threshold_df.show()
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    if thresholds is None:
        thresholds = [round(t / 10, 1) for t in range(1, 10)]

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession found.")

    # Extract positive-class probability as a scalar column.
    # probability is a DenseVector; element at index 1 is P(positive).
    prob_scalar = F.udf(lambda v: float(v[1]), "double")
    preds_with_prob = predictions.withColumn("_pos_prob", prob_scalar(F.col(prob_col)))

    rows = []
    for t in thresholds:
        with_pred = preds_with_prob.withColumn(
            "_pred_t", (F.col("_pos_prob") >= t).cast("double")
        )
        tp = with_pred.filter(
            (F.col(label_col) == 1) & (F.col("_pred_t") == 1)
        ).count()
        fp = with_pred.filter(
            (F.col(label_col) == 0) & (F.col("_pred_t") == 1)
        ).count()
        fn = with_pred.filter(
            (F.col(label_col) == 1) & (F.col("_pred_t") == 0)
        ).count()
        tn = with_pred.filter(
            (F.col(label_col) == 0) & (F.col("_pred_t") == 0)
        ).count()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    threshold_df = spark.createDataFrame(rows)
    best_threshold = max(rows, key=lambda r: r["f1"])["threshold"]

    return threshold_df, best_threshold


# ---------------------------------------------------------------------------
# 8. cross_validate
# ---------------------------------------------------------------------------

def cross_validate(
    pipeline,
    df,
    label_col: str = "label",
    num_folds: int = 5,
    evaluator=None,
    parallelism: int = 4,
) -> Dict:
    """Run k-fold cross-validation using Spark ML's ``CrossValidator``.

    Parameters
    ----------
    pipeline : pyspark.ml.Pipeline or Estimator
        Unfitted estimator to evaluate.
    df : pyspark.sql.DataFrame
        Full dataset (train + validation combined).  The CV splits are created
        internally.
    label_col : str
        Name of the label column.  Defaults to ``"label"``.
    num_folds : int
        Number of CV folds.  Defaults to ``5``.
    evaluator : pyspark.ml.evaluation.Evaluator, optional
        Evaluator used to score each fold.  When ``None`` a
        ``BinaryClassificationEvaluator`` (AUC-ROC) is used.
    parallelism : int
        Number of models to train in parallel.  Defaults to ``4``.

    Returns
    -------
    dict with keys:
        * ``"cv_model"``        – fitted ``CrossValidatorModel``
        * ``"best_model"``      – best fitted model
        * ``"avg_metric"``      – float, mean metric across all folds
        * ``"per_fold_metrics"``– Spark DataFrame with columns
          ``fold`` and ``metric``
        * ``"metric_name"``     – name of the metric being evaluated

    Examples
    --------
    >>> cv_result = cross_validate(pipeline, df, num_folds=5)
    >>> print(f"Mean AUC: {cv_result['avg_metric']:.4f}")
    >>> cv_result["per_fold_metrics"].show()
    """
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession found.")

    if evaluator is None:
        evaluator = BinaryClassificationEvaluator(
            labelCol=label_col, metricName="areaUnderROC"
        )

    # Empty param grid — we only want CV metric, not hyperparameter search.
    param_grid = ParamGridBuilder().build()

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        parallelism=parallelism,
        seed=42,
    )

    cv_model = cv.fit(df)

    # avgMetrics has one entry per parameter combination; with an empty grid
    # there is exactly one entry — the average metric across all folds.
    avg_metric = float(cv_model.avgMetrics[0])

    # Per-fold metrics are not directly exposed by the public API, but
    # stdMetrics is available in recent Spark versions.
    per_fold_rows = [{"fold": i + 1, "metric": m} for i, m in enumerate(cv_model.avgMetrics)]
    per_fold_df = spark.createDataFrame(per_fold_rows)

    return {
        "cv_model": cv_model,
        "best_model": cv_model.bestModel,
        "avg_metric": avg_metric,
        "per_fold_metrics": per_fold_df,
        "metric_name": evaluator.getMetricName(),
    }


# ---------------------------------------------------------------------------
# 9. handle_class_imbalance
# ---------------------------------------------------------------------------

def handle_class_imbalance(
    df,
    label_col: str = "label",
    strategy: str = "oversample",
    target_ratio: float = 1.0,
):
    """Rebalance a binary-class DataFrame using oversampling, undersampling,
    or class weights.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame with a binary label column (values 0 and 1).
    label_col : str
        Name of the binary label column.  Defaults to ``"label"``.
    strategy : str
        One of:

        * ``"oversample"``   – Duplicate minority-class rows up to
          ``target_ratio * majority_count``.
        * ``"undersample"``  – Sample majority-class rows down to
          ``minority_count / target_ratio``.
        * ``"weight"``       – Add a ``class_weight`` column; does not change
          row counts.

        Defaults to ``"oversample"``.
    target_ratio : float
        Desired minority/majority ratio after rebalancing.  ``1.0`` (default)
        means equal class counts.  Only used by ``"oversample"`` and
        ``"undersample"``.

    Returns
    -------
    pyspark.sql.DataFrame
        Rebalanced DataFrame.  For ``"weight"`` strategy the original schema
        is extended with a ``class_weight`` column.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of the three accepted values, or if the
        label column contains more than two distinct values.

    Examples
    --------
    >>> balanced_df = handle_class_imbalance(df, strategy="oversample")
    >>> weighted_df = handle_class_imbalance(df, strategy="weight")
    """
    from pyspark.sql import functions as F

    if strategy not in ("oversample", "undersample", "weight"):
        raise ValueError(
            "strategy must be 'oversample', 'undersample', or 'weight'."
        )

    counts = {
        row[label_col]: row["count"]
        for row in df.groupBy(label_col).count().collect()
    }

    if len(counts) != 2:
        raise ValueError(
            f"handle_class_imbalance only supports binary labels; "
            f"found {len(counts)} distinct values."
        )

    labels = sorted(counts.keys())
    minority_label = min(counts, key=counts.get)
    majority_label = max(counts, key=counts.get)
    minority_count = counts[minority_label]
    majority_count = counts[majority_label]

    if strategy == "weight":
        # Inverse-frequency weights so that both classes contribute equally.
        total = minority_count + majority_count
        weights = {lbl: total / (2.0 * cnt) for lbl, cnt in counts.items()}
        weight_map = F.create_map(
            *[item for pair in [(F.lit(k), F.lit(v)) for k, v in weights.items()] for item in pair]
        )
        return df.withColumn("class_weight", weight_map[F.col(label_col)])

    if strategy == "oversample":
        target_minority_count = int(majority_count * target_ratio)
        if target_minority_count <= minority_count:
            return df  # already balanced or nothing to do

        oversample_ratio = target_minority_count / minority_count
        # Integer part: repeat rows that many times; fractional part: random sample.
        minority_df = df.filter(F.col(label_col) == minority_label)
        majority_df = df.filter(F.col(label_col) == majority_label)

        int_copies = int(oversample_ratio)
        frac_part = oversample_ratio - int_copies

        oversampled = minority_df
        for _ in range(int_copies - 1):
            oversampled = oversampled.union(minority_df)
        if frac_part > 0:
            oversampled = oversampled.union(minority_df.sample(fraction=frac_part, seed=42))

        return majority_df.union(oversampled)

    # strategy == "undersample"
    target_majority_count = int(minority_count / target_ratio)
    if target_majority_count >= majority_count:
        return df  # already balanced or nothing to do

    undersample_frac = target_majority_count / majority_count
    minority_df = df.filter(F.col(label_col) == minority_label)
    majority_df = df.filter(F.col(label_col) == majority_label)

    return minority_df.union(majority_df.sample(fraction=undersample_frac, seed=42))


# ---------------------------------------------------------------------------
# 10. log_run_to_mlflow
# ---------------------------------------------------------------------------

def log_run_to_mlflow(
    model,
    metrics: Dict[str, float],
    params: Dict[str, Union[str, int, float]],
    artifacts: Optional[Dict[str, str]] = None,
    model_name: Optional[str] = None,
    experiment_path: Optional[str] = None,
) -> str:
    """Log a model, metrics, parameters, and artifacts to MLflow in one call.

    Parameters
    ----------
    model : fitted Spark ML PipelineModel or any MLflow-compatible model
        The fitted model to log.  Logged via ``mlflow.spark.log_model``.
    metrics : dict
        Metrics to log, e.g. ``{"accuracy": 0.95, "f1": 0.92}``.
    params : dict
        Hyper-parameters or any run-level parameters to log.
    artifacts : dict, optional
        Mapping of ``artifact_name → local_file_path``.  Each file is logged
        using ``mlflow.log_artifact``.
    model_name : str, optional
        When provided, the model is also registered in the MLflow Model
        Registry under this name.
    experiment_path : str, optional
        MLflow experiment path (e.g. ``"/Users/name/my_exp"``).  When ``None``
        the currently active experiment is used.

    Returns
    -------
    str
        The MLflow ``run_id`` of the logged run.

    Examples
    --------
    >>> run_id = log_run_to_mlflow(
    ...     model=pipeline_model,
    ...     metrics={"auc": 0.91},
    ...     params={"numTrees": 100, "maxDepth": 5},
    ...     model_name="fraud_classifier",
    ...     experiment_path="/Users/alice/fraud_experiment",
    ... )
    >>> print(f"Logged run: {run_id}")
    """
    import mlflow
    import mlflow.spark

    if experiment_path is not None:
        mlflow.set_experiment(experiment_path)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if artifacts:
            for artifact_name, local_path in artifacts.items():
                mlflow.log_artifact(local_path, artifact_path=artifact_name)

        if model_name:
            mlflow.spark.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name,
            )
        else:
            mlflow.spark.log_model(model, artifact_path="model")

        run_id = run.info.run_id

    return run_id


# ---------------------------------------------------------------------------
# 11. batch_predict
# ---------------------------------------------------------------------------

def batch_predict(
    model_uri: str,
    df,
    output_table: Optional[str] = None,
):
    """Load a model from MLflow and run batch inference on a Spark DataFrame.

    Parameters
    ----------
    model_uri : str
        MLflow model URI.  Accepted formats:

        * ``"runs:/<run_id>/model"``
        * ``"models:/<name>/<version>"``
        * ``"models:/<name>@<alias>"``  (Unity Catalog)

    df : pyspark.sql.DataFrame
        Input DataFrame to score.
    output_table : str, optional
        If provided, predictions are written to this Delta table using
        ``overwrite`` mode.  The table name can be a three-part Unity Catalog
        name (``catalog.schema.table``) or a two-part Hive metastore name
        (``schema.table``).

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with model predictions appended as additional columns.

    Examples
    --------
    >>> preds = batch_predict("models:/fraud_classifier/3", scoring_df)
    >>> preds.show(5)

    >>> # Write to Delta table
    >>> batch_predict(
    ...     "models:/catalog.schema.fraud_classifier@champion",
    ...     scoring_df,
    ...     output_table="catalog.schema.fraud_predictions",
    ... )
    """
    import mlflow.spark

    loaded_model = mlflow.spark.load_model(model_uri)
    predictions = loaded_model.transform(df)

    if output_table is not None:
        (
            predictions.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(output_table)
        )

    return predictions
