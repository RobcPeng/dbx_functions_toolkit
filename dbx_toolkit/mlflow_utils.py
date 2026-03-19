"""
mlflow_utils.py
---------------
Reusable MLflow utility functions for Databricks POCs.

Covers experiment setup, run logging, model registration, comparison,
and cleanup — all designed to work with Unity Catalog (UC) model registry
and Databricks-hosted MLflow tracking.

All MLflow imports are deferred to function scope so the module can be
imported in environments where mlflow is not yet installed without raising
an ImportError at import time.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. setup_experiment
# ---------------------------------------------------------------------------

def setup_experiment(
    experiment_path: str,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Set up an MLflow experiment, creating it if it does not already exist.

    Parameters
    ----------
    experiment_path:
        Absolute workspace path for the experiment, e.g.
        ``"/Users/me@company.com/my_experiment"``.
    tags:
        Optional key/value tags to apply to the experiment.

    Returns
    -------
    str
        The ``experiment_id`` of the experiment.

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the experiment cannot be created or retrieved.
    """
    import mlflow

    mlflow.set_experiment(experiment_path)
    experiment = mlflow.get_experiment_by_name(experiment_path)

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{experiment_path}' could not be created or retrieved."
        )

    if tags:
        client = mlflow.tracking.MlflowClient()
        for key, value in tags.items():
            client.set_experiment_tag(experiment.experiment_id, key, value)

    logger.info("Experiment ready: %s (id=%s)", experiment_path, experiment.experiment_id)
    return experiment.experiment_id


# ---------------------------------------------------------------------------
# 2. log_run
# ---------------------------------------------------------------------------

def log_run(
    model: Any,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
    model_name: Optional[str] = None,
    run_name: Optional[str] = None,
    experiment_path: Optional[str] = None,
) -> str:
    """Log a complete ML run in a single call.

    Logs parameters, metrics, a model (with auto-inferred signature when
    possible), arbitrary file artifacts, and tags.  Optionally registers the
    model to Unity Catalog.

    Parameters
    ----------
    model:
        A scikit-learn, XGBoost, LightGBM, PyTorch, or other MLflow-supported
        model object.
    metrics:
        Dictionary of metric name → numeric value.
    params:
        Optional dictionary of hyper-parameters / run configuration values.
    artifacts:
        Optional mapping of ``{artifact_name: local_file_path}``.  Each file
        is uploaded under ``artifact_name`` inside the run artifact store.
    tags:
        Optional run-level tags.
    model_name:
        If provided, register the logged model to UC under this name
        (``catalog.schema.model_name`` format recommended).
    run_name:
        Human-readable name for the MLflow run.
    experiment_path:
        Workspace path of the target experiment.  If *None*, the currently
        active experiment is used.

    Returns
    -------
    str
        The ``run_id`` of the created run.
    """
    import mlflow

    if experiment_path:
        mlflow.set_experiment(experiment_path)

    with mlflow.start_run(run_name=run_name) as run:
        # --- params ---
        if params:
            mlflow.log_params(params)

        # --- metrics ---
        mlflow.log_metrics(metrics)

        # --- tags ---
        if tags:
            mlflow.set_tags(tags)

        # --- model ---
        _log_model_generic(model, model_name=model_name)

        # --- artifacts ---
        if artifacts:
            for artifact_name, local_path in artifacts.items():
                mlflow.log_artifact(local_path, artifact_path=artifact_name)

        run_id = run.info.run_id

    logger.info("Run logged: run_id=%s", run_id)
    return run_id


def _log_model_generic(model: Any, model_name: Optional[str]) -> None:
    """Internal helper: detect model flavour and log via the correct MLflow API."""
    import mlflow

    registered_model_name = model_name  # may be None

    try:
        module = type(model).__module__

        if "sklearn" in module or "xgboost" in module or "lightgbm" in module:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
        elif "torch" in module or "pytorch" in module:
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
        elif "tensorflow" in module or "keras" in module:
            mlflow.tensorflow.log_model(
                model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
        elif "pyspark" in module:
            mlflow.spark.log_model(
                model,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
        else:
            # Fall back to pyfunc
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                registered_model_name=registered_model_name,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model logging failed (%s); skipping model artifact.", exc)


# ---------------------------------------------------------------------------
# 3. log_spark_model
# ---------------------------------------------------------------------------

def log_spark_model(
    spark_model: Any,
    metrics: Dict[str, float],
    feature_names: Optional[List[str]] = None,
    train_df_sample: Optional[Any] = None,
    experiment_path: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    """Log a Spark ML model with automatic feature importance extraction.

    Parameters
    ----------
    spark_model:
        A fitted PySpark ML model (``Pipeline``, ``PipelineModel``, or any
        single stage with a ``featureImportances`` attribute).
    metrics:
        Dictionary of metric name → numeric value to log.
    feature_names:
        Optional ordered list of feature names used during training.  Used to
        annotate feature importance values.
    train_df_sample:
        Optional PySpark or pandas DataFrame used to infer the MLflow model
        signature.  Only the first 5 rows are used for signature inference.
    experiment_path:
        Workspace path of the target experiment.
    run_name:
        Human-readable name for the MLflow run.

    Returns
    -------
    str
        The ``run_id`` of the created run.
    """
    import mlflow
    import mlflow.spark

    if experiment_path:
        mlflow.set_experiment(experiment_path)

    with mlflow.start_run(run_name=run_name or "spark_model_run") as run:
        mlflow.log_metrics(metrics)

        # --- feature importances ---
        importances = _extract_feature_importances(spark_model)
        if importances is not None:
            if feature_names and len(feature_names) == len(importances):
                importance_dict = dict(zip(feature_names, importances))
            else:
                importance_dict = {f"feature_{i}": v for i, v in enumerate(importances)}

            # Log each importance as a metric for easy querying
            for feat, imp in importance_dict.items():
                safe_key = feat.replace(" ", "_")[:250]
                mlflow.log_metric(f"importance_{safe_key}", float(imp))

            # Also persist a CSV artifact for convenient offline analysis
            imp_df = pd.DataFrame(
                list(importance_dict.items()), columns=["feature", "importance"]
            ).sort_values("importance", ascending=False)
            buf = io.StringIO()
            imp_df.to_csv(buf, index=False)
            mlflow.log_text(buf.getvalue(), "feature_importances.csv")

        # --- model signature ---
        signature = None
        if train_df_sample is not None:
            try:
                from mlflow.models.signature import infer_signature

                if hasattr(train_df_sample, "toPandas"):
                    sample_pd = train_df_sample.limit(5).toPandas()
                else:
                    sample_pd = train_df_sample.head(5)

                signature = infer_signature(sample_pd)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not infer model signature: %s", exc)

        mlflow.spark.log_model(
            spark_model,
            artifact_path="spark-model",
            signature=signature,
        )

        run_id = run.info.run_id

    logger.info("Spark model run logged: run_id=%s", run_id)
    return run_id


def _extract_feature_importances(model: Any) -> Optional[List[float]]:
    """Return feature importances from a Spark ML model or PipelineModel, if available."""
    # Direct attribute
    if hasattr(model, "featureImportances"):
        return list(model.featureImportances)

    # PipelineModel — look for the last stage with importances
    if hasattr(model, "stages"):
        for stage in reversed(model.stages):
            if hasattr(stage, "featureImportances"):
                return list(stage.featureImportances)

    return None


# ---------------------------------------------------------------------------
# 4. log_grid_search
# ---------------------------------------------------------------------------

def log_grid_search(
    cv_model: Any,
    param_grid: List[Dict[str, Any]],
    experiment_path: Optional[str] = None,
    parent_run_name: str = "grid_search",
) -> str:
    """Log all grid search results as nested MLflow runs.

    Creates one *parent* run named ``parent_run_name`` and one child run per
    parameter combination, each containing the parameter values and the
    average cross-validation metric reported by the CV model.

    Compatible with PySpark ``CrossValidatorModel`` (``avgMetrics``) and
    scikit-learn ``GridSearchCV`` (``cv_results_``).

    Parameters
    ----------
    cv_model:
        A fitted cross-validator model.  Must expose either ``avgMetrics``
        (PySpark) or ``cv_results_`` (scikit-learn).
    param_grid:
        The list of parameter dictionaries that was passed to the CV model.
        For scikit-learn ``GridSearchCV`` you can pass
        ``list(ParameterGrid(estimator.param_grid))``.
    experiment_path:
        Workspace path of the target experiment.
    parent_run_name:
        Name for the parent MLflow run that groups all child runs.

    Returns
    -------
    str
        The ``run_id`` of the parent run.
    """
    import mlflow

    if experiment_path:
        mlflow.set_experiment(experiment_path)

    # Determine metric values per parameter combination
    avg_metrics = _extract_cv_avg_metrics(cv_model)

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.set_tag("mlflow.runName", parent_run_name)

        if avg_metrics is None:
            logger.warning(
                "Could not extract average metrics from cv_model; "
                "child runs will have no metric values."
            )
            avg_metrics = [None] * len(param_grid)

        for i, (params, avg_metric) in enumerate(zip(param_grid, avg_metrics)):
            with mlflow.start_run(
                run_name=f"child_{i:03d}", nested=True
            ):
                # Flatten param values to strings for safe logging
                flat_params = {
                    str(k): str(v) for k, v in params.items()
                }
                mlflow.log_params(flat_params)

                if avg_metric is not None:
                    mlflow.log_metric("avg_cv_metric", float(avg_metric))

    logger.info(
        "Grid search logged: %d child runs under parent run_id=%s",
        len(param_grid),
        parent_run_id,
    )
    return parent_run_id


def _extract_cv_avg_metrics(cv_model: Any) -> Optional[List[float]]:
    """Return a list of average CV metrics, one per parameter combination."""
    # PySpark CrossValidatorModel
    if hasattr(cv_model, "avgMetrics"):
        return list(cv_model.avgMetrics)

    # scikit-learn GridSearchCV
    if hasattr(cv_model, "cv_results_"):
        return list(cv_model.cv_results_.get("mean_test_score", []))

    return None


# ---------------------------------------------------------------------------
# 5. get_best_run
# ---------------------------------------------------------------------------

def get_best_run(
    experiment_path: Optional[str] = None,
    experiment_id: Optional[str] = None,
    metric: str = "metrics.accuracy",
    ascending: bool = False,
    filter_string: Optional[str] = None,
) -> Dict[str, Any]:
    """Find the best run in an experiment by a chosen metric.

    Exactly one of ``experiment_path`` or ``experiment_id`` must be provided
    (``experiment_path`` takes precedence).

    Parameters
    ----------
    experiment_path:
        Workspace path of the experiment.
    experiment_id:
        MLflow experiment ID (used when ``experiment_path`` is not given).
    metric:
        Metric column to rank by, e.g. ``"metrics.accuracy"`` or
        ``"metrics.rmse"``.
    ascending:
        If *True*, the run with the **lowest** metric value is returned
        (useful for loss/RMSE).  Defaults to *False* (highest is best).
    filter_string:
        Optional MLflow search filter, e.g. ``"params.model_type = 'xgb'"``.

    Returns
    -------
    dict
        A dictionary with keys ``run_id``, ``experiment_id``, ``start_time``,
        ``metrics``, ``params``, ``tags``, and ``artifact_uri``.

    Raises
    ------
    ValueError
        If neither ``experiment_path`` nor ``experiment_id`` is supplied, or
        if no runs match the filter.
    """
    import mlflow

    exp_id = _resolve_experiment_id(experiment_path, experiment_id)

    order_by = f"{metric} ASC" if ascending else f"{metric} DESC"

    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=filter_string or "",
        order_by=[order_by],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(
            f"No runs found in experiment '{exp_id}' "
            f"(filter: {filter_string!r})."
        )

    best = runs.iloc[0]
    return _run_row_to_dict(best)


# ---------------------------------------------------------------------------
# 6. get_run_comparison
# ---------------------------------------------------------------------------

def get_run_comparison(
    experiment_path: Optional[str] = None,
    experiment_id: Optional[str] = None,
    metric_columns: Optional[List[str]] = None,
    param_columns: Optional[List[str]] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return a DataFrame comparing top runs side by side.

    Parameters
    ----------
    experiment_path:
        Workspace path of the experiment.
    experiment_id:
        MLflow experiment ID.
    metric_columns:
        List of metric names to include, e.g. ``["accuracy", "f1"]``.
        If *None*, all metric columns are included.
    param_columns:
        List of param names to include, e.g. ``["learning_rate", "max_depth"]``.
        If *None*, all param columns are included.
    top_n:
        Maximum number of runs to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``run_id`` as index and selected params + metrics as
        columns, sorted by start time (most recent first).
    """
    import mlflow

    exp_id = _resolve_experiment_id(experiment_path, experiment_id)

    runs_df = mlflow.search_runs(
        experiment_ids=[exp_id],
        max_results=top_n,
        order_by=["start_time DESC"],
    )

    if runs_df.empty:
        logger.warning("No runs found in experiment '%s'.", exp_id)
        return pd.DataFrame()

    selected_cols = ["run_id", "start_time", "status"]

    if metric_columns:
        selected_cols += [f"metrics.{m}" for m in metric_columns]
    else:
        selected_cols += [c for c in runs_df.columns if c.startswith("metrics.")]

    if param_columns:
        selected_cols += [f"params.{p}" for p in param_columns]
    else:
        selected_cols += [c for c in runs_df.columns if c.startswith("params.")]

    # Keep only columns that actually exist in the DataFrame
    available = [c for c in selected_cols if c in runs_df.columns]
    result = runs_df[available].copy()
    result = result.rename(columns={"run_id": "run_id"}).set_index("run_id")

    return result


# ---------------------------------------------------------------------------
# 7. promote_model
# ---------------------------------------------------------------------------

def promote_model(
    model_name: str,
    version: Optional[Union[int, str]] = None,
    alias: str = "champion",
) -> None:
    """Set an alias on a Unity Catalog model version.

    Parameters
    ----------
    model_name:
        Fully-qualified UC model name, e.g.
        ``"catalog.schema.my_model"``.
    version:
        Model version number.  If *None*, the latest registered version
        is used.
    alias:
        The alias to assign, e.g. ``"champion"`` or ``"challenger"``.
    """
    import mlflow

    client = mlflow.tracking.MlflowClient()

    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(
                f"No versions found for model '{model_name}'."
            )
        # search_model_versions returns versions sorted by creation time desc
        version = max(int(v.version) for v in versions)
        logger.info(
            "No version specified; using latest version %s for model '%s'.",
            version,
            model_name,
        )

    client.set_registered_model_alias(model_name, alias, str(version))
    logger.info(
        "Alias '%s' set on model '%s' version %s.", alias, model_name, version
    )


# ---------------------------------------------------------------------------
# 8. load_model
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    alias: str = "champion",
    version: Optional[Union[int, str]] = None,
) -> Any:
    """Load a model from the Unity Catalog registry by alias or version.

    ``version`` takes precedence over ``alias`` when both are supplied.

    Parameters
    ----------
    model_name:
        Fully-qualified UC model name, e.g.
        ``"catalog.schema.my_model"``.
    alias:
        The alias to resolve, e.g. ``"champion"``.  Ignored if ``version``
        is provided.
    version:
        Specific model version to load.

    Returns
    -------
    Any
        The loaded model object (returned by ``mlflow.pyfunc.load_model``).

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the model or alias/version cannot be found.
    """
    import mlflow

    if version is not None:
        uri = f"models:/{model_name}/{version}"
    else:
        uri = f"models:/{model_name}@{alias}"

    logger.info("Loading model from URI: %s", uri)
    return mlflow.pyfunc.load_model(uri)


# ---------------------------------------------------------------------------
# 9. cleanup_runs
# ---------------------------------------------------------------------------

def cleanup_runs(
    experiment_path: Optional[str] = None,
    experiment_id: Optional[str] = None,
    keep_top_n: int = 5,
    metric: str = "metrics.accuracy",
    delete: bool = False,
) -> List[str]:
    """Identify (and optionally delete) runs below the top-N threshold.

    Parameters
    ----------
    experiment_path:
        Workspace path of the experiment.
    experiment_id:
        MLflow experiment ID.
    keep_top_n:
        Number of top runs (by ``metric``) to retain.
    metric:
        Metric column used to rank runs, e.g. ``"metrics.accuracy"``.
    delete:
        If *True*, the identified runs are permanently deleted from MLflow.
        Defaults to *False* (dry-run: only the list is returned).

    Returns
    -------
    list[str]
        List of ``run_id`` values that are (or would be) deleted.

    Notes
    -----
    Runs that do not have the specified metric recorded are placed at the
    bottom of the ranking and are candidates for deletion first.
    """
    import mlflow

    exp_id = _resolve_experiment_id(experiment_path, experiment_id)

    runs_df = mlflow.search_runs(
        experiment_ids=[exp_id],
        order_by=[f"{metric} DESC"],
    )

    if runs_df.empty:
        logger.info("No runs found in experiment '%s'.", exp_id)
        return []

    # Runs with missing metric end up with NaN; sort them to the bottom
    if metric in runs_df.columns:
        runs_df = runs_df.sort_values(metric, ascending=False, na_position="last")
    else:
        logger.warning("Metric column '%s' not found; ordering by start_time.", metric)
        runs_df = runs_df.sort_values("start_time", ascending=False)

    to_delete_ids: List[str] = runs_df.iloc[keep_top_n:]["run_id"].tolist()

    if not to_delete_ids:
        logger.info(
            "Nothing to clean up; only %d run(s) found (keep_top_n=%d).",
            len(runs_df),
            keep_top_n,
        )
        return []

    if delete:
        client = mlflow.tracking.MlflowClient()
        for run_id in to_delete_ids:
            client.delete_run(run_id)
        logger.info(
            "Deleted %d run(s) from experiment '%s'.",
            len(to_delete_ids),
            exp_id,
        )
    else:
        logger.info(
            "Dry-run: %d run(s) would be deleted (set delete=True to confirm).",
            len(to_delete_ids),
        )

    return to_delete_ids


# ---------------------------------------------------------------------------
# 10. compare_model_versions
# ---------------------------------------------------------------------------

def compare_model_versions(
    model_name: str,
    version_a: Union[int, str],
    version_b: Union[int, str],
    test_df: pd.DataFrame,
    label_col: str = "label",
    task: str = "classification",
) -> pd.DataFrame:
    """Load two model versions, predict on the same test set, compare metrics.

    Parameters
    ----------
    model_name:
        Fully-qualified UC model name.
    version_a:
        First model version number to compare.
    version_b:
        Second model version number to compare.
    test_df:
        Pandas DataFrame containing features **and** the label column.
    label_col:
        Name of the ground-truth label column in ``test_df``.
    task:
        ``"classification"`` or ``"regression"``.  Determines which metrics
        are computed.

    Returns
    -------
    pd.DataFrame
        Two-row DataFrame indexed by version label (e.g. ``"v1"``, ``"v2"``)
        with metric columns.

    Raises
    ------
    ValueError
        If ``task`` is not ``"classification"`` or ``"regression"``.
    """
    import mlflow

    if task not in ("classification", "regression"):
        raise ValueError(
            f"task must be 'classification' or 'regression', got '{task}'."
        )

    if label_col not in test_df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in test_df."
        )

    feature_df = test_df.drop(columns=[label_col])
    y_true = test_df[label_col].values

    records = []
    for version in (version_a, version_b):
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
        y_pred = model.predict(feature_df)
        metrics = _compute_comparison_metrics(y_true, y_pred, task)
        metrics["version"] = str(version)
        records.append(metrics)

    result = pd.DataFrame(records).set_index("version")
    return result


def _compute_comparison_metrics(
    y_true: Any,
    y_pred: Any,
    task: str,
) -> Dict[str, float]:
    """Compute classification or regression metrics without hard sklearn dependency."""
    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task == "regression":
        residuals = y_true - y_pred
        mse = float(np.mean(residuals ** 2))
        mae = float(np.mean(np.abs(residuals)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
        return {"rmse": float(np.sqrt(mse)), "mae": mae, "r2": r2}

    # classification
    correct = int(np.sum(y_true == y_pred))
    total = len(y_true)
    accuracy = correct / total if total > 0 else float("nan")
    metrics: Dict[str, float] = {"accuracy": accuracy}

    # Attempt precision / recall / f1 via sklearn if available
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score

        metrics["precision"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    except ImportError:
        pass

    return metrics


# ---------------------------------------------------------------------------
# 11. log_data_snapshot
# ---------------------------------------------------------------------------

def log_data_snapshot(
    df: Any,
    name: str = "training_data",
    sample_size: int = 1000,
) -> None:
    """Log a sample of a DataFrame as an artifact for reproducibility.

    Accepts both PySpark and pandas DataFrames.  The sample is written as a
    CSV to the active MLflow run's artifact store under the path ``data/``.
    A basic schema / summary is also logged as a JSON artifact.

    Parameters
    ----------
    df:
        PySpark or pandas DataFrame to sample.
    name:
        Base name used for the artifact files (no extension).
    sample_size:
        Number of rows to include in the snapshot.

    Raises
    ------
    RuntimeError
        If there is no active MLflow run when this function is called.
    """
    import json

    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError(
            "log_data_snapshot must be called inside an active MLflow run "
            "(use 'with mlflow.start_run():' or call mlflow.start_run() first)."
        )

    # Convert to pandas
    if hasattr(df, "toPandas"):
        # PySpark — use sample to avoid full collect
        total_rows = df.count()
        fraction = min(1.0, sample_size / max(total_rows, 1))
        sample_pd = df.sample(fraction=fraction, seed=42).limit(sample_size).toPandas()
    elif isinstance(df, pd.DataFrame):
        sample_pd = df.head(sample_size).copy()
        total_rows = len(df)
    else:
        raise TypeError(
            f"df must be a pandas or PySpark DataFrame, got {type(df).__name__}."
        )

    actual_sample = len(sample_pd)

    # --- CSV snapshot ---
    csv_content = sample_pd.to_csv(index=False)
    mlflow.log_text(csv_content, f"data/{name}_sample.csv")

    # --- Schema / summary JSON ---
    schema_info: Dict[str, Any] = {
        "name": name,
        "total_rows_source": total_rows,
        "sample_rows_logged": actual_sample,
        "columns": list(sample_pd.columns),
        "dtypes": {col: str(dtype) for col, dtype in sample_pd.dtypes.items()},
        "null_counts": sample_pd.isnull().sum().to_dict(),
    }

    # Add basic numeric stats
    try:
        numeric_summary = sample_pd.describe().to_dict()
        # Convert numpy floats to native Python floats for JSON serialisation
        schema_info["numeric_summary"] = {
            col: {k: float(v) for k, v in stats.items()}
            for col, stats in numeric_summary.items()
        }
    except Exception:  # noqa: BLE001
        pass

    mlflow.log_text(
        json.dumps(schema_info, indent=2, default=str),
        f"data/{name}_schema.json",
    )

    logger.info(
        "Data snapshot logged: %d rows sampled from %d total (%s).",
        actual_sample,
        total_rows,
        name,
    )


# ---------------------------------------------------------------------------
# Private helpers shared across functions
# ---------------------------------------------------------------------------

def _resolve_experiment_id(
    experiment_path: Optional[str],
    experiment_id: Optional[str],
) -> str:
    """Return an experiment_id, resolving from path when needed."""
    import mlflow

    if experiment_path is not None:
        experiment = mlflow.get_experiment_by_name(experiment_path)
        if experiment is None:
            raise ValueError(
                f"Experiment not found at path '{experiment_path}'. "
                "Call setup_experiment() first."
            )
        return experiment.experiment_id

    if experiment_id is not None:
        return experiment_id

    raise ValueError(
        "Provide either experiment_path or experiment_id."
    )


def _run_row_to_dict(row: pd.Series) -> Dict[str, Any]:
    """Convert a row from mlflow.search_runs() DataFrame to a clean dict."""
    metrics = {
        k[len("metrics."):]: v
        for k, v in row.items()
        if k.startswith("metrics.")
    }
    params = {
        k[len("params."):]: v
        for k, v in row.items()
        if k.startswith("params.")
    }
    tags = {
        k[len("tags."):]: v
        for k, v in row.items()
        if k.startswith("tags.")
    }
    return {
        "run_id": row.get("run_id"),
        "experiment_id": row.get("experiment_id"),
        "start_time": row.get("start_time"),
        "status": row.get("status"),
        "artifact_uri": row.get("artifact_uri"),
        "metrics": metrics,
        "params": params,
        "tags": tags,
    }
