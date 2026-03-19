"""
feature_engineering.py
-----------------------
Reusable PySpark utility functions for feature engineering in Databricks POCs.

All functions are designed to return fitted stages/models alongside transformed
DataFrames so the same transformations can be applied consistently to test data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Bucketizer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    QuantileDiscretizer,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)


# ---------------------------------------------------------------------------
# 1. encode_categoricals
# ---------------------------------------------------------------------------

def encode_categoricals(
    df: DataFrame,
    columns: List[str],
    method: str = "index",
    handle_invalid: str = "keep",
) -> Tuple[DataFrame, List[Any]]:
    """Encode categorical string columns using StringIndexer or
    StringIndexer + OneHotEncoder.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    columns : list[str]
        Names of the categorical columns to encode.
    method : str, optional
        ``"index"``  – apply StringIndexer only (default).
        ``"onehot"`` – apply StringIndexer then OneHotEncoder.
    handle_invalid : str, optional
        How to handle unseen labels during transform.
        Passed directly to StringIndexer. One of ``"error"``,
        ``"skip"``, or ``"keep"`` (default).

    Returns
    -------
    transformed_df : DataFrame
        DataFrame with encoded columns appended.
    fitted_stages : list
        Fitted StringIndexer models (and OHE models when
        ``method="onehot"``). Pass these to ``transform`` on new data.

    Examples
    --------
    >>> encoded_df, stages = encode_categoricals(df, ["color", "size"])
    >>> test_encoded_df, _ = encode_categoricals(test_df, [], method="index")
    >>> # Re-apply fitted stages to test data:
    >>> for stage in stages:
    ...     test_df = stage.transform(test_df)
    """
    if method not in ("index", "onehot"):
        raise ValueError(f"method must be 'index' or 'onehot', got '{method}'")

    fitted_stages: List[Any] = []
    result_df = df

    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_indexed",
            handleInvalid=handle_invalid,
        )
        for col in columns
    ]

    fitted_indexers = [indexer.fit(result_df) for indexer in indexers]

    for fitted_indexer in fitted_indexers:
        result_df = fitted_indexer.transform(result_df)
        fitted_stages.append(fitted_indexer)

    if method == "onehot":
        encoder = OneHotEncoder(
            inputCols=[f"{col}_indexed" for col in columns],
            outputCols=[f"{col}_onehot" for col in columns],
            handleInvalid=handle_invalid,
        )
        fitted_encoder = encoder.fit(result_df)
        result_df = fitted_encoder.transform(result_df)
        fitted_stages.append(fitted_encoder)

    return result_df, fitted_stages


# ---------------------------------------------------------------------------
# 2. scale_features
# ---------------------------------------------------------------------------

def scale_features(
    df: DataFrame,
    input_col: str,
    output_col: str = "scaled_features",
    method: str = "standard",
) -> Tuple[DataFrame, Any]:
    """Scale a vector column using a chosen scaler.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame. ``input_col`` must be a vector column.
    input_col : str
        Name of the vector column to scale.
    output_col : str, optional
        Name of the output scaled vector column (default ``"scaled_features"``).
    method : str, optional
        Scaling method. One of:
        ``"standard"`` – StandardScaler (zero mean, unit variance, default).
        ``"minmax"``   – MinMaxScaler (scales to [0, 1]).
        ``"maxabs"``   – MaxAbsScaler (scales each feature by its max absolute value).

    Returns
    -------
    transformed_df : DataFrame
        DataFrame with the scaled column appended.
    fitted_scaler : fitted scaler model
        The fitted scaler. Call ``fitted_scaler.transform(test_df)``
        to apply the same scaling to new data.

    Examples
    --------
    >>> scaled_df, scaler_model = scale_features(df, "features")
    >>> scaled_test_df = scaler_model.transform(test_df)
    """
    method_map = {
        "standard": StandardScaler(
            inputCol=input_col, outputCol=output_col, withMean=True, withStd=True
        ),
        "minmax": MinMaxScaler(inputCol=input_col, outputCol=output_col),
        "maxabs": MaxAbsScaler(inputCol=input_col, outputCol=output_col),
    }

    if method not in method_map:
        raise ValueError(
            f"method must be one of {list(method_map.keys())}, got '{method}'"
        )

    scaler = method_map[method]
    fitted_scaler = scaler.fit(df)
    transformed_df = fitted_scaler.transform(df)
    return transformed_df, fitted_scaler


# ---------------------------------------------------------------------------
# 3. assemble_features
# ---------------------------------------------------------------------------

def assemble_features(
    df: DataFrame,
    input_cols: List[str],
    output_col: str = "features",
    handle_invalid: str = "skip",
) -> DataFrame:
    """Assemble multiple numeric/vector columns into a single feature vector.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    input_cols : list[str]
        Columns to assemble into the feature vector.
    output_col : str, optional
        Name of the assembled vector column (default ``"features"``).
    handle_invalid : str, optional
        How to handle invalid (null) values. One of ``"error"``,
        ``"skip"`` (default), or ``"keep"``.

    Returns
    -------
    DataFrame
        DataFrame with the assembled feature vector column appended.

    Examples
    --------
    >>> features_df = assemble_features(df, ["age", "salary", "size_indexed"])
    """
    assembler = VectorAssembler(
        inputCols=input_cols,
        outputCol=output_col,
        handleInvalid=handle_invalid,
    )
    return assembler.transform(df)


# ---------------------------------------------------------------------------
# 4. create_bins
# ---------------------------------------------------------------------------

def create_bins(
    df: DataFrame,
    column: str,
    method: str = "quantile",
    n_bins: int = 4,
    custom_splits: Optional[List[float]] = None,
    output_col: Optional[str] = None,
) -> DataFrame:
    """Bin a numeric column into discrete buckets.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    column : str
        Name of the numeric column to bin.
    method : str, optional
        Binning strategy:
        ``"quantile"`` – equal-frequency bins via QuantileDiscretizer (default).
        ``"custom"``   – user-defined split points via Bucketizer.
    n_bins : int, optional
        Number of bins for quantile method (default ``4``).
    custom_splits : list[float], optional
        Explicit split points when ``method="custom"``. Must include
        ``-inf`` and ``inf`` as the boundary values, e.g.
        ``[float("-inf"), 18, 35, 60, float("inf")]``.
    output_col : str, optional
        Name of the output column. Defaults to ``{column}_binned``.

    Returns
    -------
    DataFrame
        DataFrame with the binned column appended.

    Examples
    --------
    >>> binned_df = create_bins(df, "age", method="quantile", n_bins=5)
    >>> custom_df = create_bins(
    ...     df, "age", method="custom",
    ...     custom_splits=[float("-inf"), 18, 35, 60, float("inf")]
    ... )
    """
    if output_col is None:
        output_col = f"{column}_binned"

    if method == "quantile":
        discretizer = QuantileDiscretizer(
            numBuckets=n_bins,
            inputCol=column,
            outputCol=output_col,
            handleInvalid="keep",
        )
        return discretizer.fit(df).transform(df)

    elif method == "custom":
        if custom_splits is None:
            raise ValueError("custom_splits must be provided when method='custom'")
        bucketizer = Bucketizer(
            splits=custom_splits,
            inputCol=column,
            outputCol=output_col,
            handleInvalid="keep",
        )
        return bucketizer.transform(df)

    else:
        raise ValueError(f"method must be 'quantile' or 'custom', got '{method}'")


# ---------------------------------------------------------------------------
# 5. add_date_features
# ---------------------------------------------------------------------------

def add_date_features(
    df: DataFrame,
    date_col: str,
    features: Optional[List[str]] = None,
) -> DataFrame:
    """Extract calendar features from a date or timestamp column.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    date_col : str
        Name of the date/timestamp column to extract features from.
    features : list[str], optional
        Subset of features to extract. Defaults to all:
        ``["year", "month", "day", "day_of_week", "quarter", "is_weekend"]``.
        Valid options also include ``"hour"``, ``"minute"``, ``"week_of_year"``.

    Returns
    -------
    DataFrame
        DataFrame with new date feature columns appended, each prefixed
        with ``{date_col}_``.

    Examples
    --------
    >>> dated_df = add_date_features(df, "event_date")
    >>> dated_df = add_date_features(df, "ts", features=["year", "month", "hour"])
    """
    _default_features = ["year", "month", "day", "day_of_week", "quarter", "is_weekend"]
    feature_list = features if features is not None else _default_features

    _extractor_map = {
        "year":         lambda c: F.year(c),
        "month":        lambda c: F.month(c),
        "day":          lambda c: F.dayofmonth(c),
        "day_of_week":  lambda c: F.dayofweek(c),
        "quarter":      lambda c: F.quarter(c),
        "week_of_year": lambda c: F.weekofyear(c),
        "hour":         lambda c: F.hour(c),
        "minute":       lambda c: F.minute(c),
        "is_weekend":   lambda c: (F.dayofweek(c).isin([1, 7])).cast("int"),
    }

    invalid = set(feature_list) - set(_extractor_map.keys())
    if invalid:
        raise ValueError(
            f"Unknown feature(s): {invalid}. "
            f"Valid options: {list(_extractor_map.keys())}"
        )

    result_df = df
    for feat in feature_list:
        out_col = f"{date_col}_{feat}"
        result_df = result_df.withColumn(out_col, _extractor_map[feat](date_col))

    return result_df


# ---------------------------------------------------------------------------
# 6. add_lag_features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: DataFrame,
    column: str,
    partition_by: Union[str, List[str]],
    order_by: Union[str, List[str]],
    lags: List[int] = None,
) -> DataFrame:
    """Add lag columns for a numeric column within a partition.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    column : str
        Name of the column to lag.
    partition_by : str or list[str]
        Column(s) that define the partition (e.g., entity ID).
    order_by : str or list[str]
        Column(s) used to define the time ordering within each partition.
    lags : list[int], optional
        Lag offsets to compute (default ``[1, 7, 30]``).

    Returns
    -------
    DataFrame
        DataFrame with lag columns appended, named
        ``{column}_lag_{n}`` for each lag ``n``.

    Examples
    --------
    >>> lag_df = add_lag_features(
    ...     df, "sales", partition_by="store_id", order_by="date"
    ... )
    """
    if lags is None:
        lags = [1, 7, 30]

    if isinstance(partition_by, str):
        partition_by = [partition_by]
    if isinstance(order_by, str):
        order_by = [order_by]

    window_spec = Window.partitionBy(*partition_by).orderBy(*order_by)

    result_df = df
    for lag in lags:
        out_col = f"{column}_lag_{lag}"
        result_df = result_df.withColumn(out_col, F.lag(column, lag).over(window_spec))

    return result_df


# ---------------------------------------------------------------------------
# 7. add_rolling_features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: DataFrame,
    column: str,
    partition_by: Union[str, List[str]],
    order_by: Union[str, List[str]],
    windows: List[int] = None,
    funcs: List[str] = None,
) -> DataFrame:
    """Add rolling window aggregate columns for a numeric column.

    The window is defined as ``rowsBetween(-window_size + 1, 0)``
    (i.e., the current row and the preceding ``window_size - 1`` rows).

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    column : str
        Name of the column to aggregate.
    partition_by : str or list[str]
        Column(s) that define the partition.
    order_by : str or list[str]
        Column(s) used to define the ordering within each partition.
    windows : list[int], optional
        Rolling window sizes in rows (default ``[7, 30]``).
    funcs : list[str], optional
        Aggregation functions to apply (default ``["avg", "min", "max"]``).
        Supported: ``"avg"``, ``"min"``, ``"max"``, ``"sum"``, ``"stddev"``.

    Returns
    -------
    DataFrame
        DataFrame with rolling columns appended, named
        ``{column}_rolling_{func}_{n}`` for each function and window size.

    Examples
    --------
    >>> rolled_df = add_rolling_features(
    ...     df, "sales",
    ...     partition_by="store_id",
    ...     order_by="date",
    ...     windows=[7, 14],
    ...     funcs=["avg", "sum"],
    ... )
    """
    if windows is None:
        windows = [7, 30]
    if funcs is None:
        funcs = ["avg", "min", "max"]

    _func_map = {
        "avg":    F.avg,
        "min":    F.min,
        "max":    F.max,
        "sum":    F.sum,
        "stddev": F.stddev,
    }

    invalid_funcs = set(funcs) - set(_func_map.keys())
    if invalid_funcs:
        raise ValueError(
            f"Unknown function(s): {invalid_funcs}. "
            f"Supported: {list(_func_map.keys())}"
        )

    if isinstance(partition_by, str):
        partition_by = [partition_by]
    if isinstance(order_by, str):
        order_by = [order_by]

    result_df = df
    for window_size in windows:
        window_spec = (
            Window.partitionBy(*partition_by)
            .orderBy(*order_by)
            .rowsBetween(-window_size + 1, 0)
        )
        for func_name in funcs:
            out_col = f"{column}_rolling_{func_name}_{window_size}"
            result_df = result_df.withColumn(
                out_col, _func_map[func_name](column).over(window_spec)
            )

    return result_df


# ---------------------------------------------------------------------------
# 8. add_interaction_features
# ---------------------------------------------------------------------------

def add_interaction_features(
    df: DataFrame,
    column_pairs: List[Tuple[str, str]],
) -> DataFrame:
    """Create multiplicative interaction features for pairs of numeric columns.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    column_pairs : list of (str, str)
        Pairs of column names whose product forms the interaction feature.
        Example: ``[("age", "salary"), ("height", "weight")]``.

    Returns
    -------
    DataFrame
        DataFrame with interaction columns appended, named
        ``{col_a}_x_{col_b}`` for each pair.

    Examples
    --------
    >>> inter_df = add_interaction_features(
    ...     df, [("age", "salary"), ("height", "weight")]
    ... )
    """
    result_df = df
    for col_a, col_b in column_pairs:
        out_col = f"{col_a}_x_{col_b}"
        result_df = result_df.withColumn(
            out_col, F.col(col_a) * F.col(col_b)
        )
    return result_df


# ---------------------------------------------------------------------------
# 9. add_ratio_features
# ---------------------------------------------------------------------------

def add_ratio_features(
    df: DataFrame,
    ratios: List[Tuple[str, str, str]],
) -> DataFrame:
    """Create ratio features by dividing one column by another.

    Division by zero produces ``null`` (via ``nullif`` semantics using
    a conditional expression).

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    ratios : list of (str, str, str)
        Each tuple is ``(numerator_col, denominator_col, output_col_name)``.
        Example: ``[("revenue", "employees", "revenue_per_employee")]``.

    Returns
    -------
    DataFrame
        DataFrame with ratio columns appended.

    Examples
    --------
    >>> ratio_df = add_ratio_features(
    ...     df, [("revenue", "employees", "revenue_per_employee")]
    ... )
    """
    result_df = df
    for numerator, denominator, out_col in ratios:
        result_df = result_df.withColumn(
            out_col,
            F.when(F.col(denominator) != 0, F.col(numerator) / F.col(denominator)).otherwise(
                F.lit(None).cast("double")
            ),
        )
    return result_df


# ---------------------------------------------------------------------------
# 10. add_missing_indicators
# ---------------------------------------------------------------------------

def add_missing_indicators(
    df: DataFrame,
    columns: Optional[List[str]] = None,
) -> DataFrame:
    """Add binary indicator columns flagging which values were null.

    This should be called **before** imputation so the indicator columns
    capture the original missingness pattern.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    columns : list[str], optional
        Columns to create indicators for. Defaults to all columns
        that contain at least one null value.

    Returns
    -------
    DataFrame
        DataFrame with ``{col}_missing`` indicator columns (1 = was null,
        0 = was not null) appended for each specified column.

    Examples
    --------
    >>> indicator_df = add_missing_indicators(df, ["age", "salary"])
    >>> indicator_df = add_missing_indicators(df)  # auto-detects nulls
    """
    if columns is None:
        # Auto-detect columns with any nulls
        null_counts = df.select(
            [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
        ).collect()[0].asDict()
        columns = [c for c, cnt in null_counts.items() if cnt > 0]

    result_df = df
    for col in columns:
        out_col = f"{col}_missing"
        result_df = result_df.withColumn(
            out_col, F.when(F.col(col).isNull(), F.lit(1)).otherwise(F.lit(0))
        )
    return result_df


# ---------------------------------------------------------------------------
# 11. impute_columns
# ---------------------------------------------------------------------------

def impute_columns(
    df: DataFrame,
    strategy: str = "median",
    columns: Optional[List[str]] = None,
    group_by: Optional[Union[str, List[str]]] = None,
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Impute null values with mean, median, or mode, optionally within groups.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    strategy : str, optional
        Imputation strategy: ``"mean"``, ``"median"``, or ``"mode"``
        (default ``"median"``).
    columns : list[str], optional
        Columns to impute. Defaults to all numeric columns (for
        mean/median) or all columns (for mode).
    group_by : str or list[str], optional
        If provided, imputation values are computed per group and rows
        are filled using group-specific values.

    Returns
    -------
    imputed_df : DataFrame
        DataFrame with nulls filled.
    impute_values : dict
        Mapping of ``column -> impute_value`` (or
        ``column -> {group -> value}`` when ``group_by`` is set).
        Pass to :func:`pyspark.sql.DataFrame.fillna` or reuse on test data.

    Examples
    --------
    >>> imputed_df, fill_vals = impute_columns(df, strategy="median")
    >>> # Reuse on test set (global strategy, no group_by):
    >>> test_imputed = test_df.fillna(fill_vals)
    """
    if strategy not in ("mean", "median", "mode"):
        raise ValueError(
            f"strategy must be 'mean', 'median', or 'mode', got '{strategy}'"
        )

    # Default to numeric columns for mean/median; all columns for mode
    if columns is None:
        if strategy in ("mean", "median"):
            numeric_types = ("int", "bigint", "float", "double", "decimal", "long", "short")
            columns = [
                f.name for f in df.schema.fields
                if any(t in f.dataType.simpleString() for t in numeric_types)
            ]
        else:
            columns = df.columns

    if not columns:
        return df, {}

    if group_by is not None:
        return _impute_with_groups(df, columns, strategy, group_by)

    # --- global imputation ---
    impute_values: Dict[str, Any] = {}

    if strategy == "mean":
        agg_exprs = [F.mean(F.col(c)).alias(c) for c in columns]
        row = df.agg(*agg_exprs).collect()[0]
        impute_values = {c: row[c] for c in columns if row[c] is not None}

    elif strategy == "median":
        for col in columns:
            median_val = df.approxQuantile(col, [0.5], 0.01)
            if median_val:
                impute_values[col] = median_val[0]

    elif strategy == "mode":
        for col in columns:
            mode_row = (
                df.groupBy(col)
                .count()
                .orderBy(F.desc("count"))
                .filter(F.col(col).isNotNull())
                .limit(1)
                .collect()
            )
            if mode_row:
                impute_values[col] = mode_row[0][col]

    imputed_df = df.fillna(impute_values)
    return imputed_df, impute_values


def _impute_with_groups(
    df: DataFrame,
    columns: List[str],
    strategy: str,
    group_by: Union[str, List[str]],
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Internal helper: group-aware imputation using window functions."""
    if isinstance(group_by, str):
        group_by = [group_by]

    window_spec = Window.partitionBy(*group_by)
    result_df = df
    impute_meta: Dict[str, Any] = {}

    for col in columns:
        temp_col = f"__impute_{col}__"

        if strategy == "mean":
            result_df = result_df.withColumn(
                temp_col, F.mean(F.col(col)).over(window_spec)
            )
        elif strategy == "median":
            result_df = result_df.withColumn(
                temp_col,
                F.percentile_approx(F.col(col), 0.5).over(window_spec),
            )
        elif strategy == "mode":
            # mode within a window is complex; fall back to dense_rank approach
            # using a subquery per group
            result_df = result_df.withColumn(
                temp_col,
                F.first(F.col(col), ignorenulls=True).over(
                    window_spec.orderBy(F.col(col))
                ),
            )

        result_df = result_df.withColumn(
            col,
            F.when(F.col(col).isNull(), F.col(temp_col)).otherwise(F.col(col)),
        ).drop(temp_col)

        impute_meta[col] = f"group-wise {strategy} by {group_by}"

    return result_df, impute_meta


# ---------------------------------------------------------------------------
# 12. prepare_features
# ---------------------------------------------------------------------------

def prepare_features(
    df: DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    label_col: Optional[str] = None,
    impute_strategy: str = "median",
    scale_method: str = "standard",
) -> Tuple[DataFrame, Dict[str, Any]]:
    """End-to-end feature preparation: impute → encode → assemble → scale.

    The function applies the following steps in order:

    1. Add missing-value indicator columns for all numeric and categorical
       columns that contain nulls.
    2. Impute numeric columns using the chosen strategy.
    3. Encode categorical columns with StringIndexer.
    4. Assemble all numeric and encoded columns into a single feature vector.
    5. Scale the feature vector using the chosen scaler.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame.
    numeric_cols : list[str]
        Names of numeric feature columns.
    categorical_cols : list[str]
        Names of categorical (string) feature columns.
    label_col : str, optional
        If provided, this column is kept as-is and excluded from features.
    impute_strategy : str, optional
        Imputation strategy passed to :func:`impute_columns`
        (default ``"median"``).
    scale_method : str, optional
        Scaling method passed to :func:`scale_features`
        (default ``"standard"``).

    Returns
    -------
    transformed_df : DataFrame
        Fully prepared DataFrame with ``"features"`` (unscaled) and
        ``"scaled_features"`` vector columns.
    pipeline_artifacts : dict
        Dictionary containing all fitted objects needed to reproduce the
        transformation on new data:

        * ``"impute_values"`` – dict returned by :func:`impute_columns`.
        * ``"indexer_stages"`` – fitted StringIndexer models.
        * ``"fitted_scaler"`` – fitted scaler model.
        * ``"feature_cols"`` – list of column names fed to VectorAssembler.

    Examples
    --------
    >>> prepped_df, artifacts = prepare_features(
    ...     df,
    ...     numeric_cols=["age", "salary"],
    ...     categorical_cols=["department", "city"],
    ...     label_col="churn",
    ... )
    >>> # Apply same pipeline to test data:
    >>> test_df = test_df.fillna(artifacts["impute_values"])
    >>> for stage in artifacts["indexer_stages"]:
    ...     test_df = stage.transform(test_df)
    >>> test_df = assemble_features(test_df, artifacts["feature_cols"])
    >>> test_df = artifacts["fitted_scaler"].transform(test_df)
    """
    pipeline_artifacts: Dict[str, Any] = {}

    # Step 1: missing indicators (before imputation)
    all_feature_cols = numeric_cols + categorical_cols
    result_df = add_missing_indicators(df, columns=all_feature_cols)

    indicator_cols = [f"{c}_missing" for c in all_feature_cols]

    # Step 2: impute numeric columns
    result_df, impute_values = impute_columns(
        result_df, strategy=impute_strategy, columns=numeric_cols
    )
    pipeline_artifacts["impute_values"] = impute_values

    # Step 3: encode categorical columns
    if categorical_cols:
        result_df, fitted_stages = encode_categoricals(
            result_df, columns=categorical_cols, method="index", handle_invalid="keep"
        )
        pipeline_artifacts["indexer_stages"] = fitted_stages
    else:
        pipeline_artifacts["indexer_stages"] = []

    encoded_cols = [f"{c}_indexed" for c in categorical_cols]

    # Step 4: assemble feature vector
    feature_cols = numeric_cols + encoded_cols + indicator_cols
    pipeline_artifacts["feature_cols"] = feature_cols

    result_df = assemble_features(
        result_df, input_cols=feature_cols, output_col="features", handle_invalid="skip"
    )

    # Step 5: scale
    result_df, fitted_scaler = scale_features(
        result_df,
        input_col="features",
        output_col="scaled_features",
        method=scale_method,
    )
    pipeline_artifacts["fitted_scaler"] = fitted_scaler

    return result_df, pipeline_artifacts
