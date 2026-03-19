"""
data_profiling.py — Part of dbx_toolkit

Reusable PySpark utility functions for data profiling in Databricks POCs stored
on a Volume. Provides composable, DataFrame-returning helpers for null analysis,
cardinality checks, numeric summaries, outlier detection, schema diffing, and
overall data quality scoring.

All functions accept and return PySpark DataFrames for easy chaining and
downstream use in notebooks or pipelines.

Usage example:
    from dbx_toolkit.data_profiling import profile_table, null_report

    profile_df = profile_table(df, sample_size=100_000)
    profile_df.display()
"""

from __future__ import annotations

from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_spark() -> SparkSession:
    """Return the active SparkSession, raising if none exists."""
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(
            "No active SparkSession found. "
            "Create or retrieve one before calling dbx_toolkit functions."
        )
    return spark


def _numeric_columns(df: DataFrame) -> List[str]:
    """Return a list of column names whose types are numeric."""
    numeric_types = (
        T.ByteType,
        T.ShortType,
        T.IntegerType,
        T.LongType,
        T.FloatType,
        T.DoubleType,
        T.DecimalType,
    )
    return [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, numeric_types)
    ]


def _orderable_column(field: T.StructField) -> bool:
    """Return True if the column type supports min/max ordering."""
    non_orderable = (T.MapType, T.ArrayType, T.StructType)
    return not isinstance(field.dataType, non_orderable)


def _safe_sample(df: DataFrame, sample_size: Optional[int]) -> DataFrame:
    """Return a sampled DataFrame if sample_size is provided, otherwise the original."""
    if sample_size is None:
        return df
    total = df.count()
    if total == 0 or sample_size >= total:
        return df
    fraction = min(sample_size / total, 1.0)
    return df.sample(withReplacement=False, fraction=fraction, seed=42)


# ---------------------------------------------------------------------------
# 1. profile_table
# ---------------------------------------------------------------------------

def profile_table(
    df: DataFrame,
    sample_size: Optional[int] = None,
) -> DataFrame:
    """
    Return a profiling summary DataFrame with one row per column.

    Columns in the result:
        column_name, data_type, non_null_count, null_count, null_pct,
        distinct_count, min, max, mean, stddev, sample_values

    Parameters
    ----------
    df : DataFrame
        Input DataFrame to profile.
    sample_size : int, optional
        If set, profile a random sample of this many rows instead of the full
        DataFrame. Useful for large tables where an approximate profile is
        acceptable.

    Returns
    -------
    DataFrame
        One row per column with the profiling metrics described above.
    """
    spark = _get_spark()
    working_df = _safe_sample(df, sample_size)

    total_rows = working_df.count()
    if total_rows == 0:
        # Return an empty schema-correct DataFrame
        schema = T.StructType([
            T.StructField("column_name",    T.StringType(),  True),
            T.StructField("data_type",      T.StringType(),  True),
            T.StructField("non_null_count", T.LongType(),    True),
            T.StructField("null_count",     T.LongType(),    True),
            T.StructField("null_pct",       T.DoubleType(),  True),
            T.StructField("distinct_count", T.LongType(),    True),
            T.StructField("min",            T.StringType(),  True),
            T.StructField("max",            T.StringType(),  True),
            T.StructField("mean",           T.DoubleType(),  True),
            T.StructField("stddev",         T.DoubleType(),  True),
            T.StructField("sample_values",  T.StringType(),  True),
        ])
        return spark.createDataFrame([], schema)

    numeric_cols = set(_numeric_columns(working_df))
    rows = []

    for field in working_df.schema.fields:
        col_name = field.name
        dtype_str = field.dataType.simpleString()
        col_expr = F.col(f"`{col_name}`")

        is_orderable = _orderable_column(field)

        agg_exprs = [
            F.count(col_expr).alias("non_null_count"),
            F.count(F.when(col_expr.isNull(), 1)).alias("null_count"),
            F.countDistinct(col_expr).alias("distinct_count"),
        ]

        # min/max only work on orderable types (not MAP, ARRAY, STRUCT)
        if is_orderable:
            agg_exprs += [
                F.min(col_expr).cast(T.StringType()).alias("col_min"),
                F.max(col_expr).cast(T.StringType()).alias("col_max"),
            ]

        if col_name in numeric_cols:
            agg_exprs += [
                F.mean(col_expr).alias("col_mean"),
                F.stddev(col_expr).alias("col_stddev"),
            ]

        agg_result = working_df.agg(*agg_exprs).collect()[0]

        non_null  = agg_result["non_null_count"]
        null_cnt  = agg_result["null_count"]
        null_pct  = round(null_cnt / total_rows * 100, 4) if total_rows > 0 else 0.0
        distinct  = agg_result["distinct_count"]
        col_min   = agg_result["col_min"] if is_orderable else None
        col_max   = agg_result["col_max"] if is_orderable else None
        col_mean  = float(agg_result["col_mean"])   if col_name in numeric_cols and agg_result["col_mean"]   is not None else None
        col_std   = float(agg_result["col_stddev"]) if col_name in numeric_cols and agg_result["col_stddev"] is not None else None

        # Sample up to 5 distinct non-null values
        sample_vals = (
            working_df
            .select(col_expr.cast(T.StringType()).alias("v"))
            .filter(F.col("v").isNotNull())
            .dropDuplicates(["v"])
            .limit(5)
            .agg(F.concat_ws(", ", F.collect_list("v")).alias("samples"))
            .collect()[0]["samples"]
        )

        rows.append((
            col_name,
            dtype_str,
            non_null,
            null_cnt,
            null_pct,
            distinct,
            col_min,
            col_max,
            col_mean,
            col_std,
            sample_vals,
        ))

    schema = T.StructType([
        T.StructField("column_name",    T.StringType(),  True),
        T.StructField("data_type",      T.StringType(),  True),
        T.StructField("non_null_count", T.LongType(),    True),
        T.StructField("null_count",     T.LongType(),    True),
        T.StructField("null_pct",       T.DoubleType(),  True),
        T.StructField("distinct_count", T.LongType(),    True),
        T.StructField("min",            T.StringType(),  True),
        T.StructField("max",            T.StringType(),  True),
        T.StructField("mean",           T.DoubleType(),  True),
        T.StructField("stddev",         T.DoubleType(),  True),
        T.StructField("sample_values",  T.StringType(),  True),
    ])
    return spark.createDataFrame(rows, schema)


# ---------------------------------------------------------------------------
# 2. null_report
# ---------------------------------------------------------------------------

def null_report(df: DataFrame) -> DataFrame:
    """
    Return a DataFrame with the null count and null percentage per column,
    sorted by null percentage descending.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    DataFrame
        Columns: column_name, null_count, null_pct
        Sorted by null_pct descending.
    """
    spark = _get_spark()
    total_rows = df.count()

    if total_rows == 0:
        schema = T.StructType([
            T.StructField("column_name", T.StringType(), True),
            T.StructField("null_count",  T.LongType(),   True),
            T.StructField("null_pct",    T.DoubleType(), True),
        ])
        return spark.createDataFrame([], schema)

    agg_exprs = [
        F.count(F.when(F.col(f"`{c}`").isNull(), 1)).alias(c)
        for c in df.columns
    ]
    null_counts = df.agg(*agg_exprs).collect()[0].asDict()

    rows = [
        (col, int(null_counts[col]), round(null_counts[col] / total_rows * 100, 4))
        for col in df.columns
    ]

    schema = T.StructType([
        T.StructField("column_name", T.StringType(), True),
        T.StructField("null_count",  T.LongType(),   True),
        T.StructField("null_pct",    T.DoubleType(), True),
    ])
    return (
        spark.createDataFrame(rows, schema)
        .orderBy(F.col("null_pct").desc())
    )


# ---------------------------------------------------------------------------
# 3. cardinality_report
# ---------------------------------------------------------------------------

def cardinality_report(df: DataFrame) -> DataFrame:
    """
    Return the distinct count and cardinality ratio (distinct / total rows)
    for every column.

    A cardinality_ratio near 1.0 suggests the column is a good candidate key;
    near 0.0 suggests a low-cardinality categorical.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    DataFrame
        Columns: column_name, distinct_count, cardinality_ratio
        Sorted by cardinality_ratio descending.
    """
    spark = _get_spark()
    total_rows = df.count()

    if total_rows == 0:
        schema = T.StructType([
            T.StructField("column_name",       T.StringType(), True),
            T.StructField("distinct_count",    T.LongType(),   True),
            T.StructField("cardinality_ratio", T.DoubleType(), True),
        ])
        return spark.createDataFrame([], schema)

    agg_exprs = [
        F.countDistinct(F.col(f"`{c}`")).alias(c)
        for c in df.columns
    ]
    distinct_counts = df.agg(*agg_exprs).collect()[0].asDict()

    rows = [
        (col, int(distinct_counts[col]), round(distinct_counts[col] / total_rows, 6))
        for col in df.columns
    ]

    schema = T.StructType([
        T.StructField("column_name",       T.StringType(), True),
        T.StructField("distinct_count",    T.LongType(),   True),
        T.StructField("cardinality_ratio", T.DoubleType(), True),
    ])
    return (
        spark.createDataFrame(rows, schema)
        .orderBy(F.col("cardinality_ratio").desc())
    )


# ---------------------------------------------------------------------------
# 4. value_counts
# ---------------------------------------------------------------------------

def value_counts(
    df: DataFrame,
    column: str,
    top_n: int = 20,
) -> DataFrame:
    """
    Return value counts with row percentage for the given column, limited to
    the top_n most frequent values.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column : str
        Column to count values for.
    top_n : int, default 20
        Maximum number of values to return.

    Returns
    -------
    DataFrame
        Columns: value, count, pct
        Sorted by count descending.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available: {df.columns}")

    total_rows = df.count()
    if total_rows == 0:
        schema = T.StructType([
            T.StructField("value", T.StringType(), True),
            T.StructField("count", T.LongType(),   True),
            T.StructField("pct",   T.DoubleType(), True),
        ])
        spark = _get_spark()
        return spark.createDataFrame([], schema)

    return (
        df.select(F.col(f"`{column}`").cast(T.StringType()).alias("value"))
        .groupBy("value")
        .agg(F.count("*").alias("count"))
        .withColumn("pct", F.round(F.col("count") / total_rows * 100, 4))
        .orderBy(F.col("count").desc())
        .limit(top_n)
    )


# ---------------------------------------------------------------------------
# 5. numeric_summary
# ---------------------------------------------------------------------------

def numeric_summary(
    df: DataFrame,
    columns: Optional[List[str]] = None,
) -> DataFrame:
    """
    Return extended descriptive statistics for numeric columns including
    percentiles, skewness, and kurtosis.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    columns : list of str, optional
        Specific numeric columns to summarise. Defaults to all numeric columns.

    Returns
    -------
    DataFrame
        Columns: column_name, mean, stddev, min, p25, p50, p75, max,
                 skewness, kurtosis
    """
    spark = _get_spark()
    target_cols = columns if columns is not None else _numeric_columns(df)
    target_cols = [c for c in target_cols if c in df.columns]

    if not target_cols:
        schema = T.StructType([
            T.StructField("column_name", T.StringType(), True),
            T.StructField("mean",        T.DoubleType(), True),
            T.StructField("stddev",      T.DoubleType(), True),
            T.StructField("min",         T.DoubleType(), True),
            T.StructField("p25",         T.DoubleType(), True),
            T.StructField("p50",         T.DoubleType(), True),
            T.StructField("p75",         T.DoubleType(), True),
            T.StructField("max",         T.DoubleType(), True),
            T.StructField("skewness",    T.DoubleType(), True),
            T.StructField("kurtosis",    T.DoubleType(), True),
        ])
        return spark.createDataFrame([], schema)

    if df.count() == 0:
        schema = T.StructType([
            T.StructField("column_name", T.StringType(), True),
            T.StructField("mean",        T.DoubleType(), True),
            T.StructField("stddev",      T.DoubleType(), True),
            T.StructField("min",         T.DoubleType(), True),
            T.StructField("p25",         T.DoubleType(), True),
            T.StructField("p50",         T.DoubleType(), True),
            T.StructField("p75",         T.DoubleType(), True),
            T.StructField("max",         T.DoubleType(), True),
            T.StructField("skewness",    T.DoubleType(), True),
            T.StructField("kurtosis",    T.DoubleType(), True),
        ])
        return spark.createDataFrame([], schema)

    # Percentiles via approxQuantile (single pass over all target columns)
    quantiles = df.approxQuantile(target_cols, [0.25, 0.50, 0.75], 0.01)
    percentile_map = {col: q for col, q in zip(target_cols, quantiles)}

    agg_exprs = []
    for c in target_cols:
        col_expr = F.col(f"`{c}`").cast(T.DoubleType())
        agg_exprs += [
            F.mean(col_expr).alias(f"{c}__mean"),
            F.stddev(col_expr).alias(f"{c}__stddev"),
            F.min(col_expr).alias(f"{c}__min"),
            F.max(col_expr).alias(f"{c}__max"),
            F.skewness(col_expr).alias(f"{c}__skewness"),
            F.kurtosis(col_expr).alias(f"{c}__kurtosis"),
        ]

    agg_result = df.agg(*agg_exprs).collect()[0].asDict()

    def _safe(val):
        return float(val) if val is not None else None

    rows = []
    for c in target_cols:
        p = percentile_map.get(c, [None, None, None])
        rows.append((
            c,
            _safe(agg_result.get(f"{c}__mean")),
            _safe(agg_result.get(f"{c}__stddev")),
            _safe(agg_result.get(f"{c}__min")),
            _safe(p[0]) if p else None,
            _safe(p[1]) if p else None,
            _safe(p[2]) if p else None,
            _safe(agg_result.get(f"{c}__max")),
            _safe(agg_result.get(f"{c}__skewness")),
            _safe(agg_result.get(f"{c}__kurtosis")),
        ))

    schema = T.StructType([
        T.StructField("column_name", T.StringType(), True),
        T.StructField("mean",        T.DoubleType(), True),
        T.StructField("stddev",      T.DoubleType(), True),
        T.StructField("min",         T.DoubleType(), True),
        T.StructField("p25",         T.DoubleType(), True),
        T.StructField("p50",         T.DoubleType(), True),
        T.StructField("p75",         T.DoubleType(), True),
        T.StructField("max",         T.DoubleType(), True),
        T.StructField("skewness",    T.DoubleType(), True),
        T.StructField("kurtosis",    T.DoubleType(), True),
    ])
    return spark.createDataFrame(rows, schema)


# ---------------------------------------------------------------------------
# 6. correlation_matrix
# ---------------------------------------------------------------------------

def correlation_matrix(
    df: DataFrame,
    columns: Optional[List[str]] = None,
) -> DataFrame:
    """
    Return pairwise Pearson correlations for numeric columns as a tidy DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    columns : list of str, optional
        Numeric columns to include. Defaults to all numeric columns.

    Returns
    -------
    DataFrame
        Columns: column_a, column_b, pearson_correlation
        All pairs (a, b) with a != b are included exactly once (lower triangle).
    """
    spark = _get_spark()
    target_cols = columns if columns is not None else _numeric_columns(df)
    target_cols = [c for c in target_cols if c in df.columns]

    schema = T.StructType([
        T.StructField("column_a",            T.StringType(), True),
        T.StructField("column_b",            T.StringType(), True),
        T.StructField("pearson_correlation", T.DoubleType(), True),
    ])

    if len(target_cols) < 2 or df.count() == 0:
        return spark.createDataFrame([], schema)

    # Cast target columns to double to ensure corr() works correctly
    cast_df = df.select([F.col(f"`{c}`").cast(T.DoubleType()).alias(c) for c in target_cols])
    # Drop rows where all target cols are null
    cast_df = cast_df.dropna(how="all", subset=target_cols)

    if cast_df.count() < 2:
        return spark.createDataFrame([], schema)

    import math

    rows = []
    for i, col_a in enumerate(target_cols):
        for col_b in target_cols[i + 1:]:
            corr_val = cast_df.stat.corr(col_a, col_b, method="pearson")
            if corr_val is not None and not math.isnan(corr_val):
                rows.append((col_a, col_b, float(corr_val)))
            else:
                rows.append((col_a, col_b, None))

    return spark.createDataFrame(rows, schema).orderBy(
        F.abs(F.col("pearson_correlation")).desc()
    )


# ---------------------------------------------------------------------------
# 7. detect_outliers_iqr
# ---------------------------------------------------------------------------

def detect_outliers_iqr(
    df: DataFrame,
    column: str,
    factor: float = 1.5,
) -> DataFrame:
    """
    Flag outliers in a numeric column using the IQR (Tukey fences) method.

    A value is an outlier if it falls below (Q1 - factor * IQR) or above
    (Q3 + factor * IQR).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column : str
        Numeric column to evaluate.
    factor : float, default 1.5
        IQR multiplier. Use 3.0 for "extreme outlier" detection.

    Returns
    -------
    DataFrame
        Original DataFrame with three additional columns:
            iqr_lower_bound  — lower fence value
            iqr_upper_bound  — upper fence value
            is_outlier       — boolean flag (True when value is outside bounds)
        Rows where the column value is null are flagged as is_outlier = False.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available: {df.columns}")

    if df.count() == 0:
        return df.withColumn("iqr_lower_bound", F.lit(None).cast(T.DoubleType())) \
                 .withColumn("iqr_upper_bound", F.lit(None).cast(T.DoubleType())) \
                 .withColumn("is_outlier",      F.lit(False))

    quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    col_expr = F.col(f"`{column}`").cast(T.DoubleType())

    return (
        df
        .withColumn("iqr_lower_bound", F.lit(lower).cast(T.DoubleType()))
        .withColumn("iqr_upper_bound", F.lit(upper).cast(T.DoubleType()))
        .withColumn(
            "is_outlier",
            F.when(col_expr.isNull(), F.lit(False))
             .otherwise((col_expr < lower) | (col_expr > upper))
        )
    )


# ---------------------------------------------------------------------------
# 8. compare_dataframes
# ---------------------------------------------------------------------------

def compare_dataframes(
    df1: DataFrame,
    df2: DataFrame,
    key_columns: List[str],
) -> DataFrame:
    """
    Compare two DataFrames and return a summary of differences.

    The result contains all rows that differ between df1 and df2, annotated
    with a ``diff_type`` column:
        - ``only_in_df1``  — key exists in df1 but not df2
        - ``only_in_df2``  — key exists in df2 but not df1
        - ``changed``      — key exists in both but at least one value differs

    Both DataFrames must share the same schema (column names and types).
    The comparison is performed on the union of all non-key columns.

    Parameters
    ----------
    df1 : DataFrame
        First (e.g., "before") DataFrame.
    df2 : DataFrame
        Second (e.g., "after") DataFrame.
    key_columns : list of str
        Columns that together form the row identifier.

    Returns
    -------
    DataFrame
        Columns: all columns from df1/df2, plus ``diff_type`` and ``source``
        (``df1`` or ``df2`` for the ``changed`` rows, showing both versions).
    """
    if not key_columns:
        raise ValueError("key_columns must contain at least one column name.")

    missing_in_df1 = [c for c in key_columns if c not in df1.columns]
    missing_in_df2 = [c for c in key_columns if c not in df2.columns]
    if missing_in_df1:
        raise ValueError(f"Key columns missing from df1: {missing_in_df1}")
    if missing_in_df2:
        raise ValueError(f"Key columns missing from df2: {missing_in_df2}")

    value_cols = [c for c in df1.columns if c not in key_columns]

    # Rows only in df1
    only_in_1 = (
        df1.join(df2.select(key_columns), on=key_columns, how="left_anti")
        .withColumn("diff_type", F.lit("only_in_df1"))
        .withColumn("source", F.lit("df1"))
    )

    # Rows only in df2
    only_in_2 = (
        df2.join(df1.select(key_columns), on=key_columns, how="left_anti")
        .withColumn("diff_type", F.lit("only_in_df2"))
        .withColumn("source", F.lit("df2"))
    )

    # Changed rows — keys present in both but at least one value differs
    if value_cols:
        df1_keyed = df1.alias("_d1")
        df2_keyed = df2.alias("_d2")

        join_cond = [F.col(f"_d1.{k}") == F.col(f"_d2.{k}") for k in key_columns]

        # Build a filter: any value column differs (treating null == null as equal)
        any_diff = F.lit(False)
        for vc in value_cols:
            c1 = F.col(f"_d1.{vc}")
            c2 = F.col(f"_d2.{vc}")
            # eqNullSafe returns True when both are null — negate to detect actual diffs
            any_diff = any_diff | (~c1.eqNullSafe(c2))

        joined = df1_keyed.join(df2_keyed, join_cond, how="inner").filter(any_diff)

        changed_df1 = (
            joined.select([F.col(f"_d1.{c}").alias(c) for c in df1.columns])
            .withColumn("diff_type", F.lit("changed"))
            .withColumn("source", F.lit("df1"))
        )
        changed_df2 = (
            joined.select([F.col(f"_d2.{c}").alias(c) for c in df2.columns])
            .withColumn("diff_type", F.lit("changed"))
            .withColumn("source", F.lit("df2"))
        )
        changed = changed_df1.union(changed_df2)
    else:
        # No value columns; nothing can "change"
        changed = only_in_1.filter(F.lit(False))  # empty with correct schema

    return only_in_1.union(only_in_2).union(changed)


# ---------------------------------------------------------------------------
# 9. schema_diff
# ---------------------------------------------------------------------------

def schema_diff(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """
    Compare the schemas of two DataFrames and surface columns that were added,
    removed, or had their data type changed.

    Parameters
    ----------
    df1 : DataFrame
        "Before" or reference DataFrame.
    df2 : DataFrame
        "After" or target DataFrame.

    Returns
    -------
    DataFrame
        Columns: column_name, status, type_in_df1, type_in_df2
        ``status`` is one of: ``added``, ``removed``, ``type_changed``, ``unchanged``
    """
    spark = _get_spark()

    schema1 = {f.name: f.dataType.simpleString() for f in df1.schema.fields}
    schema2 = {f.name: f.dataType.simpleString() for f in df2.schema.fields}

    all_cols = sorted(set(schema1) | set(schema2))
    rows = []
    for col in all_cols:
        t1 = schema1.get(col)
        t2 = schema2.get(col)
        if t1 is None:
            status = "added"
        elif t2 is None:
            status = "removed"
        elif t1 != t2:
            status = "type_changed"
        else:
            status = "unchanged"
        rows.append((col, status, t1, t2))

    schema = T.StructType([
        T.StructField("column_name",  T.StringType(), True),
        T.StructField("status",       T.StringType(), True),
        T.StructField("type_in_df1",  T.StringType(), True),
        T.StructField("type_in_df2",  T.StringType(), True),
    ])
    return (
        spark.createDataFrame(rows, schema)
        .orderBy(
            F.when(F.col("status") == "removed",      F.lit(0))
             .when(F.col("status") == "added",         F.lit(1))
             .when(F.col("status") == "type_changed",  F.lit(2))
             .otherwise(F.lit(3)),
            "column_name",
        )
    )


# ---------------------------------------------------------------------------
# 10. data_quality_score
# ---------------------------------------------------------------------------

def data_quality_score(df: DataFrame) -> DataFrame:
    """
    Return a 0–100 overall data quality score for the DataFrame, along with
    the three component scores used to compute it.

    Scoring methodology
    -------------------
    Three equally-weighted dimensions (each 0–100):

    completeness  (33.3%)
        Mean non-null rate across all columns.
        100 = no nulls anywhere; 0 = every value is null.

    uniqueness    (33.3%)
        Proportion of columns whose cardinality_ratio > 0.01 (not near-constant).
        Penalises DataFrames with many constant or near-constant columns.
        Note: this is a heuristic — adjust the threshold for your domain.

    validity      (33.3%)
        Proportion of numeric columns that have a non-zero standard deviation
        (i.e. they are not entirely the same value). Non-numeric columns are
        excluded from this component; if there are no numeric columns the
        component defaults to 100.

    overall_score = (completeness + uniqueness + validity) / 3

    Parameters
    ----------
    df : DataFrame
        Input DataFrame to score.

    Returns
    -------
    DataFrame
        Single-row DataFrame with columns:
            total_rows, total_columns, completeness_score,
            uniqueness_score, validity_score, overall_score
    """
    spark = _get_spark()

    schema = T.StructType([
        T.StructField("total_rows",          T.LongType(),   True),
        T.StructField("total_columns",       T.IntegerType(),True),
        T.StructField("completeness_score",  T.DoubleType(), True),
        T.StructField("uniqueness_score",    T.DoubleType(), True),
        T.StructField("validity_score",      T.DoubleType(), True),
        T.StructField("overall_score",       T.DoubleType(), True),
    ])

    total_rows = df.count()
    total_cols = len(df.columns)

    if total_rows == 0 or total_cols == 0:
        empty_row = [(0, 0, 0.0, 0.0, 0.0, 0.0)]
        return spark.createDataFrame(empty_row, schema)

    # --- Completeness ---
    null_agg_exprs = [
        F.count(F.when(F.col(f"`{c}`").isNull(), 1)).alias(c)
        for c in df.columns
    ]
    null_counts = df.agg(*null_agg_exprs).collect()[0].asDict()
    non_null_rates = [
        (total_rows - null_counts[c]) / total_rows for c in df.columns
    ]
    completeness_score = round(sum(non_null_rates) / total_cols * 100, 4)

    # --- Uniqueness ---
    distinct_agg_exprs = [
        F.countDistinct(F.col(f"`{c}`")).alias(c)
        for c in df.columns
    ]
    distinct_counts = df.agg(*distinct_agg_exprs).collect()[0].asDict()
    cardinality_ratios = [distinct_counts[c] / total_rows for c in df.columns]
    high_cardinality = sum(1 for r in cardinality_ratios if r > 0.01)
    uniqueness_score = round(high_cardinality / total_cols * 100, 4)

    # --- Validity (numeric stddev check) ---
    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        validity_score = 100.0
    else:
        std_agg_exprs = [
            F.stddev(F.col(f"`{c}`")).alias(c)
            for c in numeric_cols
        ]
        std_results = df.agg(*std_agg_exprs).collect()[0].asDict()
        nonzero_std = sum(
            1 for c in numeric_cols
            if std_results[c] is not None and std_results[c] > 0
        )
        validity_score = round(nonzero_std / len(numeric_cols) * 100, 4)

    overall_score = round((completeness_score + uniqueness_score + validity_score) / 3, 4)

    return spark.createDataFrame(
        [(total_rows, total_cols, completeness_score, uniqueness_score, validity_score, overall_score)],
        schema,
    )
