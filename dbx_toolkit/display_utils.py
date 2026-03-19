"""
display_utils.py — Part of dbx_toolkit

Reusable PySpark utility functions for displaying and formatting data in
Databricks notebooks. Provides composable, DataFrame-returning helpers for
quick previews, schema inspection, number formatting, pivot tables, histograms,
and grouped ranking.

All functions accept and return PySpark DataFrames for easy chaining and
downstream use in notebooks or pipelines.

Usage example:
    from dbx_toolkit.display_utils import peek, summary_table, top_n_by_group

    peek(df)
    summary_table(df).display()
    top_n_by_group(df, group_col="category", order_col="revenue").display()
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_spark() -> SparkSession:
    """Return the active SparkSession, raising a clear error if none exists."""
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(
            "No active SparkSession found. "
            "Create one with SparkSession.builder.getOrCreate()."
        )
    return spark


def _numeric_columns(df: DataFrame) -> List[str]:
    """Return names of all numeric columns in *df*."""
    numeric_types = (
        T.ByteType, T.ShortType, T.IntegerType, T.LongType,
        T.FloatType, T.DoubleType, T.DecimalType,
    )
    return [f.name for f in df.schema.fields if isinstance(f.dataType, numeric_types)]


# ---------------------------------------------------------------------------
# 1. show_df
# ---------------------------------------------------------------------------

def show_df(
    df: DataFrame,
    n: int = 20,
    truncate: bool = False,
    vertical: bool = False,
) -> DataFrame:
    """Display a DataFrame with sensible defaults and return it for chaining.

    Wraps ``DataFrame.show`` with truncation disabled by default so that long
    string values are not silently cut off in notebook output.

    Args:
        df:       The PySpark DataFrame to display.
        n:        Maximum number of rows to show. Defaults to 20.
        truncate: Whether to truncate long cell values. Defaults to False.
        vertical: Display each row vertically (one field per line).
                  Defaults to False.

    Returns:
        The original DataFrame unchanged, enabling method chaining.

    Example:
        show_df(df, n=50).filter("status = 'active'").display()
    """
    df.show(n=n, truncate=truncate, vertical=vertical)
    return df


# ---------------------------------------------------------------------------
# 2. peek
# ---------------------------------------------------------------------------

def peek(df: DataFrame, n: int = 5) -> DataFrame:
    """Print a quick preview: first *n* rows, schema, row count, column count.

    Designed for interactive exploration at the top of a notebook cell. All
    diagnostic output is printed to stdout; the DataFrame is returned unchanged
    for further chaining.

    Args:
        df: The PySpark DataFrame to inspect.
        n:  Number of sample rows to display. Defaults to 5.

    Returns:
        The original DataFrame unchanged, enabling method chaining.

    Example:
        peek(raw_df).filter("year = 2024")
    """
    row_count = df.count()
    col_count = len(df.columns)

    print(f"Shape: {row_count:,} rows x {col_count} columns")
    print("-" * 60)
    print("Schema:")
    df.printSchema()
    print(f"First {n} rows:")
    df.show(n=n, truncate=False)
    return df


# ---------------------------------------------------------------------------
# 3. summary_table
# ---------------------------------------------------------------------------

def summary_table(df: DataFrame) -> DataFrame:
    """Build a formatted summary DataFrame combining schema metadata and stats.

    For each column the summary includes:
    - ``column``        – column name
    - ``data_type``     – PySpark data type string
    - ``nullable``      – whether the column allows nulls
    - ``null_count``    – absolute number of null/NaN values
    - ``null_pct``      – null percentage (0–100)
    - ``distinct_count``– approximate distinct value count
    - ``sample_values`` – up to 3 non-null sample values as a string

    Args:
        df: The PySpark DataFrame to summarise.

    Returns:
        A new DataFrame with one row per source column and the columns
        described above. Suitable for ``.display()`` in a Databricks notebook.

    Example:
        summary_table(df).display()
    """
    spark = _get_spark()
    total_rows = df.count()

    rows = []
    for field in df.schema.fields:
        col_name = field.name
        dtype = str(field.dataType)
        nullable = field.nullable

        null_count = df.filter(F.col(col_name).isNull()).count()
        null_pct = round(null_count / total_rows * 100, 2) if total_rows > 0 else 0.0

        distinct_count = df.agg(F.approx_count_distinct(col_name)).collect()[0][0]

        sample_vals = [
            str(row[0])
            for row in (
                df.filter(F.col(col_name).isNotNull())
                .select(col_name)
                .limit(3)
                .collect()
            )
        ]
        sample_str = ", ".join(sample_vals)

        rows.append((col_name, dtype, nullable, null_count, null_pct, distinct_count, sample_str))

    schema = T.StructType([
        T.StructField("column",         T.StringType(),  True),
        T.StructField("data_type",      T.StringType(),  True),
        T.StructField("nullable",       T.BooleanType(), True),
        T.StructField("null_count",     T.LongType(),    True),
        T.StructField("null_pct",       T.DoubleType(),  True),
        T.StructField("distinct_count", T.LongType(),    True),
        T.StructField("sample_values",  T.StringType(),  True),
    ])

    return spark.createDataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# 4. compare_side_by_side
# ---------------------------------------------------------------------------

def compare_side_by_side(
    df1: DataFrame,
    df2: DataFrame,
    name1: str = "df1",
    name2: str = "df2",
) -> DataFrame:
    """Compare two DataFrames and return a structured summary of differences.

    Prints a shape comparison and schema differences to stdout, then returns a
    DataFrame detailing per-column presence and type agreement between the two
    inputs.

    The returned DataFrame has the following columns:
    - ``column``        – column name (union of both schemas)
    - ``in_<name1>``    – whether the column exists in df1
    - ``in_<name2>``    – whether the column exists in df2
    - ``type_<name1>``  – data type in df1 (empty string if absent)
    - ``type_<name2>``  – data type in df2 (empty string if absent)
    - ``types_match``   – True when both DataFrames share the column and type

    Args:
        df1:   First DataFrame.
        df2:   Second DataFrame.
        name1: Label for df1 in output. Defaults to "df1".
        name2: Label for df2 in output. Defaults to "df2".

    Returns:
        A summary DataFrame with one row per distinct column name.

    Example:
        compare_side_by_side(raw, cleaned, name1="raw", name2="cleaned").display()
    """
    spark = _get_spark()

    count1, count2 = df1.count(), df2.count()
    cols1, cols2 = len(df1.columns), len(df2.columns)

    print(f"{'':30s}  {name1:>15s}  {name2:>15s}")
    print(f"{'Row count':30s}  {count1:>15,}  {count2:>15,}")
    print(f"{'Column count':30s}  {cols1:>15}  {cols2:>15}")

    schema1 = {f.name: str(f.dataType) for f in df1.schema.fields}
    schema2 = {f.name: str(f.dataType) for f in df2.schema.fields}
    all_cols = sorted(set(schema1) | set(schema2))

    only_in_1 = sorted(set(schema1) - set(schema2))
    only_in_2 = sorted(set(schema2) - set(schema1))
    type_diff = sorted(
        c for c in set(schema1) & set(schema2) if schema1[c] != schema2[c]
    )

    if only_in_1:
        print(f"\nColumns only in {name1}: {only_in_1}")
    if only_in_2:
        print(f"Columns only in {name2}: {only_in_2}")
    if type_diff:
        print(f"\nType mismatches:")
        for c in type_diff:
            print(f"  {c}: {name1}={schema1[c]}  {name2}={schema2[c]}")
    if not only_in_1 and not only_in_2 and not type_diff:
        print("\nSchemas are identical.")

    rows = []
    for c in all_cols:
        in1 = c in schema1
        in2 = c in schema2
        t1 = schema1.get(c, "")
        t2 = schema2.get(c, "")
        match = in1 and in2 and t1 == t2
        rows.append((c, in1, in2, t1, t2, match))

    schema = T.StructType([
        T.StructField("column",                   T.StringType(),  True),
        T.StructField(f"in_{name1}",              T.BooleanType(), True),
        T.StructField(f"in_{name2}",              T.BooleanType(), True),
        T.StructField(f"type_{name1}",            T.StringType(),  True),
        T.StructField(f"type_{name2}",            T.StringType(),  True),
        T.StructField("types_match",              T.BooleanType(), True),
    ])

    return spark.createDataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# 5. format_number_columns
# ---------------------------------------------------------------------------

def format_number_columns(
    df: DataFrame,
    columns: Optional[Sequence[str]] = None,
    decimals: int = 2,
) -> DataFrame:
    """Round numeric columns to *decimals* decimal places for cleaner display.

    When *columns* is ``None`` every numeric column in the DataFrame is
    rounded. Non-numeric columns are left untouched.

    Args:
        df:       The PySpark DataFrame to transform.
        columns:  Explicit list of column names to round. Pass ``None`` to
                  round all numeric columns automatically.
        decimals: Number of decimal places to keep. Defaults to 2.

    Returns:
        A new DataFrame with the specified columns rounded.

    Example:
        format_number_columns(df, columns=["price", "discount"], decimals=4)
    """
    targets = list(columns) if columns is not None else _numeric_columns(df)
    for col_name in targets:
        df = df.withColumn(col_name, F.round(F.col(col_name), decimals))
    return df


# ---------------------------------------------------------------------------
# 6. format_pct_columns
# ---------------------------------------------------------------------------

def format_pct_columns(
    df: DataFrame,
    columns: Sequence[str],
    decimals: int = 1,
) -> DataFrame:
    """Format columns as percentages: multiply by 100 and append a '%' suffix.

    The original numeric column is replaced by a string column of the form
    ``"12.3%"``. Null values remain null after transformation.

    Args:
        df:       The PySpark DataFrame to transform.
        columns:  Column names whose values represent fractions (0–1) to be
                  converted to percentage strings.
        decimals: Number of decimal places in the formatted string.
                  Defaults to 1.

    Returns:
        A new DataFrame with the specified columns replaced by formatted
        percentage strings.

    Example:
        format_pct_columns(df, columns=["conversion_rate", "churn_rate"])
    """
    fmt = f"%.{decimals}f%%"
    for col_name in columns:
        df = df.withColumn(
            col_name,
            F.when(
                F.col(col_name).isNull(),
                F.lit(None).cast(T.StringType()),
            ).otherwise(
                F.format_string(fmt, F.round(F.col(col_name) * 100, decimals))
            ),
        )
    return df


# ---------------------------------------------------------------------------
# 7. add_row_numbers
# ---------------------------------------------------------------------------

def add_row_numbers(
    df: DataFrame,
    col_name: str = "row_num",
) -> DataFrame:
    """Add a sequential row-number column to *df*.

    Uses ``monotonically_increasing_id`` which guarantees uniqueness and
    monotonic ordering within each partition but is *not* contiguous across
    partitions. For strict 1-based sequential numbers on ordered data, supply
    an ``orderBy`` before calling this function and use the ``Window``-based
    approach described in the note below.

    Note:
        For a strictly contiguous 1-based index you can do::

            from pyspark.sql import Window
            from pyspark.sql import functions as F

            w = Window.orderBy("some_sort_col")
            df = df.withColumn("row_num", F.row_number().over(w))

        This function intentionally avoids a full sort so it remains
        performant on very large DataFrames.

    Args:
        df:       The PySpark DataFrame to augment.
        col_name: Name of the new row-number column. Defaults to "row_num".

    Returns:
        A new DataFrame with *col_name* prepended as the first column.

    Example:
        add_row_numbers(df, col_name="id").display()
    """
    df_with_id = df.withColumn(col_name, F.monotonically_increasing_id())
    # Move the new column to the front for readability
    return df_with_id.select([col_name] + [c for c in df.columns])


# ---------------------------------------------------------------------------
# 8. pivot_summary
# ---------------------------------------------------------------------------

def pivot_summary(
    df: DataFrame,
    group_col: str,
    value_col: str,
    agg_func: Literal["sum", "avg", "mean", "count", "min", "max"] = "sum",
    pivot_col: Optional[str] = None,
) -> DataFrame:
    """Create a quick pivot table with sensible defaults.

    When *pivot_col* is ``None`` the function returns a simple grouped
    aggregation of *value_col* by *group_col*. When *pivot_col* is provided
    the result is a true pivot with one column per distinct value of
    *pivot_col*.

    Args:
        df:        The PySpark DataFrame to aggregate.
        group_col: Column name to group by (rows of the pivot).
        value_col: Column name to aggregate.
        agg_func:  Aggregation function: one of ``"sum"``, ``"avg"``/
                   ``"mean"``, ``"count"``, ``"min"``, ``"max"``.
                   Defaults to ``"sum"``.
        pivot_col: Column whose distinct values become new columns.
                   Pass ``None`` for a plain grouped aggregation.

    Returns:
        A new aggregated/pivoted DataFrame ordered by *group_col*.

    Example:
        pivot_summary(df, group_col="region", value_col="revenue",
                      agg_func="sum", pivot_col="product_category").display()
    """
    _agg_map = {
        "sum":   F.sum,
        "avg":   F.avg,
        "mean":  F.avg,
        "count": F.count,
        "min":   F.min,
        "max":   F.max,
    }
    if agg_func not in _agg_map:
        raise ValueError(
            f"agg_func must be one of {list(_agg_map)}; got {agg_func!r}."
        )

    agg_fn = _agg_map[agg_func]

    if pivot_col is None:
        return (
            df.groupBy(group_col)
            .agg(agg_fn(value_col).alias(f"{agg_func}_{value_col}"))
            .orderBy(group_col)
        )

    return (
        df.groupBy(group_col)
        .pivot(pivot_col)
        .agg(agg_fn(value_col))
        .orderBy(group_col)
    )


# ---------------------------------------------------------------------------
# 9. histogram_data
# ---------------------------------------------------------------------------

def histogram_data(
    df: DataFrame,
    column: str,
    n_bins: int = 20,
) -> DataFrame:
    """Generate histogram bin data as a DataFrame for charting.

    Computes the min/max of *column*, divides the range into *n_bins* equal-
    width buckets using ``Bucketizer``, then counts the rows in each bucket.
    The result can be passed directly to a charting library (e.g., pandas +
    matplotlib) after ``.toPandas()``.

    Args:
        df:     The PySpark DataFrame containing the column to histogram.
        column: Name of the numeric column to bin.
        n_bins: Number of equal-width bins. Defaults to 20.

    Returns:
        A DataFrame with columns:
        - ``bin_index``  – 0-based bin index
        - ``bin_start``  – inclusive lower bound of the bin
        - ``bin_end``    – exclusive upper bound of the bin
        - ``count``      – number of rows in the bin

        The DataFrame is ordered by ``bin_index``.

    Example:
        histogram_data(df, column="amount", n_bins=30).display()
    """
    from pyspark.ml.feature import Bucketizer

    spark = _get_spark()

    stats = df.agg(
        F.min(F.col(column)).alias("min_val"),
        F.max(F.col(column)).alias("max_val"),
    ).collect()[0]

    min_val = float(stats["min_val"])
    max_val = float(stats["max_val"])

    if min_val == max_val:
        # All values identical — return a single bin
        single_schema = T.StructType([
            T.StructField("bin_index", T.IntegerType(), False),
            T.StructField("bin_start", T.DoubleType(),  False),
            T.StructField("bin_end",   T.DoubleType(),  False),
            T.StructField("count",     T.LongType(),    False),
        ])
        return spark.createDataFrame(
            [(0, min_val, min_val, df.count())], schema=single_schema
        )

    step = (max_val - min_val) / n_bins
    # Bucketizer requires splits to be strictly increasing; extend slightly
    # past max_val so the maximum value falls inside the last bin.
    splits = [min_val + i * step for i in range(n_bins + 1)]
    splits[-1] = splits[-1] + 1e-10  # avoid "value == upper boundary" errors

    bucket_col = f"__bucket_{column}__"
    bucketizer = Bucketizer(
        splits=splits,
        inputCol=column,
        outputCol=bucket_col,
        handleInvalid="skip",
    )

    bucketed = bucketizer.transform(df.filter(F.col(column).isNotNull()))
    counts = (
        bucketed.groupBy(bucket_col)
        .count()
        .withColumnRenamed(bucket_col, "bin_index_double")
    )

    # Build a reference DataFrame of all bins (so empty bins appear as 0)
    bin_rows = [
        (i, splits[i], splits[i + 1] - (1e-10 if i == n_bins - 1 else 0))
        for i in range(n_bins)
    ]
    bins_schema = T.StructType([
        T.StructField("bin_index",        T.IntegerType(), False),
        T.StructField("bin_start",        T.DoubleType(),  False),
        T.StructField("bin_end",          T.DoubleType(),  False),
    ])
    bins_df = spark.createDataFrame(bin_rows, schema=bins_schema)

    result = (
        bins_df
        .join(
            counts.withColumn(
                "bin_index", F.col("bin_index_double").cast(T.IntegerType())
            ).drop("bin_index_double"),
            on="bin_index",
            how="left",
        )
        .withColumn("count", F.coalesce(F.col("count"), F.lit(0)))
        .select("bin_index", "bin_start", "bin_end", "count")
        .orderBy("bin_index")
    )

    return result


# ---------------------------------------------------------------------------
# 10. top_n_by_group
# ---------------------------------------------------------------------------

def top_n_by_group(
    df: DataFrame,
    group_col: str,
    order_col: str,
    n: int = 5,
    ascending: bool = False,
) -> DataFrame:
    """Return the top *n* rows per group ordered by *order_col*.

    Uses a ``Window`` function with ``row_number`` so ties within a group are
    broken arbitrarily (stable across reruns within the same Spark job).

    Args:
        df:         The PySpark DataFrame to filter.
        group_col:  Column name that defines each group.
        order_col:  Column name used to rank rows within each group.
        n:          Number of rows to keep per group. Defaults to 5.
        ascending:  Sort direction. ``False`` (default) returns the *highest*
                    values first; ``True`` returns the lowest.

    Returns:
        A new DataFrame containing at most *n* rows per distinct value of
        *group_col*, with an additional ``rank`` column (1 = best).

    Example:
        top_n_by_group(df, group_col="category", order_col="revenue",
                       n=3).display()
    """
    sort_expr = (
        F.col(order_col).asc_nulls_last()
        if ascending
        else F.col(order_col).desc_nulls_last()
    )
    w = Window.partitionBy(group_col).orderBy(sort_expr)
    return (
        df.withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= n)
        .orderBy(group_col, "rank")
    )


# ---------------------------------------------------------------------------
# 11. crosstab_pct
# ---------------------------------------------------------------------------

def crosstab_pct(
    df: DataFrame,
    col1: str,
    col2: str,
) -> DataFrame:
    """Cross-tabulation with percentages instead of raw counts.

    Each cell value represents the percentage of the *grand total* that falls
    in that ``(col1, col2)`` combination. Column names in the result follow
    the PySpark ``crosstab`` convention: the first column is named
    ``<col1>_<col2>`` and subsequent columns are the distinct values of *col2*.

    Args:
        df:   The PySpark DataFrame to cross-tabulate.
        col1: Row dimension column name.
        col2: Column dimension column name.

    Returns:
        A new DataFrame where numeric cells contain percentages (0–100,
        rounded to 2 decimal places) instead of raw counts. The index column
        (first column) is unchanged.

    Example:
        crosstab_pct(df, col1="region", col2="product_category").display()
    """
    ct = df.stat.crosstab(col1, col2)

    # The first column is the row-label column (string); the rest are counts.
    label_col = ct.columns[0]
    value_cols = ct.columns[1:]

    # Compute grand total
    total_expr = sum(F.sum(F.col(c)) for c in value_cols)
    grand_total = ct.agg(total_expr.alias("total")).collect()[0]["total"]

    if grand_total == 0:
        return ct

    pct_cols = [
        F.round(F.col(c) / grand_total * 100, 2).alias(c)
        for c in value_cols
    ]

    return ct.select([F.col(label_col)] + pct_cols)
