"""
datetime_utils.py
-----------------
Reusable PySpark utility functions for datetime operations in Databricks POCs.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, TimestampType


# ---------------------------------------------------------------------------
# 1. parse_dates
# ---------------------------------------------------------------------------

def parse_dates(
    df: DataFrame,
    column: str,
    formats: Optional[List[str]] = None,
    output_col: Optional[str] = None,
) -> DataFrame:
    """Try parsing a string column as a date using multiple formats via coalesce.

    Each format is attempted in order; the first successful parse is used.
    Rows that do not match any format will contain ``null`` in the output column.

    Args:
        df: Input DataFrame.
        column: Name of the string column to parse.
        formats: List of date format strings (Java ``SimpleDateFormat`` patterns).
            Defaults to ``["yyyy-MM-dd", "MM/dd/yyyy", "dd-MMM-yyyy",
            "yyyyMMdd", "MM-dd-yyyy"]``.
        output_col: Name of the output column. Defaults to ``column`` (overwrites
            the source column).

    Returns:
        DataFrame with the parsed date column added or replaced.

    Example::

        df = parse_dates(df, "event_date", output_col="event_date_parsed")
    """
    if formats is None:
        formats = [
            "yyyy-MM-dd",
            "MM/dd/yyyy",
            "dd-MMM-yyyy",
            "yyyyMMdd",
            "MM-dd-yyyy",
        ]

    if output_col is None:
        output_col = column

    parse_exprs = [F.expr(f"try_to_date(`{column}`, '{fmt}')") for fmt in formats]
    parsed = F.coalesce(*parse_exprs)
    return df.withColumn(output_col, parsed)


# ---------------------------------------------------------------------------
# 2. add_date_parts
# ---------------------------------------------------------------------------

def add_date_parts(
    df: DataFrame,
    date_col: str,
    parts: Optional[List[str]] = None,
) -> DataFrame:
    """Add individual date-part columns derived from a date/timestamp column.

    Each new column is prefixed with ``<date_col>_``.

    Args:
        df: Input DataFrame.
        date_col: Name of the date or timestamp column.
        parts: List of parts to extract. Supported values: ``"year"``,
            ``"month"``, ``"day"``, ``"quarter"``, ``"day_of_week"``,
            ``"day_of_year"``, ``"week_of_year"``. Defaults to all seven.

    Returns:
        DataFrame with the requested date-part columns appended.

    Example::

        df = add_date_parts(df, "order_date", parts=["year", "month", "day"])
    """
    if parts is None:
        parts = [
            "year",
            "month",
            "day",
            "quarter",
            "day_of_week",
            "day_of_year",
            "week_of_year",
        ]

    col_ref = F.col(date_col)
    part_map = {
        "year": F.year(col_ref),
        "month": F.month(col_ref),
        "day": F.dayofmonth(col_ref),
        "quarter": F.quarter(col_ref),
        "day_of_week": F.dayofweek(col_ref),
        "day_of_year": F.dayofyear(col_ref),
        "week_of_year": F.weekofyear(col_ref),
    }

    for part in parts:
        if part not in part_map:
            raise ValueError(
                f"Unsupported date part '{part}'. "
                f"Supported parts: {list(part_map.keys())}"
            )
        df = df.withColumn(f"{date_col}_{part}", part_map[part])

    return df


# ---------------------------------------------------------------------------
# 3. add_time_parts
# ---------------------------------------------------------------------------

def add_time_parts(
    df: DataFrame,
    timestamp_col: str,
    parts: Optional[List[str]] = None,
) -> DataFrame:
    """Add individual time-part columns derived from a timestamp column.

    Each new column is prefixed with ``<timestamp_col>_``.

    Args:
        df: Input DataFrame.
        timestamp_col: Name of the timestamp column.
        parts: List of parts to extract. Supported values: ``"hour"``,
            ``"minute"``, ``"second"``. Defaults to all three.

    Returns:
        DataFrame with the requested time-part columns appended.

    Example::

        df = add_time_parts(df, "event_ts")
    """
    if parts is None:
        parts = ["hour", "minute", "second"]

    col_ref = F.col(timestamp_col)
    part_map = {
        "hour": F.hour(col_ref),
        "minute": F.minute(col_ref),
        "second": F.second(col_ref),
    }

    for part in parts:
        if part not in part_map:
            raise ValueError(
                f"Unsupported time part '{part}'. "
                f"Supported parts: {list(part_map.keys())}"
            )
        df = df.withColumn(f"{timestamp_col}_{part}", part_map[part])

    return df


# ---------------------------------------------------------------------------
# 4. date_diff_columns
# ---------------------------------------------------------------------------

def date_diff_columns(
    df: DataFrame,
    start_col: str,
    end_col: str,
    units: str = "days",
) -> DataFrame:
    """Add a column with the difference between two date/timestamp columns.

    The result column is named ``<start_col>_to_<end_col>_diff_<units>``.
    A positive value means ``end_col`` is later than ``start_col``.

    Args:
        df: Input DataFrame.
        start_col: Name of the start date/timestamp column.
        end_col: Name of the end date/timestamp column.
        units: Unit of the difference. One of ``"days"``, ``"hours"``,
            ``"minutes"``, ``"seconds"``, ``"months"``, ``"years"``.
            Defaults to ``"days"``.

    Returns:
        DataFrame with the difference column appended.

    Example::

        df = date_diff_columns(df, "start_date", "end_date", units="months")
    """
    output_col = f"{start_col}_to_{end_col}_diff_{units}"
    start = F.col(start_col)
    end = F.col(end_col)

    units_lower = units.lower()
    if units_lower == "days":
        diff_expr = F.datediff(end, start)
    elif units_lower == "months":
        diff_expr = F.months_between(end, start).cast("int")
    elif units_lower == "years":
        diff_expr = (F.months_between(end, start) / 12).cast("int")
    elif units_lower == "hours":
        diff_expr = (
            (F.unix_timestamp(end) - F.unix_timestamp(start)) / 3600
        ).cast("long")
    elif units_lower == "minutes":
        diff_expr = (
            (F.unix_timestamp(end) - F.unix_timestamp(start)) / 60
        ).cast("long")
    elif units_lower == "seconds":
        diff_expr = (
            F.unix_timestamp(end) - F.unix_timestamp(start)
        ).cast("long")
    else:
        raise ValueError(
            f"Unsupported units '{units}'. "
            "Supported: 'days', 'hours', 'minutes', 'seconds', 'months', 'years'."
        )

    return df.withColumn(output_col, diff_expr)


# ---------------------------------------------------------------------------
# 5. add_business_days
# ---------------------------------------------------------------------------

def add_business_days(
    df: DataFrame,
    date_col: str,
    n_days: int,
    output_col: Optional[str] = None,
) -> DataFrame:
    """Add *n* business days (Monday–Friday) to a date column.

    This is an approximation that skips weekends but does **not** account for
    public holidays.  The algorithm:

    1. Determine the ISO day-of-week of the start date (1 = Monday … 7 = Sunday).
    2. Use integer arithmetic to compute how many calendar days must be added in
       order to advance by exactly ``n_days`` business days.

    Args:
        df: Input DataFrame.
        date_col: Name of the date column.
        n_days: Number of business days to add. May be negative.
        output_col: Name of the output column. Defaults to
            ``<date_col>_plus_<n_days>bd``.

    Returns:
        DataFrame with the result date column appended.

    Example::

        df = add_business_days(df, "invoice_date", 5)
    """
    if output_col is None:
        output_col = f"{date_col}_plus_{n_days}bd"

    # ``dayofweek`` in Spark: 1 = Sunday … 7 = Saturday.
    # Convert to ISO: Monday = 1 … Sunday = 7.
    dow = ((F.dayofweek(F.col(date_col)) + 5) % 7) + 1  # 1=Mon … 7=Sun

    # Number of full weeks + remaining days
    full_weeks = abs(n_days) // 5
    remaining = abs(n_days) % 5

    if n_days >= 0:
        # Days until end of current business week (Friday)
        days_until_friday = 5 - dow  # 0 on Friday, can be negative (weekend)
        # calendar days to add:
        # full_weeks * 7 + remaining + weekend bumps
        calendar_days = (
            full_weeks * 7
            + remaining
            + F.when(
                (dow + remaining) > 5,
                F.lit(2),  # crosses a weekend
            ).otherwise(F.lit(0))
        )
        result = F.date_add(F.col(date_col), calendar_days.cast("int"))
    else:
        # Subtract business days
        calendar_days = (
            full_weeks * 7
            + remaining
            + F.when(
                (dow - remaining) < 1,
                F.lit(2),  # crosses a weekend going backwards
            ).otherwise(F.lit(0))
        )
        result = F.date_sub(F.col(date_col), calendar_days.cast("int"))

    return df.withColumn(output_col, result)


# ---------------------------------------------------------------------------
# 6. is_business_day
# ---------------------------------------------------------------------------

def is_business_day(
    df: DataFrame,
    date_col: str,
    output_col: Optional[str] = None,
) -> DataFrame:
    """Add a boolean column indicating whether the date falls on a weekday.

    Monday–Friday are considered business days; Saturday and Sunday are not.
    Public holidays are **not** taken into account.

    Args:
        df: Input DataFrame.
        date_col: Name of the date column.
        output_col: Name of the output boolean column. Defaults to
            ``<date_col>_is_business_day``.

    Returns:
        DataFrame with the boolean column appended.

    Example::

        df = is_business_day(df, "transaction_date")
    """
    if output_col is None:
        output_col = f"{date_col}_is_business_day"

    # Spark dayofweek: 1 = Sunday, 2 = Monday, …, 7 = Saturday
    dow = F.dayofweek(F.col(date_col))
    is_bday = (dow >= 2) & (dow <= 6)  # Mon-Fri
    return df.withColumn(output_col, is_bday)


# ---------------------------------------------------------------------------
# 7. generate_date_spine
# ---------------------------------------------------------------------------

def generate_date_spine(
    spark: SparkSession,
    start_date: str,
    end_date: str,
    interval: str = "day",
) -> DataFrame:
    """Generate a DataFrame with one row per interval between *start_date* and
    *end_date* (inclusive).

    Args:
        spark: Active ``SparkSession``.
        start_date: Start of the range, as an ISO-8601 string (``"YYYY-MM-DD"``
            or ``"YYYY-MM-DD HH:MM:SS"``).
        end_date: End of the range (inclusive).
        interval: Granularity of the spine. One of ``"day"``, ``"week"``,
            ``"month"``, ``"hour"``. Defaults to ``"day"``.

    Returns:
        DataFrame with a single column named after the chosen interval:

        * ``"day"``   → column ``date`` (``DateType``)
        * ``"week"``  → column ``week_start`` (``DateType``)
        * ``"month"`` → column ``month_start`` (``DateType``)
        * ``"hour"``  → column ``ts`` (``TimestampType``)

    Example::

        spine = generate_date_spine(spark, "2024-01-01", "2024-12-31")
    """
    interval_lower = interval.lower()

    if interval_lower == "day":
        # Use sequence of dates
        df = spark.sql(
            f"""
            SELECT explode(
                sequence(
                    to_date('{start_date}'),
                    to_date('{end_date}'),
                    interval 1 day
                )
            ) AS date
            """
        )
    elif interval_lower == "week":
        df = spark.sql(
            f"""
            SELECT explode(
                sequence(
                    to_date('{start_date}'),
                    to_date('{end_date}'),
                    interval 1 week
                )
            ) AS week_start
            """
        )
    elif interval_lower == "month":
        df = spark.sql(
            f"""
            SELECT explode(
                sequence(
                    to_date('{start_date}'),
                    to_date('{end_date}'),
                    interval 1 month
                )
            ) AS month_start
            """
        )
    elif interval_lower == "hour":
        df = spark.sql(
            f"""
            SELECT explode(
                sequence(
                    to_timestamp('{start_date}'),
                    to_timestamp('{end_date}'),
                    interval 1 hour
                )
            ) AS ts
            """
        )
    else:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            "Supported: 'day', 'week', 'month', 'hour'."
        )

    return df


# ---------------------------------------------------------------------------
# 8. fill_date_gaps
# ---------------------------------------------------------------------------

def fill_date_gaps(
    df: DataFrame,
    date_col: str,
    partition_cols: Optional[List[str]] = None,
    interval: str = "day",
    fill_value: Union[int, float, str] = 0,
    value_cols: Optional[List[str]] = None,
) -> DataFrame:
    """Join a time-series DataFrame with a date spine to fill temporal gaps.

    Missing rows introduced by the join have their ``value_cols`` filled with
    ``fill_value``.

    Args:
        df: Input DataFrame containing at least ``date_col`` and optionally
            ``partition_cols`` and ``value_cols``.
        date_col: Name of the date column used for the spine join.
        partition_cols: Optional list of columns that define independent
            time-series groups (e.g., ``["store_id", "product_id"]``).
        interval: Spine interval — same options as :func:`generate_date_spine`.
            Defaults to ``"day"``.
        fill_value: Value used to fill ``null``s in ``value_cols`` after the
            join. Defaults to ``0``.
        value_cols: Columns whose nulls should be filled. Defaults to all
            non-key columns (all columns except ``date_col`` and
            ``partition_cols``).

    Returns:
        DataFrame with no gaps in the date dimension; missing value columns are
        filled with ``fill_value``.

    Example::

        df_filled = fill_date_gaps(
            df, "sale_date",
            partition_cols=["store_id"],
            value_cols=["revenue", "units"],
        )
    """
    spark = df.sparkSession

    # Determine the date range from the data itself
    date_range = df.agg(
        F.min(F.col(date_col)).alias("min_date"),
        F.max(F.col(date_col)).alias("max_date"),
    ).collect()[0]

    min_date = str(date_range["min_date"])
    max_date = str(date_range["max_date"])

    spine = generate_date_spine(spark, min_date, max_date, interval=interval)

    # The spine column name depends on the interval
    interval_col_map = {
        "day": "date",
        "week": "week_start",
        "month": "month_start",
        "hour": "ts",
    }
    spine_col = interval_col_map.get(interval.lower(), "date")

    if partition_cols:
        # Cross-join the spine with distinct partition values, then left-join data
        partitions = df.select(*partition_cols).distinct()
        spine_full = spine.crossJoin(partitions)
        join_keys = [spine_col] + list(partition_cols)
        # Rename date_col in original df to match spine_col for the join
        df_renamed = df.withColumnRenamed(date_col, spine_col)
        result = spine_full.join(df_renamed, on=join_keys, how="left")
    else:
        df_renamed = df.withColumnRenamed(date_col, spine_col)
        result = spine.join(df_renamed, on=spine_col, how="left")

    # Determine which columns to fill
    if value_cols is None:
        key_cols = {spine_col} | (set(partition_cols) if partition_cols else set())
        value_cols = [c for c in result.columns if c not in key_cols]

    for vc in value_cols:
        result = result.withColumn(
            vc, F.coalesce(F.col(vc), F.lit(fill_value))
        )

    return result


# ---------------------------------------------------------------------------
# 9. add_fiscal_year
# ---------------------------------------------------------------------------

def add_fiscal_year(
    df: DataFrame,
    date_col: str,
    fiscal_start_month: int = 7,
    output_col: str = "fiscal_year",
) -> DataFrame:
    """Add a fiscal year column based on a configurable fiscal year start month.

    The fiscal year label is the *calendar year in which the fiscal year ends*.
    For example, with ``fiscal_start_month=7`` (July), a date of 2024-08-15
    belongs to fiscal year 2025 (FY2025 runs July 2024 – June 2025).

    Args:
        df: Input DataFrame.
        date_col: Name of the date column.
        fiscal_start_month: Month number (1–12) on which the fiscal year starts.
            Defaults to ``7`` (July).
        output_col: Name of the output column. Defaults to ``"fiscal_year"``.

    Returns:
        DataFrame with the fiscal year integer column appended.

    Example::

        df = add_fiscal_year(df, "transaction_date", fiscal_start_month=4)
    """
    if not 1 <= fiscal_start_month <= 12:
        raise ValueError("fiscal_start_month must be between 1 and 12.")

    col_ref = F.col(date_col)
    cal_year = F.year(col_ref)
    cal_month = F.month(col_ref)

    if fiscal_start_month == 1:
        # Fiscal year equals calendar year
        fiscal_year_expr = cal_year
    else:
        # If month >= fiscal_start_month → FY is cal_year + 1, else cal_year
        fiscal_year_expr = F.when(
            cal_month >= fiscal_start_month, cal_year + 1
        ).otherwise(cal_year)

    return df.withColumn(output_col, fiscal_year_expr)


# ---------------------------------------------------------------------------
# 10. add_age
# ---------------------------------------------------------------------------

def add_age(
    df: DataFrame,
    date_col: str,
    as_of_date: Optional[str] = None,
    output_col: str = "age_years",
) -> DataFrame:
    """Calculate age in whole years from a date column.

    Age is computed as the number of complete years elapsed between
    ``date_col`` and ``as_of_date``.

    Args:
        df: Input DataFrame.
        date_col: Name of the birth-date (or reference date) column.
        as_of_date: The reference date against which age is computed, as an
            ISO-8601 string (``"YYYY-MM-DD"``). Defaults to
            ``current_date()``.
        output_col: Name of the output column. Defaults to ``"age_years"``.

    Returns:
        DataFrame with the integer age column appended.

    Example::

        df = add_age(df, "date_of_birth")
        df = add_age(df, "date_of_birth", as_of_date="2025-01-01")
    """
    ref = F.to_date(F.lit(as_of_date)) if as_of_date else F.current_date()
    col_ref = F.col(date_col)

    # months_between gives fractional months; floor-divide by 12
    age_expr = F.floor(F.months_between(ref, col_ref) / 12).cast("int")
    return df.withColumn(output_col, age_expr)


# ---------------------------------------------------------------------------
# 11. add_time_since
# ---------------------------------------------------------------------------

def add_time_since(
    df: DataFrame,
    date_col: str,
    as_of_date: Optional[str] = None,
    units: str = "days",
    output_col: Optional[str] = None,
) -> DataFrame:
    """Calculate the elapsed time between a date column and a reference date.

    Args:
        df: Input DataFrame.
        date_col: Name of the date or timestamp column.
        as_of_date: Reference date/timestamp string. Defaults to
            ``current_date()`` (or ``current_timestamp()`` for sub-day units).
        units: Unit of measurement. One of ``"days"``, ``"hours"``,
            ``"minutes"``, ``"seconds"``, ``"months"``, ``"years"``.
            Defaults to ``"days"``.
        output_col: Name of the output column. Defaults to
            ``<date_col>_since_<units>``.

    Returns:
        DataFrame with the time-since column appended.

    Example::

        df = add_time_since(df, "last_login", units="days")
    """
    if output_col is None:
        output_col = f"{date_col}_since_{units}"

    units_lower = units.lower()

    # Choose appropriate reference expression
    if as_of_date:
        if units_lower in ("hours", "minutes", "seconds"):
            ref: F.Column = F.to_timestamp(F.lit(as_of_date))
        else:
            ref = F.to_date(F.lit(as_of_date))
    else:
        if units_lower in ("hours", "minutes", "seconds"):
            ref = F.current_timestamp()
        else:
            ref = F.current_date()

    col_ref = F.col(date_col)

    if units_lower == "days":
        result_expr = F.datediff(ref, col_ref)
    elif units_lower == "months":
        result_expr = F.floor(F.months_between(ref, col_ref)).cast("int")
    elif units_lower == "years":
        result_expr = F.floor(F.months_between(ref, col_ref) / 12).cast("int")
    elif units_lower == "hours":
        result_expr = (
            (F.unix_timestamp(ref) - F.unix_timestamp(col_ref)) / 3600
        ).cast("long")
    elif units_lower == "minutes":
        result_expr = (
            (F.unix_timestamp(ref) - F.unix_timestamp(col_ref)) / 60
        ).cast("long")
    elif units_lower == "seconds":
        result_expr = (
            F.unix_timestamp(ref) - F.unix_timestamp(col_ref)
        ).cast("long")
    else:
        raise ValueError(
            f"Unsupported units '{units}'. "
            "Supported: 'days', 'hours', 'minutes', 'seconds', 'months', 'years'."
        )

    return df.withColumn(output_col, result_expr)


# ---------------------------------------------------------------------------
# 12. bucket_time_of_day
# ---------------------------------------------------------------------------

def bucket_time_of_day(
    df: DataFrame,
    timestamp_col: str,
    output_col: str = "time_of_day",
) -> DataFrame:
    """Categorize timestamps into named periods of the day.

    Buckets
    -------
    * **morning**   — 06:00 (inclusive) to 12:00 (exclusive)
    * **afternoon** — 12:00 (inclusive) to 17:00 (exclusive)
    * **evening**   — 17:00 (inclusive) to 21:00 (exclusive)
    * **night**     — 21:00 (inclusive) to 06:00 (exclusive, wraps past midnight)

    Args:
        df: Input DataFrame.
        timestamp_col: Name of the timestamp column.
        output_col: Name of the output string column. Defaults to
            ``"time_of_day"``.

    Returns:
        DataFrame with the time-of-day category column appended.

    Example::

        df = bucket_time_of_day(df, "event_timestamp")
    """
    hr = F.hour(F.col(timestamp_col))

    bucket_expr = (
        F.when((hr >= 6) & (hr < 12), F.lit("morning"))
        .when((hr >= 12) & (hr < 17), F.lit("afternoon"))
        .when((hr >= 17) & (hr < 21), F.lit("evening"))
        .otherwise(F.lit("night"))  # 21–23 and 0–5
    )

    return df.withColumn(output_col, bucket_expr)
