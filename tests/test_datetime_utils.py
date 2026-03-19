"""
test_datetime_utils.py
----------------------
Pytest tests for dbx_toolkit.datetime_utils module.
"""

import pytest
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DateType,
    TimestampType,
)

from dbx_toolkit.datetime_utils import (
    parse_dates,
    add_date_parts,
    add_time_parts,
    date_diff_columns,
    add_business_days,
    is_business_day,
    generate_date_spine,
    fill_date_gaps,
    add_fiscal_year,
    add_age,
    add_time_since,
    bucket_time_of_day,
)


# ---------------------------------------------------------------------------
# 1. parse_dates
# ---------------------------------------------------------------------------

class TestParseDates:
    def test_default_formats_iso(self, spark):
        """ISO yyyy-MM-dd strings are parsed correctly."""
        df = spark.createDataFrame([("2024-03-15",)], ["date_str"])
        result = parse_dates(df, "date_str", output_col="parsed")
        row = result.first()
        assert row["parsed"] is not None
        assert str(row["parsed"]) == "2024-03-15"

    def test_default_formats_slash(self, spark):
        """MM/dd/yyyy strings are parsed correctly."""
        df = spark.createDataFrame([("03/15/2024",)], ["date_str"])
        result = parse_dates(df, "date_str", output_col="parsed")
        row = result.first()
        assert row["parsed"] is not None
        assert str(row["parsed"]) == "2024-03-15"

    def test_default_formats_compact(self, spark):
        """yyyyMMdd strings are parsed correctly."""
        df = spark.createDataFrame([("20240315",)], ["date_str"])
        result = parse_dates(df, "date_str", output_col="parsed")
        row = result.first()
        assert row["parsed"] is not None
        assert str(row["parsed"]) == "2024-03-15"

    def test_invalid_string_becomes_null(self, spark):
        """Strings that match no format produce null."""
        df = spark.createDataFrame([("not-a-date",)], ["date_str"])
        result = parse_dates(df, "date_str", output_col="parsed")
        row = result.first()
        assert row["parsed"] is None

    def test_custom_format(self, spark):
        """A custom format list is used instead of defaults."""
        df = spark.createDataFrame([("15.03.2024",)], ["date_str"])
        result = parse_dates(df, "date_str", formats=["dd.MM.yyyy"], output_col="parsed")
        row = result.first()
        assert row["parsed"] is not None
        assert str(row["parsed"]) == "2024-03-15"

    def test_output_col_default_overwrites_source(self, spark):
        """When output_col is omitted the source column is overwritten."""
        df = spark.createDataFrame([("2024-03-15",)], ["date_str"])
        result = parse_dates(df, "date_str")
        # Column name unchanged, but type is now DateType
        assert "date_str" in result.columns
        row = result.first()
        assert str(row["date_str"]) == "2024-03-15"

    def test_mixed_valid_and_invalid(self, spark):
        """Rows with valid dates parse; rows with invalid strings yield null."""
        df = spark.createDataFrame(
            [("2024-01-01",), ("bad-date",), ("2024-06-30",)],
            ["date_str"],
        )
        result = parse_dates(df, "date_str", output_col="parsed")
        rows = {r["date_str"]: r["parsed"] for r in result.collect()}
        assert rows["2024-01-01"] is not None
        assert rows["bad-date"] is None
        assert rows["2024-06-30"] is not None

    def test_first_matching_format_wins(self, spark):
        """With multiple valid formats the first match in the list is used."""
        # "01/02/2024" is ambiguous; with MM/dd/yyyy it should be Jan 2.
        df = spark.createDataFrame([("01/02/2024",)], ["date_str"])
        result = parse_dates(
            df, "date_str",
            formats=["MM/dd/yyyy", "dd/MM/yyyy"],
            output_col="parsed",
        )
        row = result.first()
        assert str(row["parsed"]) == "2024-01-02"


# ---------------------------------------------------------------------------
# 2. add_date_parts
# ---------------------------------------------------------------------------

class TestAddDateParts:
    """2024-03-15 is a Friday (day_of_week=6 in Spark, where 1=Sun).
    It is in Q1, day 75 of year, week 11."""

    def _base_df(self, spark):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("event_date", F.to_date(F.lit("2024-03-15")))
        )

    def test_year(self, spark):
        df = add_date_parts(self._base_df(spark), "event_date", parts=["year"])
        assert df.first()["event_date_year"] == 2024

    def test_month(self, spark):
        df = add_date_parts(self._base_df(spark), "event_date", parts=["month"])
        assert df.first()["event_date_month"] == 3

    def test_day(self, spark):
        df = add_date_parts(self._base_df(spark), "event_date", parts=["day"])
        assert df.first()["event_date_day"] == 15

    def test_quarter(self, spark):
        df = add_date_parts(self._base_df(spark), "event_date", parts=["quarter"])
        assert df.first()["event_date_quarter"] == 1

    def test_day_of_week(self, spark):
        # Spark dayofweek: 1=Sun, 2=Mon, ..., 6=Fri, 7=Sat
        df = add_date_parts(self._base_df(spark), "event_date", parts=["day_of_week"])
        assert df.first()["event_date_day_of_week"] == 6  # Friday

    def test_day_of_year(self, spark):
        # 2024 is a leap year. Jan=31, Feb=29, Mar 15 => 31+29+15 = 75
        df = add_date_parts(self._base_df(spark), "event_date", parts=["day_of_year"])
        assert df.first()["event_date_day_of_year"] == 75

    def test_week_of_year(self, spark):
        df = add_date_parts(self._base_df(spark), "event_date", parts=["week_of_year"])
        assert df.first()["event_date_week_of_year"] == 11

    def test_all_parts_default(self, spark):
        """Calling with no parts argument adds all seven columns."""
        df = add_date_parts(self._base_df(spark), "event_date")
        for part in ["year", "month", "day", "quarter", "day_of_week", "day_of_year", "week_of_year"]:
            assert f"event_date_{part}" in df.columns

    def test_unsupported_part_raises(self, spark):
        with pytest.raises(ValueError, match="Unsupported date part"):
            add_date_parts(self._base_df(spark), "event_date", parts=["decade"])


# ---------------------------------------------------------------------------
# 3. add_time_parts
# ---------------------------------------------------------------------------

class TestAddTimeParts:
    """2024-03-15 14:30:45 → hour=14, minute=30, second=45."""

    def _base_df(self, spark):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("event_ts", F.to_timestamp(F.lit("2024-03-15 14:30:45")))
        )

    def test_hour(self, spark):
        df = add_time_parts(self._base_df(spark), "event_ts", parts=["hour"])
        assert df.first()["event_ts_hour"] == 14

    def test_minute(self, spark):
        df = add_time_parts(self._base_df(spark), "event_ts", parts=["minute"])
        assert df.first()["event_ts_minute"] == 30

    def test_second(self, spark):
        df = add_time_parts(self._base_df(spark), "event_ts", parts=["second"])
        assert df.first()["event_ts_second"] == 45

    def test_all_parts_default(self, spark):
        """Calling with no parts argument adds hour, minute, second."""
        df = add_time_parts(self._base_df(spark), "event_ts")
        for part in ["hour", "minute", "second"]:
            assert f"event_ts_{part}" in df.columns

    def test_unsupported_part_raises(self, spark):
        with pytest.raises(ValueError, match="Unsupported time part"):
            add_time_parts(self._base_df(spark), "event_ts", parts=["millisecond"])

    def test_midnight(self, spark):
        """Midnight timestamp yields hour=0, minute=0, second=0."""
        df = (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("ts", F.to_timestamp(F.lit("2024-01-01 00:00:00")))
        )
        result = add_time_parts(df, "ts")
        row = result.first()
        assert row["ts_hour"] == 0
        assert row["ts_minute"] == 0
        assert row["ts_second"] == 0


# ---------------------------------------------------------------------------
# 4. date_diff_columns
# ---------------------------------------------------------------------------

class TestDateDiffColumns:
    def _base_df(self, spark, start="2024-01-01", end="2024-03-15"):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("start_date", F.to_date(F.lit(start)))
            .withColumn("end_date", F.to_date(F.lit(end)))
        )

    def test_days(self, spark):
        # Jan(31) + Feb(29, leap) + 14 = 74 days
        df = date_diff_columns(self._base_df(spark), "start_date", "end_date", units="days")
        assert df.first()["start_date_to_end_date_diff_days"] == 74

    def test_months(self, spark):
        df = date_diff_columns(self._base_df(spark), "start_date", "end_date", units="months")
        # Jan 1 to Mar 15 is just over 2 months; months_between truncated to int = 2
        assert df.first()["start_date_to_end_date_diff_months"] == 2

    def test_years(self, spark):
        df = date_diff_columns(
            self._base_df(spark, "2020-01-01", "2024-03-15"),
            "start_date", "end_date", units="years",
        )
        # ~4.2 years → int = 4
        assert df.first()["start_date_to_end_date_diff_years"] == 4

    def test_hours(self, spark):
        df = (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("start_ts", F.to_timestamp(F.lit("2024-03-15 00:00:00")))
            .withColumn("end_ts", F.to_timestamp(F.lit("2024-03-15 06:00:00")))
        )
        result = date_diff_columns(df, "start_ts", "end_ts", units="hours")
        assert result.first()["start_ts_to_end_ts_diff_hours"] == 6

    def test_output_col_name(self, spark):
        """Result column follows the naming convention."""
        df = date_diff_columns(self._base_df(spark), "start_date", "end_date", units="days")
        assert "start_date_to_end_date_diff_days" in df.columns

    def test_negative_diff(self, spark):
        """end before start produces a negative value."""
        df = date_diff_columns(
            self._base_df(spark, "2024-03-15", "2024-01-01"),
            "start_date", "end_date", units="days",
        )
        assert df.first()["start_date_to_end_date_diff_days"] < 0

    def test_unsupported_units_raises(self, spark):
        with pytest.raises(ValueError, match="Unsupported units"):
            date_diff_columns(self._base_df(spark), "start_date", "end_date", units="decades")


# ---------------------------------------------------------------------------
# 5. add_business_days
# ---------------------------------------------------------------------------

class TestAddBusinessDays:
    """
    Reference dates:
      2024-03-15 = Friday
      2024-03-18 = Monday
      2024-03-11 = Monday
    """

    def _df(self, spark, date_str):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("ref_date", F.to_date(F.lit(date_str)))
        )

    def test_add_1_from_friday_skips_weekend(self, spark):
        """1 business day after Friday should be Monday."""
        df = add_business_days(self._df(spark, "2024-03-15"), "ref_date", 1)
        result = str(df.first()["ref_date_plus_1bd"])
        assert result == "2024-03-18"

    def test_add_5_from_monday(self, spark):
        """5 business days from Monday = next Monday."""
        df = add_business_days(self._df(spark, "2024-03-11"), "ref_date", 5)
        result = str(df.first()["ref_date_plus_5bd"])
        assert result == "2024-03-18"

    def test_add_0_no_change(self, spark):
        """Adding 0 business days returns the same date."""
        df = add_business_days(self._df(spark, "2024-03-15"), "ref_date", 0)
        result = str(df.first()["ref_date_plus_0bd"])
        assert result == "2024-03-15"

    def test_negative_days(self, spark):
        """Subtracting business days skips weekends going backwards."""
        # 1 business day before Monday = Friday
        df = add_business_days(self._df(spark, "2024-03-18"), "ref_date", -1)
        result = str(df.first()["ref_date_plus_-1bd"])
        assert result == "2024-03-15"

    def test_custom_output_col(self, spark):
        """Custom output_col name is respected."""
        df = add_business_days(
            self._df(spark, "2024-03-15"), "ref_date", 2, output_col="result_date"
        )
        assert "result_date" in df.columns

    def test_add_10_crosses_two_weekends(self, spark):
        """10 business days from Monday spans two full weeks."""
        # 2024-03-11 (Mon) + 10 bd = 2024-03-25 (Mon)
        df = add_business_days(self._df(spark, "2024-03-11"), "ref_date", 10)
        result = str(df.first()["ref_date_plus_10bd"])
        assert result == "2024-03-25"


# ---------------------------------------------------------------------------
# 6. is_business_day
# ---------------------------------------------------------------------------

class TestIsBusinessDay:
    def _df(self, spark, date_str):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("ref_date", F.to_date(F.lit(date_str)))
        )

    def test_monday_is_business_day(self, spark):
        df = is_business_day(self._df(spark, "2024-03-11"), "ref_date")
        assert df.first()["ref_date_is_business_day"] is True

    def test_friday_is_business_day(self, spark):
        df = is_business_day(self._df(spark, "2024-03-15"), "ref_date")
        assert df.first()["ref_date_is_business_day"] is True

    def test_saturday_is_not_business_day(self, spark):
        df = is_business_day(self._df(spark, "2024-03-16"), "ref_date")
        assert df.first()["ref_date_is_business_day"] is False

    def test_sunday_is_not_business_day(self, spark):
        df = is_business_day(self._df(spark, "2024-03-17"), "ref_date")
        assert df.first()["ref_date_is_business_day"] is False

    def test_wednesday_is_business_day(self, spark):
        df = is_business_day(self._df(spark, "2024-03-13"), "ref_date")
        assert df.first()["ref_date_is_business_day"] is True

    def test_custom_output_col(self, spark):
        df = is_business_day(
            self._df(spark, "2024-03-15"), "ref_date", output_col="is_bday"
        )
        assert "is_bday" in df.columns
        assert "ref_date_is_business_day" not in df.columns


# ---------------------------------------------------------------------------
# 7. generate_date_spine
# ---------------------------------------------------------------------------

class TestGenerateDateSpine:
    def test_day_interval_row_count(self, spark):
        """Day spine from Jan 1 to Jan 7 inclusive = 7 rows."""
        df = generate_date_spine(spark, "2024-01-01", "2024-01-07", interval="day")
        assert df.count() == 7

    def test_day_interval_column_name(self, spark):
        df = generate_date_spine(spark, "2024-01-01", "2024-01-03", interval="day")
        assert "date" in df.columns

    def test_week_interval_row_count(self, spark):
        """Weekly spine over 4 weeks = 4 rows (start dates: Jan 1, 8, 15, 22)."""
        df = generate_date_spine(spark, "2024-01-01", "2024-01-22", interval="week")
        assert df.count() == 4

    def test_week_interval_column_name(self, spark):
        df = generate_date_spine(spark, "2024-01-01", "2024-01-08", interval="week")
        assert "week_start" in df.columns

    def test_month_interval_row_count(self, spark):
        """Monthly spine Jan–Mar 2024 = 3 rows."""
        df = generate_date_spine(spark, "2024-01-01", "2024-03-01", interval="month")
        assert df.count() == 3

    def test_month_interval_column_name(self, spark):
        df = generate_date_spine(spark, "2024-01-01", "2024-03-01", interval="month")
        assert "month_start" in df.columns

    def test_hour_interval_row_count(self, spark):
        """Hourly spine over 6 hours = 7 rows (0,1,2,3,4,5,6)."""
        df = generate_date_spine(
            spark,
            "2024-01-01 00:00:00",
            "2024-01-01 06:00:00",
            interval="hour",
        )
        assert df.count() == 7

    def test_hour_interval_column_name(self, spark):
        df = generate_date_spine(
            spark,
            "2024-01-01 00:00:00",
            "2024-01-01 02:00:00",
            interval="hour",
        )
        assert "ts" in df.columns

    def test_single_day_spine(self, spark):
        """When start == end the spine has exactly 1 row."""
        df = generate_date_spine(spark, "2024-03-15", "2024-03-15", interval="day")
        assert df.count() == 1

    def test_unsupported_interval_raises(self, spark):
        with pytest.raises(ValueError, match="Unsupported interval"):
            generate_date_spine(spark, "2024-01-01", "2024-01-07", interval="decade")


# ---------------------------------------------------------------------------
# 8. fill_date_gaps
# ---------------------------------------------------------------------------

class TestFillDateGaps:
    def test_fills_missing_days(self, spark):
        """A gap of 1 day is filled with fill_value=0."""
        schema = StructType([
            StructField("sale_date", DateType(), True),
            StructField("revenue", IntegerType(), True),
        ])
        # Jan 1 and Jan 3 provided; Jan 2 is missing
        rows = [
            ("2024-01-01", 100),
            ("2024-01-03", 200),
        ]
        df = spark.createDataFrame(rows, ["sale_date_str", "revenue"])
        df = df.withColumn("sale_date", F.to_date(F.col("sale_date_str"))).drop("sale_date_str")

        result = fill_date_gaps(df, "sale_date", value_cols=["revenue"])
        assert result.count() == 3

        filled_row = result.filter(F.col("date") == F.lit("2024-01-02")).first()
        assert filled_row is not None
        assert filled_row["revenue"] == 0

    def test_custom_fill_value(self, spark):
        """Missing rows are filled with the supplied fill_value."""
        rows = [("2024-01-01", 10), ("2024-01-03", 30)]
        df = spark.createDataFrame(rows, ["sale_date_str", "revenue"])
        df = df.withColumn("sale_date", F.to_date(F.col("sale_date_str"))).drop("sale_date_str")

        result = fill_date_gaps(df, "sale_date", fill_value=-1, value_cols=["revenue"])
        filled_row = result.filter(F.col("date") == F.lit("2024-01-02")).first()
        assert filled_row["revenue"] == -1

    def test_partition_cols(self, spark):
        """Each partition group gets its own gap-filled spine."""
        rows = [
            ("2024-01-01", "A", 10),
            ("2024-01-03", "A", 30),
            ("2024-01-01", "B", 5),
            ("2024-01-03", "B", 15),
        ]
        df = spark.createDataFrame(rows, ["sale_date_str", "store_id", "revenue"])
        df = df.withColumn("sale_date", F.to_date(F.col("sale_date_str"))).drop("sale_date_str")

        result = fill_date_gaps(
            df, "sale_date",
            partition_cols=["store_id"],
            value_cols=["revenue"],
        )
        # 3 dates x 2 stores = 6 rows total
        assert result.count() == 6

    def test_no_gaps_unchanged_count(self, spark):
        """If the data has no gaps the row count stays the same."""
        rows = [("2024-01-01", 10), ("2024-01-02", 20), ("2024-01-03", 30)]
        df = spark.createDataFrame(rows, ["sale_date_str", "revenue"])
        df = df.withColumn("sale_date", F.to_date(F.col("sale_date_str"))).drop("sale_date_str")

        result = fill_date_gaps(df, "sale_date", value_cols=["revenue"])
        assert result.count() == 3


# ---------------------------------------------------------------------------
# 9. add_fiscal_year
# ---------------------------------------------------------------------------

class TestAddFiscalYear:
    def _df(self, spark, date_str):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("txn_date", F.to_date(F.lit(date_str)))
        )

    def test_fiscal_start_july_before_start_month(self, spark):
        """March 2024 is in FY2024 when fiscal year starts in July."""
        df = add_fiscal_year(self._df(spark, "2024-03-15"), "txn_date", fiscal_start_month=7)
        assert df.first()["fiscal_year"] == 2024

    def test_fiscal_start_july_on_or_after_start_month(self, spark):
        """August 2024 is in FY2025 when fiscal year starts in July."""
        df = add_fiscal_year(self._df(spark, "2024-08-15"), "txn_date", fiscal_start_month=7)
        assert df.first()["fiscal_year"] == 2025

    def test_fiscal_start_january_equals_calendar_year(self, spark):
        """With fiscal_start_month=1 fiscal year == calendar year."""
        df = add_fiscal_year(self._df(spark, "2024-06-15"), "txn_date", fiscal_start_month=1)
        assert df.first()["fiscal_year"] == 2024

    def test_fiscal_start_april(self, spark):
        """April 2024 is in FY2025 when fiscal year starts in April."""
        df = add_fiscal_year(self._df(spark, "2024-04-01"), "txn_date", fiscal_start_month=4)
        assert df.first()["fiscal_year"] == 2025

    def test_fiscal_start_april_march_before_start(self, spark):
        """March 2024 is still in FY2024 when fiscal year starts in April."""
        df = add_fiscal_year(self._df(spark, "2024-03-31"), "txn_date", fiscal_start_month=4)
        assert df.first()["fiscal_year"] == 2024

    def test_custom_output_col(self, spark):
        df = add_fiscal_year(
            self._df(spark, "2024-03-15"), "txn_date",
            fiscal_start_month=7, output_col="fy",
        )
        assert "fy" in df.columns
        assert "fiscal_year" not in df.columns

    def test_invalid_start_month_raises(self, spark):
        with pytest.raises(ValueError, match="fiscal_start_month"):
            add_fiscal_year(self._df(spark, "2024-03-15"), "txn_date", fiscal_start_month=13)


# ---------------------------------------------------------------------------
# 10. add_age
# ---------------------------------------------------------------------------

class TestAddAge:
    def _df(self, spark, dob_str):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("dob", F.to_date(F.lit(dob_str)))
        )

    def test_exact_age_calculation(self, spark):
        """Born on 2000-01-01, as of 2030-01-01 = age 30."""
        df = add_age(self._df(spark, "2000-01-01"), "dob", as_of_date="2030-01-01")
        assert df.first()["age_years"] == 30

    def test_age_before_birthday_in_year(self, spark):
        """Born 1990-06-15; as of 2024-03-15 hasn't had birthday yet → age 33."""
        df = add_age(self._df(spark, "1990-06-15"), "dob", as_of_date="2024-03-15")
        assert df.first()["age_years"] == 33

    def test_age_on_birthday(self, spark):
        """As of exactly the birthday the full year counts."""
        df = add_age(self._df(spark, "1990-03-15"), "dob", as_of_date="2024-03-15")
        assert df.first()["age_years"] == 34

    def test_age_after_birthday_in_year(self, spark):
        """Born 1990-01-01; as of 2024-03-15 has had birthday → age 34."""
        df = add_age(self._df(spark, "1990-01-01"), "dob", as_of_date="2024-03-15")
        assert df.first()["age_years"] == 34

    def test_custom_output_col(self, spark):
        df = add_age(
            self._df(spark, "2000-01-01"), "dob",
            as_of_date="2024-01-01", output_col="customer_age",
        )
        assert "customer_age" in df.columns
        assert "age_years" not in df.columns


# ---------------------------------------------------------------------------
# 11. add_time_since
# ---------------------------------------------------------------------------

class TestAddTimeSince:
    def _df(self, spark, date_str):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("last_event", F.to_date(F.lit(date_str)))
        )

    def _ts_df(self, spark, ts_str):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("last_event", F.to_timestamp(F.lit(ts_str)))
        )

    def test_days_since(self, spark):
        """Days since 2024-01-01 as of 2024-03-15 = 74."""
        df = add_time_since(
            self._df(spark, "2024-01-01"),
            "last_event", as_of_date="2024-03-15", units="days",
        )
        assert df.first()["last_event_since_days"] == 74

    def test_months_since(self, spark):
        """Months since 2024-01-01 as of 2024-03-15 = 2 (floor)."""
        df = add_time_since(
            self._df(spark, "2024-01-01"),
            "last_event", as_of_date="2024-03-15", units="months",
        )
        assert df.first()["last_event_since_months"] == 2

    def test_years_since(self, spark):
        """Years since 2020-01-01 as of 2024-03-15 = 4 (floor)."""
        df = add_time_since(
            self._df(spark, "2020-01-01"),
            "last_event", as_of_date="2024-03-15", units="years",
        )
        assert df.first()["last_event_since_years"] == 4

    def test_hours_since(self, spark):
        """6 hours between timestamps."""
        df = add_time_since(
            self._ts_df(spark, "2024-03-15 08:00:00"),
            "last_event",
            as_of_date="2024-03-15 14:00:00",
            units="hours",
        )
        assert df.first()["last_event_since_hours"] == 6

    def test_minutes_since(self, spark):
        """90 minutes between timestamps."""
        df = add_time_since(
            self._ts_df(spark, "2024-03-15 12:00:00"),
            "last_event",
            as_of_date="2024-03-15 13:30:00",
            units="minutes",
        )
        assert df.first()["last_event_since_minutes"] == 90

    def test_seconds_since(self, spark):
        """120 seconds between timestamps."""
        df = add_time_since(
            self._ts_df(spark, "2024-03-15 12:00:00"),
            "last_event",
            as_of_date="2024-03-15 12:02:00",
            units="seconds",
        )
        assert df.first()["last_event_since_seconds"] == 120

    def test_custom_output_col(self, spark):
        df = add_time_since(
            self._df(spark, "2024-01-01"),
            "last_event",
            as_of_date="2024-03-15",
            units="days",
            output_col="days_elapsed",
        )
        assert "days_elapsed" in df.columns

    def test_default_output_col_name(self, spark):
        df = add_time_since(
            self._df(spark, "2024-01-01"),
            "last_event",
            as_of_date="2024-03-15",
            units="months",
        )
        assert "last_event_since_months" in df.columns

    def test_unsupported_units_raises(self, spark):
        with pytest.raises(ValueError, match="Unsupported units"):
            add_time_since(
                self._df(spark, "2024-01-01"),
                "last_event",
                as_of_date="2024-03-15",
                units="decades",
            )


# ---------------------------------------------------------------------------
# 12. bucket_time_of_day
# ---------------------------------------------------------------------------

class TestBucketTimeOfDay:
    """
    Buckets:
      morning   06:00 – 11:59
      afternoon 12:00 – 16:59
      evening   17:00 – 20:59
      night     21:00 – 05:59  (wraps midnight)
    """

    def _df(self, spark, ts_str):
        return (
            spark.createDataFrame([(1,)], ["id"])
            .withColumn("event_ts", F.to_timestamp(F.lit(ts_str)))
        )

    def test_morning_at_0600(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 06:00:00"), "event_ts")
        assert df.first()["time_of_day"] == "morning"

    def test_morning_at_1159(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 11:59:00"), "event_ts")
        assert df.first()["time_of_day"] == "morning"

    def test_afternoon_at_1200(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 12:00:00"), "event_ts")
        assert df.first()["time_of_day"] == "afternoon"

    def test_afternoon_at_1430(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 14:30:00"), "event_ts")
        assert df.first()["time_of_day"] == "afternoon"

    def test_afternoon_at_1659(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 16:59:00"), "event_ts")
        assert df.first()["time_of_day"] == "afternoon"

    def test_evening_at_1700(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 17:00:00"), "event_ts")
        assert df.first()["time_of_day"] == "evening"

    def test_evening_at_2059(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 20:59:00"), "event_ts")
        assert df.first()["time_of_day"] == "evening"

    def test_night_at_2100(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 21:00:00"), "event_ts")
        assert df.first()["time_of_day"] == "night"

    def test_night_at_2359(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 23:59:00"), "event_ts")
        assert df.first()["time_of_day"] == "night"

    def test_night_at_midnight(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 00:00:00"), "event_ts")
        assert df.first()["time_of_day"] == "night"

    def test_night_at_0559(self, spark):
        df = bucket_time_of_day(self._df(spark, "2024-03-15 05:59:00"), "event_ts")
        assert df.first()["time_of_day"] == "night"

    def test_custom_output_col(self, spark):
        df = bucket_time_of_day(
            self._df(spark, "2024-03-15 14:30:00"),
            "event_ts",
            output_col="day_period",
        )
        assert "day_period" in df.columns
        assert "time_of_day" not in df.columns

    def test_all_four_buckets_present(self, spark):
        """A DataFrame with one timestamp in each bucket produces all four labels."""
        rows = [
            ("2024-03-15 08:00:00",),  # morning
            ("2024-03-15 13:00:00",),  # afternoon
            ("2024-03-15 18:00:00",),  # evening
            ("2024-03-15 22:00:00",),  # night
        ]
        df = spark.createDataFrame(rows, ["ts_str"])
        df = df.withColumn("event_ts", F.to_timestamp(F.col("ts_str")))
        result = bucket_time_of_day(df, "event_ts")
        labels = {r["time_of_day"] for r in result.collect()}
        assert labels == {"morning", "afternoon", "evening", "night"}
