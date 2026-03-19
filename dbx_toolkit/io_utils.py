"""
io_utils.py — Reusable PySpark I/O utility functions for Databricks POCs.

Covers Delta table reads/writes, Unity Catalog Volume I/O, table introspection,
MERGE (upsert), SCD Type 2, backup, and catalog listing helpers.
"""

from __future__ import annotations

import fnmatch
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. read_table_safe
# ---------------------------------------------------------------------------

def read_table_safe(
    spark: SparkSession,
    table_name: str,
    columns: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    limit: Optional[int] = None,
) -> Optional[DataFrame]:
    """Read a Delta table with optional column selection, filter, and row limit.

    Handles the case where the table does not exist by returning ``None``
    instead of raising an exception, making it safe to call in exploratory
    or conditional code paths.

    Args:
        spark: Active ``SparkSession``.
        table_name: Fully-qualified table name (``catalog.schema.table`` or
            ``schema.table``).
        columns: List of column names to select.  ``None`` returns all columns.
        filter_expr: SQL-style filter expression, e.g. ``"year = 2024"``.
        limit: Maximum number of rows to return.  ``None`` returns all rows.

    Returns:
        A ``DataFrame`` if the table exists, otherwise ``None``.
    """
    if not table_exists(spark, table_name):
        logger.warning("read_table_safe: table '%s' does not exist — returning None.", table_name)
        return None

    df: DataFrame = spark.table(table_name)

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"read_table_safe: columns {missing} not found in '{table_name}'. "
                f"Available columns: {df.columns}"
            )
        df = df.select(*columns)

    if filter_expr:
        df = df.filter(filter_expr)

    if limit is not None:
        if not isinstance(limit, int) or limit < 0:
            raise ValueError(f"read_table_safe: 'limit' must be a non-negative integer, got {limit!r}.")
        df = df.limit(limit)

    return df


# ---------------------------------------------------------------------------
# 2. write_table
# ---------------------------------------------------------------------------

def write_table(
    df: DataFrame,
    table_name: str,
    mode: str = "overwrite",
    partition_by: Optional[List[str]] = None,
    optimize_after: bool = True,
) -> None:
    """Write a DataFrame to a Delta table.

    Args:
        df: Source ``DataFrame``.
        table_name: Fully-qualified target table name.
        mode: Spark write mode — ``"overwrite"``, ``"append"``, ``"ignore"``,
            or ``"error"``/``"errorifexists"``.
        partition_by: List of column names to partition the table by.
        optimize_after: When ``True``, run ``OPTIMIZE`` on the table after the
            write completes to compact small files.

    Raises:
        ValueError: If *mode* is not a recognised Spark write mode.
    """
    valid_modes = {"overwrite", "append", "ignore", "error", "errorifexists"}
    if mode.lower() not in valid_modes:
        raise ValueError(f"write_table: invalid mode '{mode}'. Must be one of {valid_modes}.")

    writer = df.write.format("delta").mode(mode)

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.saveAsTable(table_name)
    logger.info("write_table: wrote %s rows to '%s' (mode=%s).", df.count(), table_name, mode)

    if optimize_after:
        spark = df.sparkSession
        spark.sql(f"OPTIMIZE {table_name}")
        logger.info("write_table: OPTIMIZE completed on '%s'.", table_name)


# ---------------------------------------------------------------------------
# 3. read_csv_with_schema
# ---------------------------------------------------------------------------

def read_csv_with_schema(
    spark: SparkSession,
    path: str,
    schema: Optional[StructType] = None,
    header: bool = True,
    delimiter: str = ",",
    infer_schema: bool = True,
    multiline: bool = False,
    null_values: Optional[List[str]] = None,
) -> DataFrame:
    """Read one or more CSV files with common options bundled into a single call.

    When *schema* is provided, ``inferSchema`` is ignored and the supplied
    schema is enforced directly — this is the recommended approach for
    production pipelines as it avoids an extra scan pass.

    Args:
        spark: Active ``SparkSession``.
        path: File path or glob pattern (e.g. ``/Volumes/cat/sch/vol/data/*.csv``).
        schema: Optional explicit ``StructType`` schema.
        header: Whether the first row contains column names.
        delimiter: Field separator character.
        infer_schema: Infer column types when ``schema`` is not provided.
            Requires an extra scan pass; disable for large files when the
            schema is known.
        multiline: Enable multiline mode for CSV fields that contain embedded
            newlines.
        null_values: List of string tokens to interpret as ``null``, e.g.
            ``["NA", "N/A", "NULL", ""]``.

    Returns:
        A ``DataFrame`` containing the CSV data.
    """
    reader = (
        spark.read.format("csv")
        .option("header", str(header).lower())
        .option("delimiter", delimiter)
        .option("multiLine", str(multiline).lower())
    )

    if schema is not None:
        reader = reader.schema(schema)
    else:
        reader = reader.option("inferSchema", str(infer_schema).lower())

    if null_values:
        # Spark accepts a single string; join multiple values with the pipe
        # separator (Spark CSV treats each listed value as null).
        for nv in null_values:
            reader = reader.option("nullValue", nv)

    df = reader.load(path)
    logger.info("read_csv_with_schema: loaded path '%s' → %d columns.", path, len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 4. read_files_from_volume
# ---------------------------------------------------------------------------

def read_files_from_volume(
    spark: SparkSession,
    volume_path: str,
    file_format: str = "csv",
    options: Optional[Dict[str, Any]] = None,
) -> DataFrame:
    """Read files from a Unity Catalog Volume.

    Args:
        spark: Active ``SparkSession``.
        volume_path: Full Volume path, e.g.
            ``/Volumes/catalog/schema/volume_name/`` or a glob such as
            ``/Volumes/catalog/schema/volume_name/*.parquet``.
        file_format: One of ``"csv"``, ``"json"``, ``"parquet"``, ``"delta"``,
            ``"text"``.
        options: Arbitrary reader options passed directly to
            ``spark.read.options(**options)``.

    Returns:
        A ``DataFrame``.

    Raises:
        ValueError: If *file_format* is not supported.
    """
    supported = {"csv", "json", "parquet", "delta", "text"}
    fmt = file_format.lower()
    if fmt not in supported:
        raise ValueError(
            f"read_files_from_volume: unsupported format '{file_format}'. "
            f"Must be one of {supported}."
        )

    reader = spark.read.format(fmt)

    if options:
        reader = reader.options(**options)

    # Sensible defaults per format
    if fmt == "csv" and not options:
        reader = reader.option("header", "true").option("inferSchema", "true")
    elif fmt == "json" and not options:
        reader = reader.option("multiLine", "true")

    df = reader.load(volume_path)
    logger.info(
        "read_files_from_volume: loaded '%s' (format=%s) → %d columns.",
        volume_path, fmt, len(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# 5. write_to_volume
# ---------------------------------------------------------------------------

def write_to_volume(
    df: DataFrame,
    volume_path: str,
    file_format: str = "parquet",
    mode: str = "overwrite",
    options: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a DataFrame to a Unity Catalog Volume path.

    Args:
        df: Source ``DataFrame``.
        volume_path: Target directory inside the Volume, e.g.
            ``/Volumes/catalog/schema/volume_name/output/``.
        file_format: One of ``"csv"``, ``"json"``, ``"parquet"``, ``"delta"``,
            ``"text"``.
        mode: Spark write mode — ``"overwrite"`` or ``"append"``.
        options: Arbitrary writer options passed to
            ``df.write.options(**options)``.

    Raises:
        ValueError: If *file_format* or *mode* is not supported.
    """
    supported_formats = {"csv", "json", "parquet", "delta", "text"}
    fmt = file_format.lower()
    if fmt not in supported_formats:
        raise ValueError(
            f"write_to_volume: unsupported format '{file_format}'. "
            f"Must be one of {supported_formats}."
        )

    valid_modes = {"overwrite", "append"}
    if mode.lower() not in valid_modes:
        raise ValueError(
            f"write_to_volume: unsupported mode '{mode}'. Must be one of {valid_modes}."
        )

    writer = df.write.format(fmt).mode(mode)

    if options:
        writer = writer.options(**options)

    # Sensible defaults per format
    if fmt == "csv" and not options:
        writer = writer.option("header", "true")

    writer.save(volume_path)
    logger.info("write_to_volume: wrote to '%s' (format=%s, mode=%s).", volume_path, fmt, mode)


# ---------------------------------------------------------------------------
# 6. table_exists
# ---------------------------------------------------------------------------

def table_exists(spark: SparkSession, table_name: str) -> bool:
    """Check whether a table (or view) exists in the catalog.

    Works with both two-part (``schema.table``) and three-part
    (``catalog.schema.table``) identifiers.

    Args:
        spark: Active ``SparkSession``.
        table_name: Table identifier to check.

    Returns:
        ``True`` if the table exists, ``False`` otherwise.
    """
    try:
        parts = table_name.strip("`").split(".")
        if len(parts) == 3:
            catalog, database, table = parts
            return spark.catalog.tableExists(table, f"{catalog}.{database}")
        elif len(parts) == 2:
            database, table = parts
            return spark.catalog.tableExists(table, database)
        else:
            return spark.catalog.tableExists(table_name)
    except Exception:
        # Fall back to a SQL probe — handles edge cases with special characters
        # or Unity Catalog quirks.
        try:
            spark.sql(f"DESCRIBE TABLE {table_name}")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 7. get_table_info
# ---------------------------------------------------------------------------

def get_table_info(spark: SparkSession, table_name: str) -> Dict[str, Any]:
    """Return metadata about a Delta table.

    Args:
        spark: Active ``SparkSession``.
        table_name: Fully-qualified table name.

    Returns:
        A dictionary with the following keys:

        - ``row_count`` (int | None): Approximate row count from table
          statistics, or ``None`` if stats are unavailable.
        - ``column_count`` (int): Number of columns.
        - ``columns`` (list[str]): Column names.
        - ``size_bytes`` (int | None): On-disk size in bytes, or ``None``.
        - ``partitioned_by`` (list[str]): Partition columns, empty if
          unpartitioned.
        - ``last_modified`` (str | None): ISO-8601 timestamp of the last write
          operation, or ``None``.

    Raises:
        ValueError: If the table does not exist.
    """
    if not table_exists(spark, table_name):
        raise ValueError(f"get_table_info: table '{table_name}' does not exist.")

    df = spark.table(table_name)
    columns = df.columns
    column_count = len(columns)

    # Partition columns
    partitioned_by: List[str] = []
    try:
        detail_row = spark.sql(f"DESCRIBE DETAIL {table_name}").collect()[0]
        partitioned_by = detail_row["partitionColumns"] or []
        size_bytes: Optional[int] = detail_row["sizeInBytes"]
        last_modified_raw = detail_row.get("lastModified") or detail_row.get("createdAt")
        last_modified: Optional[str] = (
            last_modified_raw.isoformat() if last_modified_raw else None
        )
    except Exception:
        size_bytes = None
        last_modified = None

    # Row count — prefer table statistics to avoid a full scan
    row_count: Optional[int] = None
    try:
        stats_row = spark.sql(f"ANALYZE TABLE {table_name} COMPUTE STATISTICS").collect()
        desc_rows = spark.sql(f"DESCRIBE EXTENDED {table_name}").collect()
        for row in desc_rows:
            if row["col_name"] == "Statistics":
                stat_str = row["data_type"]  # e.g. "12345 rows, 678 bytes"
                for part in stat_str.split(","):
                    part = part.strip()
                    if "rows" in part:
                        row_count = int(part.split()[0])
                        break
                break
    except Exception:
        row_count = None

    return {
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "size_bytes": size_bytes,
        "partitioned_by": partitioned_by,
        "last_modified": last_modified,
    }


# ---------------------------------------------------------------------------
# 8. merge_into
# ---------------------------------------------------------------------------

def merge_into(
    spark: SparkSession,
    target_table: str,
    source_df: DataFrame,
    merge_keys: List[str],
    update_columns: Optional[List[str]] = None,
    insert_all: bool = True,
) -> None:
    """Perform a Delta MERGE (upsert) operation.

    Matched rows are updated; unmatched rows are inserted when
    ``insert_all=True``.

    Args:
        spark: Active ``SparkSession``.
        target_table: Fully-qualified name of the existing Delta target table.
        source_df: ``DataFrame`` containing the source (new/updated) data.
        merge_keys: Column names used to match rows between source and target.
        update_columns: Columns to update on match.  When ``None``, all
            non-key columns present in *source_df* are updated.
        insert_all: When ``True``, insert every unmatched source row.

    Raises:
        ValueError: If the target table does not exist or *merge_keys* is empty.
    """
    if not merge_keys:
        raise ValueError("merge_into: 'merge_keys' must contain at least one column.")
    if not table_exists(spark, target_table):
        raise ValueError(f"merge_into: target table '{target_table}' does not exist.")

    target = DeltaTable.forName(spark, target_table)

    join_condition = " AND ".join(
        [f"target.{k} = source.{k}" for k in merge_keys]
    )

    if update_columns is None:
        update_columns = [c for c in source_df.columns if c not in merge_keys]

    update_map = {col: F.col(f"source.{col}") for col in update_columns}

    merge_builder = (
        target.alias("target")
        .merge(source_df.alias("source"), join_condition)
        .whenMatchedUpdate(set=update_map)
    )

    if insert_all:
        merge_builder = merge_builder.whenNotMatchedInsertAll()

    merge_builder.execute()
    logger.info(
        "merge_into: MERGE completed on '%s' (keys=%s, update_cols=%d, insert_all=%s).",
        target_table, merge_keys, len(update_columns), insert_all,
    )


# ---------------------------------------------------------------------------
# 9. scd_type2
# ---------------------------------------------------------------------------

def scd_type2(
    spark: SparkSession,
    target_table: str,
    source_df: DataFrame,
    key_columns: List[str],
    tracked_columns: List[str],
    effective_date_col: str = "effective_date",
    end_date_col: str = "end_date",
    current_flag_col: str = "is_current",
) -> None:
    """Implement a Slowly Changing Dimension Type 2 (SCD2) merge pattern.

    For each incoming source row:

    - If no matching key exists in the target, insert as a new current record.
    - If a matching key exists and any *tracked_columns* have changed, close
      the existing current record (set ``end_date`` and ``is_current=False``)
      and insert a new current record with the updated values.
    - Unchanged rows are left untouched.

    Args:
        spark: Active ``SparkSession``.
        target_table: Fully-qualified Delta target table name.  Must already
            exist with the SCD2 control columns present.
        source_df: Incoming source ``DataFrame``.  Must contain *key_columns*
            and *tracked_columns*.
        key_columns: Business key columns used to identify a record.
        tracked_columns: Columns whose changes trigger a new SCD2 version.
        effective_date_col: Column name for the record's effective start date.
        end_date_col: Column name for the record's effective end date
            (``null`` for current records).
        current_flag_col: Boolean column name indicating the current record.

    Raises:
        ValueError: If the target table does not exist.
    """
    if not table_exists(spark, target_table):
        raise ValueError(f"scd_type2: target table '{target_table}' does not exist.")

    today = datetime.now(tz=timezone.utc).date().isoformat()

    target = DeltaTable.forName(spark, target_table)

    # Build join condition on business keys
    key_join = " AND ".join(
        [f"target.{k} = source.{k}" for k in key_columns]
    )

    # Build a change-detection expression across tracked columns
    change_expr = " OR ".join(
        [f"target.{c} <> source.{c} OR (target.{c} IS NULL AND source.{c} IS NOT NULL) "
         f"OR (target.{c} IS NOT NULL AND source.{c} IS NULL)"
         for c in tracked_columns]
    )

    full_match_condition = f"({key_join}) AND target.{current_flag_col} = true"
    changed_condition = f"({change_expr})"

    # --- Step 1: Close existing current records that have changed ---
    (
        target.alias("target")
        .merge(source_df.alias("source"), full_match_condition)
        .whenMatchedUpdate(
            condition=changed_condition,
            set={
                end_date_col: F.lit(today),
                current_flag_col: F.lit(False),
            },
        )
        .execute()
    )

    # --- Step 2: Insert new versions for changed records + brand new records ---
    # Re-read target after step 1 so we reflect closed rows
    target_df = spark.table(target_table)

    # Rows in source that have no current match in target (new or just closed)
    new_records = source_df.alias("src").join(
        target_df.filter(F.col(current_flag_col)).alias("tgt"),
        on=key_columns,
        how="left_anti",
    )

    if new_records.rdd.isEmpty():
        logger.info("scd_type2: no new or changed rows to insert into '%s'.", target_table)
        return

    new_records = (
        new_records
        .withColumn(effective_date_col, F.lit(today))
        .withColumn(end_date_col, F.lit(None).cast("string"))
        .withColumn(current_flag_col, F.lit(True))
    )

    new_records.write.format("delta").mode("append").saveAsTable(target_table)
    logger.info(
        "scd_type2: SCD2 merge completed on '%s'. Inserted %d new/updated record(s).",
        target_table, new_records.count(),
    )


# ---------------------------------------------------------------------------
# 10. create_table_if_not_exists
# ---------------------------------------------------------------------------

def create_table_if_not_exists(
    spark: SparkSession,
    df: DataFrame,
    table_name: str,
    partition_by: Optional[List[str]] = None,
    comment: Optional[str] = None,
) -> bool:
    """Create a Delta table from a DataFrame only if it does not already exist.

    Args:
        spark: Active ``SparkSession``.
        df: ``DataFrame`` whose schema (and data) will be used to create the
            table.
        table_name: Fully-qualified target table name.
        partition_by: Partition columns for the new table.
        comment: Optional table-level comment / description.

    Returns:
        ``True`` if the table was created, ``False`` if it already existed.
    """
    if table_exists(spark, table_name):
        logger.info(
            "create_table_if_not_exists: '%s' already exists — skipping creation.", table_name
        )
        return False

    writer = df.write.format("delta").mode("errorifexists")

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.saveAsTable(table_name)

    if comment:
        spark.sql(f"COMMENT ON TABLE {table_name} IS '{comment}'")

    logger.info("create_table_if_not_exists: created table '%s'.", table_name)
    return True


# ---------------------------------------------------------------------------
# 11. backup_table
# ---------------------------------------------------------------------------

def backup_table(
    spark: SparkSession,
    source_table: str,
    backup_suffix: Optional[str] = None,
) -> str:
    """Create a backup copy of a Delta table with a timestamp suffix.

    The backup table is created in the same catalog/schema as the source.

    Args:
        spark: Active ``SparkSession``.
        source_table: Fully-qualified name of the table to back up.
        backup_suffix: Custom suffix appended to the backup table name.
            Defaults to ``_bkp_YYYYMMDD_HHMMSS`` (UTC).

    Returns:
        The fully-qualified name of the backup table that was created.

    Raises:
        ValueError: If the source table does not exist.
    """
    if not table_exists(spark, source_table):
        raise ValueError(f"backup_table: source table '{source_table}' does not exist.")

    if backup_suffix is None:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_suffix = f"_bkp_{timestamp}"

    backup_table_name = f"{source_table}{backup_suffix}"

    spark.sql(
        f"CREATE TABLE {backup_table_name} "
        f"DEEP CLONE {source_table}"
    )
    logger.info(
        "backup_table: '%s' cloned to '%s'.", source_table, backup_table_name
    )
    return backup_table_name


# ---------------------------------------------------------------------------
# 12. list_tables
# ---------------------------------------------------------------------------

def list_tables(
    spark: SparkSession,
    database: Optional[str] = None,
    pattern: Optional[str] = None,
) -> List[Dict[str, str]]:
    """List tables in a database, optionally filtered by a glob-style pattern.

    Args:
        spark: Active ``SparkSession``.
        database: Database / schema name (two-part ``catalog.schema`` or bare
            ``schema``).  Defaults to the current database.
        pattern: Glob-style filter applied to the bare table name, e.g.
            ``"sales_*"`` or ``"*_staging"``.

    Returns:
        A list of dictionaries, each with keys:

        - ``catalog`` (str): Catalog name.
        - ``database`` (str): Schema / database name.
        - ``name`` (str): Table name.
        - ``table_type`` (str): e.g. ``"MANAGED"``, ``"EXTERNAL"``, ``"VIEW"``.
        - ``is_temporary`` (bool as str): Whether the table is a temp view.
    """
    if database:
        rows = spark.catalog.listTables(database)
    else:
        rows = spark.catalog.listTables()

    results: List[Dict[str, str]] = []
    for row in rows:
        if pattern and not fnmatch.fnmatch(row.name, pattern):
            continue
        results.append(
            {
                "catalog": getattr(row, "catalog", "") or "",
                "database": row.database or "",
                "name": row.name,
                "table_type": row.tableType or "",
                "is_temporary": str(row.isTemporary),
            }
        )

    logger.info(
        "list_tables: found %d table(s) in database='%s' matching pattern='%s'.",
        len(results), database or "<current>", pattern or "*",
    )
    return results
