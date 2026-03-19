"""Shared fixtures for dbx_toolkit tests."""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Return the active SparkSession (Databricks) or create a local one."""
    # Try to use existing session (Databricks)
    existing = SparkSession.getActiveSession()
    if existing is not None:
        return existing

    # Fall back to local session for local testing
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("dbx_toolkit_tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse-test")
        .config("spark.driver.extraJavaOptions", "-Dderby.system.home=/tmp/derby-test")
        .getOrCreate()
    )
