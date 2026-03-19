"""Shared fixtures for dbx_toolkit tests."""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Return the active SparkSession (Databricks) or create a local one."""
    existing = SparkSession.getActiveSession()
    if existing is not None:
        return existing

    return (
        SparkSession.builder
        .master("local[2]")
        .appName("dbx_toolkit_tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .getOrCreate()
    )
