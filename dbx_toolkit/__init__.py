"""
dbx_toolkit - Reusable PySpark utility library for Databricks POCs.

A collection of production-ready helper functions for data profiling,
cleansing, feature engineering, ML workflows, I/O operations, datetime
handling, MLflow integration, and display formatting.

Usage on Databricks:
    # From a Unity Catalog Volume
    import sys
    sys.path.insert(0, "/Volumes/catalog/schema/volume/dbx_toolkit")
    from dbx_toolkit import data_profiling, data_cleansing, ml_utils

    # Or install as a wheel
    %pip install /Volumes/catalog/schema/volume/dbx_toolkit-0.1.0-py3-none-any.whl
    from dbx_toolkit import data_profiling
"""

__version__ = "0.1.0"

__all__ = [
    "data_profiling",
    "data_cleansing",
    "feature_engineering",
    "ml_utils",
    "io_utils",
    "datetime_utils",
    "mlflow_utils",
    "display_utils",
]
