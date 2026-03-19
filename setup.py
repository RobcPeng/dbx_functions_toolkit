"""
Setup for building dbx_toolkit as a wheel.

Build the wheel:
    cd dbx_toolkit
    python setup.py bdist_wheel

The wheel will be in dist/dbx_toolkit-0.1.0-py3-none-any.whl

Upload to a Unity Catalog Volume:
    databricks fs cp dist/dbx_toolkit-0.1.0-py3-none-any.whl \
        dbfs:/Volumes/catalog/schema/volume/wheels/dbx_toolkit-0.1.0-py3-none-any.whl

Install in a notebook:
    %pip install /Volumes/catalog/schema/volume/wheels/dbx_toolkit-0.1.0-py3-none-any.whl
"""

from setuptools import setup, find_packages

setup(
    name="dbx_toolkit",
    version="0.1.0",
    description="Reusable PySpark utility library for Databricks POCs",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],  # All deps are pre-installed on Databricks
)
