# dbx_toolkit

A reusable PySpark utility library for Databricks POCs. Sync it via Git folder, and get a head start on any new project.

## Installation

### Recommended: Databricks Git Folder (Git Sync)

1. Push this repo to GitHub (or any Git provider connected to your Databricks workspace).

2. In Databricks, go to **Workspace > Users > your_user > Git folders** and click **Add Git folder**.

3. Paste the repo URL and click **Create Git folder**.

4. In the Databricks sidebar, right-click your Git folder and select **Copy path**. It will look something like:
   ```
   /Workspace/Users/<your_username>/dbx_toolkit
   ```

5. In any notebook, paste the path into `sys.path.insert` as the first cell, then import:

```python
# Cell 1 — Add the Git folder to the Python path
import sys
sys.path.insert(0, "/Workspace/Users/<your_username>/dbx_toolkit")  # paste your full path here
```

```python
# Cell 2 — Import and use
from dbx_toolkit.data_profiling import profile_table, null_report
from dbx_toolkit.ml_utils import compare_models, classification_report
```

6. To update, open the Git folder in Databricks and click **Pull** (or it syncs automatically if you configure auto-pull).

### Alternative Deployment Options

<details>
<summary><b>Wheel on a Volume</b> (best for Jobs, pipelines, and serverless)</summary>

Build locally:

```bash
cd dbx_toolkit
pip install wheel setuptools
python setup.py bdist_wheel
```

Upload to a Volume:

```bash
databricks fs cp dist/dbx_toolkit-0.1.0-py3-none-any.whl \
    dbfs:/Volumes/catalog/schema/volume/wheels/dbx_toolkit-0.1.0-py3-none-any.whl
```

Install in a notebook:

```python
%pip install /Volumes/catalog/schema/volume/wheels/dbx_toolkit-0.1.0-py3-none-any.whl
```

**When to use wheels over Git sync:**
- Databricks Jobs and Workflows (Git folders aren't available in job tasks by default)
- Lakeflow Declarative Pipelines (DLT)
- Serverless compute where you need pinned versions
- Sharing across workspaces that don't have Git provider access
- CI/CD pipelines where you want immutable, versioned artifacts

</details>

<details>
<summary><b>Cluster-scoped library</b> (classic compute, team-wide)</summary>

Upload the wheel via **Compute > your_cluster > Libraries > Install New > Upload** so it's available on every notebook attached to that cluster.

</details>

<details>
<summary><b>sys.path</b> (quick and dirty)</summary>

Upload the folder to a Volume and add to path:

```python
import sys
sys.path.insert(0, "/Volumes/my_catalog/my_schema/libs")
from dbx_toolkit import data_profiling
```

Not recommended for anything beyond a quick throwaway POC.

</details>

## Best Practices for Shared Libraries on Databricks

### Repo Structure

Keep utility libraries **separate from notebooks**. Notebooks go in their own repo or folder; shared libs get their own repo that multiple projects import from. This avoids coupling POC-specific notebooks with reusable code.

```
# Good — library is its own repo
github.com/your-org/dbx_toolkit     ← this repo (shared lib)
github.com/your-org/churn-poc       ← POC repo, imports dbx_toolkit
github.com/your-org/fraud-detection ← another POC, same import

# Bad — everything in one giant repo
github.com/your-org/data-science/
├── notebooks/
├── utils/
└── poc_1/
```

### Versioning Strategy

- **Git sync (interactive development):** Always works off `main` or a release branch. Pull to get latest.
- **Wheels (production/jobs):** Tag releases (`v0.1.0`, `v0.2.0`), build wheels from tags, upload versioned wheels to a Volume. This gives you reproducibility — a job always runs the same version.

```
/Volumes/catalog/schema/libs/
├── dbx_toolkit-0.1.0-py3-none-any.whl   ← stable
├── dbx_toolkit-0.2.0-py3-none-any.whl   ← latest
└── dbx_toolkit-latest-py3-none-any.whl  ← symlink/copy for convenience
```

### Git Sync vs Wheels: When to Use Which

| Scenario | Use Git Sync | Use Wheel |
|----------|:------------:|:---------:|
| Interactive notebook development | Yes | |
| Quick POCs and exploration | Yes | |
| Databricks Jobs / Workflows | | Yes |
| Lakeflow Declarative Pipelines (DLT) | | Yes |
| Serverless with pinned versions | | Yes |
| Multi-workspace sharing | | Yes |
| CI/CD with immutable artifacts | | Yes |
| Rapid iteration during development | Yes | |
| Production with audit requirements | | Yes |

### Import Pattern for Notebooks

Add a standard preamble at the top of every POC notebook:

```python
# === Setup ===
# Option A: Git folder (interactive)
from dbx_toolkit.data_profiling import profile_table, null_report
from dbx_toolkit.data_cleansing import clean_column_names, deduplicate
from dbx_toolkit.ml_utils import split_data, compare_models

# Option B: Wheel (jobs/production)
# %pip install /Volumes/catalog/schema/libs/dbx_toolkit-0.1.0-py3-none-any.whl
# from dbx_toolkit.data_profiling import profile_table, null_report
```

### Testing

**Locally** (before pushing):

```bash
cd dbx_toolkit
pip install pyspark pytest
python -m pytest tests/ -v
```

**In Databricks** — run all tests from a notebook cell:

```python
import os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # prevents __pycache__ errors on Workspace filesystem

sys.path.insert(0, "/Workspace/Users/<your_username>/dbx_toolkit")

import pytest
pytest.main([
    "/Workspace/Users/<your_username>/dbx_toolkit/tests/",
    "-v", "--tb=short", "-p", "no:cacheprovider"
])
```

Run a specific test file:

```python
import os, sys
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

sys.path.insert(0, "/Workspace/Users/<your_username>/dbx_toolkit")

import pytest
pytest.main([
    "/Workspace/Users/<your_username>/dbx_toolkit/tests/test_data_profiling.py",
    "-v", "-p", "no:cacheprovider"
])
```

> **Note:** Tests run in-process (not via `subprocess`) to reuse the existing Databricks SparkSession. `PYTHONDONTWRITEBYTECODE=1` prevents `__pycache__` writes that the Workspace filesystem doesn't support.

For Delta/Unity Catalog tests (`merge_into`, `scd_type2`, `backup_table`, etc.), run directly in a notebook against a test catalog.

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/release.yml
name: Build and Upload Wheel
on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install wheel setuptools
      - run: python setup.py bdist_wheel
      - name: Upload to Databricks Volume
        run: |
          databricks fs cp dist/*.whl \
            dbfs:/Volumes/catalog/schema/libs/ \
            --overwrite
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
```

## Modules

For detailed examples with sample inputs and outputs for every function, see **[EXAMPLES.md](EXAMPLES.md)**.

### `data_profiling` — Understand your data fast

```python
from dbx_toolkit.data_profiling import profile_table, null_report, correlation_matrix

# Full column-level profile
profile_table(df).display()

# Null analysis
null_report(df).display()

# Correlation matrix for numeric columns
correlation_matrix(df, ["age", "salary", "tenure"]).display()

# Outlier detection
detect_outliers_iqr(df, "salary").filter("is_outlier").display()

# Compare two DataFrames
result = compare_dataframes(df_old, df_new, key_columns=["id"])
result["only_in_left"].display()
result["changed"].display()
```

### `data_cleansing` — Clean and validate data

```python
from dbx_toolkit.data_cleansing import (
    clean_column_names, deduplicate, standardize_strings,
    remove_outliers, split_valid_invalid
)

# Clean column names (lowercase, underscores, no special chars)
df = clean_column_names(df)

# Deduplicate keeping the latest record per customer
df = deduplicate(df, subset=["customer_id"], order_by="updated_at", keep="last")

# Trim and normalize string columns
df = standardize_strings(df, columns=["name", "email"])

# Validate and split into valid/invalid
rules = {
    "age": "col('age').between(0, 120)",
    "email": "col('email').rlike('@')"
}
valid_df, invalid_df = split_valid_invalid(df, rules)
```

### `feature_engineering` — Build features for ML

```python
from dbx_toolkit.feature_engineering import (
    prepare_features, add_date_features, add_lag_features,
    add_rolling_features, add_missing_indicators
)

# End-to-end feature preparation
transformed_df, pipeline_model = prepare_features(
    df,
    numeric_cols=["age", "salary", "tenure"],
    categorical_cols=["department", "region"],
    label_col="churn",
    impute_strategy="median",
    scale_method="standard"
)

# Apply same transformations to test data
test_transformed = pipeline_model.transform(test_df)

# Date features
df = add_date_features(df, "order_date")

# Lag + rolling features for time series
df = add_lag_features(df, "sales", partition_by="store_id", order_by="date", lags=[1, 7, 30])
df = add_rolling_features(df, "sales", partition_by="store_id", order_by="date", windows=[7, 30])
```

### `ml_utils` — Train, evaluate, compare models

```python
from dbx_toolkit.ml_utils import (
    split_data, compare_models, classification_report,
    feature_importance, handle_class_imbalance
)

# Stratified train/test split
train_df, test_df = split_data(df, ratios=[0.8, 0.2], stratify_col="label")

# Handle class imbalance
train_df = handle_class_imbalance(train_df, strategy="oversample")

# Compare multiple models side by side
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
results = compare_models(
    {"RF": rf_pipeline, "GBT": gbt_pipeline},
    train_df, test_df,
    task="classification"
)
results.display()

# Detailed classification report
report = classification_report(predictions)
print(report["metrics"])
report["confusion_matrix"].display()

# Feature importances
feature_importance(model, feature_names).display()
```

### `io_utils` — Read, write, merge Delta tables

```python
from dbx_toolkit.io_utils import (
    read_table_safe, write_table, merge_into,
    table_exists, get_table_info, backup_table
)

# Safe table read with filter
df = read_table_safe(spark, "catalog.schema.customers", filter_expr="status = 'active'")

# Write with auto-optimize
write_table(df, "catalog.schema.output", partition_by=["region"])

# Upsert (Delta MERGE)
merge_into(spark, "catalog.schema.customers", new_data, merge_keys=["customer_id"])

# Backup before a risky operation
backup_table(spark, "catalog.schema.customers")
```

### `datetime_utils` — Date and time operations

```python
from dbx_toolkit.datetime_utils import (
    parse_dates, add_date_parts, generate_date_spine,
    fill_date_gaps, add_fiscal_year, bucket_time_of_day
)

# Parse messy date strings (tries multiple formats)
df = parse_dates(df, "date_str")

# Generate a date spine for gap-free time series
spine = generate_date_spine(spark, "2024-01-01", "2024-12-31")

# Fill gaps in daily data
df = fill_date_gaps(df, "event_date", partition_cols=["store_id"], value_cols=["sales"])

# Fiscal year calculation
df = add_fiscal_year(df, "order_date", fiscal_start_month=7)

# Categorize by time of day
df = bucket_time_of_day(df, "event_timestamp")
```

### `mlflow_utils` — MLflow experiment tracking

```python
from dbx_toolkit.mlflow_utils import (
    setup_experiment, log_run, log_grid_search,
    get_best_run, promote_model
)

# Set up experiment
setup_experiment("/Users/me/churn_prediction", tags={"project": "churn"})

# Log a complete run in one call
log_run(
    model=pipeline_model,
    metrics={"accuracy": 0.95, "f1": 0.93},
    params={"model": "rf", "n_trees": 100},
    model_name="catalog.schema.churn_model"
)

# Log grid search results as nested runs
log_grid_search(cv_model, param_grid)

# Find and promote the best model
best = get_best_run(metric="metrics.f1")
promote_model("catalog.schema.churn_model", alias="champion")
```

### `display_utils` — Display and format data

```python
from dbx_toolkit.display_utils import (
    peek, summary_table, top_n_by_group, histogram_data
)

# Quick peek at a DataFrame
peek(df)

# Full summary table
summary_table(df).display()

# Top 5 products per category
top_n_by_group(df, "category", "revenue", n=5).display()

# Histogram bins for charting
histogram_data(df, "salary", n_bins=20).display()
```

## Serverless Compatibility

All modules are designed to work on serverless compute:

- No native/JVM dependencies — pure Python + PySpark
- No SparkSession creation — uses the existing `spark` object
- No local filesystem writes — all persistent output goes to Volumes or MLflow
- Wheel install works with `%pip install` on serverless
- Git sync works on serverless notebooks

## Project Structure

```
dbx_toolkit/
├── README.md
├── setup.py                       # For building wheels
├── pyproject.toml
└── dbx_toolkit/
    ├── __init__.py
    ├── data_profiling.py          # Table profiling, null analysis, outlier detection
    ├── data_cleansing.py          # Column cleaning, dedup, validation, outlier handling
    ├── feature_engineering.py     # Encoding, scaling, binning, lag/rolling features
    ├── ml_utils.py                # Train/test split, model comparison, evaluation
    ├── io_utils.py                # Delta read/write, merge, SCD Type 2, backups
    ├── datetime_utils.py          # Date parsing, date spine, fiscal year, time buckets
    ├── mlflow_utils.py            # Experiment setup, run logging, model promotion
    └── display_utils.py           # Peek, summary tables, histogram data, formatting
```

## Requirements

- Databricks Runtime 13.0+ (all dependencies pre-installed)
- PySpark 3.4+
- MLflow (pre-installed on Databricks)
- No additional pip installs needed

## License

MIT
