# dbx_toolkit — Function Examples with Inputs & Outputs

All examples assume this setup:

```python
import sys
sys.path.insert(0, "/Workspace/Users/<your_username>/dbx_toolkit")

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
```

---

## data_profiling

### `profile_table(df, sample_size=None)`

```python
from dbx_toolkit.data_profiling import profile_table

df = spark.createDataFrame([
    ("Alice", 30, 75000.0),
    ("Bob",   25, None),
    ("Carol", None, 90000.0),
], ["name", "age", "salary"])

profile_table(df).display()
```

**Output:**

| column | data_type | non_null_count | null_count | null_pct | distinct_count | min   | max   | mean    | stddev  | sample_values         |
|--------|-----------|---------------|------------|----------|----------------|-------|-------|---------|---------|----------------------|
| name   | string    | 3             | 0          | 0.0      | 3              | Alice | Carol | null    | null    | [Alice, Bob, Carol]  |
| age    | bigint    | 2             | 1          | 33.33    | 2              | 25    | 30    | 27.5    | 3.54    | [30, 25]             |
| salary | double    | 2             | 1          | 33.33    | 2              | 75000 | 90000 | 82500.0 | 10606.6 | [75000.0, 90000.0]   |

---

### `null_report(df)`

```python
from dbx_toolkit.data_profiling import null_report

null_report(df).display()
```

**Output:**

| column | null_count | null_pct |
|--------|-----------|----------|
| age    | 1         | 33.33    |
| salary | 1         | 33.33    |
| name   | 0         | 0.0      |

---

### `cardinality_report(df)`

```python
from dbx_toolkit.data_profiling import cardinality_report

cardinality_report(df).display()
```

**Output:**

| column | distinct_count | total_count | cardinality_ratio |
|--------|---------------|-------------|-------------------|
| name   | 3             | 3           | 1.0               |
| age    | 2             | 3           | 0.67              |
| salary | 2             | 3           | 0.67              |

---

### `value_counts(df, column, top_n=20)`

```python
from dbx_toolkit.data_profiling import value_counts

df = spark.createDataFrame([
    ("East",), ("East",), ("West",), ("West",), ("West",), ("North",)
], ["region"])

value_counts(df, "region").display()
```

**Output:**

| region | count | pct   |
|--------|-------|-------|
| West   | 3     | 50.0  |
| East   | 2     | 33.33 |
| North  | 1     | 16.67 |

---

### `numeric_summary(df, columns=None)`

```python
from dbx_toolkit.data_profiling import numeric_summary

df = spark.createDataFrame([
    (25, 50000.0), (30, 60000.0), (35, 75000.0),
    (40, 90000.0), (28, 55000.0),
], ["age", "salary"])

numeric_summary(df).display()
```

**Output:**

| column | mean  | stddev | min   | max   | p25   | p50   | p75   | skewness | kurtosis |
|--------|-------|--------|-------|-------|-------|-------|-------|----------|----------|
| age    | 31.6  | 5.94   | 25    | 40    | 28    | 30    | 35    | 0.57     | -1.15    |
| salary | 66000 | 16432  | 50000 | 90000 | 55000 | 60000 | 75000 | 0.54     | -1.31    |

---

### `correlation_matrix(df, columns=None)`

```python
from dbx_toolkit.data_profiling import correlation_matrix

correlation_matrix(df, ["age", "salary"]).display()
```

**Output:**

| column_1 | column_2 | correlation |
|----------|----------|-------------|
| age      | salary   | 0.9856      |

---

### `detect_outliers_iqr(df, column, factor=1.5)`

```python
from dbx_toolkit.data_profiling import detect_outliers_iqr

df = spark.createDataFrame(
    [(x,) for x in [10, 12, 14, 13, 15, 100]], ["value"]
)

detect_outliers_iqr(df, "value").display()
```

**Output:**

| value | is_outlier | lower_bound | upper_bound |
|-------|-----------|-------------|-------------|
| 10    | false     | 7.0         | 19.5        |
| 12    | false     | 7.0         | 19.5        |
| 14    | false     | 7.0         | 19.5        |
| 13    | false     | 7.0         | 19.5        |
| 15    | false     | 7.0         | 19.5        |
| 100   | true      | 7.0         | 19.5        |

---

### `compare_dataframes(df1, df2, key_columns)`

```python
from dbx_toolkit.data_profiling import compare_dataframes

df1 = spark.createDataFrame([(1, "Alice"), (2, "Bob"), (3, "Carol")], ["id", "name"])
df2 = spark.createDataFrame([(1, "Alice"), (2, "Robert"), (4, "Dave")], ["id", "name"])

result = compare_dataframes(df1, df2, key_columns=["id"])

result["only_in_left"].display()   # rows only in df1
result["only_in_right"].display()  # rows only in df2
result["changed"].display()        # same key, different values
```

**Output — only_in_left:**

| id | name  |
|----|-------|
| 3  | Carol |

**Output — only_in_right:**

| id | name |
|----|------|
| 4  | Dave |

**Output — changed:**

| id | name   |
|----|--------|
| 2  | Bob    |

---

### `schema_diff(df1, df2)`

```python
from dbx_toolkit.data_profiling import schema_diff

df1 = spark.createDataFrame([(1, "a", 1.0)], ["id", "name", "score"])
df2 = spark.createDataFrame([(1, 100)], ["id", "score"])

schema_diff(df1, df2).display()
```

**Output:**

| column | status       | type_df1 | type_df2 |
|--------|-------------|----------|----------|
| name   | removed     | string   | null     |
| score  | type_changed| double   | bigint   |
| id     | unchanged   | bigint   | bigint   |

---

### `data_quality_score(df)`

```python
from dbx_toolkit.data_profiling import data_quality_score

data_quality_score(df).display()
```

**Output:**

| completeness | uniqueness | validity | overall_score |
|-------------|-----------|----------|---------------|
| 88.89       | 77.78     | 100.0    | 88.89         |

---

## data_cleansing

### `clean_column_names(df)`

```python
from dbx_toolkit.data_cleansing import clean_column_names

df = spark.createDataFrame([(1, 2, 3)], ["First Name", "Last  Name!!", "AGE"])
df = clean_column_names(df)
print(df.columns)
```

**Output:**
```
['first_name', 'last_name', 'age']
```

---

### `drop_null_columns(df, threshold=1.0)`

```python
from dbx_toolkit.data_cleansing import drop_null_columns

df = spark.createDataFrame([
    (1, None, "a"), (2, None, "b"), (3, None, None)
], ["id", "empty_col", "status"])

df = drop_null_columns(df, threshold=1.0)  # drop columns that are 100% null
print(df.columns)
```

**Output:**
```
['id', 'status']
```

---

### `drop_constant_columns(df)`

```python
from dbx_toolkit.data_cleansing import drop_constant_columns

df = spark.createDataFrame([
    (1, "active", "US"), (2, "active", "UK"), (3, "active", "FR")
], ["id", "status", "country"])

df = drop_constant_columns(df)
print(df.columns)
```

**Output:**
```
['id', 'country']   # 'status' dropped — only one distinct value
```

---

### `fill_nulls_by_type(df, numeric_fill=0, string_fill="unknown")`

```python
from dbx_toolkit.data_cleansing import fill_nulls_by_type

df = spark.createDataFrame([
    (1, None, "a"), (None, 100.0, None)
], ["id", "salary", "name"])

fill_nulls_by_type(df, numeric_fill=0, string_fill="unknown").display()
```

**Output:**

| id | salary | name    |
|----|--------|---------|
| 1  | 0.0    | a       |
| 0  | 100.0  | unknown |

---

### `deduplicate(df, subset=None, order_by=None, keep="first")`

```python
from dbx_toolkit.data_cleansing import deduplicate

df = spark.createDataFrame([
    (1, "2024-01-01", "Alice"),
    (1, "2024-06-15", "Alice Updated"),
    (2, "2024-03-01", "Bob"),
], ["customer_id", "updated_at", "name"])

deduplicate(df, subset=["customer_id"], order_by="updated_at", keep="last").display()
```

**Output:**

| customer_id | updated_at | name          |
|-------------|-----------|---------------|
| 1           | 2024-06-15| Alice Updated |
| 2           | 2024-03-01| Bob           |

---

### `standardize_strings(df, columns=None)`

```python
from dbx_toolkit.data_cleansing import standardize_strings

df = spark.createDataFrame([
    ("  Alice  SMITH  ",), ("  BOB jones",)
], ["name"])

standardize_strings(df, columns=["name"]).display()
```

**Output:**

| name        |
|-------------|
| alice smith |
| bob jones   |

---

### `remove_outliers(df, column, method="iqr")`

```python
from dbx_toolkit.data_cleansing import remove_outliers

df = spark.createDataFrame([(x,) for x in [10, 12, 14, 13, 15, 200]], ["salary"])

remove_outliers(df, "salary", method="iqr").display()
```

**Output:**

| salary |
|--------|
| 10     |
| 12     |
| 14     |
| 13     |
| 15     |

---

### `cap_outliers(df, column)`

```python
from dbx_toolkit.data_cleansing import cap_outliers

df = spark.createDataFrame([(x,) for x in [10, 12, 14, 13, 15, 200]], ["salary"])

cap_outliers(df, "salary", method="iqr").display()
```

**Output:**

| salary |
|--------|
| 10     |
| 12     |
| 14     |
| 13     |
| 15     |
| 19.5   |

(200 was capped to the upper IQR bound)

---

### `validate_schema(df, expected_schema)`

```python
from dbx_toolkit.data_cleansing import validate_schema

df = spark.createDataFrame([(1, "Alice", 30.0)], ["id", "name", "age"])

result = validate_schema(df, {"id": "bigint", "name": "string", "email": "string", "age": "int"})
print(result)
```

**Output:**
```python
{
    "missing_cols": ["email"],
    "extra_cols": [],
    "type_mismatches": {"age": {"expected": "int", "actual": "double"}}
}
```

---

### `enforce_types(df, type_map)`

```python
from dbx_toolkit.data_cleansing import enforce_types

df = spark.createDataFrame([("25", "100.50"), ("bad", "200.00")], ["age", "price"])

enforce_types(df, {"age": "integer", "price": "double"}).display()
```

**Output:**

| age  | price  |
|------|--------|
| 25   | 100.5  |
| null | 200.0  |

("bad" couldn't cast to integer → null)

---

### `normalize_values(df, column, mapping)`

```python
from dbx_toolkit.data_cleansing import normalize_values

df = spark.createDataFrame([("Y",), ("yes",), ("YES",), ("N",), ("no",)], ["active"])

normalize_values(df, "active", {"Y": "yes", "YES": "yes", "N": "no", "NO": "no"}).display()
```

**Output:**

| active |
|--------|
| yes    |
| yes    |
| yes    |
| no     |
| no     |

---

### `split_valid_invalid(df, rules)`

```python
from dbx_toolkit.data_cleansing import split_valid_invalid

df = spark.createDataFrame([
    (25, "a@b.com"), (150, "bad"), (30, "c@d.com")
], ["age", "email"])

rules = {
    "age": "col('age').between(0, 120)",
    "email": "col('email').rlike('@')"
}

valid_df, invalid_df = split_valid_invalid(df, rules)
valid_df.display()
invalid_df.display()
```

**Output — valid_df:**

| age | email   |
|-----|---------|
| 25  | a@b.com |
| 30  | c@d.com |

**Output — invalid_df:**

| age | email |
|-----|-------|
| 150 | bad   |

---

## feature_engineering

### `encode_categoricals(df, columns, method="index")`

```python
from dbx_toolkit.feature_engineering import encode_categoricals

df = spark.createDataFrame([
    ("red", "S"), ("blue", "M"), ("red", "L"), ("green", "S")
], ["color", "size"])

encoded_df, stages = encode_categoricals(df, ["color", "size"], method="index")
encoded_df.display()
```

**Output:**

| color | size | color_idx | size_idx |
|-------|------|-----------|----------|
| red   | S    | 0.0       | 0.0      |
| blue  | M    | 1.0       | 1.0      |
| red   | L    | 0.0       | 2.0      |
| green | S    | 2.0       | 0.0      |

---

### `scale_features(df, input_col, output_col, method="standard")`

```python
from dbx_toolkit.feature_engineering import scale_features, assemble_features

df = spark.createDataFrame([(25, 50000.0), (30, 60000.0), (35, 75000.0)], ["age", "salary"])
df = assemble_features(df, ["age", "salary"], output_col="features")

scaled_df, scaler = scale_features(df, "features", "scaled_features", method="standard")
scaled_df.select("age", "salary", "scaled_features").display()
```

**Output:**

| age | salary | scaled_features      |
|-----|--------|---------------------|
| 25  | 50000  | [-1.0, -0.87]       |
| 30  | 60000  | [0.0, -0.07]        |
| 35  | 75000  | [1.0, 1.13]         |

---

### `create_bins(df, column, method="quantile", n_bins=4)`

```python
from dbx_toolkit.feature_engineering import create_bins

df = spark.createDataFrame([(x,) for x in range(1, 101)], ["score"])

create_bins(df, "score", method="quantile", n_bins=4).groupBy("score_binned").count().orderBy("score_binned").display()
```

**Output:**

| score_binned | count |
|-------------|-------|
| 0.0         | 25    |
| 1.0         | 25    |
| 2.0         | 25    |
| 3.0         | 25    |

---

### `add_date_features(df, date_col)`

```python
from dbx_toolkit.feature_engineering import add_date_features

df = spark.createDataFrame([("2024-03-15",), ("2024-12-25",)], ["order_date"])
df = df.withColumn("order_date", F.to_date("order_date"))

add_date_features(df, "order_date").display()
```

**Output:**

| order_date | order_date_year | order_date_month | order_date_day | order_date_quarter | order_date_day_of_week | order_date_is_weekend |
|-----------|----------------|-----------------|---------------|-------------------|----------------------|----------------------|
| 2024-03-15| 2024           | 3               | 15            | 1                 | 6                    | false                |
| 2024-12-25| 2024           | 12              | 25            | 4                 | 4                    | false                |

---

### `add_lag_features(df, column, partition_by, order_by, lags=[1, 7, 30])`

```python
from dbx_toolkit.feature_engineering import add_lag_features

df = spark.createDataFrame([
    ("A", "2024-01-01", 100), ("A", "2024-01-02", 110),
    ("A", "2024-01-03", 105), ("A", "2024-01-04", 120),
], ["store", "date", "sales"])

add_lag_features(df, "sales", partition_by="store", order_by="date", lags=[1, 2]).display()
```

**Output:**

| store | date       | sales | sales_lag_1 | sales_lag_2 |
|-------|-----------|-------|-------------|-------------|
| A     | 2024-01-01| 100   | null        | null        |
| A     | 2024-01-02| 110   | 100         | null        |
| A     | 2024-01-03| 105   | 110         | 100         |
| A     | 2024-01-04| 120   | 105         | 110         |

---

### `add_rolling_features(df, column, partition_by, order_by, windows=[7, 30])`

```python
from dbx_toolkit.feature_engineering import add_rolling_features

add_rolling_features(df, "sales", partition_by="store", order_by="date",
                     windows=[3], funcs=["avg", "max"]).display()
```

**Output:**

| store | date       | sales | sales_rolling_avg_3 | sales_rolling_max_3 |
|-------|-----------|-------|--------------------|--------------------|
| A     | 2024-01-01| 100   | 100.0              | 100                |
| A     | 2024-01-02| 110   | 105.0              | 110                |
| A     | 2024-01-03| 105   | 105.0              | 110                |
| A     | 2024-01-04| 120   | 111.67             | 120                |

---

### `add_missing_indicators(df, columns=None)`

```python
from dbx_toolkit.feature_engineering import add_missing_indicators

df = spark.createDataFrame([(1, None), (None, 50.0), (3, 75.0)], ["age", "salary"])

add_missing_indicators(df).display()
```

**Output:**

| age  | salary | age_missing | salary_missing |
|------|--------|-------------|----------------|
| 1    | null   | 0           | 1              |
| null | 50.0   | 1           | 0              |
| 3    | 75.0   | 0           | 0              |

---

### `impute_columns(df, strategy="median", columns=None)`

```python
from dbx_toolkit.feature_engineering import impute_columns

df = spark.createDataFrame([(1, 50.0), (None, 60.0), (3, None), (4, 80.0)], ["age", "salary"])

imputed_df, impute_values = impute_columns(df, strategy="median")
imputed_df.display()
print(impute_values)
```

**Output:**

| age | salary |
|-----|--------|
| 1   | 50.0   |
| 3   | 60.0   |
| 3   | 65.0   |
| 4   | 80.0   |

```python
{"age": 3.0, "salary": 65.0}  # median values used for imputation
```

---

### `prepare_features(df, numeric_cols, categorical_cols, label_col=None)`

```python
from dbx_toolkit.feature_engineering import prepare_features

df = spark.createDataFrame([
    (25, 50000.0, "East", 1), (30, None, "West", 0),
    (None, 75000.0, "East", 1), (40, 90000.0, "West", 0),
], ["age", "salary", "region", "label"])

transformed_df, pipeline_model = prepare_features(
    df,
    numeric_cols=["age", "salary"],
    categorical_cols=["region"],
    label_col="label"
)
transformed_df.select("label", "features").display()
```

**Output:**

| label | features                    |
|-------|-----------------------------|
| 1     | [25.0, 50000.0, 0.0]       |
| 0     | [30.0, 71666.7, 1.0]       |
| 1     | [31.67, 75000.0, 0.0]      |
| 0     | [40.0, 90000.0, 1.0]       |

(nulls imputed, categoricals encoded, features assembled)

---

## ml_utils

### `split_data(df, ratios=[0.8, 0.2], stratify_col=None)`

```python
from dbx_toolkit.ml_utils import split_data

train_df, test_df = split_data(df, ratios=[0.8, 0.2], seed=42)
print(f"Train: {train_df.count()}, Test: {test_df.count()}")

# Stratified split
train_df, test_df = split_data(df, ratios=[0.8, 0.2], stratify_col="label")

# Three-way split
train_df, val_df, test_df = split_data(df, ratios=[0.7, 0.15, 0.15])
```

**Output:**
```
Train: 800, Test: 200
```

---

### `compare_models(models_dict, train_df, test_df, task="classification")`

```python
from dbx_toolkit.ml_utils import compare_models
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["age", "salary"], outputCol="features")
rf_pipeline = Pipeline(stages=[assembler, RandomForestClassifier(labelCol="label")])
lr_pipeline = Pipeline(stages=[assembler, LogisticRegression(labelCol="label")])

results = compare_models(
    {"Random Forest": rf_pipeline, "Logistic Regression": lr_pipeline},
    train_df, test_df,
    task="classification"
)
results.display()
```

**Output:**

| model               | accuracy | f1   | auc_roc | weighted_precision | weighted_recall |
|---------------------|----------|------|---------|-------------------|-----------------|
| Random Forest       | 0.92     | 0.91 | 0.96    | 0.92              | 0.92            |
| Logistic Regression | 0.88     | 0.87 | 0.93    | 0.88              | 0.88            |

---

### `classification_report(predictions)`

```python
from dbx_toolkit.ml_utils import classification_report

report = classification_report(predictions, label_col="label")
print(report["metrics"])
report["confusion_matrix"].display()
```

**Output — metrics:**
```python
{"accuracy": 0.92, "f1": 0.91, "weighted_precision": 0.92,
 "weighted_recall": 0.92, "auc_roc": 0.96, "auc_pr": 0.94}
```

**Output — confusion_matrix:**

| label | prediction | count |
|-------|-----------|-------|
| 0     | 0         | 85    |
| 0     | 1         | 8     |
| 1     | 0         | 7     |
| 1     | 1         | 100   |

---

### `regression_report(predictions)`

```python
from dbx_toolkit.ml_utils import regression_report

report = regression_report(predictions, label_col="price", prediction_col="prediction")
print(report)
```

**Output:**
```python
{"rmse": 12450.32, "mse": 155010462.5, "mae": 9876.54, "r2": 0.87, "mape": 0.12}
```

---

### `feature_importance(model, feature_names)`

```python
from dbx_toolkit.ml_utils import feature_importance

feature_importance(rf_model, ["age", "salary", "tenure", "region_idx"]).display()
```

**Output:**

| feature    | importance |
|-----------|-----------|
| salary    | 0.42      |
| age       | 0.28      |
| tenure    | 0.21      |
| region_idx| 0.09      |

---

### `handle_class_imbalance(df, label_col="label", strategy="oversample")`

```python
from dbx_toolkit.ml_utils import handle_class_imbalance

# Before
df.groupBy("label").count().display()
# | label | count |
# |-------|-------|
# | 0     | 900   |
# | 1     | 100   |

balanced_df = handle_class_imbalance(df, strategy="oversample")
balanced_df.groupBy("label").count().display()
```

**Output:**

| label | count |
|-------|-------|
| 0     | 900   |
| 1     | 900   |

---

### `find_best_threshold(predictions, label_col="label")`

```python
from dbx_toolkit.ml_utils import find_best_threshold

threshold_df, best_threshold = find_best_threshold(predictions)
print(f"Best threshold: {best_threshold}")
threshold_df.display()
```

**Output:**
```
Best threshold: 0.4
```

| threshold | precision | recall | f1   | tp  | fp  | fn  | tn  |
|-----------|----------|--------|------|-----|-----|-----|-----|
| 0.3       | 0.78     | 0.95   | 0.86 | 95  | 27  | 5   | 73  |
| 0.4       | 0.85     | 0.90   | 0.87 | 90  | 16  | 10  | 84  |
| 0.5       | 0.91     | 0.82   | 0.86 | 82  | 8   | 18  | 92  |
| 0.6       | 0.94     | 0.72   | 0.82 | 72  | 4   | 28  | 96  |

---

### `batch_predict(model_uri, df, output_table=None)`

```python
from dbx_toolkit.ml_utils import batch_predict

predictions = batch_predict(
    model_uri="models:/catalog.schema.my_model@champion",
    df=new_data,
    output_table="catalog.schema.predictions"
)
predictions.display()
```

**Output:**

| id  | age | salary | prediction |
|-----|-----|--------|-----------|
| 101 | 34  | 72000  | 1.0       |
| 102 | 28  | 55000  | 0.0       |

(Also saved to `catalog.schema.predictions`)

---

## io_utils

### `read_table_safe(spark, table_name, columns=None, filter_expr=None)`

```python
from dbx_toolkit.io_utils import read_table_safe

# Table exists
df = read_table_safe(spark, "catalog.schema.customers",
                     columns=["id", "name"], filter_expr="status = 'active'", limit=100)

# Table doesn't exist — returns None instead of throwing
df = read_table_safe(spark, "catalog.schema.nonexistent")
print(df)  # None
```

---

### `write_table(df, table_name, mode="overwrite")`

```python
from dbx_toolkit.io_utils import write_table

write_table(df, "catalog.schema.output", mode="overwrite", partition_by=["region"])
# Writes the table and runs OPTIMIZE automatically
```

---

### `merge_into(spark, target_table, source_df, merge_keys)`

```python
from dbx_toolkit.io_utils import merge_into

new_data = spark.createDataFrame([
    (1, "Alice Updated", "active"),
    (4, "Dave", "active"),
], ["id", "name", "status"])

merge_into(spark, "catalog.schema.customers", new_data, merge_keys=["id"])
# id=1: updated name to "Alice Updated"
# id=4: inserted as new row
```

---

### `table_exists(spark, table_name)`

```python
from dbx_toolkit.io_utils import table_exists

print(table_exists(spark, "catalog.schema.customers"))  # True
print(table_exists(spark, "catalog.schema.fake_table")) # False
```

---

### `get_table_info(spark, table_name)`

```python
from dbx_toolkit.io_utils import get_table_info

info = get_table_info(spark, "catalog.schema.customers")
print(info)
```

**Output:**
```python
{
    "row_count": 15000,
    "column_count": 8,
    "columns": ["id", "name", "email", "age", "status", "region", "created_at", "updated_at"],
    "size_bytes": 1048576,
    "partitioned_by": ["region"],
    "last_modified": "2024-03-15T10:30:00"
}
```

---

### `backup_table(spark, source_table)`

```python
from dbx_toolkit.io_utils import backup_table

backup_table(spark, "catalog.schema.customers")
# Creates: catalog.schema.customers_backup_20240315_103000
```

---

### `scd_type2(spark, target_table, source_df, key_columns, tracked_columns)`

```python
from dbx_toolkit.io_utils import scd_type2

scd_type2(
    spark,
    target_table="catalog.schema.dim_customers",
    source_df=new_customer_data,
    key_columns=["customer_id"],
    tracked_columns=["name", "email", "address"]
)
```

**Result in target table:**

| customer_id | name          | effective_date | end_date   | is_current |
|------------|---------------|---------------|------------|------------|
| 1          | Alice Smith   | 2024-01-01    | 2024-06-15 | false      |
| 1          | Alice Johnson | 2024-06-15    | null       | true       |
| 2          | Bob Jones     | 2024-03-01    | null       | true       |

---

## datetime_utils

### `parse_dates(df, column, formats=None)`

```python
from dbx_toolkit.datetime_utils import parse_dates

df = spark.createDataFrame([
    ("2024-03-15",), ("03/15/2024",), ("15-Mar-2024",), ("20240315",), ("garbage",)
], ["date_str"])

parse_dates(df, "date_str").display()
```

**Output:**

| date_str    |
|-------------|
| 2024-03-15  |
| 2024-03-15  |
| 2024-03-15  |
| 2024-03-15  |
| null        |

(All valid formats parsed to the same date; "garbage" becomes null)

---

### `add_date_parts(df, date_col)`

```python
from dbx_toolkit.datetime_utils import add_date_parts

df = spark.createDataFrame([("2024-03-15",)], ["event_date"])
df = df.withColumn("event_date", F.to_date("event_date"))

add_date_parts(df, "event_date").display()
```

**Output:**

| event_date | event_date_year | event_date_month | event_date_day | event_date_quarter | event_date_day_of_week | event_date_day_of_year | event_date_week_of_year |
|-----------|----------------|-----------------|---------------|-------------------|----------------------|----------------------|------------------------|
| 2024-03-15| 2024           | 3               | 15            | 1                 | 6                    | 75                   | 11                     |

---

### `generate_date_spine(spark, start_date, end_date, interval="day")`

```python
from dbx_toolkit.datetime_utils import generate_date_spine

spine = generate_date_spine(spark, "2024-01-01", "2024-01-05")
spine.display()
```

**Output:**

| date       |
|-----------|
| 2024-01-01|
| 2024-01-02|
| 2024-01-03|
| 2024-01-04|
| 2024-01-05|

---

### `fill_date_gaps(df, date_col, partition_cols=None, value_cols=None)`

```python
from dbx_toolkit.datetime_utils import fill_date_gaps

df = spark.createDataFrame([
    ("A", "2024-01-01", 100),
    ("A", "2024-01-03", 150),  # Jan 2 is missing
    ("A", "2024-01-05", 200),  # Jan 4 is missing
], ["store", "date", "sales"])

fill_date_gaps(df, "date", partition_cols=["store"], value_cols=["sales"], fill_value=0).display()
```

**Output:**

| store | date       | sales |
|-------|-----------|-------|
| A     | 2024-01-01| 100   |
| A     | 2024-01-02| 0     |
| A     | 2024-01-03| 150   |
| A     | 2024-01-04| 0     |
| A     | 2024-01-05| 200   |

---

### `add_fiscal_year(df, date_col, fiscal_start_month=7)`

```python
from dbx_toolkit.datetime_utils import add_fiscal_year

df = spark.createDataFrame([
    ("2024-03-15",), ("2024-08-20",), ("2024-12-01",)
], ["order_date"])
df = df.withColumn("order_date", F.to_date("order_date"))

add_fiscal_year(df, "order_date", fiscal_start_month=7).display()
```

**Output:**

| order_date | fiscal_year |
|-----------|-------------|
| 2024-03-15| 2024        |
| 2024-08-20| 2025        |
| 2024-12-01| 2025        |

(Fiscal year starts July — Aug 2024 belongs to FY2025)

---

### `bucket_time_of_day(df, timestamp_col)`

```python
from dbx_toolkit.datetime_utils import bucket_time_of_day

df = spark.createDataFrame([
    ("2024-01-01 07:30:00",), ("2024-01-01 14:00:00",),
    ("2024-01-01 19:45:00",), ("2024-01-01 02:15:00",),
], ["event_ts"])
df = df.withColumn("event_ts", F.to_timestamp("event_ts"))

bucket_time_of_day(df, "event_ts").display()
```

**Output:**

| event_ts            | time_of_day |
|--------------------|-------------|
| 2024-01-01 07:30:00| morning     |
| 2024-01-01 14:00:00| afternoon   |
| 2024-01-01 19:45:00| evening     |
| 2024-01-01 02:15:00| night       |

---

### `is_business_day(df, date_col)`

```python
from dbx_toolkit.datetime_utils import is_business_day

df = spark.createDataFrame([
    ("2024-03-15",), ("2024-03-16",), ("2024-03-17",), ("2024-03-18",)
], ["d"])
df = df.withColumn("d", F.to_date("d"))

is_business_day(df, "d").display()
```

**Output:**

| d          | is_business_day |
|-----------|-----------------|
| 2024-03-15| true            |
| 2024-03-16| false           |
| 2024-03-17| false           |
| 2024-03-18| true            |

(Saturday and Sunday → false)

---

### `add_age(df, date_col)`

```python
from dbx_toolkit.datetime_utils import add_age

df = spark.createDataFrame([("1990-05-20",), ("2000-11-10",)], ["birth_date"])
df = df.withColumn("birth_date", F.to_date("birth_date"))

add_age(df, "birth_date").display()
```

**Output (as of 2026-03-19):**

| birth_date | age_years |
|-----------|-----------|
| 1990-05-20| 35        |
| 2000-11-10| 25        |

---

## mlflow_utils

### `setup_experiment(experiment_path, tags=None)`

```python
from dbx_toolkit.mlflow_utils import setup_experiment

experiment_id = setup_experiment(
    "/Users/me@company.com/churn_prediction",
    tags={"project": "churn", "team": "data-science"}
)
print(experiment_id)  # "1234567890"
```

---

### `log_run(model, metrics, params=None, model_name=None)`

```python
from dbx_toolkit.mlflow_utils import log_run

log_run(
    model=pipeline_model,
    metrics={"accuracy": 0.95, "f1": 0.93, "auc": 0.97},
    params={"model": "random_forest", "n_trees": 100, "max_depth": 10},
    tags={"version": "v2"},
    model_name="catalog.schema.churn_model",  # registers to UC
    run_name="rf_v2"
)
```

**Result:** MLflow run created with all params, metrics, model artifact, and model registered in Unity Catalog.

---

### `log_grid_search(cv_model, param_grid)`

```python
from dbx_toolkit.mlflow_utils import log_grid_search

log_grid_search(cv_model, param_grid, parent_run_name="rf_grid_search")
```

**Result:** Parent run "rf_grid_search" with nested child runs:

| run_name | numTrees | maxDepth | avg_auc |
|----------|---------|----------|---------|
| combo_0  | 50      | 5        | 0.91    |
| combo_1  | 50      | 10       | 0.93    |
| combo_2  | 100     | 5        | 0.94    |
| combo_3  | 100     | 10       | 0.96    |

---

### `get_best_run(experiment_path=None, metric="metrics.accuracy")`

```python
from dbx_toolkit.mlflow_utils import get_best_run

best = get_best_run(
    experiment_path="/Users/me/churn_prediction",
    metric="metrics.f1"
)
print(best)
```

**Output:**
```python
{"run_id": "abc123", "metrics.f1": 0.96, "params.model": "rf", "params.n_trees": "100"}
```

---

### `promote_model(model_name, alias="champion")`

```python
from dbx_toolkit.mlflow_utils import promote_model

# Promote the latest version to champion
promote_model("catalog.schema.churn_model", alias="champion")

# Promote a specific version
promote_model("catalog.schema.churn_model", version=3, alias="champion")
```

---

### `load_model(model_name, alias="champion")`

```python
from dbx_toolkit.mlflow_utils import load_model

model = load_model("catalog.schema.churn_model", alias="champion")
predictions = model.predict(test_data)
```

---

## display_utils

### `peek(df, n=5)`

```python
from dbx_toolkit.display_utils import peek

peek(df)
```

**Output:**
```
Shape: 1000 rows × 5 columns

root
 |-- id: long
 |-- name: string
 |-- age: long
 |-- salary: double
 |-- region: string

+---+-------+---+--------+------+
| id|   name|age|  salary|region|
+---+-------+---+--------+------+
|  1|  Alice| 30| 75000.0|  East|
|  2|    Bob| 25| 55000.0|  West|
|  3|  Carol| 35| 90000.0|  East|
|  4|  David| 28| 62000.0| North|
|  5|    Eve| 40| 85000.0|  West|
+---+-------+---+--------+------+
```

---

### `summary_table(df)`

```python
from dbx_toolkit.display_utils import summary_table

summary_table(df).display()
```

**Output:**

| column | data_type | non_null | null_count | distinct_count | sample_values          |
|--------|-----------|---------|------------|----------------|------------------------|
| id     | bigint    | 1000    | 0          | 1000           | [1, 2, 3, 4, 5]       |
| name   | string    | 1000    | 0          | 200            | [Alice, Bob, Carol]    |
| age    | bigint    | 985     | 15         | 45             | [25, 30, 35, 40]       |
| salary | double    | 970     | 30         | 500            | [55000.0, 75000.0]     |

---

### `top_n_by_group(df, group_col, order_col, n=5)`

```python
from dbx_toolkit.display_utils import top_n_by_group

top_n_by_group(df, "department", "salary", n=3).display()
```

**Output:**

| department  | name    | salary  | rank |
|------------|---------|---------|------|
| Engineering| Alice   | 150000  | 1    |
| Engineering| Bob     | 140000  | 2    |
| Engineering| Carol   | 135000  | 3    |
| Sales      | Dave    | 120000  | 1    |
| Sales      | Eve     | 115000  | 2    |
| Sales      | Frank   | 110000  | 3    |

---

### `histogram_data(df, column, n_bins=20)`

```python
from dbx_toolkit.display_utils import histogram_data

histogram_data(df, "salary", n_bins=5).display()
```

**Output:**

| bin_start | bin_end | count |
|-----------|---------|-------|
| 40000     | 52000   | 150   |
| 52000     | 64000   | 280   |
| 64000     | 76000   | 310   |
| 76000     | 88000   | 180   |
| 88000     | 100000  | 80    |

---

### `crosstab_pct(df, col1, col2)`

```python
from dbx_toolkit.display_utils import crosstab_pct

crosstab_pct(df, "department", "status").display()
```

**Output:**

| department  | active | inactive | pending |
|------------|--------|----------|---------|
| Engineering| 45.2   | 12.3     | 8.5     |
| Sales      | 18.7   | 10.1     | 5.2     |

(Values are percentages of total)

---

### `pivot_summary(df, group_col, value_col, agg_func="sum")`

```python
from dbx_toolkit.display_utils import pivot_summary

pivot_summary(df, "department", "revenue", agg_func="sum", pivot_col="quarter").display()
```

**Output:**

| department  | Q1      | Q2      | Q3      | Q4      |
|------------|---------|---------|---------|---------|
| Engineering| 250000  | 280000  | 310000  | 290000  |
| Sales      | 180000  | 200000  | 195000  | 220000  |

---

### `format_number_columns(df, columns=None, decimals=2)`

```python
from dbx_toolkit.display_utils import format_number_columns

df = spark.createDataFrame([(1.23456, 78.90123)], ["score", "pct"])

format_number_columns(df, decimals=2).display()
```

**Output:**

| score | pct   |
|-------|-------|
| 1.23  | 78.90 |

---

### `compare_side_by_side(df1, df2, name1="df1", name2="df2")`

```python
from dbx_toolkit.display_utils import compare_side_by_side

compare_side_by_side(df1, df2, name1="before", name2="after").display()
```

**Output:**

| column | in_before | type_before | in_after | type_after | status       |
|--------|-----------|-------------|----------|------------|-------------|
| id     | true      | bigint      | true     | bigint     | unchanged   |
| name   | true      | string      | true     | string     | unchanged   |
| score  | true      | double      | true     | integer    | type_changed|
| email  | false     | null        | true     | string     | added       |
| temp   | true      | string      | false    | null       | removed     |
