# EDA Agent — Vertex AI Edition

Domain-agnostic automated EDA. Point it at any tabular dataset — fraud,
tennis, sales, medical, logistics — and it generates a complete, runnable
Jupyter notebook with charts, statistics, and AI-written insights tailored
to your data.

Uses **Gemini 2.0 Flash** on Vertex AI. No Anthropic key needed — billed
entirely to your GCP project.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate (local dev)
gcloud auth application-default login

# 3. Set your project
export GOOGLE_CLOUD_PROJECT=my-gcp-project

# 4. Run
python example_usage.py

# 5. Open the notebook
jupyter notebook eda_report.ipynb
```

---

## Usage — all parameters optional except df

```python
import pandas as pd
from fraud_eda_agent import EDAAgent

df = pd.read_csv("any_dataset.csv")

agent = EDAAgent(project="my-gcp-project", location="us-central1")

# Minimal — agent infers everything
agent.run(df=df)

# With a goal
agent.run(df=df, goal="find the main drivers of customer churn")

# With goal + explicit target
agent.run(df=df, goal="find churn drivers", target_col="churned")

# Full detail — most accurate notebook
agent.run(
    df=df,
    goal="find churn drivers",
    target_col="churned",
    context="""
        Telecom customer dataset. churned=1 means cancelled within 90 days.
        tenure_months = how long they have been a customer.
        plan_type: Basic / Pro / Enterprise.
        monthly_charge is in USD, excluding taxes.
    """,
    output="churn_eda_report.ipynb",
)
```

### Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | DataFrame | required | Any tabular dataset |
| `goal` | str | None | Plain-text analysis goal |
| `target_col` | str | None | Target column — agent infers if omitted |
| `context` | str | None | Dataset description: column meanings, value encodings, domain notes |
| `output` | str | `"eda_report.ipynb"` | Output notebook path |
| `verbose` | bool | True | Print progress to stdout |

The more you provide, the better the notebook. But everything beyond `df` is optional.

---

## Reading from BigQuery

```python
from google.cloud import bigquery
from fraud_eda_agent import EDAAgent

df = bigquery.Client().query("""
    SELECT * FROM `my-project.my_dataset.my_table`
    WHERE DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
""").to_dataframe()

EDAAgent(project="my-gcp-project").run(
    df=df,
    goal="identify seasonal revenue patterns",
)
```

---

## Saving the notebook to GCS (for Vertex Workbench)

```python
from google.cloud import storage

agent.run(df=df, output="/tmp/eda_report.ipynb")

storage.Client().bucket("my-notebooks-bucket") \
    .blob("eda/eda_report.ipynb") \
    .upload_from_filename("/tmp/eda_report.ipynb")
```

---

## Vertex AI Pipelines component

```python
from kfp import dsl
from kfp.dsl import Dataset, Output

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-aiplatform>=1.49.0", "vertexai>=1.49.0",
        "nbformat", "pandas", "matplotlib", "seaborn", "scipy",
    ],
)
def eda_component(
    project:    str,
    bq_table:   str,
    goal:       str,
    context:    str,
    notebook:   Output[Dataset],
):
    from google.cloud import bigquery
    from fraud_eda_agent import EDAAgent

    df = bigquery.Client(project=project).query(
        f"SELECT * FROM `{bq_table}`"
    ).to_dataframe()

    EDAAgent(project=project).run(
        df=df,
        goal=goal,
        context=context,
        output=notebook.path,
    )
```

---

## IAM requirements

| Role | Purpose |
|---|---|
| `roles/aiplatform.user` | Call Gemini 2.0 Flash via Vertex AI |
| `roles/bigquery.dataViewer` | Read source tables |
| `roles/bigquery.jobUser` | Run BigQuery queries |
| `roles/storage.objectCreator` | Write notebooks to GCS |

---

## What the notebook contains

Sections are generated dynamically based on the detected domain and data.
At minimum every notebook includes:

| Section | Content |
|---|---|
| Executive summary | Domain, key findings, data quality overview |
| Dataset overview | Shape, dtypes, missing values |
| Target distribution | Class balance or regression distribution (if target exists) |
| Numeric distributions | Histograms / KDE per feature, split by target class |
| Categorical breakdown | Value frequency charts per categorical column |
| Correlation heatmap | Numeric feature correlation matrix |
| Statistical significance | Mann-Whitney U p-values for binary targets |
| Domain-specific sections | e.g. serve stats for tennis, velocity for fraud, cohorts for sales |
| Key findings | AI-written bullets with specific numbers from the data |

---

## Supported domains (auto-detected)

`fraud` · `sports` · `sales` · `medical` · `finance` · `marketing` · `hr` · `logistics` · `generic`

Detection is based on column names and value patterns in Phase 1.
Use `context=` to correct or guide detection for ambiguous datasets.
