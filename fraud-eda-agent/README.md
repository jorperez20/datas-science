# Fraud EDA Agent — Vertex AI Edition

Generates a complete, runnable Jupyter notebook from any transaction DataFrame,
using **Gemini 2.0 Flash** on Vertex AI as the reasoning engine.
No Anthropic API key needed — billed entirely to your GCP project.

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
jupyter notebook fraud_eda_report.ipynb
```

---

## Usage in code

```python
import pandas as pd
from fraud_eda_agent import FraudEDAAgent

df = pd.read_csv("transactions.csv")

agent = FraudEDAAgent(
    project="my-gcp-project",
    location="us-central1",
)
agent.run(
    df=df,
    target_col="is_fraud",
    output="fraud_eda_report.ipynb",
    focus="focus on velocity and off-hours patterns",  # optional
    verbose=True,
)
```

---

## Reading from BigQuery

```python
from google.cloud import bigquery
from fraud_eda_agent import FraudEDAAgent

df = bigquery.Client().query("""
    SELECT * FROM `my-project.fraud_dataset.transactions`
    WHERE DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
""").to_dataframe()

FraudEDAAgent(project="my-gcp-project").run(df, target_col="is_fraud")
```

---

## Saving the notebook to GCS (for Vertex Workbench)

```python
from google.cloud import storage

agent.run(df, output="/tmp/fraud_eda_report.ipynb")

storage.Client().bucket("my-notebooks-bucket") \
    .blob("fraud/fraud_eda_report.ipynb") \
    .upload_from_filename("/tmp/fraud_eda_report.ipynb")
```

---

## Vertex AI Pipelines component

```python
from kfp import dsl
from kfp.dsl import Dataset, Output

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-aiplatform", "nbformat",
        "pandas", "matplotlib", "seaborn", "scipy"
    ],
)
def fraud_eda_component(
    project: str,
    bq_table: str,
    target_col: str,
    notebook: Output[Dataset],
):
    from google.cloud import bigquery
    from fraud_eda_agent import FraudEDAAgent

    df = bigquery.Client(project=project).query(
        f"SELECT * FROM `{bq_table}`"
    ).to_dataframe()

    FraudEDAAgent(project=project).run(
        df=df, target_col=target_col, output=notebook.path
    )
```

---

## IAM requirements

The service account running this agent needs:

| Role | Purpose |
|---|---|
| `roles/aiplatform.user` | Call Gemini via Vertex AI |
| `roles/bigquery.dataViewer` | Read transaction tables |
| `roles/storage.objectCreator` | Write notebooks to GCS |

---

## What the notebook contains

| Section | Content |
|---|---|
| Executive summary | Fraud rate, top risk signals, data quality |
| Dataset overview | Shape, dtypes, df.info() |
| Missing values | Seaborn heatmap of nulls |
| Class imbalance | Bar chart with % annotations |
| Numeric distributions | KDE overlays — fraud vs legit |
| Feature lift table | Ranked by fraud_mean / legit_mean |
| Categorical breakdown | Grouped bar charts |
| Correlation heatmap | Numeric feature correlations |
| Statistical significance | Mann-Whitney U p-values |
| Key findings | AI-written bullets citing actual numbers |
