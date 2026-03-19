"""
example_usage.py
----------------
Demonstrates FraudEDAAgent with a synthetic transaction dataset.
Run with:  python example_usage.py
"""

import numpy as np
import pandas as pd
from fraud_eda_agent import FraudEDAAgent

# ── Generate synthetic fraud dataset ──────────────────────────────────
np.random.seed(42)
N = 2000
fraud_mask = np.random.rand(N) < 0.048

df = pd.DataFrame({
    "transaction_id":   [f"TXN{i:06d}" for i in range(N)],
    "amount":           np.where(fraud_mask,
                            np.random.exponential(800, N),
                            np.random.exponential(85, N)).round(2),
    "transaction_hour": np.where(fraud_mask,
                            np.random.choice([1,2,3,23,0], N),
                            np.random.randint(7, 22, N)),
    "velocity_24h":     np.where(fraud_mask,
                            np.random.poisson(12, N),
                            np.random.poisson(2, N)),
    "customer_age":     np.random.randint(18, 75, N),
    "merchant_category": np.where(fraud_mask,
                            np.random.choice(["online_retail","atm_withdrawal","crypto"], N, p=[0.4,0.4,0.2]),
                            np.random.choice(["grocery","restaurant","gas_station","online_retail"], N)),
    "card_type":        np.random.choice(["visa","mastercard","amex"], N, p=[0.5,0.35,0.15]),
    "country":          np.where(fraud_mask,
                            np.random.choice(["US","NG","RO","BR","MX"], N, p=[0.3,0.2,0.2,0.15,0.15]),
                            np.random.choice(["US","CA","UK"], N, p=[0.8,0.12,0.08])),
    "is_fraud":         fraud_mask.astype(int),
})

# Introduce some missing values for realism
df.loc[np.random.choice(N, 40, replace=False), "customer_age"] = np.nan
df.loc[np.random.choice(N, 15, replace=False), "merchant_category"] = np.nan

print(f"Dataset: {len(df):,} rows | fraud rate: {df['is_fraud'].mean()*100:.2f}%")

# ── Run the agent ──────────────────────────────────────────────────────
agent = FraudEDAAgent(
    project="YOUR_GCP_PROJECT_ID",   # or set GOOGLE_CLOUD_PROJECT env var
    location="us-central1",
)

agent.run(
    df=df,
    target_col="is_fraud",
    output="fraud_eda_report.ipynb",
    focus="Pay special attention to transaction velocity and off-hours patterns.",
    verbose=True,
)
