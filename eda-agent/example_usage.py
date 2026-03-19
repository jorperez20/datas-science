"""
example_usage.py
----------------
Shows EDAAgent working on three completely different datasets:
  1. Credit card fraud  (binary classification — auto-detected)
  2. Tennis match stats (domain-specific analysis)
  3. E-commerce sales   (regression / unsupervised)

Run with:  python example_usage.py
"""

import numpy as np
import pandas as pd
from fraud_eda_agent import EDAAgent

agent = EDAAgent(
    project="YOUR_GCP_PROJECT_ID",
    location="us-central1",
)


# ── Example 1: Credit card fraud ─────────────────────────────────────────────
def example_fraud():
    np.random.seed(42)
    N = 2000
    fraud = np.random.rand(N) < 0.048
    df = pd.DataFrame({
        "transaction_id":    [f"TXN{i:06d}" for i in range(N)],
        "amount":            np.where(fraud, np.random.exponential(800,N),
                                            np.random.exponential(85,N)).round(2),
        "transaction_hour":  np.where(fraud, np.random.choice([1,2,3,23,0],N),
                                            np.random.randint(7,22,N)),
        "velocity_24h":      np.where(fraud, np.random.poisson(12,N),
                                            np.random.poisson(2,N)),
        "customer_age":      np.random.randint(18,75,N),
        "merchant_category": np.where(fraud,
                                np.random.choice(["atm_withdrawal","crypto"],N),
                                np.random.choice(["grocery","restaurant","gas_station"],N)),
        "country":           np.where(fraud,
                                np.random.choice(["US","NG","RO","BR"],N,p=[.3,.25,.25,.2]),
                                np.random.choice(["US","CA","UK"],N,p=[.8,.12,.08])),
        "is_fraud":          fraud.astype(int),
    })
    agent.run(
        df=df,
        goal="identify strongest fraud signals and flag high-risk patterns",
        context="""
Credit card transaction dataset from a Latin American bank (2024).
Columns:
  - amount       : transaction value in USD
  - transaction_hour : hour of day (0-23) the transaction occurred
  - velocity_24h : number of card uses in the past 24 hours
  - customer_age : cardholder age in years
  - merchant_category : type of merchant (grocery, restaurant, atm_withdrawal, crypto, etc.)
  - country      : ISO country code of the merchant
  - is_fraud     : 1 = confirmed fraudulent, 0 = legitimate (ground truth label)
Fraud cases were confirmed by the bank's dispute team within 30 days of the transaction.
        """,
        output="fraud_eda_report.ipynb",
    )


# ── Example 2: Tennis match statistics ───────────────────────────────────────
def example_tennis():
    np.random.seed(7)
    N = 1500
    surfaces   = np.random.choice(["Hard","Clay","Grass","Carpet"],N,p=[.55,.25,.15,.05])
    w_rank     = np.random.randint(1,200,N)
    l_rank     = w_rank + np.random.randint(1,150,N)
    df = pd.DataFrame({
        "match_id":               [f"M{i:05d}" for i in range(N)],
        "surface":                surfaces,
        "round":                  np.random.choice(["R128","R64","R32","R16","QF","SF","F"],N),
        "tournament_level":       np.random.choice(["Grand Slam","Masters","ATP 500","ATP 250"],N,
                                                    p=[.12,.2,.28,.4]),
        "winner_rank":            w_rank,
        "loser_rank":             l_rank,
        "rank_diff":              l_rank - w_rank,
        "winner_first_serve_pct": np.random.normal(62,6,N).clip(40,80).round(1),
        "winner_ace_count":       np.random.poisson(7,N),
        "winner_double_faults":   np.random.poisson(2,N),
        "winner_break_pts_won":   np.random.randint(0,8,N),
        "loser_first_serve_pct":  np.random.normal(58,7,N).clip(38,78).round(1),
        "loser_ace_count":        np.random.poisson(5,N),
        "match_duration_min":     np.random.normal(95,28,N).clip(45,220).round(0).astype(int),
        "sets_played":            np.random.choice([2,3],N,p=[.55,.45]),
        "upset":                  (l_rank < w_rank).astype(int),
    })
    agent.run(
        df=df,
        goal="analyse serve efficiency, upset patterns, and performance by surface and level",
        context="""
ATP (men's professional tennis) match results dataset.
Each row is one completed match. Columns:
  - surface         : court surface — Hard, Clay, Grass, or Carpet
  - round           : tournament round (R128=first round, F=final)
  - tournament_level: Grand Slam > Masters > ATP 500 > ATP 250 (prestige order)
  - winner_rank / loser_rank : ATP world ranking at time of match (lower = better)
  - rank_diff       : loser_rank minus winner_rank (positive = favourite won)
  - winner_first_serve_pct : % of first serves in (higher is better)
  - winner_ace_count : number of aces served by the winner
  - winner_double_faults : unforced double faults by the winner
  - winner_break_pts_won : break points converted by winner
  - match_duration_min : total match length in minutes
  - sets_played     : number of sets (best of 3 at ATP level)
  - upset           : 1 = lower-ranked player won, 0 = favourite won
        """,
        output="tennis_eda_report.ipynb",
    )


# ── Example 3: E-commerce sales ──────────────────────────────────────────────
def example_sales():
    np.random.seed(99)
    N = 3000
    cats   = np.random.choice(["Electronics","Clothing","Home","Sports","Books"],N,
                               p=[.25,.3,.2,.15,.1])
    prices = np.array([{"Electronics":350,"Clothing":65,"Home":120,
                         "Sports":85,"Books":22}[c]*(0.7+np.random.rand()*0.8)
                        for c in cats]).round(2)
    disc   = np.random.choice([0,5,10,15,20,25,30],N,p=[.4,.15,.15,.1,.1,.05,.05])
    df = pd.DataFrame({
        "order_id":        [f"ORD{i:06d}" for i in range(N)],
        "order_date":      pd.date_range("2023-01-01",periods=N,freq="3H").strftime("%Y-%m-%d"),
        "category":        cats,
        "product_rating":  np.random.normal(4.1,0.6,N).clip(1,5).round(1),
        "price":           prices,
        "discount_pct":    disc,
        "quantity":        np.random.choice([1,2,3,4,5],N,p=[.5,.25,.13,.07,.05]),
        "revenue":         (prices*(1-disc/100)*np.random.randint(1,6,N)).round(2),
        "customer_region": np.random.choice(["North","South","East","West","Central"],N),
        "return_flag":     np.random.choice([0,1],N,p=[.88,.12]),
        "days_to_ship":    np.random.choice([1,2,3,4,5,6,7],N,p=[.1,.2,.3,.2,.1,.06,.04]),
        "customer_type":   np.random.choice(["New","Returning","VIP"],N,p=[.4,.45,.15]),
    })
    agent.run(
        df=df,
        goal="identify revenue drivers, return patterns, and regional performance",
        context="""
E-commerce order dataset for a US online retailer (2023).
Each row is one customer order. Columns:
  - order_date     : date the order was placed (YYYY-MM-DD)
  - category       : product category — Electronics, Clothing, Home, Sports, Books
  - product_rating : customer review score (1-5 stars)
  - price          : listed unit price in USD before discount
  - discount_pct   : percentage discount applied at checkout (0, 5, 10, 15, 20, 25, or 30)
  - quantity       : number of units ordered
  - revenue        : actual revenue collected after discount × quantity
  - customer_region: US region — North, South, East, West, Central
  - return_flag    : 1 = item was returned within 30 days, 0 = kept
  - days_to_ship   : days from order placement to shipment
  - customer_type  : New (first purchase), Returning, or VIP (loyalty programme member)
        """,
        output="sales_eda_report.ipynb",
    )


# ── Your own CSV ─────────────────────────────────────────────────────────────
def example_csv(path: str, goal: str = None, target_col: str = None, context: str = None):
    # Run EDA on any CSV.
    # Parameters:
    #   path       - path to CSV file
    #   goal       - plain-text analysis goal, e.g. "find churn drivers"
    #   target_col - explicit target column (optional, agent will infer if omitted)
    #   context    - plain-text description of the dataset and column meanings
    #
    # Example:
    #   example_csv(
    #       "patient_data.csv",
    #       goal="identify risk factors for hospital readmission",
    #       target_col="readmitted_30d",
    #       context=(
    #           "Hospital discharge records 2020-2023. "
    #           "age=patient age, los_days=length of stay, "
    #           "readmitted_30d=1 if readmitted within 30 days."
    #       ),
    #   )
    df = pd.read_csv(path)
    agent.run(
        df=df,
        goal=goal,
        target_col=target_col,
        context=context,
        output=path.replace(".csv", "_eda_report.ipynb"),
    )


if __name__ == "__main__":
    print("Running tennis example...")
    example_tennis()

    # Uncomment to run others:
    # example_fraud()
    # example_sales()
    # example_csv("my_data.csv", goal="understand the key drivers of churn")
