# ============================================================
#  LightGBM Overfitting Lab — Credit Risk (Fictitious Data)
#  700k rows | 20% charge-off rate
# ============================================================
# Run each section independently like a Jupyter notebook.
# Dependencies: pip install lightgbm scikit-learn pandas numpy matplotlib seaborn imbalanced-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb

np.random.seed(42)
plt.style.use("dark_background")
COLORS = {"train": "#00e5ff", "val": "#ff3e6c", "test": "#a259ff", "good": "#00e5a0", "warn": "#ffb930"}

# ============================================================
#  SECTION 0 — Generate Fictitious Credit Risk Dataset
# ============================================================
print("=" * 60)
print("SECTION 0 — Generating Fictitious Credit Risk Dataset")
print("=" * 60)

N = 700_000
rng = np.random.default_rng(42)

# ── Applicant Demographics ──────────────────────────────────
age                 = rng.integers(18, 75, N)
income              = np.clip(rng.lognormal(10.8, 0.6, N), 15_000, 500_000).astype(int)
employment_years    = np.clip(rng.exponential(5, N), 0, 40)
num_dependents      = rng.integers(0, 6, N)
education_level     = rng.choice([0, 1, 2, 3], N, p=[0.15, 0.35, 0.35, 0.15])  # 0=HS, 1=Some, 2=Bachelor, 3=Grad

# ── Loan Characteristics ────────────────────────────────────
loan_amount         = np.clip(rng.lognormal(10.0, 0.8, N), 1_000, 100_000).astype(int)
loan_term_months    = rng.choice([12, 24, 36, 48, 60], N)
interest_rate       = np.clip(rng.normal(12, 4, N), 3, 30)
loan_purpose        = rng.choice([0, 1, 2, 3, 4], N)  # debt_consol, home, auto, edu, other
debt_to_income      = np.clip(rng.beta(2, 5, N) * 0.8, 0.01, 0.75)

# ── Credit History ──────────────────────────────────────────
credit_score        = np.clip(rng.normal(680, 80, N), 300, 850).astype(int)
num_credit_lines    = rng.integers(1, 20, N)
oldest_account_yrs  = np.clip(rng.exponential(8, N), 0, 40)
num_late_payments   = rng.integers(0, 15, N)
num_hard_inquiries  = rng.integers(0, 10, N)
credit_utilization  = np.clip(rng.beta(2, 4, N), 0.01, 0.99)
has_bankruptcy      = rng.choice([0, 1], N, p=[0.93, 0.07])
has_collections     = rng.choice([0, 1], N, p=[0.88, 0.12])

# ── Behavioral Features ─────────────────────────────────────
months_employed     = employment_years * 12
payment_history_pct = np.clip(1 - (num_late_payments / (num_credit_lines + 1)) * 0.3, 0.3, 1.0)
revolving_balance   = (credit_utilization * income * 0.2).astype(int)
savings_balance     = np.clip(rng.lognormal(8, 1.5, N), 0, 200_000).astype(int)
checking_balance    = np.clip(rng.lognormal(7, 1.2, N), 0, 50_000).astype(int)

# ── Derived / Interaction Features ──────────────────────────
monthly_payment_est = loan_amount / loan_term_months
payment_to_income   = monthly_payment_est / (income / 12)

# ── Target: charge-off (1 = charged off) ────────────────────
# Logistic relationship — realistic risk drivers
log_odds = (
    -4.5
    + (700 - credit_score) * 0.008
    + debt_to_income * 3.5
    + num_late_payments * 0.15
    + has_bankruptcy * 1.8
    + has_collections * 0.9
    + credit_utilization * 1.2
    + payment_to_income * 2.0
    - payment_history_pct * 1.5
    - (income / 100_000) * 0.8
    + num_hard_inquiries * 0.1
    - oldest_account_yrs * 0.02
    + rng.normal(0, 0.5, N)        # noise
)
prob_chargeoff = 1 / (1 + np.exp(-log_odds))

# Calibrate to exactly 20% charge-off rate
threshold_co = np.percentile(prob_chargeoff, 80)
target = (prob_chargeoff > threshold_co).astype(int)

# ── Assemble DataFrame ──────────────────────────────────────
df = pd.DataFrame({
    "age": age, "income": income, "employment_years": employment_years,
    "num_dependents": num_dependents, "education_level": education_level,
    "loan_amount": loan_amount, "loan_term_months": loan_term_months,
    "interest_rate": interest_rate, "loan_purpose": loan_purpose,
    "debt_to_income": debt_to_income, "credit_score": credit_score,
    "num_credit_lines": num_credit_lines, "oldest_account_yrs": oldest_account_yrs,
    "num_late_payments": num_late_payments, "num_hard_inquiries": num_hard_inquiries,
    "credit_utilization": credit_utilization, "has_bankruptcy": has_bankruptcy,
    "has_collections": has_collections, "months_employed": months_employed,
    "payment_history_pct": payment_history_pct, "revolving_balance": revolving_balance,
    "savings_balance": savings_balance, "checking_balance": checking_balance,
    "payment_to_income": payment_to_income,
    "charge_off": target
})

# ── Split ────────────────────────────────────────────────────
FEATURES = [c for c in df.columns if c != "charge_off"]
X, y = df[FEATURES], df["charge_off"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print(f"  Train : {len(X_train):>7,}  rows | charge-off: {y_train.mean():.1%}")
print(f"  Val   : {len(X_val):>7,}  rows | charge-off: {y_val.mean():.1%}")
print(f"  Test  : {len(X_test):>7,}  rows | charge-off: {y_test.mean():.1%}")
print(f"  Total features : {len(FEATURES)}")
print()

train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

# ── Helper: evaluate ────────────────────────────────────────
def evaluate(model, tag=""):
    train_auc = roc_auc_score(y_train, model.predict(X_train))
    val_auc   = roc_auc_score(y_val,   model.predict(X_val))
    test_auc  = roc_auc_score(y_test,  model.predict(X_test))
    gap       = train_auc - val_auc
    print(f"  {tag:<30}  Train={train_auc:.4f}  Val={val_auc:.4f}  Test={test_auc:.4f}  Gap={gap:.4f}")
    return {"train": train_auc, "val": val_auc, "test": test_auc, "gap": gap}


# ============================================================
#  SECTION 1 — BASELINE (OVERFIT)
# ============================================================
print("=" * 60)
print("SECTION 1 — Baseline: Intentionally Overfit Model")
print("=" * 60)
print("""
SCENARIO: A data scientist runs LightGBM with default-ish params —
high num_leaves, no depth limit, tiny min_child_samples. The model
memorises the 490k training rows and fails to generalise.
""")

params_baseline = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "num_leaves": 400,         # ← way too high
    "max_depth": -1,           # ← no depth limit
    "min_child_samples": 5,    # ← very small = memorises
    "learning_rate": 0.1,
    "n_estimators": 500,
}

callbacks_baseline = [lgb.record_evaluation(evals_result := {}), lgb.log_evaluation(-1)]

model_baseline = lgb.train(
    params_baseline,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    callbacks=callbacks_baseline,
)

res_baseline = evaluate(model_baseline, "Baseline (overfit)")

# ── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Section 1 — Baseline: Overfit Model", fontsize=14, color="white", y=1.01)

ax = axes[0]
ax.plot(evals_result["train"]["auc"], color=COLORS["train"], lw=2, label="Train AUC")
ax.plot(evals_result["val"]["auc"],   color=COLORS["val"],   lw=2, label="Val AUC")
ax.fill_between(range(500),
                evals_result["train"]["auc"],
                evals_result["val"]["auc"],
                alpha=0.12, color=COLORS["val"], label="Overfit Gap")
ax.set_title("Training Curves — Baseline", color="white")
ax.set_xlabel("Iteration"); ax.set_ylabel("AUC")
ax.legend(); ax.grid(alpha=0.15)

# Feature importance
ax2 = axes[1]
fi = pd.Series(model_baseline.feature_importance(importance_type="gain"), index=FEATURES).nlargest(15)
fi.sort_values().plot(kind="barh", ax=ax2, color=COLORS["train"], alpha=0.8)
ax2.set_title("Top 15 Feature Importances (Gain)", color="white")
ax2.grid(alpha=0.15, axis="x")

plt.tight_layout()
plt.savefig("/home/claude/sec1_baseline.png", dpi=120, bbox_inches="tight", facecolor="#0a0c10")
plt.close()
print(f"\n  ⚠️  Gap of {res_baseline['gap']:.4f} — model is heavily overfit!")
print("  Saved: sec1_baseline.png\n")


# ============================================================
#  SECTION 2 — REGULARIZATION
# ============================================================
print("=" * 60)
print("SECTION 2 — Fix with Regularization Parameters")
print("=" * 60)
print("""
SCENARIO: We constrain tree complexity directly:
  • Lower num_leaves (31) — fewer leaf nodes per tree
  • max_depth=6          — explicit depth ceiling
  • min_child_samples=100 — each leaf needs 100 samples minimum
  • lambda_l1/l2=0.1     — L1/L2 weight penalties
  • min_gain_to_split=0.01 — don't split unless gain > threshold
""")

params_reg = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "num_leaves": 31,            # ← down from 400
    "max_depth": 6,              # ← explicit cap
    "min_child_samples": 100,    # ← up from 5 to 100
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_gain_to_split": 0.01,
    "learning_rate": 0.05,
    "n_estimators": 500,
}

evals_reg = {}
model_reg = lgb.train(
    params_reg, train_data, num_boost_round=500,
    valid_sets=[train_data, val_data], valid_names=["train", "val"],
    callbacks=[lgb.record_evaluation(evals_reg), lgb.log_evaluation(-1)],
)
res_reg = evaluate(model_reg, "Regularized")

# ── Compare curves side-by-side ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Section 2 — Regularization: Before vs After", fontsize=14, color="white")

for ax, evals, title, gap in [
    (axes[0], evals_result, f"Baseline  (Gap={res_baseline['gap']:.4f})", res_baseline["gap"]),
    (axes[1], evals_reg,    f"Regularized (Gap={res_reg['gap']:.4f})",    res_reg["gap"]),
]:
    ax.plot(evals["train"]["auc"], color=COLORS["train"], lw=2, label="Train")
    ax.plot(evals["val"]["auc"],   color=COLORS["val"],   lw=2, label="Val")
    ax.fill_between(range(len(evals["train"]["auc"])),
                    evals["train"]["auc"], evals["val"]["auc"],
                    alpha=0.15, color=COLORS["val"])
    ax.set_title(title, color="white")
    ax.set_xlabel("Iteration"); ax.set_ylabel("AUC")
    ax.legend(); ax.grid(alpha=0.15)
    ax.set_ylim(0.65, 1.0)

plt.tight_layout()
plt.savefig("/home/claude/sec2_regularization.png", dpi=120, bbox_inches="tight", facecolor="#0a0c10")
plt.close()
print(f"\n  ✅ Gap reduced: {res_baseline['gap']:.4f} → {res_reg['gap']:.4f}  "
      f"({(1-res_reg['gap']/res_baseline['gap'])*100:.0f}% improvement)")
print("  Saved: sec2_regularization.png\n")


# ============================================================
#  SECTION 3 — SUBSAMPLING (Stochastic Boosting)
# ============================================================
print("=" * 60)
print("SECTION 3 — Fix with Subsampling")
print("=" * 60)
print("""
SCENARIO: We add randomness per-tree to reduce variance:
  • subsample=0.8       — each tree sees only 80% of rows
  • colsample_bytree=0.8 — each tree sees only 80% of features
  • subsample_freq=1    — apply every iteration

With 490k training rows, even 50% subsample = 245k per tree.
This is more than enough data while breaking correlation between trees.
""")

params_sub = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "num_leaves": 63, "max_depth": 7, "min_child_samples": 50,
    "subsample": 0.8,          # ← row subsampling
    "subsample_freq": 1,
    "colsample_bytree": 0.8,   # ← column subsampling
    "learning_rate": 0.05,
}

# Try multiple subsample values to show the effect
sub_results = {}
for ss in [0.3, 0.5, 0.7, 0.9, 1.0]:
    p = {**params_sub, "subsample": ss, "subsample_freq": 1 if ss < 1.0 else 0}
    ev = {}
    m = lgb.train(p, train_data, num_boost_round=300,
                  valid_sets=[train_data, val_data], valid_names=["train","val"],
                  callbacks=[lgb.record_evaluation(ev), lgb.log_evaluation(-1)])
    gap = roc_auc_score(y_train, m.predict(X_train)) - roc_auc_score(y_val, m.predict(X_val))
    sub_results[ss] = {"gap": gap, "val_auc": roc_auc_score(y_val, m.predict(X_val)),
                       "train_curves": ev["train"]["auc"], "val_curves": ev["val"]["auc"]}
    print(f"  subsample={ss:.1f}  Val={sub_results[ss]['val_auc']:.4f}  Gap={gap:.4f}")

evals_sub = {}
model_sub = lgb.train(params_sub, train_data, num_boost_round=400,
                      valid_sets=[train_data, val_data], valid_names=["train","val"],
                      callbacks=[lgb.record_evaluation(evals_sub), lgb.log_evaluation(-1)])
res_sub = evaluate(model_sub, "Subsampled (0.8/0.8)")

# ── 5-fold CV variance ───────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_full, y_full = df[FEATURES], df["charge_off"]
fold_scores_before, fold_scores_after = [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full)):
    Xtr, Xva = X_full.iloc[tr_idx], X_full.iloc[va_idx]
    ytr, yva = y_full.iloc[tr_idx], y_full.iloc[va_idx]
    td = lgb.Dataset(Xtr, label=ytr)

    # Before (no subsample)
    m1 = lgb.train({**params_baseline, "n_estimators": 200}, td,
                   num_boost_round=200, callbacks=[lgb.log_evaluation(-1)])
    fold_scores_before.append(roc_auc_score(yva, m1.predict(Xva)))

    # After (with subsample)
    m2 = lgb.train({**params_sub}, td,
                   num_boost_round=200, callbacks=[lgb.log_evaluation(-1)])
    fold_scores_after.append(roc_auc_score(yva, m2.predict(Xva)))

print(f"\n  5-Fold CV Val AUC — Without subsample: {np.mean(fold_scores_before):.4f} ± {np.std(fold_scores_before):.4f}")
print(f"  5-Fold CV Val AUC — With    subsample: {np.mean(fold_scores_after):.4f} ± {np.std(fold_scores_after):.4f}")

# ── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Section 3 — Subsampling", fontsize=14, color="white")

# Subsample rate vs gap
ax = axes[0]
ss_vals  = list(sub_results.keys())
gap_vals = [sub_results[s]["gap"] for s in ss_vals]
val_vals = [sub_results[s]["val_auc"] for s in ss_vals]
ax.plot(ss_vals, gap_vals, "o-", color=COLORS["val"], lw=2, label="Overfit Gap")
ax.plot(ss_vals, val_vals, "s-", color=COLORS["train"], lw=2, label="Val AUC")
ax.set_title("Subsample Rate vs. Performance", color="white")
ax.set_xlabel("subsample"); ax.legend(); ax.grid(alpha=0.15)

# CV fold bars
ax = axes[1]
x = np.arange(5)
w = 0.35
ax.bar(x - w/2, fold_scores_before, w, color=COLORS["val"],   alpha=0.8, label="Without subsample")
ax.bar(x + w/2, fold_scores_after,  w, color=COLORS["train"], alpha=0.8, label="With subsample")
ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
ax.set_title("5-Fold CV Variance", color="white")
ax.set_ylabel("Val AUC"); ax.legend(); ax.grid(alpha=0.15, axis="y")
ax.set_ylim(0.8, 0.9)

# Training curves
ax = axes[2]
ax.plot(evals_sub["train"]["auc"], color=COLORS["train"], lw=2, label="Train (ss=0.8)")
ax.plot(evals_sub["val"]["auc"],   color=COLORS["val"],   lw=2, label="Val (ss=0.8)")
ax.set_title("Training Curves with Subsampling", color="white")
ax.set_xlabel("Iteration"); ax.set_ylabel("AUC"); ax.legend(); ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig("/home/claude/sec3_subsampling.png", dpi=120, bbox_inches="tight", facecolor="#0a0c10")
plt.close()
print("  Saved: sec3_subsampling.png\n")


# ============================================================
#  SECTION 4 — EARLY STOPPING
# ============================================================
print("=" * 60)
print("SECTION 4 — Early Stopping")
print("=" * 60)
print("""
SCENARIO: We set n_estimators=2000 (ceiling) and let early stopping
decide when to stop. The model halts when val AUC hasn't improved
for `stopping_rounds` consecutive iterations.

This eliminates the need to manually tune n_estimators and
prevents the model from fitting noise in later iterations.
""")

params_es = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "num_leaves": 63, "max_depth": 6, "min_child_samples": 50,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "lambda_l1": 0.05, "lambda_l2": 0.05,
    "learning_rate": 0.05,
}

evals_es = {}
model_es = lgb.train(
    params_es,
    train_data,
    num_boost_round=2000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.record_evaluation(evals_es),
        lgb.log_evaluation(-1),
    ],
)

best_iter    = model_es.best_iteration
best_val_auc = max(evals_es["val"]["auc"])
final_iter   = len(evals_es["val"]["auc"])

print(f"  Best iteration  : {best_iter}")
print(f"  Stopped at      : {final_iter}  (saved {2000 - final_iter} unused trees)")
print(f"  Best Val AUC    : {best_val_auc:.4f}")
res_es = evaluate(model_es, "Early Stopping")

# ── Compare stopping_rounds values ──────────────────────────
print("\n  Effect of stopping_rounds:")
for sr in [10, 25, 50, 100, 200]:
    ev2 = {}
    m2 = lgb.train({**params_es}, train_data, num_boost_round=2000,
                   valid_sets=[train_data, val_data], valid_names=["train","val"],
                   callbacks=[lgb.early_stopping(sr, verbose=False),
                               lgb.record_evaluation(ev2), lgb.log_evaluation(-1)])
    v = max(ev2["val"]["auc"])
    print(f"    stopping_rounds={sr:>3}  best_iter={m2.best_iteration:>4}  val_auc={v:.4f}")

# ── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Section 4 — Early Stopping", fontsize=14, color="white")

ax = axes[0]
iters = range(len(evals_es["train"]["auc"]))
ax.plot(iters, evals_es["train"]["auc"], color=COLORS["train"], lw=2, label="Train AUC")
ax.plot(iters, evals_es["val"]["auc"],   color=COLORS["val"],   lw=2, label="Val AUC")
ax.axvline(best_iter, color=COLORS["good"], lw=1.5, linestyle="--", label=f"Best iter ({best_iter})")
ax.axvline(final_iter - 1, color=COLORS["warn"], lw=1.5, linestyle=":", label=f"Stopped at ({final_iter})")
ax.set_title("Val AUC — Early Stopping in Action", color="white")
ax.set_xlabel("Iteration"); ax.set_ylabel("AUC"); ax.legend(); ax.grid(alpha=0.15)

# Without early stopping — trained full 500
ax = axes[1]
iters_b = range(len(evals_result["train"]["auc"]))
ax.plot(iters_b, evals_result["train"]["auc"], color=COLORS["train"], lw=2, label="Train (no early stop)")
ax.plot(iters_b, evals_result["val"]["auc"],   color=COLORS["val"],   lw=2, label="Val (no early stop)")
ax.fill_between(iters_b, evals_result["train"]["auc"], evals_result["val"]["auc"], alpha=0.1, color=COLORS["val"])
ax.set_title("Without Early Stopping (Full 500 iters)", color="white")
ax.set_xlabel("Iteration"); ax.set_ylabel("AUC"); ax.legend(); ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig("/home/claude/sec4_earlystop.png", dpi=120, bbox_inches="tight", facecolor="#0a0c10")
plt.close()
print("\n  Saved: sec4_earlystop.png\n")


# ============================================================
#  SECTION 5 — CLASS IMBALANCE (20% charge-off)
# ============================================================
print("=" * 60)
print("SECTION 5 — Handling 20% Class Imbalance")
print("=" * 60)
print("""
SCENARIO: A naive model maximises accuracy by predicting 'no charge-off'
most of the time. We compare 4 strategies to boost minority recall:
  1. No correction (baseline)
  2. scale_pos_weight = 4  (≈ 80/20 ratio)
  3. is_unbalance = True   (auto-weights)
  4. SMOTE oversampling    (synthetic minority samples)
""")

from imblearn.over_sampling import SMOTE

base_params = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "num_leaves": 31, "max_depth": 6, "min_child_samples": 100,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "lambda_l1": 0.1, "lambda_l2": 0.1, "learning_rate": 0.05,
}

imb_results = {}

# Strategy 1: No correction
m1 = lgb.train({**base_params}, train_data, num_boost_round=300, callbacks=[lgb.log_evaluation(-1)])
preds1 = (m1.predict(X_val) > 0.5).astype(int)
imb_results["No correction"] = {
    "auc":  roc_auc_score(y_val, m1.predict(X_val)),
    "prec": precision_score(y_val, preds1, zero_division=0),
    "rec":  recall_score(y_val, preds1, zero_division=0),
    "f1":   f1_score(y_val, preds1, zero_division=0),
}

# Strategy 2: scale_pos_weight
m2 = lgb.train({**base_params, "scale_pos_weight": 4}, train_data,
               num_boost_round=300, callbacks=[lgb.log_evaluation(-1)])
preds2 = (m2.predict(X_val) > 0.5).astype(int)
imb_results["scale_pos_weight=4"] = {
    "auc":  roc_auc_score(y_val, m2.predict(X_val)),
    "prec": precision_score(y_val, preds2, zero_division=0),
    "rec":  recall_score(y_val, preds2, zero_division=0),
    "f1":   f1_score(y_val, preds2, zero_division=0),
}

# Strategy 3: is_unbalance
m3 = lgb.train({**base_params, "is_unbalance": True}, train_data,
               num_boost_round=300, callbacks=[lgb.log_evaluation(-1)])
preds3 = (m3.predict(X_val) > 0.5).astype(int)
imb_results["is_unbalance=True"] = {
    "auc":  roc_auc_score(y_val, m3.predict(X_val)),
    "prec": precision_score(y_val, preds3, zero_division=0),
    "rec":  recall_score(y_val, preds3, zero_division=0),
    "f1":   f1_score(y_val, preds3, zero_division=0),
}

# Strategy 4: SMOTE (on a 50k subsample for speed)
X_tr_sm = X_train.iloc[:50_000]; y_tr_sm = y_train.iloc[:50_000]
smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_tr_sm, y_tr_sm)
td_sm = lgb.Dataset(X_res, label=y_res)
m4 = lgb.train({**base_params}, td_sm, num_boost_round=300, callbacks=[lgb.log_evaluation(-1)])
preds4 = (m4.predict(X_val) > 0.5).astype(int)
imb_results["SMOTE"] = {
    "auc":  roc_auc_score(y_val, m4.predict(X_val)),
    "prec": precision_score(y_val, preds4, zero_division=0),
    "rec":  recall_score(y_val, preds4, zero_division=0),
    "f1":   f1_score(y_val, preds4, zero_division=0),
}

print(f"\n  {'Strategy':<22} {'AUC':>6} {'Precision':>10} {'Recall':>8} {'F1':>6}")
print("  " + "-" * 55)
for name, r in imb_results.items():
    print(f"  {name:<22} {r['auc']:>6.4f} {r['prec']:>10.4f} {r['rec']:>8.4f} {r['f1']:>6.4f}")

# ── Confusion matrices ───────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Section 5 — Confusion Matrices by Imbalance Strategy", fontsize=13, color="white")
models_preds = [("No correction", preds1), ("scale_pos_weight=4", preds2),
                ("is_unbalance", preds3),   ("SMOTE", preds4)]
for ax, (name, preds) in zip(axes, models_preds):
    cm = confusion_matrix(y_val, preds, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2%", ax=ax, cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"],
                linewidths=0.5, cbar=False)
    r = imb_results[name]
    ax.set_title(f"{name}\nF1={r['f1']:.3f}  Recall={r['rec']:.3f}", color="white", fontsize=10)

plt.tight_layout()
plt.savefig("/home/claude/sec5_imbalance.png", dpi=120, bbox_inches="tight", facecolor="#0a0c10")
plt.close()
print("\n  Saved: sec5_imbalance.png\n")


# ============================================================
#  SECTION 6 — CROSS-VALIDATION
# ============================================================
print("=" * 60)
print("SECTION 6 — Stratified K-Fold Cross-Validation")
print("=" * 60)
print("""
SCENARIO: Instead of a single train/val split, we use 5-fold
stratified CV to get a more reliable estimate of generalisation.
Each fold preserves the 20% charge-off ratio.
""")

best_params = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "num_leaves": 31, "max_depth": 6, "min_child_samples": 100,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "lambda_l1": 0.1, "lambda_l2": 0.1,
    "scale_pos_weight": 4,
    "learning_rate": 0.05,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs, fold_gaps = [], []
print(f"\n  {'Fold':<6} {'Train AUC':>10} {'Val AUC':>9} {'Gap':>8}")
print("  " + "-" * 36)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full)):
    Xtr, Xva = X_full.iloc[tr_idx], X_full.iloc[va_idx]
    ytr, yva = y_full.iloc[tr_idx], y_full.iloc[va_idx]
    td = lgb.Dataset(Xtr, label=ytr)
    vd = lgb.Dataset(Xva, label=yva, reference=td)
    ev = {}
    m = lgb.train(best_params, td, num_boost_round=500,
                  valid_sets=[td, vd], valid_names=["train","val"],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.record_evaluation(ev), lgb.log_evaluation(-1)])
    tr_auc = roc_auc_score(ytr, m.predict(Xtr))
    va_auc = roc_auc_score(yva, m.predict(Xva))
    gap = tr_auc - va_auc
    fold_aucs.append(va_auc); fold_gaps.append(gap)
    print(f"  Fold {fold+1}    {tr_auc:.4f}     {va_auc:.4f}   {gap:.4f}")

print(f"\n  Mean Val AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"  Mean Gap     : {np.mean(fold_gaps):.4f} ± {np.std(fold_gaps):.4f}")

# ── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Section 6 — 5-Fold Stratified CV", fontsize=14, color="white")

ax = axes[0]
ax.bar(range(1,6), fold_aucs, color=COLORS["train"], alpha=0.8, edgecolor="white", lw=0.5)
ax.axhline(np.mean(fold_aucs), color=COLORS["warn"], lw=2, linestyle="--",
           label=f"Mean={np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
ax.set_xticks(range(1,6)); ax.set_xticklabels([f"Fold {i}" for i in range(1,6)])
ax.set_title("Val AUC per Fold", color="white"); ax.set_ylabel("AUC")
ax.legend(); ax.grid(alpha=0.15, axis="y"); ax.set_ylim(0.80, 0.92)

ax = axes[1]
ax.bar(range(1,6), fold_gaps, color=COLORS["val"], alpha=0.8, edgecolor="white", lw=0.5)
ax.axhline(np.mean(fold_gaps), color=COLORS["warn"], lw=2, linestyle="--",
           label=f"Mean gap={np.mean(fold_gaps):.4f}")
ax.set_xticks(range(1,6)); ax.set_xticklabels([f"Fold {i}" for i in range(1,6)])
ax.set_title("Overfit Gap per Fold", color="white"); ax.set_ylabel("Train AUC − Val AUC")
ax.legend(); ax.grid(alpha=0.15, axis="y")

plt.tight_layout()
plt.savefig("/home/claude/sec6_cv.png", dpi=120, bbox_inches="tight", facecolor="#0a0c10")
plt.close()
print("\n  Saved: sec6_cv.png\n")


# ============================================================
#  SECTION 7 — FINAL COMPARISON
# ============================================================
print("=" * 60)
print("SECTION 7 — Final Comparison: All Experiments")
print("=" * 60)

all_results = {
    "Baseline (overfit)":   res_baseline,
    "Regularization":       res_reg,
    "Subsampling (0.8)":    res_sub,
    "Early Stopping":       res_es,
    "Best (CV mean)":       {"train": None, "val": np.mean(fold_aucs),
                              "test": None, "gap": np.mean(fold_gaps)},
}

print(f"\n  {'Model':<24} {'Val AUC':>9} {'Test AUC':>10} {'Overfit Gap':>13}")
print("  " + "-" * 60)
for name, r in all_results.items():
    val_str  = f"{r['val']:.4f}" if r['val']  else "   —  "
    test_str = f"{r['test']:.4f}" if r.get('test') and r['test'] else "   —  "
    flag = "⚠️ " if r["gap"] > 0.05 else ("✅" if r["gap"] < 0.02 else "🟡")
    print(f"  {name:<24} {val_str:>9} {test_str:>10} {r['gap']:>10.4f}  {flag}")

# ── Final summary plot ───────────────────────────────────────
names     = list(all_results.keys())
val_aucs  = [r["val"] for r in all_results.values()]
gaps      = [r["gap"] for r in all_results.values()]
bar_colors = [COLORS["val"] if g > 0.05 else (COLORS["warn"] if g > 0.02 else COLORS["good"]) for g in gaps]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Section 7 — All Experiments: Final Comparison", fontsize=14, color="white")

ax = axes[0]
bars = ax.barh(names, val_aucs, color=bar_colors, alpha=0.85, edgecolor="white", lw=0.5)
for bar, v in zip(bars, val_aucs):
    ax.text(v + 0.001, bar.get_y() + bar.get_height()/2, f"{v:.4f}",
            va="center", color="white", fontsize=10)
ax.set_title("Val AUC by Experiment", color="white")
ax.set_xlabel("AUC"); ax.grid(alpha=0.15, axis="x")
ax.set_xlim(0.78, 0.93)

ax = axes[1]
bars2 = ax.barh(names, gaps, color=bar_colors, alpha=0.85, edgecolor="white", lw=0.5)
for bar, v in zip(bars2, gaps):
    ax.text(v + 0.0005, bar.get_y() + bar.get_height()/2, f"{v:.4f}",
            va="center", color="white", fontsize=10)
ax.axvline(0.05, color=COLORS["val"],  lw=1.5, linestyle="--", label="Severe overfit threshold")
ax.axvline(0.02, color=COLORS["good"], lw=1.5, linestyle="--", label="Target gap")
ax.set_title("Overfit Gap by Experiment (lower = better)", color="white")
ax.set_xlabel("Train AUC − Val AUC"); ax.legend(fontsize=9); ax.grid(alpha=0.15, axis="x")

plt.tight_layout()
plt.savefig("/home/claude/sec7_comparison.png", dpi=120, bbox_inches="tight", facecolor="#0a0c10")
plt.close()
print("\n  Saved: sec7_comparison.png")

# ============================================================
print("\n" + "=" * 60)
print("  RECOMMENDED FINAL CONFIG  (copy-paste ready)")
print("=" * 60)
print("""
params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    # ── Tree complexity ────────────────────────────
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 100,
    # ── Regularization ────────────────────────────
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_gain_to_split": 0.01,
    # ── Stochastic boosting ───────────────────────
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    # ── Class imbalance (20% charge-off) ─────────
    "scale_pos_weight": 4,      # ≈ 80/20
    # ── Learning ──────────────────────────────────
    "learning_rate": 0.05,
    "n_estimators": 2000,       # ceiling — early stopping decides
    "random_state": 42,
}

model = lgb.train(
    params, train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(100),
    ],
)
""")
print("All plots saved to /home/claude/sec*.png")
