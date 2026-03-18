# LightGBM Overfit Lab — Credit Risk (Fictitious Data)

## Quick Start

```bash
pip install lightgbm scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
python lgbm_overfit_experiments.py
```

## What's Inside

| Section | Topic | Key Params |
|---------|-------|------------|
| 0 | Generate 700k fictitious credit-risk rows | 24 features, 20% charge-off |
| 1 | Baseline — intentional overfit | num_leaves=400, min_child_samples=5 |
| 2 | Fix with Regularization | L1/L2, max_depth, min_gain_to_split |
| 3 | Fix with Subsampling | subsample=0.8, colsample_bytree=0.8 |
| 4 | Fix with Early Stopping | stopping_rounds=50, lr=0.05 |
| 5 | Handle 20% Class Imbalance | scale_pos_weight, is_unbalance, SMOTE |
| 6 | Stratified 5-Fold CV | StratifiedKFold, gap stability |
| 7 | Final Comparison + Best Config | Copy-paste ready params |

## Output Charts
- `sec1_baseline.png`    — training curves + feature importance
- `sec2_regularization.png` — before/after comparison
- `sec3_subsampling.png` — subsample rate sweep + CV variance
- `sec4_earlystop.png`   — best iteration detection
- `sec5_imbalance.png`   — confusion matrices per strategy
- `sec6_cv.png`          — fold-by-fold AUC + gap
- `sec7_comparison.png`  — all experiments side-by-side
