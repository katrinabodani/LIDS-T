# =============================================================================
# PHASE 2 — WRAPPER-BASED FEATURE SELECTION (RF + XGBoost)
# Project: Explainable Multiclass Network Intrusion Detection
# Base paper: DOI 10.1038/s41598-025-11348-5 (Scientific Reports, Q1, 2025)
# =============================================================================
# Base paper uses: SVM + Random Forest + Naive Bayes → top 20 features
# Our improvement: Random Forest + XGBoost → top 20 features
#
# Why RF + XGBoost instead of RF + NB + SVM:
#   - Naive Bayes assumes feature independence — wrong for network traffic
#     where features are deeply correlated (e.g. sbytes and sload)
#   - SVM feature importance is not natively supported — requires workarounds
#   - XGBoost is a gradient boosting method — different family from RF,
#     giving genuinely diverse importance scores
#   - RF + XGBoost = two strong, diverse, natively importance-aware selectors
#
# Install: pip install xgboost
# Run:     python phase2_feature_selection.py
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

TARGET_FEATURES = 20    # matching base paper's 20 feature target
SEED            = 42
N_ESTIMATORS    = 100   # trees for RF and XGBoost

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading preprocessed data")
print("=" * 60)

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train_binary.csv').values.ravel()

with open('feature_cols.json') as f:
    feature_cols = json.load(f)

print(f"Training samples: {X_train.shape[0]}")
print(f"Total features:   {X_train.shape[1]}")
print(f"Feature names:    {feature_cols}")

# Use a subsample for speed — 50k samples is enough for stable importance
# Full dataset would take too long for RF fitting
SAMPLE_SIZE = 50000
if len(X_train) > SAMPLE_SIZE:
    idx = np.random.RandomState(SEED).choice(
        len(X_train), SAMPLE_SIZE, replace=False
    )
    X_sample = X_train.values[idx]
    y_sample = y_train[idx]
    print(f"\nUsing {SAMPLE_SIZE} samples for feature selection (faster)")
else:
    X_sample = X_train.values
    y_sample = y_train

# =============================================================================
# STEP 2 — RANDOM FOREST FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Random Forest feature importance")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=SEED,
    n_jobs=-1,          # use all CPU cores
    max_depth=15,
    min_samples_leaf=5
)
rf.fit(X_sample, y_sample)

rf_importances = pd.Series(
    rf.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nRandom Forest — top 25 features by importance:")
for i, (feat, imp) in enumerate(rf_importances.head(25).items()):
    bar = "█" * int(imp * 500)
    print(f"  {i+1:>2}. {feat:<25} {imp:.6f}  {bar}")

# =============================================================================
# STEP 3 — XGBOOST FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: XGBoost feature importance")
print("=" * 60)

xgb_model = xgb.XGBClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=SEED,
    n_jobs=-1,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_sample, y_sample)

xgb_importances = pd.Series(
    xgb_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nXGBoost — top 25 features by importance:")
for i, (feat, imp) in enumerate(xgb_importances.head(25).items()):
    bar = "█" * int(imp * 300)
    print(f"  {i+1:>2}. {feat:<25} {imp:.6f}  {bar}")

# =============================================================================
# STEP 4 — COMBINE: UNION OF TOP FEATURES (matching base paper method)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Combining selectors — union of top features")
print("=" * 60)

# Normalize importances to [0,1] so both selectors contribute equally
rf_norm  = rf_importances  / rf_importances.max()
xgb_norm = xgb_importances / xgb_importances.max()

# Combined score = average of normalized importances from both selectors
combined_score = (rf_norm + xgb_norm) / 2
combined_score = combined_score.sort_values(ascending=False)

print("\nCombined importance score (RF + XGBoost normalized average):")
print(f"\n  {'Rank':<6} {'Feature':<28} {'RF rank':<10} {'XGB rank':<10} {'Combined'}")
print(f"  {'-'*65}")

rf_ranks  = {feat: i+1 for i, feat in enumerate(rf_importances.index)}
xgb_ranks = {feat: i+1 for i, feat in enumerate(xgb_importances.index)}

for rank, (feat, score) in enumerate(combined_score.items(), 1):
    marker = " ←" if rank <= TARGET_FEATURES else ""
    print(f"  {rank:<6} {feat:<28} {rf_ranks[feat]:<10} "
          f"{xgb_ranks[feat]:<10} {score:.4f}{marker}")

# Select top N features
selected_features = list(combined_score.head(TARGET_FEATURES).index)
print(f"\nSelected {TARGET_FEATURES} features:")
print(selected_features)

# =============================================================================
# STEP 5 — ANALYZE WHAT WAS SELECTED VS DROPPED
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Analysis — selected vs dropped features")
print("=" * 60)

dropped_features = [f for f in feature_cols if f not in selected_features]
print(f"\nKept ({len(selected_features)}):   {selected_features}")
print(f"\nDropped ({len(dropped_features)}): {dropped_features}")

# Check if categorical features made it
cats = ['proto', 'service', 'state']
print(f"\nCategorical features status:")
for cat in cats:
    status = "✓ KEPT" if cat in selected_features else "✗ DROPPED"
    rank   = list(combined_score.index).index(cat) + 1
    print(f"  {cat:<12} {status}  (rank {rank}/{len(feature_cols)})")

# =============================================================================
# STEP 6 — SAVE REDUCED FEATURE SETS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Saving reduced feature datasets")
print("=" * 60)

# Load full datasets and reduce
X_train_full = pd.read_csv('X_train.csv')
X_test_full  = pd.read_csv('X_test.csv')

X_train_reduced = X_train_full[selected_features]
X_test_reduced  = X_test_full[selected_features]

X_train_reduced.to_csv('X_train_20.csv', index=False)
X_test_reduced.to_csv('X_test_20.csv', index=False)

with open('selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2)

# Save importance scores for paper figures
importance_df = pd.DataFrame({
    'feature':   combined_score.index,
    'rf_importance':  rf_importances[combined_score.index].values,
    'xgb_importance': xgb_importances[combined_score.index].values,
    'combined_score': combined_score.values,
    'selected':  [f in selected_features for f in combined_score.index]
})
importance_df.to_csv('feature_importances.csv', index=False)

print("Saved:")
print("  X_train_20.csv        — reduced training features (20 cols)")
print("  X_test_20.csv         — reduced test features (20 cols)")
print("  selected_features.json — list of selected feature names")
print("  feature_importances.csv — full importance scores")

# =============================================================================
# STEP 7 — VISUALIZATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Generating feature importance plot")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    'Wrapper-Based Feature Selection — RF + XGBoost\n'
    '(Improvement over base paper: replaces Naïve Bayes with XGBoost)',
    fontsize=12, fontweight='bold'
)

top_n = 20  # show top 20 in plots

# RF importance
ax = axes[0]
top_rf = rf_importances.head(top_n)
colors = ['#2ecc71' if f in selected_features else '#95a5a6'
          for f in top_rf.index]
ax.barh(range(len(top_rf)), top_rf.values[::-1], color=colors[::-1])
ax.set_yticks(range(len(top_rf)))
ax.set_yticklabels(top_rf.index[::-1], fontsize=8)
ax.set_title('Random Forest', fontweight='bold')
ax.set_xlabel('Importance score')

# XGBoost importance
ax = axes[1]
top_xgb = xgb_importances.head(top_n)
colors = ['#2ecc71' if f in selected_features else '#95a5a6'
          for f in top_xgb.index]
ax.barh(range(len(top_xgb)), top_xgb.values[::-1], color=colors[::-1])
ax.set_yticks(range(len(top_xgb)))
ax.set_yticklabels(top_xgb.index[::-1], fontsize=8)
ax.set_title('XGBoost', fontweight='bold')
ax.set_xlabel('Importance score')

# Combined score — all features
ax = axes[2]
colors = ['#27ae60' if f in selected_features else '#e74c3c'
          for f in combined_score.index]
ax.barh(range(len(combined_score)),
        combined_score.values[::-1],
        color=colors[::-1])
ax.set_yticks(range(len(combined_score)))
ax.set_yticklabels(combined_score.index[::-1], fontsize=7.5)
ax.set_title(f'Combined Score\n(green = selected top {TARGET_FEATURES})',
             fontweight='bold')
ax.set_xlabel('Normalized combined importance')
ax.axhline(y=len(combined_score) - TARGET_FEATURES - 0.5,
           color='black', linestyle='--', linewidth=1.5, alpha=0.7)

plt.tight_layout()
plt.savefig('feature_selection.png', dpi=150, bbox_inches='tight')
print("Saved: feature_selection.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FEATURE SELECTION COMPLETE")
print("=" * 60)
print(f"  Original features:  {len(feature_cols)}")
print(f"  Selected features:  {len(selected_features)}")
print(f"  Dropped features:   {len(dropped_features)}")
print(f"\n  Selected: {selected_features}")
print(f"\n  Method: RF + XGBoost normalized average")
print(f"  Improvement over base paper: replaced Naive Bayes with XGBoost")
print(f"\n  Next: retrain Phase 2A using X_train_20.csv / X_test_20.csv")
print("=" * 60)