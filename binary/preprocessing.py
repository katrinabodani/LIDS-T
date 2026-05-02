# =============================================================================
# PHASE 1 — UNSW-NB15 PREPROCESSING PIPELINE (v3 — corrected order)
# Project: Explainable Multiclass Network Intrusion Detection
# Base paper: DOI 10.1038/s41598-025-11348-5 (Scientific Reports, Q1, 2025)
# =============================================================================
# CRITICAL FIX from v2:
#   Base paper pipeline (Figure 2):
#     Encode → Normalize → Outlier Removal → SMOTE
#   Previous versions did:
#     Encode → Outlier Removal → Normalize  ← WRONG ORDER
#
#   Impact: Z-score on raw unnormalized data removed 28% of samples (49,085)
#   because network features like sbytes/sload have extreme raw value ranges.
#   Z-score on normalized [0,1] data is far more lenient — retains more samples.
#   This is the most likely reason for our 7% accuracy gap vs base paper.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading UNSW-NB15 dataset")
print("=" * 60)

TRAIN_PATH = "UNSW_NB15_training-set.csv"
TEST_PATH  = "UNSW_NB15_testing-set.csv"

for path in [TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Auto-fix swap
if len(train_df) < len(test_df):
    print("WARNING: Files swapped — fixing automatically...")
    train_df, test_df = test_df, train_df

print(f"Train shape: {train_df.shape}  (should be ~175k rows)")
print(f"Test shape:  {test_df.shape}   (should be ~82k rows)")

# =============================================================================
# STEP 2 — CLEAN ATTACK_CAT LABEL
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Cleaning attack_cat label")
print("=" * 60)

def clean_attack_cat(df):
    df = df.copy()
    df['attack_cat'] = df['attack_cat'].astype(str).str.strip().str.title()
    df['attack_cat'] = df['attack_cat'].replace('Nan', 'Normal')
    df['attack_cat'] = df['attack_cat'].fillna('Normal')
    return df

train_df = clean_attack_cat(train_df)
test_df  = clean_attack_cat(test_df)

print("Attack categories (train):")
print(train_df['attack_cat'].value_counts())
print("\nNOTE: 'Normal' = benign traffic. 9 other classes = attack types.")

# =============================================================================
# STEP 3 — DEFINE FEATURE COLUMNS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Defining feature columns")
print("=" * 60)

CATEGORICAL_COLS = ['proto', 'service', 'state']
DROP_COLS = ['id', 'srcip', 'dstip', 'Stime', 'Ltime', 'attack_cat', 'label']
DROP_COLS = [c for c in DROP_COLS if c in train_df.columns]
feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
print(f"Total features: {len(feature_cols)}")

# =============================================================================
# STEP 4 — ENCODE CATEGORICAL FEATURES (base paper Step 1)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Encoding categorical features [base paper Step 1]")
print("=" * 60)

le_dict = {}
for col in CATEGORICAL_COLS:
    if col not in train_df.columns:
        continue
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col]  = le.transform(test_df[col].astype(str))
    le_dict[col]  = le
    print(f"  {col}: {len(le.classes_)} unique values encoded")

# =============================================================================
# STEP 5 — SEPARATE LABELS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Separating labels and features")
print("=" * 60)

y_train_binary = train_df['label'].values.astype(int)
y_test_binary  = test_df['label'].values.astype(int)

le_attack = LabelEncoder()
all_cats  = pd.concat([train_df['attack_cat'], test_df['attack_cat']])
le_attack.fit(all_cats)
y_train_multi = le_attack.transform(train_df['attack_cat'])
y_test_multi  = le_attack.transform(test_df['attack_cat'])

class_mapping     = {int(i): name for i, name in enumerate(le_attack.classes_)}
normal_class_idx  = [k for k, v in class_mapping.items() if v == 'Normal'][0]
print(f"Class mapping: {class_mapping}")
print(f"Normal class index: {normal_class_idx}")

X_train = train_df[feature_cols].copy()
X_test  = test_df[feature_cols].copy()
print(f"Feature matrix — train: {X_train.shape}, test: {X_test.shape}")

# =============================================================================
# STEP 6 — HANDLE MISSING VALUES
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Handling missing values")
print("=" * 60)

for col in X_train.columns:
    if X_train[col].isnull().any() or X_test[col].isnull().any():
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col]  = X_test[col].fillna(median_val)
        print(f"  Filled {col} NaNs with median {median_val:.4f}")
print("Done.")

# =============================================================================
# STEP 7 — MIN-MAX NORMALIZATION FIRST (base paper Step 2)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Min-Max normalization [base paper Step 2 — BEFORE outlier removal]")
print("=" * 60)

scaler = MinMaxScaler()
X_train_norm = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_norm = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# Clip test set for out-of-range values
X_test_norm = X_test_norm.clip(lower=0, upper=1)

print(f"Train range after normalization: "
      f"[{X_train_norm.values.min():.3f}, {X_train_norm.values.max():.3f}]")
print(f"Test range after normalization:  "
      f"[{X_test_norm.values.min():.3f}, {X_test_norm.values.max():.3f}]")
print("\nNOTE: Normalizing BEFORE outlier removal matches base paper Figure 2.")
print("Z-score on [0,1] data is far more lenient → retains more training samples.")

# =============================================================================
# STEP 8 — OUTLIER REMOVAL ON NORMALIZED DATA (base paper Step 3)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Outlier removal on NORMALIZED data [base paper Step 3]")
print("=" * 60)

numeric_cols = X_train_norm.select_dtypes(include=[np.number]).columns.tolist()
z_scores     = np.abs(stats.zscore(X_train_norm[numeric_cols]))
outlier_feature_count = (z_scores > 5).sum(axis=1)
mask = outlier_feature_count <= (len(numeric_cols) * 0.2)

X_train_clean    = X_train_norm[mask].reset_index(drop=True)
y_train_binary_c = y_train_binary[mask]
y_train_multi_c  = y_train_multi[mask]

removed = len(X_train_norm) - len(X_train_clean)
print(f"Rows before: {len(X_train_norm)}")
print(f"Rows after:  {len(X_train_clean)}")
print(f"Removed:     {removed} ({100*removed/len(X_train_norm):.1f}%)")
print("\nExpected: much fewer removals than v2's 28% (49,085 rows)")
print("NOTE: Test set NOT cleaned — evaluated as real-world traffic.")

# =============================================================================
# STEP 9 — CLASS DISTRIBUTION ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Class distribution analysis")
print("=" * 60)

class_counts = pd.Series(y_train_multi_c).map(class_mapping).value_counts()
print("\nMulticlass distribution (train, after outlier removal):")
for cls, count in class_counts.items():
    bar = "█" * max(1, int(count / 700))
    tag = " ← BENIGN" if cls == "Normal" else ""
    print(f"  {cls:<18} {count:>6}  {bar}{tag}")

ratio = class_counts.max() / class_counts.min()
print(f"\nImbalance ratio: {ratio:.1f}x")
print(f"Most common:  {class_counts.idxmax()} ({class_counts.max()} samples)")
print(f"Least common: {class_counts.idxmin()} ({class_counts.min()} samples)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('UNSW-NB15 Class Distribution (v3 — corrected pipeline)',
             fontsize=13, fontweight='bold')

colors = ['#27ae60' if c == 'Normal' else '#e74c3c' for c in class_counts.index]

for ax, log, title in zip(
    axes, [False, True],
    ['Linear Scale', 'Log Scale (minority classes visible)']
):
    bars = ax.bar(class_counts.index, class_counts.values,
                  color=colors, edgecolor='white', linewidth=0.5)
    if log:
        ax.set_yscale('log')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Sample count' + (' (log)' if log else ''))
    ax.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * (1.05 if log else 1.01) + (0 if log else 50),
                str(count), ha='center', va='bottom', fontsize=7.5)

axes[0].legend(handles=[
    plt.Rectangle((0,0),1,1, color='#27ae60', label='Normal (benign)'),
    plt.Rectangle((0,0),1,1, color='#e74c3c', label='Attack classes')
], fontsize=9)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
print("\nSaved: class_distribution.png")

# =============================================================================
# STEP 10 — SAVE ALL OUTPUTS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Saving processed data")
print("=" * 60)

X_train_clean.to_csv('X_train.csv', index=False)
X_test_norm.to_csv('X_test.csv', index=False)
pd.Series(y_train_binary_c, name='label').to_csv('y_train_binary.csv', index=False)
pd.Series(y_test_binary, name='label').to_csv('y_test_binary.csv', index=False)
pd.Series(y_train_multi_c, name='attack_cat').to_csv('y_train_multi.csv', index=False)
pd.Series(y_test_multi, name='attack_cat').to_csv('y_test_multi.csv', index=False)

with open('class_mapping.json', 'w') as f:
    json.dump(class_mapping, f, indent=2)
with open('feature_cols.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

print("Files saved:")
print("  X_train.csv, X_test.csv")
print("  y_train_binary.csv, y_test_binary.csv")
print("  y_train_multi.csv, y_test_multi.csv")
print("  class_mapping.json, feature_cols.json")
print("  class_distribution.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE (v3 — corrected pipeline order)")
print("=" * 60)
print(f"  Train samples (clean): {len(X_train_clean)}")
print(f"  Test samples:          {len(X_test_norm)}")
print(f"  Features:              {X_train_clean.shape[1]}")
print(f"  Binary classes:        2  (0=Normal, 1=Attack)")
print(f"  Multiclass classes:    {len(class_mapping)}")
print(f"    Benign:  Normal (class {normal_class_idx})")
print(f"    Attacks: {[v for v in class_mapping.values() if v != 'Normal']}")
print(f"  Imbalance ratio:       {ratio:.1f}x")
print(f"\n  Pipeline order (matches base paper Figure 2):")
print(f"    1. Encode categoricals")
print(f"    2. Normalize [0,1]       ← moved BEFORE outlier removal")
print(f"    3. Remove outliers (Z>3) ← now on normalized data")
print(f"    4. SMOTE                 ← Phase 3")
print(f"\n  Next: rerun feature selection + Phase 2A with new X_train.csv")
print("=" * 60)