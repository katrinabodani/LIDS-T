# =============================================================================
# PREPROCESS_FULL.PY — MULTICLASS ONLY
# Project : Explainable Multiclass Network Intrusion Detection (LIDS-T)
# Input   : UNSW-NB15_1.csv … UNSW-NB15_4.csv  (raw, no header, 49 cols)
# Output  : X_train_full.csv, X_test_full.csv
#           y_train_multi_full.csv, y_test_multi_full.csv
#           selected_features.json
#           attack_class_names.json
# =============================================================================

import pandas as pd
import numpy as np
import json
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.preprocessing     import LabelEncoder, MinMaxScaler
from sklearn.model_selection   import train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.svm               import LinearSVC
from sklearn.ensemble          import RandomForestClassifier

SEED               = 42
TEST_SIZE          = 0.20
FS_SAMPLE          = 150_000
N_FEATURES_SELECT  = 20

np.random.seed(SEED)

# =============================================================================
# COLUMN NAMES — 49 cols, no header in raw files
# Col 47 = attack_cat (string), Col 48 = label (unused here)
# =============================================================================

COL_NAMES = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'attack_cat', 'label'
]

CAT_COLS  = ['proto', 'state', 'service']
DROP_COLS = ['srcip', 'dstip', 'Stime', 'Ltime', 'sport', 'dsport',
             'attack_cat', 'label']

# =============================================================================
# STEP 1 — LOAD ALL 4 RAW CSVs
# =============================================================================

print("=" * 60)
print("STEP 1: Loading raw CSVs")
print("=" * 60)

dfs = []
for i in range(1, 5):
    fname = f'UNSW-NB15_{i}.csv'
    t0    = time.time()
    df_i  = pd.read_csv(fname, header=None, names=COL_NAMES, low_memory=False)
    print(f"  {fname}: {df_i.shape[0]:,} rows  ({time.time()-t0:.1f}s)")
    dfs.append(df_i)

df = pd.concat(dfs, ignore_index=True)
print(f"\n  Combined: {df.shape[0]:,} rows x {df.shape[1]} columns")

# =============================================================================
# STEP 2 — EXTRACT MULTICLASS LABEL
# attack_cat is a string — empty string means Normal traffic
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Extracting multiclass labels")
print("=" * 60)

df['attack_cat'] = (df['attack_cat']
                    .fillna('Normal')          # catch actual NaN first
                    .astype(str)
                    .str.strip()
                    .replace({'': 'Normal', 'Backdoors': 'Backdoor'}))

le          = LabelEncoder()
y_multi     = le.fit_transform(df['attack_cat'])
class_names = le.classes_.tolist()

print(f"  {len(class_names)} classes found:")
for i, cls in enumerate(class_names):
    count = (y_multi == i).sum()
    pct   = count / len(y_multi) * 100
    print(f"    {i}: {cls:<20} {count:>8,}  ({pct:.1f}%)")

with open('attack_class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print("\n  Saved: attack_class_names.json")

# =============================================================================
# STEP 3 — BUILD FEATURE MATRIX
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Building feature matrix")
print("=" * 60)

df_feat = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

for col in CAT_COLS:
    if col in df_feat.columns:
        enc = LabelEncoder()
        df_feat[col] = enc.fit_transform(df_feat[col].astype(str))
        print(f"  Encoded {col}: {len(enc.classes_)} unique values")

for col in df_feat.columns:
    df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')

nan_cols = df_feat.columns[df_feat.isna().any()].tolist()
if nan_cols:
    print(f"  Filling NaN in: {nan_cols}")
    for col in nan_cols:
        df_feat[col].fillna(df_feat[col].median(), inplace=True)

df_feat = df_feat.astype(np.float32)
print(f"\n  Feature matrix: {df_feat.shape}")

# =============================================================================
# STEP 4 — OUTLIER CLIPPING (Z-SCORE |z| > 3)
# Clip instead of drop — keeps X and y perfectly aligned
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Outlier clipping (|z| > 3)")
print("=" * 60)

t0      = time.time()
mean    = df_feat.mean()
std     = df_feat.std().replace(0, 1)
df_feat = df_feat.clip(lower=mean - 3*std, upper=mean + 3*std, axis=1)
print(f"  Done ({time.time()-t0:.1f}s)")

# =============================================================================
# STEP 5 — MIN-MAX NORMALIZATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Min-Max normalization [0, 1]")
print("=" * 60)

t0       = time.time()
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(df_feat).astype(np.float32)
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=0.0)
print(f"  Done ({time.time()-t0:.1f}s)  Shape: {X_scaled.shape}")

col_names_feat = df_feat.columns.tolist()

# =============================================================================
# STEP 6 — STRATIFIED 80/20 TRAIN/TEST SPLIT
# Split BEFORE feature selection — no leakage
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Stratified 80/20 train/test split")
print("=" * 60)

t0 = time.time()
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y_multi,
    test_size    = TEST_SIZE,
    random_state = SEED,
    stratify     = y_multi
)
print(f"  Train: {X_train_full.shape[0]:,} rows  ({time.time()-t0:.1f}s)")
print(f"  Test : {X_test_full.shape[0]:,} rows")

# =============================================================================
# STEP 7 — WRAPPER-BASED FEATURE SELECTION
# 3 algorithms on 150k sample from TRAIN only — no leakage
# Top 20 by selection frequency across all 3 algorithms
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Wrapper-based feature selection")
print(f"  Sample: {FS_SAMPLE:,} rows from training set")
print("=" * 60)

sss    = StratifiedShuffleSplit(n_splits=1, train_size=FS_SAMPLE,
                                 random_state=SEED)
idx, _ = next(sss.split(X_train_full, y_train))
X_fs   = X_train_full[idx]
y_fs   = y_train[idx]
print(f"  Sample shape: {X_fs.shape}")

freq = np.zeros(X_fs.shape[1], dtype=int)

# --- 1: LinearSVC + RFE ---
print("\n  [1/3] LinearSVC + RFE ...")
t0      = time.time()
rfe_svm = RFE(LinearSVC(max_iter=2000, C=0.1, random_state=SEED),
              n_features_to_select=N_FEATURES_SELECT, step=5)
rfe_svm.fit(X_fs, y_fs)
freq   += rfe_svm.support_.astype(int)
print(f"        Done ({time.time()-t0:.1f}s)")
print(f"        {np.array(col_names_feat)[rfe_svm.support_].tolist()}")

# --- 2: Random Forest importance ---
print("\n  [2/3] Random Forest importance ...")
t0      = time.time()
rf      = RandomForestClassifier(n_estimators=100, max_depth=10,
                                  n_jobs=-1, random_state=SEED)
rf.fit(X_fs, y_fs)
top_idx = np.argsort(rf.feature_importances_)[::-1][:N_FEATURES_SELECT]
rf_mask = np.zeros(X_fs.shape[1], dtype=bool)
rf_mask[top_idx] = True
freq   += rf_mask.astype(int)
print(f"        Done ({time.time()-t0:.1f}s)")
print(f"        {np.array(col_names_feat)[rf_mask].tolist()}")

# --- 3: Naive Bayes + RFE ---
# --- 3: SelectKBest with mutual information (NB-equivalent) ---
from sklearn.feature_selection import SelectKBest, mutual_info_classif
print("\n  [3/3] SelectKBest (mutual information) ...")
t0    = time.time()
skb   = SelectKBest(mutual_info_classif, k=N_FEATURES_SELECT)
skb.fit(X_fs, y_fs)
freq += skb.get_support().astype(int)
print(f"        Done ({time.time()-t0:.1f}s)")
print(f"        {np.array(col_names_feat)[skb.get_support()].tolist()}")

# --- Final: top 20 by frequency ---
freq_series  = pd.Series(freq, index=col_names_feat).sort_values(ascending=False)
top_features = freq_series.head(N_FEATURES_SELECT).index.tolist()

print(f"\n  Feature frequencies (all):")
print(freq_series.to_string())
print(f"\n  Top {N_FEATURES_SELECT} selected:")
for i, f in enumerate(top_features):
    print(f"    {i+1:2d}. {f}  ({freq_series[f]}/3 algorithms)")

with open('selected_features.json', 'w') as f:
    json.dump(top_features, f, indent=2)
print(f"\n  Saved: selected_features.json")

# =============================================================================
# STEP 8 — APPLY SELECTION + SAVE ALL FILES
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Applying selection and saving")
print("=" * 60)

feat_idx   = [col_names_feat.index(f) for f in top_features]
X_train_20 = X_train_full[:, feat_idx]
X_test_20  = X_test_full[:, feat_idx]

print(f"  X_train: {X_train_20.shape}")
print(f"  X_test : {X_test_20.shape}")

t0 = time.time()
pd.DataFrame(X_train_20, columns=top_features).to_csv('X_train_full.csv', index=False)
pd.DataFrame(X_test_20,  columns=top_features).to_csv('X_test_full.csv',  index=False)
pd.DataFrame({'label': y_train}).to_csv('y_train_multi_full.csv', index=False)
pd.DataFrame({'label': y_test}).to_csv('y_test_multi_full.csv',  index=False)
print(f"  Saved 4 files ({time.time()-t0:.1f}s)")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  Total rows : {df.shape[0]:,}")
print(f"  Train rows : {X_train_20.shape[0]:,}")
print(f"  Test rows  : {X_test_20.shape[0]:,}")
print(f"  Features   : {top_features}")
print(f"  Classes    : {class_names}")
print(f"\n  Files ready:")
print(f"    X_train_full.csv  |  X_test_full.csv")
print(f"    y_train_multi_full.csv  |  y_test_multi_full.csv")
print(f"    selected_features.json  |  attack_class_names.json")
print("=" * 60)