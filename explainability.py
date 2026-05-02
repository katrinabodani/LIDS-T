# =============================================================================
# SHAP_EXPLAINABILITY.PY — Per-Class SHAP Analysis for LIDS-T
# Project  : Explainable Multiclass Network Intrusion Detection
# =============================================================================
# Run: python shap_explainability.py
# Requires: pip install shap
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Run: pip install shap")
    exit(1)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# =============================================================================
# LIDS-T ARCHITECTURE (must match lids_t.py exactly)
# =============================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, n_features, out_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise  = nn.Conv1d(n_features, n_features,
                                     kernel_size=kernel_size,
                                     padding=pad, groups=n_features,
                                     bias=False)
        self.pointwise  = nn.Conv1d(n_features, out_channels,
                                     kernel_size=1, bias=True)
        self.bn         = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x.squeeze(-1)


class LIDST(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_layers, d_ff,
                 cnn_channels, cnn_kernel, dropout, hidden_units, n_classes):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True,
        )
        self.transformer    = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=n_layers)
        self.gap            = nn.AdaptiveAvgPool1d(1)
        self.cnn_branch     = DepthwiseSeparableConv(n_features,
                                                      cnn_channels,
                                                      cnn_kernel)
        self.cnn_proj       = nn.Linear(cnn_channels, d_model)
        self.fusion_weights = nn.Parameter(torch.ones(2))
        self.dropout        = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, n_classes)
        )

    def forward(self, x):
        a = self.input_proj(x).unsqueeze(1)
        a = self.transformer(a)
        a = self.gap(a.permute(0, 2, 1)).squeeze(-1)
        b = self.cnn_proj(self.cnn_branch(x))
        w = torch.softmax(self.fusion_weights, dim=0)
        return self.head(self.dropout(w[0] * a + w[1] * b))


CONFIG = {
    'd_model': 64, 'n_heads': 4, 'n_layers': 1,
    'd_ff': 128, 'cnn_channels': 64, 'cnn_kernel': 3,
    'dropout': 0.0, 'hidden_units': 64, 'n_classes': 10,
}
N_FEATURES = 20

# =============================================================================
# STEP 1 — LOAD MODEL
# =============================================================================

print("=" * 60)
print("SHAP Explainability — LIDS-T")
print("=" * 60)

device = torch.device('cpu')  # SHAP works best on CPU

model = LIDST(N_FEATURES, **CONFIG)
model.load_state_dict(
    torch.load('lidst_best.pt', map_location='cpu')
)
model.eval()
print(f"\nModel loaded: {sum(p.numel() for p in model.parameters()):,} params")

# =============================================================================
# STEP 2 — LOAD DATA
# =============================================================================

print("\nLoading test data...")
X_test = pd.read_csv('X_test_full.csv').values.astype(np.float32)
y_test = pd.read_csv('y_test_multi_full.csv').values.ravel().astype(np.int64)

with open('attack_class_names.json') as f:
    class_names = json.load(f)

with open('feature_cols.json') as f:
    feature_names = json.load(f)

print(f"Test set   : {X_test.shape}")
print(f"Classes    : {class_names}")
print(f"Features   : {feature_names}")

# =============================================================================
# STEP 3 — SAMPLE DATA FOR SHAP
# =============================================================================

print("\nSampling data for SHAP computation...")

np.random.seed(42)

# Background: 200 samples per class where possible (balanced background)
bg_indices = []
for cls in range(len(class_names)):
    cls_idx = np.where(y_test == cls)[0]
    n_bg    = min(50, len(cls_idx))
    bg_indices.extend(np.random.choice(cls_idx, n_bg, replace=False))
bg_indices = np.array(bg_indices)
X_bg       = torch.tensor(X_test[bg_indices], dtype=torch.float32)

# Explain: 100 samples per class where possible
exp_indices = []
for cls in range(len(class_names)):
    cls_idx = np.where(y_test == cls)[0]
    n_exp   = min(100, len(cls_idx))
    exp_indices.extend(np.random.choice(cls_idx, n_exp, replace=False))
exp_indices = np.array(exp_indices)
X_exp       = torch.tensor(X_test[exp_indices], dtype=torch.float32)
y_exp       = y_test[exp_indices]

print(f"Background samples : {len(X_bg)}")
print(f"Explanation samples: {len(X_exp)}")

# =============================================================================
# STEP 4 — COMPUTE SHAP VALUES
# =============================================================================

print("\nComputing SHAP values with DeepExplainer...")
print("This may take 2-5 minutes...")

explainer   = shap.GradientExplainer(model, X_bg)
shap_values = explainer.shap_values(X_exp)

# shap_values: list of 10 arrays, each (n_explain, 20)
# shap_values[i][j, k] = SHAP contribution of feature k
#                         to predicting class i for sample j
print(f"SHAP values computed: {len(shap_values)} classes x "
      f"{shap_values[0].shape} per class")

# =============================================================================
# STEP 5 — PER-CLASS TOP FEATURES
# =============================================================================

print("\n" + "=" * 60)
print("PER-CLASS TOP-5 FEATURES (mean |SHAP| value)")
print("=" * 60)

per_class_top = {}
for i, cls in enumerate(class_names):
    sv          = shap_values[i]         # (n_explain, 20)
    mean_abs    = np.abs(sv).mean(axis=0)  # (20,)
    top5_idx    = mean_abs.argsort()[::-1][:5]
    top5_names  = [feature_names[j] for j in top5_idx]
    top5_vals   = [mean_abs[j]       for j in top5_idx]

    per_class_top[cls] = {
        'features': top5_names,
        'values':   [round(float(v), 6) for v in top5_vals],
        'indices':  top5_idx.tolist(),
    }

    print(f"\n  {cls}:")
    for fname, fval in zip(top5_names, top5_vals):
        bar = '█' * max(1, int(fval / max(top5_vals) * 20))
        print(f"    {fname:<22} {fval:.5f}  {bar}")

# =============================================================================
# STEP 6 — VISUALIZATIONS
# =============================================================================

print("\nGenerating SHAP visualizations...")

# --- PLOT 1: Per-class top-5 SHAP bar chart (one panel per class) ---
n_cls  = len(class_names)
n_cols = 2
n_rows = (n_cls + 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(16, n_rows * 3.2))
fig.suptitle('LIDS-T SHAP Feature Importance — Per Attack Class\n'
             'Mean absolute SHAP value (higher = more influential)',
             fontsize=13, fontweight='bold', y=1.01)
axes = axes.flatten()

colors_cls = ['#e74c3c', '#c0392b', '#e67e22', '#d35400',
              '#f39c12', '#7f8c8d', '#27ae60', '#2980b9',
              '#8e44ad', '#16a085']

for i, cls in enumerate(class_names):
    ax      = axes[i]
    top     = per_class_top[cls]
    fnames  = top['features'][::-1]
    fvals   = top['values'][::-1]
    y_pos   = range(len(fnames))
    color   = colors_cls[i]

    bars = ax.barh(y_pos, fvals, color=color, alpha=0.80,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fnames, fontsize=9)
    ax.set_title(cls, fontsize=10, fontweight='bold', color=color)
    ax.set_xlabel('Mean |SHAP|', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, fvals):
        ax.text(bar.get_width() + max(fvals)*0.02,
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=7.5)

# Hide unused panels
for j in range(n_cls, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('shap_plot1_per_class_top5.png')
plt.close()
print("  Saved: shap_plot1_per_class_top5.png")

# --- PLOT 2: Global feature importance (across all classes) ---
# Mean |SHAP| per feature summed across all classes
global_importance = np.zeros(N_FEATURES)
for i in range(n_cls):
    global_importance += np.abs(shap_values[i]).mean(axis=0)
global_importance /= n_cls

sorted_idx  = global_importance.argsort()[::-1]
sorted_feat = [feature_names[j] for j in sorted_idx]
sorted_vals = [global_importance[j] for j in sorted_idx]

fig, ax = plt.subplots(figsize=(12, 7))
y_pos = range(len(sorted_feat))
bars  = ax.barh(list(y_pos)[::-1], sorted_vals,
                color='#3498db', alpha=0.82,
                edgecolor='white', linewidth=0.5)

ax.set_yticks(list(range(len(sorted_feat))))
ax.set_yticklabels(sorted_feat[::-1], fontsize=10)
ax.set_xlabel('Mean |SHAP| value (averaged across all classes)', fontsize=11)
ax.set_title('Global Feature Importance — LIDS-T\n'
             'Averaged across all 10 attack classes',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, sorted_vals[::-1]):
    ax.text(bar.get_width() + sorted_vals[0]*0.01,
            bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('shap_plot2_global_importance.png')
plt.close()
print("  Saved: shap_plot2_global_importance.png")

# --- PLOT 3: SHAP heatmap — feature x class ---
# Each cell = mean |SHAP| for that feature in that class
shap_matrix = np.zeros((N_FEATURES, n_cls))
for i in range(n_cls):
    shap_matrix[:, i] = np.abs(shap_values[i]).mean(axis=0)

# Normalize per column for visual clarity
shap_norm = shap_matrix / (shap_matrix.max(axis=0, keepdims=True) + 1e-10)

import seaborn as sns
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(
    pd.DataFrame(shap_norm,
                 index=feature_names,
                 columns=class_names),
    annot=True, fmt='.2f', cmap='YlOrRd',
    ax=ax, linewidths=0.3, linecolor='gray',
    cbar_kws={'label': 'Normalized Mean |SHAP| (per class)'}
)
ax.set_title('SHAP Feature-Class Attribution Heatmap\n'
             'Darker = feature more influential for that class',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Attack Class', fontsize=11)
ax.set_ylabel('Network Flow Feature', fontsize=11)
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig('shap_plot3_heatmap.png')
plt.close()
print("  Saved: shap_plot3_heatmap.png")

# --- PLOT 4: Security insight — top feature per attack class ---
fig, ax = plt.subplots(figsize=(12, 6))
top1_features = [per_class_top[cls]['features'][0] for cls in class_names]
top1_values   = [per_class_top[cls]['values'][0]   for cls in class_names]
x = np.arange(len(class_names))

bars = ax.bar(x, top1_values, color=colors_cls,
               alpha=0.85, edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=25, fontsize=10)
ax.set_ylabel('Mean |SHAP| of Most Important Feature', fontsize=11)
ax.set_title('Most Discriminative Feature per Attack Class\n'
             'Primary SHAP driver for each attack type',
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, feat, val in zip(bars, top1_features, top1_values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(top1_values)*0.01,
            feat, ha='center', va='bottom',
            fontsize=8, fontweight='bold', rotation=15)

plt.tight_layout()
plt.savefig('shap_plot4_top_feature_per_class.png')
plt.close()
print("  Saved: shap_plot4_top_feature_per_class.png")

# =============================================================================
# STEP 7 — SAVE RESULTS
# =============================================================================

results = {
    'per_class_top5':   per_class_top,
    'global_importance': {
        feature_names[j]: round(float(global_importance[j]), 6)
        for j in range(N_FEATURES)
    },
    'global_ranking': sorted_feat,
    'n_background':   len(X_bg),
    'n_explained':    len(X_exp),
}

with open('results_shap.json', 'w') as f:
    json.dump(results, f, indent=2)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETE")
print("=" * 60)
print(f"\n  Global top-5 most important features (across all classes):")
for j in range(5):
    print(f"    {j+1}. {sorted_feat[j]:<22} {sorted_vals[j]:.5f}")

print(f"\n  Security insights per attack type:")
for cls in class_names:
    top = per_class_top[cls]
    print(f"    {cls:<18} → driven by: "
          f"{top['features'][0]}, {top['features'][1]}, "
          f"{top['features'][2]}")

print(f"\n  Files saved:")
print(f"    shap_plot1_per_class_top5.png    — top-5 features per class")
print(f"    shap_plot2_global_importance.png — global feature ranking")
print(f"    shap_plot3_heatmap.png           — feature x class heatmap")
print(f"    shap_plot4_top_feature_per_class.png — primary driver per class")
print(f"    results_shap.json                — all SHAP values")
print("=" * 60)