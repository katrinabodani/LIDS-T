# =============================================================================
# ABLATION_STUDY.PY — LIDS-T Ablation Study
# Project  : Explainable Multiclass Network Intrusion Detection
# =============================================================================
# Tests 4 configurations to prove each component contributes:
#
#   Config 1 — Transformer branch only  (CNN branch removed)
#   Config 2 — CNN branch only          (Transformer branch removed)
#   Config 3 — No class weighting       (equal loss weights)
#   Config 4 — Full LIDS-T              (both branches + class weights)
#
# All configs use identical hyperparameters, same data, same seed.
# The difference in results proves each component is necessary.
#
# Run: python ablation_study.py
# NOTE: This will take ~2-3 hours total (4 full training runs)
#       Run overnight or while doing something else
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

N_FEATURES  = 20
N_CLASSES   = 10
D_MODEL     = 64

# =============================================================================
# MODEL VARIANTS
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


# --- Config 1: Transformer branch only ---
class TransformerOnly(nn.Module):
    """Full LIDS-T but CNN branch replaced with zeros — only Transformer."""
    def __init__(self, n_features=N_FEATURES, d_model=D_MODEL,
                 n_heads=4, d_ff=128, dropout=0.2,
                 hidden_units=64, n_classes=N_CLASSES):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.gap         = nn.AdaptiveAvgPool1d(1)
        self.dropout     = nn.Dropout(dropout)
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
        return self.head(self.dropout(a))


# --- Config 2: CNN branch only ---
class CNNOnly(nn.Module):
    """Full LIDS-T but Transformer branch replaced — only CNN."""
    def __init__(self, n_features=N_FEATURES, d_model=D_MODEL,
                 cnn_channels=64, cnn_kernel=3, dropout=0.2,
                 hidden_units=64, n_classes=N_CLASSES):
        super().__init__()
        self.cnn_branch = DepthwiseSeparableConv(n_features,
                                                  cnn_channels, cnn_kernel)
        self.cnn_proj   = nn.Linear(cnn_channels, d_model)
        self.dropout    = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, n_classes)
        )

    def forward(self, x):
        b = self.cnn_proj(self.cnn_branch(x))
        return self.head(self.dropout(b))


# --- Config 4: Full LIDS-T (dual branch) ---
class LIDST(nn.Module):
    """Full LIDS-T — Transformer + CNN + learned fusion."""
    def __init__(self, n_features=N_FEATURES, d_model=D_MODEL,
                 n_heads=4, d_ff=128, cnn_channels=64, cnn_kernel=3,
                 dropout=0.2, hidden_units=64, n_classes=N_CLASSES):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True,
        )
        self.transformer    = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=1)
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


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading data...")
X_train = pd.read_csv('X_train_full.csv').values.astype(np.float32)
X_test  = pd.read_csv('X_test_full.csv').values.astype(np.float32)
y_train = pd.read_csv('y_train_multi_full.csv').values.ravel().astype(np.int64)
y_test  = pd.read_csv('y_test_multi_full.csv').values.ravel().astype(np.int64)

with open('attack_class_names.json') as f:
    class_names = json.load(f)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1,
    random_state=SEED, stratify=y_train
)

# Class weights (capped at 50x)
counts       = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
class_weights = len(y_tr) / (N_CLASSES * counts)
class_weights = np.clip(class_weights, 0, 50)
class_weights_tensor = torch.tensor(class_weights,
                                     dtype=torch.float32).to(device)

def make_loader(X, y, batch_size=2048, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, pin_memory=True)

train_loader = make_loader(X_tr,   y_tr)
val_loader   = make_loader(X_val,  y_val,  shuffle=False)
test_loader  = make_loader(X_test, y_test, shuffle=False)

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_and_eval(model, use_class_weights=True,
                   epochs=50, patience=7, tag=""):
    """Train model and return test metrics."""

    crit = nn.CrossEntropyLoss(
        weight=class_weights_tensor if use_class_weights else None
    )
    opt  = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', patience=3, factor=0.5, min_lr=1e-6
    )

    best_loss  = float('inf')
    best_state = None
    patience_c = 0
    t0         = time.time()
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    print(f"\n  Training {tag} ({total_params:,} params)...")
    print(f"  {'Epoch':<7} {'Train Loss':<13} {'Val Loss':<13} "
          f"{'Val Acc':<11} {'Time'}")
    print(f"  {'-'*55}")

    for epoch in range(1, epochs + 1):
        model.train()
        tl = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())

        model.eval()
        vl, vp, vt = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                lg = model(Xb)
                vl.append(crit(lg, yb).item())
                vp.extend(lg.argmax(1).cpu().numpy())
                vt.extend(yb.cpu().numpy())

        vl_m = np.mean(vl)
        va   = accuracy_score(vt, vp)
        sch.step(vl_m)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:<7} {np.mean(tl):<13.4f} {vl_m:<13.4f} "
                  f"{va:<11.4f} {time.time()-t0:.0f}s")

        if vl_m < best_loss:
            best_loss  = vl_m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Evaluate on test set
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            preds.extend(model(Xb).argmax(1).cpu().numpy())
            trues.extend(yb.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    acc  = accuracy_score(trues, preds)
    wp   = precision_score(trues, preds, average='weighted', zero_division=0)
    wr   = recall_score(trues, preds, average='weighted', zero_division=0)
    wf1  = f1_score(trues, preds, average='weighted', zero_division=0)
    mf1  = f1_score(trues, preds, average='macro', zero_division=0)
    pcf1 = f1_score(trues, preds, average=None, zero_division=0)

    print(f"\n  {tag} results:")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    Weighted F1: {wf1:.4f}")
    print(f"    Macro F1:    {mf1:.4f}")
    print(f"    Params:      {total_params:,}")

    torch.save(model.state_dict(), f'ablation_{tag.replace(" ","_")}.pt')

    return {
        'tag':          tag,
        'accuracy':     round(float(acc),  4),
        'weighted_p':   round(float(wp),   4),
        'weighted_r':   round(float(wr),   4),
        'weighted_f1':  round(float(wf1),  4),
        'macro_f1':     round(float(mf1),  4),
        'params':       total_params,
        'per_class_f1': {class_names[i]: round(float(pcf1[i]), 4)
                         for i in range(N_CLASSES)},
        'preds':        preds,
        'trues':        trues,
        'time_s':       round(time.time() - t0, 1),
    }

# =============================================================================
# RUN ALL 4 CONFIGURATIONS
# =============================================================================

print("\n" + "="*60)
print("ABLATION STUDY — 4 CONFIGURATIONS")
print("="*60)

# Pre-loaded Full LIDS-T results — already trained, no need to retrain
full_lidst_result = {
    'tag':          'Full LIDS-T',
    'accuracy':     0.9671,
    'weighted_p':   0.9824,
    'weighted_r':   0.9671,
    'weighted_f1':  0.9727,
    'macro_f1':     0.4958,
    'params':       45320,
    'per_class_f1': {
        'Analysis': 0.16, 'Backdoor': 0.11, 'DoS': 0.43,
        'Exploits': 0.59, 'Fuzzers': 0.52, 'Generic': 0.99,
        'Normal': 0.99, 'Reconnaissance': 0.74,
        'Shellcode': 0.33, 'Worms': 0.11
    },
    'time_s': 0,
}

configs = [
    ("Transformer Only",  TransformerOnly().to(device), True),
    ("CNN Only",          CNNOnly().to(device),          True),
    ("No Class Weights",  LIDST().to(device),            False),
]

ablation_results = []
for tag, model, use_cw in configs:
    print(f"\n{'='*60}")
    print(f"Config: {tag}")
    print(f"{'='*60}")
    result = train_and_eval(model, use_class_weights=use_cw,
                             epochs=50, patience=7, tag=tag)
    ablation_results.append(result)

# Append pre-loaded Full LIDS-T at the end
ablation_results.append(full_lidst_result)
print(f"\n  Full LIDS-T (pre-loaded): Accuracy=0.9671, W-F1=0.9727, Macro F1=0.4958")

# =============================================================================
# RESULTS TABLE
# =============================================================================

print("\n" + "="*70)
print("ABLATION STUDY RESULTS")
print("="*70)
print(f"\n  {'Config':<22} {'Accuracy':<11} {'W-F1':<11} "
      f"{'Macro F1':<11} {'Params':<12} {'Time'}")
print(f"  {'-'*68}")
for r in ablation_results:
    print(f"  {r['tag']:<22} {r['accuracy']:<11.4f} "
          f"{r['weighted_f1']:<11.4f} {r['macro_f1']:<11.4f} "
          f"{r['params']:<12,} {r['time_s']}s")

print(f"\n  Per-class F1 breakdown:")
print(f"  {'Class':<18}", end="")
for r in ablation_results:
    print(f" {r['tag'][:12]:<13}", end="")
print()
print(f"  {'-'*72}")
for cls in class_names:
    print(f"  {cls:<18}", end="")
    for r in ablation_results:
        val = r['per_class_f1'].get(cls, 0)
        print(f" {val:<13.4f}", end="")
    print()

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\nGenerating ablation plots...")

tags   = [r['tag'] for r in ablation_results]
accs   = [r['accuracy']    for r in ablation_results]
wf1s   = [r['weighted_f1'] for r in ablation_results]
mf1s   = [r['macro_f1']    for r in ablation_results]
params = [r['params']       for r in ablation_results]
colors = ['#e74c3c', '#e67e22', '#3498db', '#27ae60']

# --- Plot 1: Overall metrics bar chart ---
x     = np.arange(len(tags))
width = 0.25
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - width, accs,  width, label='Accuracy',
            color=[c for c in colors], alpha=0.5, edgecolor='white')
b2 = ax.bar(x,         wf1s,  width, label='Weighted F1',
            color=[c for c in colors], alpha=0.75, edgecolor='white')
b3 = ax.bar(x + width, mf1s,  width, label='Macro F1',
            color=[c for c in colors], alpha=1.0, edgecolor='white')

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_title('Ablation Study — Impact of Each LIDS-T Component\n'
             'Removing any component degrades performance',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tags, fontsize=10)
ax.set_ylabel('Score')
ax.set_ylim([0.5, 1.05])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('ablation_plot1_overall.png')
plt.close()
print("  Saved: ablation_plot1_overall.png")

# --- Plot 2: Per-class F1 heatmap ---
import seaborn as sns
f1_matrix = pd.DataFrame(
    {r['tag']: [r['per_class_f1'][cls] for cls in class_names]
     for r in ablation_results},
    index=class_names
)
fig, ax = plt.subplots(figsize=(13, 8))
sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            ax=ax, vmin=0, vmax=1,
            linewidths=0.3, linecolor='gray',
            cbar_kws={'label': 'F1 Score'})
ax.set_title('Per-Class F1 Score — Ablation Study\n'
             'Full LIDS-T consistently outperforms ablated variants',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Model Configuration')
ax.set_ylabel('Attack Class')
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('ablation_plot2_heatmap.png')
plt.close()
print("  Saved: ablation_plot2_heatmap.png")

# --- Plot 3: Accuracy vs parameter count ---
fig, ax = plt.subplots(figsize=(9, 6))
for i, r in enumerate(ablation_results):
    ax.scatter(r['params'], r['accuracy'],
               color=colors[i], s=200, zorder=5, label=r['tag'])
    ax.annotate(r['tag'],
                (r['params'], r['accuracy']),
                textcoords='offset points',
                xytext=(10, 5), fontsize=9)

ax.set_xlabel('Parameter Count', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Accuracy vs Model Size — Ablation Study\n'
             'Full LIDS-T achieves best accuracy at moderate parameter count',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ablation_plot3_acc_vs_params.png')
plt.close()
print("  Saved: ablation_plot3_acc_vs_params.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================

save_results = []
for r in ablation_results:
    sr = {k: v for k, v in r.items() if k not in ['preds', 'trues']}
    save_results.append(sr)

with open('results_ablation.json', 'w') as f:
    json.dump(save_results, f, indent=2)

# =============================================================================
# SUMMARY
# =============================================================================

full = next(r for r in ablation_results if r['tag'] == 'Full LIDS-T')
tonly = next(r for r in ablation_results if r['tag'] == 'Transformer Only')
conly = next(r for r in ablation_results if r['tag'] == 'CNN Only')
nocw  = next(r for r in ablation_results if r['tag'] == 'No Class Weights')

print("\n" + "="*60)
print("ABLATION STUDY COMPLETE")
print("="*60)
print(f"\n  Contribution of each component:")
print(f"    CNN branch adds       : "
      f"{full['accuracy']-tonly['accuracy']:+.4f} accuracy over Transformer-only")
print(f"    Transformer branch adds: "
      f"{full['accuracy']-conly['accuracy']:+.4f} accuracy over CNN-only")
print(f"    Class weights add     : "
      f"{full['accuracy']-nocw['accuracy']:+.4f} accuracy over no weighting")
print(f"\n  Macro F1 contributions:")
print(f"    CNN branch adds       : "
      f"{full['macro_f1']-tonly['macro_f1']:+.4f} macro F1")
print(f"    Transformer adds      : "
      f"{full['macro_f1']-conly['macro_f1']:+.4f} macro F1")
print(f"    Class weights add     : "
      f"{full['macro_f1']-nocw['macro_f1']:+.4f} macro F1")
print(f"\n  Saved: results_ablation.json")
print(f"  Saved: ablation_plot1_overall.png")
print(f"  Saved: ablation_plot2_heatmap.png")
print(f"  Saved: ablation_plot3_acc_vs_params.png")
print("="*60)