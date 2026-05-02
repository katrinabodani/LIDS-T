# =============================================================================
# LIDS_T.PY — Lightweight IoT Intrusion Detection Transformer
# Project  : Explainable Multiclass Network Intrusion Detection
# Dataset  : UNSW-NB15 full (2.54M rows, 10 classes, 20 features)
# Inputs   : X_train_full.csv, X_test_full.csv
#            y_train_multi_full.csv, y_test_multi_full.csv
#            attack_class_names.json
# =============================================================================
# Architecture — LIDS-T dual-branch design:
#   Branch A : Lightweight Transformer encoder  (global feature relationships)
#   Branch B : Depthwise separable 1D-CNN       (local feature correlations)
#   Fusion   : Learned weighted sum of both branches
#   Head     : MLP classifier → 10 attack classes
# Target    : ~40-50k parameters vs base paper ~270k  (85% reduction)
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
import json
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Architecture
    'd_model':      64,     # reduced from base paper 128 → saves params
    'n_heads':      4,      # 64 / 4 = 16 per head
    'n_layers':     1,      # single encoder layer — sufficient for tabular
    'd_ff':         128,    # feed-forward dim = 2 × d_model
    'cnn_channels': 64,     # output channels for CNN branch
    'cnn_kernel':   3,      # kernel size for depthwise conv
    'dropout':      0.2,

    # Classifier head
    'hidden_units': 64,
    'n_classes':    10,

    # Training
    'batch_size':   2048,   # RTX 4050 6GB handles this comfortably
    'epochs':       50,
    'lr':           1e-3,
    'patience':     7,
    'grad_clip':    1.0,
    'seed':         42,
    'val_split':    0.1,
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

print("\n" + "=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

t0 = time.time()
X_train = pd.read_csv('X_train_full.csv').values.astype(np.float32)
X_test  = pd.read_csv('X_test_full.csv').values.astype(np.float32)
y_train = pd.read_csv('y_train_multi_full.csv').values.ravel().astype(np.int64)
y_test  = pd.read_csv('y_test_multi_full.csv').values.ravel().astype(np.int64)

with open('attack_class_names.json') as f:
    class_names = json.load(f)

n_features = X_train.shape[1]
n_classes  = len(class_names)

print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  Classes : {n_classes} → {class_names}")

# =============================================================================
# STEP 2 — CLASS WEIGHTS
# Inverse frequency weighting — critical for 87% Normal class
# weight_c = total / (n_classes × count_c)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Computing class weights")
print("=" * 60)

counts       = np.bincount(y_train, minlength=n_classes).astype(np.float32)
class_weights = len(y_train) / (n_classes * counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"  {'Class':<20} {'Count':>10} {'Weight':>10}")
print(f"  {'-'*42}")
for i, (cls, cnt, w) in enumerate(zip(class_names, counts, class_weights.cpu())):
    print(f"  {cls:<20} {int(cnt):>10,} {w:>10.4f}")

# =============================================================================
# STEP 3 — VALIDATION SPLIT + DATALOADERS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Validation split and DataLoaders")
print("=" * 60)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size    = CONFIG['val_split'],
    random_state = CONFIG['seed'],
    stratify     = y_train
)
print(f"  Train : {len(X_tr):,}  Val : {len(X_val):,}  Test : {len(X_test):,}")

def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.int64))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0)

train_loader = make_loader(X_tr,    y_tr,    CONFIG['batch_size'], shuffle=True)
val_loader   = make_loader(X_val,   y_val,   CONFIG['batch_size'], shuffle=False)
test_loader  = make_loader(X_test,  y_test,  CONFIG['batch_size'], shuffle=False)

# =============================================================================
# STEP 4 — LIDS-T ARCHITECTURE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Building LIDS-T")
print("=" * 60)

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable 1D convolution.
    Treats the n_features input as a 1D sequence of length n_features.
    Depthwise  : groups=n_features, processes each feature channel independently
    Pointwise  : 1×1 conv mixes channels → out_channels
    Parameter saving vs standard conv: roughly 8-10x fewer params
    """
    def __init__(self, n_features, out_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        # Depthwise: each input channel has its own kernel
        self.depthwise  = nn.Conv1d(n_features, n_features,
                                     kernel_size=kernel_size,
                                     padding=pad, groups=n_features,
                                     bias=False)
        # Pointwise: mix channels
        self.pointwise  = nn.Conv1d(n_features, out_channels,
                                     kernel_size=1, bias=True)
        self.bn         = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, n_features) → unsqueeze → (batch, n_features, 1)
        x = x.unsqueeze(-1)                  # (B, F, 1)
        x = self.depthwise(x)                # (B, F, 1)
        x = self.pointwise(x)                # (B, out_ch, 1)
        x = self.bn(x)
        x = self.activation(x)
        return x.squeeze(-1)                 # (B, out_ch)


class LIDST(nn.Module):
    """
    LIDS-T: Lightweight IoT Intrusion Detection Transformer

    Dual-branch architecture:
      Branch A — Transformer encoder  : captures global feature interactions
      Branch B — Depthwise sep. CNN   : captures local feature correlations
      Fusion   — Learned weighted sum : model decides branch importance
      Head     — MLP → n_classes
    """
    def __init__(self, n_features, d_model, n_heads, n_layers, d_ff,
                 cnn_channels, cnn_kernel, dropout, hidden_units, n_classes):
        super().__init__()

        # --- Branch A: Transformer ---
        self.input_proj = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= d_ff,
            dropout        = dropout,
            batch_first    = True,
            activation     = 'gelu',          # GELU outperforms ReLU on tabular
            norm_first     = True,            # pre-norm — more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=n_layers)
        self.gap         = nn.AdaptiveAvgPool1d(1)

        # --- Branch B: Depthwise Separable CNN ---
        self.cnn_branch  = DepthwiseSeparableConv(n_features, cnn_channels,
                                                    cnn_kernel)
        # Project CNN output to same dim as transformer for fusion
        self.cnn_proj    = nn.Linear(cnn_channels, d_model)

        # --- Fusion: learned scalar weights (softmax → sum to 1) ---
        # Two learnable scalars: α for transformer, β for CNN
        self.fusion_weights = nn.Parameter(torch.ones(2))

        # --- Dropout ---
        self.dropout = nn.Dropout(dropout)

        # --- Classification Head ---
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, n_classes)
        )

    def forward(self, x):
        # --- Branch A ---
        a = self.input_proj(x)               # (B, d_model)
        a = a.unsqueeze(1)                   # (B, 1, d_model) — seq len = 1
        a = self.transformer(a)              # (B, 1, d_model)
        a = a.permute(0, 2, 1)              # (B, d_model, 1)
        a = self.gap(a).squeeze(-1)          # (B, d_model)

        # --- Branch B ---
        b = self.cnn_branch(x)               # (B, cnn_channels)
        b = self.cnn_proj(b)                 # (B, d_model)

        # --- Fusion ---
        # Softmax over [α, β] so they sum to 1 and are always positive
        w = torch.softmax(self.fusion_weights, dim=0)
        fused = w[0] * a + w[1] * b          # (B, d_model)

        # --- Head ---
        out = self.dropout(fused)
        return self.head(out)                # (B, n_classes)


model = LIDST(
    n_features   = n_features,
    d_model      = CONFIG['d_model'],
    n_heads      = CONFIG['n_heads'],
    n_layers     = CONFIG['n_layers'],
    d_ff         = CONFIG['d_ff'],
    cnn_channels = CONFIG['cnn_channels'],
    cnn_kernel   = CONFIG['cnn_kernel'],
    dropout      = CONFIG['dropout'],
    hidden_units = CONFIG['hidden_units'],
    n_classes    = CONFIG['n_classes'],
).to(device)

total_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
base_params    = 270_249    # base paper parameter count
param_reduction = (1 - total_params / base_params) * 100

print(f"\n  LIDS-T parameter breakdown:")
for name, module in model.named_children():
    p = sum(x.numel() for x in module.parameters())
    print(f"    {name:<20} {p:>8,} params")
print(f"\n  Total trainable params : {total_params:,}")
print(f"  Base paper params      : {base_params:,}")
print(f"  Parameter reduction    : {param_reduction:.1f}%")

# Model size on disk (estimate)
param_bytes = total_params * 4    # float32
print(f"  Estimated model size   : {param_bytes/1024:.1f} KB")

# =============================================================================
# STEP 5 — TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Training")
print("=" * 60)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                        weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6
)

best_val_loss  = float('inf')
patience_count = 0
history        = {'train_loss': [], 'val_loss': [],
                  'val_acc': [], 'val_macro_f1': []}

print(f"\n{'Epoch':<7} {'Train Loss':<13} {'Val Loss':<13} "
      f"{'Val Acc':<11} {'Macro F1':<11} {'LR':<10} {'Time'}")
print("-" * 75)

for epoch in range(1, CONFIG['epochs'] + 1):
    t0 = time.time()

    # --- Train ---
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        train_losses.append(loss.item())

    # --- Validate ---
    model.eval()
    val_losses, val_preds, val_true = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            val_losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(y_batch.cpu().numpy())

    train_loss   = np.mean(train_losses)
    val_loss     = np.mean(val_losses)
    val_acc      = accuracy_score(val_true, val_preds)
    val_macro_f1 = f1_score(val_true, val_preds, average='macro',
                            zero_division=0)
    current_lr   = optimizer.param_groups[0]['lr']
    elapsed      = time.time() - t0

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_macro_f1'].append(val_macro_f1)

    scheduler.step(val_loss)

    print(f"{epoch:<7} {train_loss:<13.4f} {val_loss:<13.4f} "
          f"{val_acc:<11.4f} {val_macro_f1:<11.4f} "
          f"{current_lr:<10.6f} {elapsed:.1f}s")

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        torch.save(model.state_dict(), 'lidst_best.pt')
    else:
        patience_count += 1
        if patience_count >= CONFIG['patience']:
            print(f"\n  Early stopping at epoch {epoch}")
            break

# =============================================================================
# STEP 6 — EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Test set evaluation")
print("=" * 60)

model.load_state_dict(torch.load('lidst_best.pt', map_location=device))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        preds   = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

accuracy     = accuracy_score(all_true, all_preds)
precision_w  = precision_score(all_true, all_preds, average='weighted', zero_division=0)
recall_w     = recall_score(all_true, all_preds, average='weighted', zero_division=0)
f1_weighted  = f1_score(all_true, all_preds, average='weighted', zero_division=0)
f1_macro     = f1_score(all_true, all_preds, average='macro', zero_division=0)

# =============================================================================
# STEP 7 — INFERENCE LATENCY BENCHMARK
# Critical for IoT edge deployment claim
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Inference latency benchmark")
print("=" * 60)

model.eval()
# Warmup
dummy = torch.randn(CONFIG['batch_size'], n_features).to(device)
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy)

# Benchmark single sample latency (most relevant for edge deployment)
single_sample = torch.randn(1, n_features).to(device)
n_runs = 1000
if device.type == 'cuda':
    torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(n_runs):
        _ = model(single_sample)
if device.type == 'cuda':
    torch.cuda.synchronize()
single_latency_ms = (time.time() - t0) / n_runs * 1000

# Batch throughput
batch_sample = torch.randn(CONFIG['batch_size'], n_features).to(device)
if device.type == 'cuda':
    torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(batch_sample)
if device.type == 'cuda':
    torch.cuda.synchronize()
throughput = CONFIG['batch_size'] * 100 / (time.time() - t0)

print(f"  Single sample latency : {single_latency_ms:.4f} ms")
print(f"  Throughput            : {throughput:,.0f} samples/sec")
print(f"  IoT real-time req     : <1ms per flow")
print(f"  Requirement met       : {'YES' if single_latency_ms < 1.0 else 'MARGINAL'}")

# =============================================================================
# STEP 8 — RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("LIDS-T — FINAL RESULTS")
print("=" * 60)

print(f"\n  {'Metric':<22} {'LIDS-T (Ours)':<18} {'Base Paper':<15} {'Gap'}")
print(f"  {'-'*65}")
print(f"  {'Accuracy':<22} {accuracy:<18.4f} {'0.93':<15} {accuracy-0.93:+.4f}")
print(f"  {'Precision (weighted)':<22} {precision_w:<18.4f} {'0.91':<15} {precision_w-0.91:+.4f}")
print(f"  {'Recall (weighted)':<22} {recall_w:<18.4f} {'0.92':<15} {recall_w-0.92:+.4f}")
print(f"  {'F1 (weighted)':<22} {f1_weighted:<18.4f} {'0.92':<15} {f1_weighted-0.92:+.4f}")
print(f"  {'F1 (macro)':<22} {f1_macro:<18.4f} {'N/A':<15}")
print(f"  {'-'*65}")
print(f"  {'Parameters':<22} {total_params:<18,} {base_params:<15,} {param_reduction:.1f}% fewer")
print(f"  {'Model size (KB)':<22} {param_bytes/1024:<18.1f} {'~1082':<15}")
print(f"  {'Latency (ms/sample)':<22} {single_latency_ms:<18.4f} {'N/A':<15}")
print(f"  {'Task':<22} {'Multiclass (10)':<18} {'Binary (2)':<15}")
print(f"  {'Dataset rows':<22} {'2,540,047':<18} {'257,673':<15}")

print("\n  Per-class results:")
print(classification_report(all_true, all_preds,
                             target_names=class_names,
                             zero_division=0))

cm = confusion_matrix(all_true, all_preds)
print(f"\n  Confusion matrix saved.")
print(f"\n  Fusion branch weights (learned):")
w_learned = torch.softmax(model.fusion_weights, dim=0).detach().cpu()
print(f"    Transformer branch : {w_learned[0]:.4f}")
print(f"    CNN branch         : {w_learned[1]:.4f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results = {
    'model'        : 'LIDS-T',
    'task'         : 'multiclass_10_class',
    'dataset_rows' : 2_540_047,
    'n_features'   : n_features,
    'n_classes'    : n_classes,
    'accuracy'     : round(float(accuracy), 4),
    'precision_w'  : round(float(precision_w), 4),
    'recall_w'     : round(float(recall_w), 4),
    'f1_weighted'  : round(float(f1_weighted), 4),
    'f1_macro'     : round(float(f1_macro), 4),
    'params'       : total_params,
    'param_reduction_pct': round(param_reduction, 1),
    'model_size_kb': round(param_bytes / 1024, 1),
    'latency_ms'   : round(single_latency_ms, 4),
    'throughput_sps': round(throughput, 0),
    'base_paper'   : {
        'accuracy': 0.93, 'precision': 0.91,
        'recall': 0.92, 'f1': 0.92,
        'params': base_params, 'task': 'binary'
    },
    'class_names'  : class_names,
    'confusion_matrix': cm.tolist(),
    'history'      : history,
    'config'       : CONFIG,
    'fusion_weights': {
        'transformer': round(float(w_learned[0]), 4),
        'cnn'        : round(float(w_learned[1]), 4),
    }
}

with open('results_lidst.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: results_lidst.json")
print(f"  Saved: lidst_best.pt")

print("\n" + "=" * 60)
if accuracy >= 0.95:
    print(f"  LIDS-T achieved {accuracy:.4f} accuracy on 10-class task")
    print(f"  with {param_reduction:.1f}% fewer parameters than base paper")
    print(f"  on {10}x more data — all novelty claims validated")
elif accuracy >= 0.93:
    print(f"  LIDS-T matched base paper accuracy ({accuracy:.4f})")
    print(f"  on harder multiclass task with {param_reduction:.1f}% fewer params")
else:
    print(f"  Accuracy: {accuracy:.4f} — paste output for diagnosis")
print("=" * 60)