# =============================================================================
# MISSING_PLOTS.PY — Generate remaining paper visualizations
# Project  : Explainable Multiclass Network Intrusion Detection
# =============================================================================
# Generates 4 missing plots:
#   Plot 1 — Model size comparison (PyTorch vs ONNX vs Quantized vs Base paper)
#   Plot 2 — Parameter count comparison (LIDS-T vs Base paper)
#   Plot 3 — Deployment readiness matrix (visual table)
#   Plot 4 — Overall results comparison (LIDS-T vs Baseline vs Base paper)
#
# Run: python missing_plots.py
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'savefig.dpi':  300,
    'savefig.bbox': 'tight',
})

# =============================================================================
# PLOT 1 — MODEL SIZE COMPARISON
# =============================================================================

print("Generating Plot 1: Model size comparison...")

labels = ['Base Paper\n(PyTorch float32)',
          'LIDS-T\n(PyTorch float32)',
          'LIDS-T\n(ONNX float32)',
          'LIDS-T\n(INT8 Quantized)']
sizes  = [1082.0, 177.0, 152.4, 88.5]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#27ae60']
alphas = [0.85, 0.85, 0.85, 0.85]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(labels, sizes, color=colors, alpha=0.85,
              edgecolor='white', linewidth=0.5, width=0.55)

for bar, size in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 15,
            f'{size:.1f} KB',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# Annotation lines
ax.axhline(y=520, color='#8e44ad', linestyle='--',
           linewidth=1.8, alpha=0.8, label='ESP32 flash limit (4MB)')
ax.axhline(y=1024*4, color='#c0392b', linestyle=':',
           linewidth=1.5, alpha=0.6, label='Raspberry Pi Zero RAM (512MB)')

ax.text(3.4, 540, 'ESP32 flash limit', color='#8e44ad',
        fontsize=8.5, va='bottom')

# Reduction annotation
ax.annotate('', xy=(1, sizes[1]), xytext=(0, sizes[0]),
            arrowprops=dict(arrowstyle='<->', color='#2c3e50',
                            lw=1.5, mutation_scale=15))
ax.text(0.5, (sizes[0]+sizes[1])/2,
        f'  83.6%\n  smaller',
        ha='center', va='center', fontsize=9,
        color='#2c3e50', fontweight='bold')

ax.set_title('Model Size Comparison Across Deployment Formats\n'
             'LIDS-T is 83.6% smaller than base paper — fits IoT flash memory',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Model Size (KB)', fontsize=12)
ax.set_ylim([0, 1300])
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('viz_plot1_model_size_comparison.png')
plt.close()
print("  Saved: viz_plot1_model_size_comparison.png")

# =============================================================================
# PLOT 2 — PARAMETER COUNT + KEY METRICS COMPARISON
# =============================================================================

print("Generating Plot 2: Parameter count comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('LIDS-T vs Base Paper — Efficiency and Accuracy Comparison',
             fontsize=14, fontweight='bold')

# Left: parameter count
models_p  = ['Base Paper', 'LIDS-T (Ours)']
params    = [270249, 45320]
colors_p  = ['#e74c3c', '#27ae60']
bars = axes[0].bar(models_p, params, color=colors_p,
                    alpha=0.85, edgecolor='white', width=0.45)
for bar, p in zip(bars, params):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 3000,
                 f'{p:,}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

axes[0].set_title('Parameter Count\n83.2% reduction', fontsize=12)
axes[0].set_ylabel('Number of Parameters')
axes[0].set_ylim([0, 320000])
axes[0].grid(axis='y', alpha=0.3)

reduction_text = '83.2%\nfewer params'
axes[0].text(0.5, 150000, reduction_text,
             ha='center', va='center', fontsize=13,
             fontweight='bold', color='#27ae60',
             bbox=dict(boxstyle='round,pad=0.4',
                       fc='#d4edda', ec='#27ae60', alpha=0.8))

# Right: key metrics side by side
metrics      = ['Accuracy', 'Precision\n(Weighted)', 'Recall\n(Weighted)',
                'F1\n(Weighted)']
base_vals    = [0.93, 0.91, 0.92, 0.92]
lids_vals    = [0.9671, 0.9824, 0.9671, 0.9727]
base_note    = '(Binary, 2-class)'
lids_note    = '(Multiclass, 10-class)'

x     = np.arange(len(metrics))
width = 0.32
b1 = axes[1].bar(x - width/2, base_vals, width,
                  label=f'Base Paper {base_note}',
                  color='#e74c3c', alpha=0.85, edgecolor='white')
b2 = axes[1].bar(x + width/2, lids_vals, width,
                  label=f'LIDS-T {lids_note}',
                  color='#27ae60', alpha=0.85, edgecolor='white')

for bar, val in zip(b1, base_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(b2, lids_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

axes[1].set_title('Performance Metrics\nLIDS-T surpasses base paper on harder task',
                   fontsize=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics, fontsize=10)
axes[1].set_ylabel('Score')
axes[1].set_ylim([0.85, 1.02])
axes[1].legend(fontsize=8.5, loc='lower right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('viz_plot2_params_and_metrics.png')
plt.close()
print("  Saved: viz_plot2_params_and_metrics.png")

# =============================================================================
# PLOT 3 — DEPLOYMENT READINESS MATRIX (visual table)
# =============================================================================

print("Generating Plot 3: Deployment readiness matrix...")

tiers   = ['Cloud Server\n(AWS EC2)', 'Edge Gateway\n(Raspberry Pi 4)',
           'IoT Gateway\n(Raspberry Pi Zero)', 'Embedded MCU\n(ESP32 / STM32)']
criteria = ['Model fits\nin memory', 'Latency\nmeets req.', 'RAM\nfits device',
            'No GPU\nrequired', 'Overall\ndeployable']

# 1 = yes, 0 = no, 0.5 = marginal
base_matrix = [
    [1,   1,   1,   0,   1  ],   # Cloud — base paper needs GPU
    [1,   1,   1,   0,   0.5],   # Edge Gateway — marginal without GPU
    [0,   1,   1,   0,   0  ],   # IoT Gateway — 1082KB too large
    [0,   0,   0,   0,   0  ],   # MCU — completely fails
]
lids_matrix = [
    [1,   1,   1,   1,   1  ],   # Cloud
    [1,   1,   1,   1,   1  ],   # Edge Gateway
    [1,   1,   1,   1,   1  ],   # IoT Gateway — 177KB fits
    [1,   1,   1,   1,   1  ],   # MCU — 177KB fits 4MB flash
]

def cell_color(val):
    if val == 1:   return '#d4edda', '#27ae60', '✓'
    if val == 0.5: return '#fff3cd', '#f39c12', '~'
    return '#f8d7da', '#e74c3c', '✗'

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Deployment Readiness Matrix — LIDS-T vs Base Paper\n'
             'LIDS-T is deployable across all IoT hardware tiers',
             fontsize=13, fontweight='bold')

for ax, matrix, title in zip(
    axes,
    [base_matrix, lids_matrix],
    ['Base Paper (Umer et al., 2025)', 'LIDS-T (Proposed)']
):
    ax.set_xlim(0, len(criteria))
    ax.set_ylim(0, len(tiers))
    ax.set_xticks(np.arange(len(criteria)) + 0.5)
    ax.set_xticklabels(criteria, fontsize=9.5)
    ax.set_yticks(np.arange(len(tiers)) + 0.5)
    ax.set_yticklabels(tiers[::-1], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color='#e74c3c' if 'Base' in title else '#27ae60')
    ax.tick_params(length=0)

    for i, row in enumerate(matrix[::-1]):
        for j, val in enumerate(row):
            fc, ec, symbol = cell_color(val)
            rect = plt.Rectangle([j, i], 1, 1,
                                  facecolor=fc, edgecolor='white',
                                  linewidth=2)
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, symbol,
                    ha='center', va='center',
                    fontsize=22, color=ec, fontweight='bold')

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#d4edda', edgecolor='#27ae60', label='✓ Deployable'),
    mpatches.Patch(facecolor='#fff3cd', edgecolor='#f39c12', label='~ Marginal'),
    mpatches.Patch(facecolor='#f8d7da', edgecolor='#e74c3c', label='✗ Not deployable'),
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=3, fontsize=10, frameon=True,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('viz_plot3_deployment_matrix.png')
plt.close()
print("  Saved: viz_plot3_deployment_matrix.png")

# =============================================================================
# PLOT 4 — FULL RESULTS COMPARISON (3-way: Base Paper / Baseline / LIDS-T)
# =============================================================================

print("Generating Plot 4: Full results comparison...")

models = ['Base Paper\n(Binary)', 'Multiclass\nBaseline', 'LIDS-T\n(Proposed)']
acc    = [0.9300, 0.7162, 0.9671]
wp     = [0.9100, 0.7823, 0.9824]
wr     = [0.9200, 0.7162, 0.9671]
wf1    = [0.9200, 0.7215, 0.9727]
mf1    = [None,   0.3883, 0.4958]   # base paper never reported macro F1

x      = np.arange(len(models))
width  = 0.16
colors_m = ['#e74c3c', '#e67e22', '#27ae60']

fig, ax = plt.subplots(figsize=(15, 7))

metrics_data = [
    ('Accuracy',          acc,  -2*width),
    ('Weighted Precision', wp,   -1*width),
    ('Weighted Recall',   wr,    0),
    ('Weighted F1',       wf1,   1*width),
    ('Macro F1',          mf1,   2*width),
]

bars_list = []
for label, vals, offset in metrics_data:
    # Handle None values (base paper macro F1)
    plot_vals = [v if v is not None else 0 for v in vals]
    bars = ax.bar(x + offset, plot_vals, width,
                  label=label, alpha=0.85, edgecolor='white')
    bars_list.append((bars, vals))
    for bar, val in zip(bars, vals):
        if val is not None and val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}',
                    ha='center', va='bottom',
                    fontsize=7.5, rotation=45)
        elif val is None:
            ax.text(bar.get_x() + bar.get_width()/2,
                    0.02, 'N/A',
                    ha='center', va='bottom',
                    fontsize=7.5, color='#888888')

ax.set_title('Comprehensive Results Comparison\n'
             'Base Paper (binary) vs Multiclass Baseline vs Proposed LIDS-T',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim([0, 1.08])
ax.legend(fontsize=9.5, loc='lower right', ncol=2)
ax.grid(axis='y', alpha=0.3)

# Highlight LIDS-T column
ax.axvspan(1.6, 2.4, alpha=0.04, color='#27ae60', label='_nolegend_')

plt.tight_layout()
plt.savefig('viz_plot4_full_comparison.png')
plt.close()
print("  Saved: viz_plot4_full_comparison.png")

# =============================================================================
# PLOT 5 — EFFICIENCY SUMMARY (FLOPs + latency + RAM in one figure)
# =============================================================================

print("Generating Plot 5: Efficiency summary...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('LIDS-T Edge Deployment Efficiency Metrics',
             fontsize=14, fontweight='bold')

# FLOPs
ax = axes[0]
flops_labels = ['Base Paper\n(est.)', 'LIDS-T']
flops_vals   = [2100000, 28732]
flops_colors = ['#e74c3c', '#27ae60']
bars = ax.bar(flops_labels, flops_vals,
              color=flops_colors, alpha=0.85, edgecolor='white', width=0.45)
for bar, val in zip(bars, flops_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 20000,
            f'{val:,}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_title('FLOPs per Inference\n98.6% reduction', fontsize=11)
ax.set_ylabel('Floating Point Operations')
ax.grid(axis='y', alpha=0.3)
ax.text(0.5, 1050000, '98.6%\nfewer FLOPs',
        ha='center', va='center', fontsize=12,
        fontweight='bold', color='#27ae60',
        bbox=dict(boxstyle='round,pad=0.3', fc='#d4edda',
                  ec='#27ae60', alpha=0.8),
        transform=ax.transData)

# Latency
ax = axes[1]
lat_labels = ['CPU\n(PyTorch)', 'CPU\n(ONNX RT)', 'IoT req.\n(<10ms)']
lat_vals   = [0.7472, 0.4600, 10.0]
lat_colors = ['#3498db', '#27ae60', '#bdc3c7']
bars = ax.bar(lat_labels, lat_vals,
              color=lat_colors, alpha=0.85, edgecolor='white', width=0.45)
for bar, val in zip(bars, lat_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{val:.2f}ms', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_title('Inference Latency\nBoth meet IoT requirement', fontsize=11)
ax.set_ylabel('Latency (ms)')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 13])

# RAM
ax = axes[2]
ram_labels  = ['Peak RAM\n(LIDS-T)', 'ESP32\nRAM budget', 'Arduino\nRAM budget']
ram_vals    = [3.8, 520, 8]
ram_colors  = ['#27ae60', '#3498db', '#9b59b6']
bars = ax.bar(ram_labels, ram_vals,
              color=ram_colors, alpha=0.85, edgecolor='white', width=0.45)
for bar, val in zip(bars, ram_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f'{val:.1f} KB', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_title('Peak Inference RAM\n3.8KB fits any IoT device', fontsize=11)
ax.set_ylabel('Memory (KB)')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('viz_plot5_efficiency_summary.png')
plt.close()
print("  Saved: viz_plot5_efficiency_summary.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*55)
print("ALL MISSING VISUALIZATIONS GENERATED")
print("="*55)
print("\n  Files saved:")
print("    viz_plot1_model_size_comparison.png")
print("    viz_plot2_params_and_metrics.png")
print("    viz_plot3_deployment_matrix.png")
print("    viz_plot4_full_comparison.png")
print("    viz_plot5_efficiency_summary.png")
print("\n  Complete visualization list for paper:")
print("  --- Phase 2A (Binary Baseline) ---")
print("    plot1_confusion_matrix.png")
print("    plot2_metrics_comparison.png")
print("    plot3_roc_curve.png")
print("    plot4_precision_recall_curve.png")
print("    plot5_per_class_metrics.png")
print("    plot6_training_curves.png")
print("  --- LIDS-T Main ---")
print("    lids_t_architecture.png")
print("    viz_plot1_model_size_comparison.png")
print("    viz_plot2_params_and_metrics.png")
print("    viz_plot4_full_comparison.png")
print("    viz_plot5_efficiency_summary.png")
print("  --- Deployment ---")
print("    viz_plot3_deployment_matrix.png")
print("  --- SHAP ---")
print("    shap_plot1_per_class_top5.png")
print("    shap_plot2_global_importance.png")
print("    shap_plot3_heatmap.png")
print("    shap_plot4_top_feature_per_class.png")
print("  --- Ablation (when done) ---")
print("    ablation_plot1_overall.png")
print("    ablation_plot2_heatmap.png")
print("    ablation_plot3_acc_vs_params.png")
print("="*55)