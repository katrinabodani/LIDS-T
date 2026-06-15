# LIDS-T: Lightweight IoT Intrusion Detection Transformer

**LIDS-T** is a dual-branch neural architecture for **multiclass network intrusion detection** on tabular flow features, designed for **IoT edge deployment**. It extends the binary transformer baseline from [Scientific Reports (2025), DOI 10.1038/s41598-025-11348-5](https://doi.org/10.1038/s41598-025-11348-5) to **10 attack classes** on the full **UNSW-NB15** dataset (~2.54M flows) while reducing trainable parameters by **~83%** (~45k vs ~270k).

---

## Architecture

LIDS-T fuses two complementary encoders over a fixed **20-dimensional feature vector** per network flow:

```
Input (B, 20)
    ├─ Branch A: Linear(20→64) → TransformerEncoder(1 layer, 4 heads, d_ff=128) → GAP → (B, 64)
    └─ Branch B: DepthwiseSeparableConv1D(20→64, k=3) → Linear(64→64)         → (B, 64)
                              ↓
              Fusion: w₀·A + w₁·B   (learned softmax weights w₀, w₁)
                              ↓
              MLP Head: Linear(64→64) → GELU → Dropout(0.2) → Linear(64→10)
                              ↓
                    10-class logits
```

| Component | Role | Design choice |
|-----------|------|---------------|
| **Transformer branch** | Global feature interactions | Single encoder layer; `d_model=64`, pre-norm (`norm_first=True`), GELU |
| **CNN branch** | Local per-feature correlations | Depthwise-separable 1D conv (~8–10× fewer params than standard conv) |
| **Fusion** | Adaptive branch weighting | Two learnable scalars, softmax-normalized |
| **Classifier** | Multiclass output | Weighted cross-entropy with inverse-frequency class weights |

**Hyperparameters** (`lids_t.py`):

| Parameter | Value |
|-----------|-------|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 1 |
| `d_ff` | 128 |
| `cnn_channels` | 64 |
| `dropout` | 0.2 |
| `batch_size` | 2048 |
| `optimizer` | AdamW (lr=1e-3, weight_decay=1e-4) |
| `scheduler` | ReduceLROnPlateau (patience=3, factor=0.5) |
| `early stopping` | patience=7 on validation loss |

---

## Dataset & Preprocessing

**Source:** UNSW-NB15 (4 raw CSV parts, 49 columns, no header).

**Pipeline** (`preprocess_full.py`):

1. Load `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` and concatenate (~2,540,047 rows).
2. Derive **10-class labels** from `attack_cat` (empty string → Normal).
3. Drop identifiers and timestamps: `srcip`, `dstip`, `Stime`, `Ltime`, `sport`, `dsport`.
4. Encode categoricals (`proto`, `state`, `service`) with `LabelEncoder`.
5. **Feature selection** — RFE with `LinearSVC` on a 150k stratified subsample → **20 features** (see `selected_features.json`).
6. **MinMaxScaler** fit on train, applied to test.
7. **80/20 stratified split** → train/test CSVs.

**Attack classes** (`attack_class_names.json`):

`Analysis`, `Backdoor`, `DoS`, `Exploits`, `Fuzzers`, `Generic`, `Normal`, `Reconnaissance`, `Shellcode`, `Worms`

**Selected features** (`selected_features.json`):

`state`, `dttl`, `sttl`, `ct_dst_src_ltm`, `ct_dst_sport_ltm`, `ct_src_dport_ltm`, `ct_state_ttl`, `smeansz`, `dmeansz`, `synack`, `Dintpkt`, `ct_srv_dst`, `Sload`, `sbytes`, `dbytes`, `ct_srv_src`, `service`, `Dload`, `Sintpkt`, `Dpkts`

---

## Repository Structure

```
LIDS-T/
├── lids_t.py              # Main training & evaluation script
├── preprocess_full.py     # Full-dataset multiclass preprocessing
├── feature_selection.py   # RF + XGBoost wrapper-based feature ranking
├── ablation_study.py      # 4-config ablation (branch / class-weight variants)
├── explainability.py      # Per-class SHAP analysis
├── early_detection.py     # Progressive feature masking (early-flow detection)
├── edge_testing.py        # RAM, CPU latency, FLOPs, deployment matrix
├── lidst_export.py        # ONNX export + INT8 dynamic quantization
├── comparison.py          # Binary baseline vs LIDS-T (JSON-driven)
├── plots.py               # Paper visualization generation
├── plots_dpi.py           # High-DPI plot variants
├── binary/
│   ├── preprocessing.py   # Binary-task preprocessing
│   └── binary_baseline.py # Binary transformer baseline reproduction
├── attack_class_names.json
├── class_mapping.json
├── feature_cols.json
├── selected_features.json
├── results_*.json         # Saved experiment metrics (no re-run needed)
└── requirements.txt
```

**Not tracked in git** (see `.gitignore`): raw/processed CSVs, model checkpoints (`.pt`), ONNX files, generated PNG/PDF outputs.

---

## Installation

```bash
git clone https://github.com/katrinabodani/LIDS-T.git
cd LIDS-T
pip install -r requirements.txt
```

Download UNSW-NB15 from the [official source](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` in the project root.

---

## Usage

Run scripts in pipeline order:

```bash
# 1. Preprocess full multiclass dataset
python preprocess_full.py

# 2. Train LIDS-T (outputs lidst_best.pt, results_lidst.json)
python lids_t.py

# 3. Optional experiments
python ablation_study.py      # ~2–3 h (4 full training runs)
python explainability.py      # SHAP per-class analysis
python early_detection.py     # Feature-masking early detection
python edge_testing.py        # Edge deployment profiling
python lidst_export.py        # ONNX + INT8 quantization

# 4. Compare & visualize
python comparison.py
python plots.py
```

---

## Reported Results

Metrics from `results_lidst.json` (test set, 10-class):

| Metric | LIDS-T | Base paper (binary) |
|--------|--------|---------------------|
| Accuracy | **0.9671** | 0.93 |
| Weighted F1 | **0.9727** | 0.92 |
| Macro F1 | 0.7870 | — |
| Parameters | **45,320** | 270,249 |
| Model size | **177 KB** (float32) | ~1,082 KB |
| GPU latency | 1.21 ms/sample | — |
| CPU latency | 0.75 ms/sample | — |

**Ablation** (`results_ablation.json`):

| Configuration | Accuracy | Macro F1 | Params |
|---------------|----------|----------|--------|
| Transformer only | 0.9074 | 0.5245 | 39,626 |
| CNN only | 0.9692 | 0.5388 | 10,502 |
| No class weighting | — | — | — |
| **Full LIDS-T** | **0.9671** | **0.7870** | **45,320** |

Full LIDS-T achieves the best **macro F1**, indicating stronger minority-class detection when both branches and class weighting are combined.

**Edge profile** (`results_edge_profile.json`):

| Metric | Value |
|--------|-------|
| Peak inference RAM | ~3.8 KB |
| CPU latency (batch=1) | 0.75 ms (p95: 1.33 ms) |
| INT8 quantized size | ~88.5 KB (via `lidst_export.py`) |

---

## Export & Deployment

`lidst_export.py` produces:

1. **ONNX** model (`lidst.onnx`) — cross-platform inference via ONNX Runtime.
2. **INT8 dynamic quantization** — ~50% size reduction with validated prediction parity.
3. Latency benchmarks simulating ARM/edge CPU targets (Raspberry Pi class devices).

Deployment readiness is scored across hardware tiers in `edge_testing.py` (RAM, latency, model size, FLOPs).

---

## Binary Baseline

The `binary/` module reproduces a **2-class** transformer baseline aligned with the reference paper. `comparison.py` loads `results_binary.json` and `results_lidst.json` to compare accuracy, parameter count, and latency without re-inference.

---

## Explainability

`explainability.py` applies **SHAP** (SHapley Additive exPlanations) to the trained LIDS-T model:

- Global feature importance per attack class
- Top-5 features per class
- Heatmaps for cross-class feature attribution

Results are saved to `results_shap.json`; plots are generated locally (not committed).

---

## Early Detection

`early_detection.py` masks features progressively (ordered by selection frequency) to simulate classification before a full flow record is available. Uses the **pre-trained** checkpoint without retraining — measures accuracy vs. number of visible features.

---

## Citation

If you use this code, please cite the base paper and acknowledge this implementation:

```bibtex
@article{base_paper_2025,
  title   = {Network intrusion detection using transformer-based deep learning},
  journal = {Scientific Reports},
  year    = {2025},
  doi     = {10.1038/s41598-025-11348-5}
}
```

---

## License

Academic research project. UNSW-NB15 dataset usage is subject to the dataset provider's terms.
