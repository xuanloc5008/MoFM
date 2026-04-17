# Topo-Evidential U-Mamba

**Kiến trúc Lai cho Phân vùng Ảnh Y tế Bền bỉ dựa trên Đặc trưng Topo và Định lượng Bất định**

---

## Tổng quan Kiến trúc

```
Input (1×H×W)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│            U-Mamba Encoder (CNN-SSM Hybrid)         │
│  [32] → [64] → [128] → [256] → [512] (Bottleneck)  │
└───────────────────────┬─────────────────────────────┘
                        │  v_i  (bottleneck feature map)
              ┌─────────┴──────────┐
              ▼                    ▼
  ┌───────────────────┐  ┌──────────────────────────┐
  │  Branch 1: TDA    │  │  Branch 2: EDL Decoder   │
  │  Contrastive Head │  │  (U-Mamba Decoder +      │
  │  ProjectionMLP    │  │   ReLU evidence head)    │
  │  ──────────────── │  │  ──────────────────────  │
  │  SimPD-guided     │  │  e_p ≥ 0  (K classes)    │
  │  L_PD-SCon        │  │  α_p = e_p + 1           │
  │  (domain-         │  │  û_p = K / S_p           │
  │   invariance)     │  │  u_p ∈ (0, 1]            │
  └───────────────────┘  └──────────────────────────┘
```

**Hàm mất mát tổng hợp:**
```
L_Total = Σ_p L_EDL(p) + γ · L_PD-SCon
        = Σ_p [L_MSE + L_Var + λ_t · L_KL] + γ · L_PD-SCon
```

---

## Cấu trúc Dự án

```
topo_evidential_umamba/
├── configs/
│   └── config.yaml              ← Toàn bộ hyperparameters
├── src/
│   ├── data/
│   │   ├── acdc.py              ← ACDC loader (train/val/test split)
│   │   ├── mnm.py               ← M&Ms-1 + M&Ms-2 loader
│   │   └── transforms.py        ← MONAI augmentation pipeline
│   ├── models/
│   │   ├── ssm_block.py         ← SSM kernel + CNN-SSM hybrid block
│   │   ├── umamba.py            ← U-Mamba encoder-decoder backbone
│   │   └── topo_evidential_umamba.py  ← Full Y-shape architecture
│   ├── topology/
│   │   └── persistence.py       ← Gudhi/Ripser PD + Wasserstein W1 + SimPD
│   ├── losses/
│   │   ├── edl_loss.py          ← Pixel-wise EDL loss (MSE + Var + KL)
│   │   └── contrastive_loss.py  ← PD-guided supervised contrastive loss
│   ├── training/
│   │   └── trainer.py           ← End-to-end training engine + TensorBoard
│   ├── evaluation/
│   │   ├── metrics.py           ← Dice, HD95, ASD, ECE, AURRC
│   │   └── clinical_metrics.py  ← EDV, ESV, EF, Myo mass, Bland-Altman
│   └── visualization/
│       └── plots.py             ← All 9 plot types
├── train.py                     ← Điểm khởi chạy training
├── evaluate.py                  ← Điểm khởi chạy evaluation
├── scripts/
│   └── demo_synthetic.py        ← Demo đầy đủ pipeline trên dữ liệu tổng hợp
└── requirements.txt
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

**Thư viện chuyên biệt:**
| Thư viện | Mục đích |
|----------|----------|
| `monai`  | Medical image transforms, augmentation |
| `gudhi`  | Cubical Complex, Persistence Diagrams |
| `POT`    | Wasserstein distance (Python Optimal Transport) |
| `ripser` | Rips complex (fallback TDA) |
| `nibabel` | NIfTI file I/O |
| `medpy`  | HD95, surface distance metrics |
| `einops` | Tensor reshaping cho SSM |

---

## Chạy nhanh (Demo tổng hợp – không cần dataset thật)

```bash
cd topo_evidential_umamba
python scripts/demo_synthetic.py
```

Pipeline hoàn chỉnh trên dữ liệu tổng hợp:
1. Training 20 epoch với L_Total = L_EDL + γ·L_PD-SCon
2. Đánh giá Dice, HD95 theo từng lớp
3. Vẽ uncertainty map + segmentation overlay
4. Reliability diagram (ECE)
5. Coverage-Accuracy curve
6. Domain-shift invariance theo vendor
7. OOD uncertainty histogram
8. Bland-Altman cho EDV, ESV, EF, Myo mass
9. Persistence diagram

---

## Training trên ACDC

```bash
# Chỉnh đường dẫn trong configs/config.yaml:
# paths.acdc_root: "data/ACDC-dataset"

python train.py --config configs/config.yaml --gpu 0
```

**TensorBoard:**
```bash
tensorboard --logdir outputs/logs
```

---

## Evaluation + Domain Shift Test

```bash
python evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best_model.pth
```

Outputs:
- `outputs/eval/acdc_test_metrics.csv`   — Dice, HD95, ASD per slice
- `outputs/eval/acdc_clinical_pred.csv`  — EDV, ESV, EF, Mass per patient
- `outputs/eval/mnm1_metrics.csv`        — M&Ms-1 domain shift
- `outputs/eval/mnm2_metrics.csv`        — M&Ms-2 domain shift
- `outputs/figures/`                     — Tất cả các biểu đồ

---

## Các Chỉ số Đánh giá

### Segmentation
| Metric | Mô tả |
|--------|-------|
| DSC    | Dice Similarity Coefficient (per class + mean) |
| HD95   | 95th percentile Hausdorff Distance (mm) |
| ASD    | Average Symmetric Surface Distance (mm) |

### Calibration & Uncertainty
| Metric | Mô tả |
|--------|-------|
| ECE    | Expected Calibration Error |
| AURRC  | Area Under Risk-Rejection Curve |
| Coverage-Accuracy | Accuracy khi loại bỏ pixel bất định cao |

### Clinical (per patient)
| Metric | Đơn vị | Mô tả |
|--------|--------|-------|
| LV_EDV | mL | Left Ventricular End-Diastolic Volume |
| LV_ESV | mL | Left Ventricular End-Systolic Volume |
| LV_EF  | %  | Left Ventricular Ejection Fraction |
| RV_EDV | mL | Right Ventricular End-Diastolic Volume |
| RV_EF  | %  | Right Ventricular Ejection Fraction |
| Myo_mass | g | Myocardial Mass |

### Domain Invariance
- Per-vendor Dice boxplot (Siemens / GE / Philips / Canon)
- OOD uncertainty distribution (ID vs Domain-shifted)

---

## Ba Tình Huống Lâm Sàng

| Tình huống | Mô tả | Hành vi mô hình |
|-----------|-------|-----------------|
| **In-Distribution** | Ảnh chuẩn (ACDC Siemens) | Bằng chứng cao, u_p ≈ 0 |
| **Domain Shift** | Máy chụp khác (GE, Philips) | TDA anchor giữ v_i ổn định, u_p vẫn thấp |
| **OOD** | Cấu trúc bệnh lý lạ | Topo bị phá vỡ, KL ép e→0, u_p ≈ 1 |

---

## Tham khảo

- Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty" (NeurIPS 2018)
- Gu et al. "CA-Mamba: Channel Attention Mamba for Medical Image Segmentation" (2024)
- Carlsson, G. "Topology and Data" (Bull. Amer. Math. Soc., 2009)
- Bernard et al. "Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation" (TMI 2018) — ACDC dataset
- Campello et al. "Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Segmentation" (TMI 2021) — M&Ms-1
