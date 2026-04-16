# DUT-Mamba-MoE — Cardiac Cine MRI Segmentation

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision monai nibabel SimpleITK gudhi \
            pingouin statsmodels seaborn scikit-learn scipy \
            tqdm einops timm

# 2. Verify your datasets (no GPU needed)
python verify_data.py --data_root /Volumes/Transcend

# 3. Demo on synthetic data (no real data needed)
python demo_quick.py

# 4. Full training
python main.py --mode full --data_root /Volumes/Transcend

# 5. Test only (requires checkpoint)
python main.py --mode test_only \
    --data_root /Volumes/Transcend \
    --checkpoint ./outputs/seed_42/checkpoint_best_seed42.pt

# 6. Plot figures only
python main.py --mode plot_only --output_dir ./outputs
```

---

## Dataset Structures Expected

### ACDC
```
ACDC-dataset/
└── database/
    ├── training/
    │   └── patient001/
    │       ├── Info.cfg
    │       ├── patient001_frame01.nii.gz
    │       ├── patient001_frame01_gt.nii.gz
    │       ├── patient001_frame12.nii.gz
    │       └── patient001_frame12_gt.nii.gz
    └── testing/
        └── patient101/ ...
```

### M&Ms-1
```
M&M1/
├── 211230_M&M...endataset.csv      ← ED/ES frame indices
├── Testing/   {ID}/ {ID}_sa.nii.gz  {ID}_sa_gt.nii.gz
├── Training/
│   ├── Labeled/   {ID}/ {ID}_sa.nii.gz  {ID}_sa_gt.nii.gz
│   └── Unlabeled/ {ID}/ {ID}_sa.nii.gz
└── Validation/ {ID}/ {ID}_sa.nii.gz  {ID}_sa_gt.nii.gz
```
*Note: `_sa.nii.gz` is 4-D (H,W,D,T). ED/ES frames extracted from CSV.*

### M&Ms-2
```
MnM2/
├── dataset_information.csv
└── dataset/
    └── 027/
        ├── 027_SA_ED.nii.gz        ← 3-D Short Axis ED
        ├── 027_SA_ES.nii.gz        ← 3-D Short Axis ES
        ├── 027_SA_ED_gt.nii.gz
        ├── 027_SA_ES_gt.nii.gz
        ├── 027_SA_CINE.nii.gz      (not used)
        └── 027_LA_*.nii.gz         (not used)
```

---

## Project Structure

```
cardiac_dut/
├── configs/config.py          All hyperparameters
├── data/preprocessing.py      MONAI transforms, TPA, parsers (ACDC/MMs1/MMs2)
├── models/architecture.py     VSS Encoder + EDL Decoder + MoE Routing
├── training/
│   ├── losses.py              EDL, Betti3D, TCL, Homoscedastic, G_TTEC
│   └── trainer.py             3-phase curriculum trainer
├── ttec/ttec_engine.py        TU scorer, G_TTEC soft gating, Topo-A*
├── evaluation/metrics.py      Dice/HD95/TER/ECE/OOD/EF/σ(EF)
├── visualization/plots.py     12 figure types (paper-quality)
├── main.py                    Full workflow orchestration
├── demo_quick.py              Verify pipeline (no real data)
└── verify_data.py             Verify dataset parsers on real data
```

## Key Libraries

| Library | Purpose |
|---|---|
| MONAI | Medical transforms, metrics, DataLoader |
| GUDHI | Cubical Complex, Persistent Homology (3D Betti) |
| nibabel | NIfTI I/O |
| pingouin | Fleiss κ inter-rater agreement |
| scikit-learn | ECE, OOD AUC, classification metrics |
| scipy | Paired t-test, Wilcoxon, Pearson r |
| seaborn | Paper-quality figures |

## Why 5 Seeds?

Training results are affected by random weight initialisation, data
shuffling, and augmentation. Running 5 seeds allows reporting
`mean ± std`, which IEEE TMI reviewers require to show reproducibility.
Results from a single seed cannot be statistically compared to baselines.

## GPU Requirements

- Minimum: 16 GB VRAM (batch_size=1, spatial_size=(128,128,8))
- Recommended: 40 GB+ (batch_size=2, spatial_size=(256,256,16))
- Training time: ~5 hours/seed on A100 80 GB
