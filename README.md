# MOFMTDA-TTA: Reducing Clinical Uncertainty in Cardiac Cine MRI Segmentation

## Project Structure

```
mofmtda/
├── config.py                 # All hyperparameters (no heuristics)
├── run.py                    # Main entry point — run any phase
├── train_expert.py           # Phase 1: Train M anatomy-specialised Experts
├── train_gating.py           # Phase 2: Train Topology-Penalised Gating
├── inference.py              # Phase 3-4: TTEC + Topo-A* + Clinical Propagation
├── data/
│   ├── dataset.py            # ACDC dataset + shared augmentation + TPA filter
│   └── topology.py           # Betti numbers, Persistence Diagrams, Wasserstein
├── models/
│   ├── sam_expert.py         # SAM + AdaLoRA + 4-branch decoder
│   ├── gating.py             # Mini-UNet gating with CBAM + routing loss
│   └── losses.py             # EDL, aleatoric, contrastive, topology, UW losses
├── ttec/
│   ├── classifier.py         # TTEC 3-signal engine + inter-slice consistency
│   └── topo_astar.py         # Gradient-free A* correction for Type-III
├── utils/
│   └── metrics.py            # Dice, HD95, TER, ECE, clinical uncertainty
└── requirements.txt
```

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt

# Download MedSAM checkpoint (ViT-B, ~2.4GB)
pip install gdown
mkdir -p checkpoints
gdown 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O checkpoints/medsam_vit_b.pth
# Or download manually: https://drive.google.com/file/d/1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_/view

# Prepare ACDC data in: data/ACDC/patient001/, patient002/, ...
```

**Note**: MedSAM uses `[0,1]` normalization (not ImageNet mean/std). Already handled in `data/dataset.py`.

### 2. Preprocess (required before training)

```bash
# Basic: extract slices + compute PDs
python preprocess.py --data_root ./dataset/acdcdata/ACDC/database/training

# Full: also cache MedSAM encoder features (~40% training speedup, needs GPU)
python preprocess.py --data_root ./dataset/acdcdata/ACDC/database/training \
    --cache_encoder --sam_checkpoint ./checkpoints/medsam_vit_b.pth
```

### 3. Run Full Pipeline (single seed)

```bash
python run.py --phase all --data_root ./data/ACDC --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth
```

### 3. Run Individual Phases

```bash
# Phase 1: Train 3 experts (LV, Myo, RV specialists)
python run.py --phase 1

# Phase 2: Train gating network (experts must be trained first)
python run.py --phase 2

# Phase 3-4: Inference with TTEC + Topo-A* + evaluation
python run.py --phase eval --dataset acdc

# Cross-dataset evaluation
python run.py --phase eval --dataset mms
```

### 4. Multi-Seed Reproducibility

```bash
# Run full pipeline with 5 seeds, reports mean ± std
python run.py --phase all --seeds 5 --seed 42
```

### 5. Custom Configuration

```python
from config import Config

config = Config()
config.train.expert_epochs = 50          # Fewer epochs for debugging
config.train.expert_batch_size = 2       # Smaller batch for limited GPU
config.adalora.r_init = 8               # Lower initial rank
config.train.cache_encoder_features = True  # 40% speedup
```

## Key Design Decisions

### No Heuristic Hyperparameters
- **Loss weights**: Learned via homoscedastic uncertainty weighting (Kendall et al.)
- **LoRA ranks**: Learned via AdaLoRA (SVD-based pruning from uniform r=16)
- **Augmentation**: Shared pool for all experts, filtered by TPA

### Three-Tier Uncertainty
1. **Pixel-level**: Vacuity (epistemic) + Dissonance (aleatoric)
2. **Task-level**: Learned σ_s per structure (from uncertainty weighting)
3. **Clinical-level**: σ(EF) via analytic propagation + MC validation

### TTEC: 3 Signals (not 4)
- **DSS**: Distribution Shift Score (OOD detection)
- **PLR**: Persistence Lifetime Ratio (transient vs stable violations)
- **SPCS**: Slice-Position Context Score (anatomical expectation)
- ~~TTSS~~: Removed (requires full temporal sequence, not available in ED/ES-only setup)

### Topo-A*: Only for Type-III
- Type I → flag for human review
- Type II → accept (anatomically correct)
- Type III → correct via A* with topological cost

## GPU Requirements (with MedSAM, r_init=8, 70 epochs)

| GPU | Batch Size | Phase 1 (3 Experts) | Phase 2 (Gating) | Total |
|-----|-----------|--------------------|--------------------|-------|
| RTX 4090 (24GB) | 4 | ~25h | ~3h | ~28h |
| A100 (80GB) | 4 | ~17h | ~2h | ~19h |
| RTX 3090 (24GB) | 4 | ~28h | ~3h | ~31h |
| RTX 4080 (16GB) | 2 | ~38h | ~4h | ~42h |

Compared to vanilla SAM (r_init=16, 100 epochs), MedSAM reduces training time by ~45% due to:
- Lower initial rank (8 vs 16) → fewer trainable parameters
- Fewer epochs (70 vs 100) → encoder already has medical features
- Faster convergence → medical-domain pretraining provides better initialisation
