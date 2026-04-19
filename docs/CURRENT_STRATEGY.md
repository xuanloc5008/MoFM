# Current Strategy

This document explains the **current training strategy** of this repository on the
experimental branch.

It is meant to answer:

- what the model is optimizing right now
- where topology enters the pipeline
- which phases are already implemented
- what is still experimental vs stable

---

## Goal

The current strategy is designed for:

- **ACDC-only training and validation**
- **M&Ms-1 / M&Ms-2 as external domain-shift evaluation**
- high in-domain Dice on ACDC
- strong robustness and calibration under scanner/vendor shift

The design principle is:

- keep **segmentation quality** as the main objective
- use **topology as an auxiliary structural prior**
- introduce stronger topological machinery gradually, behind config flags

---

## Current Pipeline

The current end-to-end path is:

```text
Raw MRI volume
  -> image-only preprocessing / harmonization
  -> 2.5D slice construction
  -> U-Mamba trunk
  -> segmentation branch (EDL + Dice)
  -> topology branch(es)
       1) cached topo_vec contrastive supervision
       2) experimental barcode encoder alignment
```

There are now **two topology paths** in the repo:

1. **Stable path**
   - cached `topo_vec`
   - topology-guided contrastive loss
   - this is the original practical topology branch

2. **Experimental path**
   - cached persistence barcodes in birth-lifetime form
   - learnable barcode encoder (`torchph`-style SLayer idea)
   - topology alignment loss between image projections and barcode embeddings

The segmentation branch remains the main driver of Dice.

---

## What Is Already Implemented

### Phase 1: Image Harmonization

Implemented.

Purpose:

- reduce domain gap before the model sees the image
- make ACDC and M&Ms more comparable in spacing and intensity

Current behavior:

- crop obvious empty borders
- fixed in-plane resampling
- per-volume normalization
- final pad/crop to fixed size

Main files:

- [src/data/preprocessing.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/data/preprocessing.py)
- [src/data/acdc.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/data/acdc.py)
- [src/data/mnm.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/data/mnm.py)
- [scripts/preprocess_acdc.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/scripts/preprocess_acdc.py)

### Phase 2: 2.5D Input

Implemented.

Purpose:

- give the model local through-slice context
- improve RV/Myo consistency

Current behavior:

- input uses adjacent slices as channels
- default config is `[prev, center, next]`

Main files:

- [src/data/context.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/data/context.py)
- [src/data/acdc.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/data/acdc.py)
- [src/data/mnm.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/data/mnm.py)
- [src/models/topo_evidential_umamba.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/models/topo_evidential_umamba.py)

### Phase 3: Boundary-Aware Loss

**Not implemented yet.**

Current losses still do **not** include boundary loss / surface loss / focal Tversky.

---

## Stable Training Objective

The stable baseline objective is:

```text
L_total = L_EDL + w_dice * L_Dice + gamma * L_Con
```

Where:

- `L_EDL` = evidential segmentation loss
- `L_Dice` = soft Dice loss
- `L_Con` = topology-guided contrastive loss using cached `topo_vec`

Main files:

- [src/losses/edl_loss.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/losses/edl_loss.py)
- [src/losses/contrastive_loss.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/losses/contrastive_loss.py)
- [src/training/trainer.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/training/trainer.py)

This is still the main reason the model achieves good Dice.

---

## Stable Topology Strategy

The stable topology strategy is:

1. Compute persistence information **offline**
2. Convert it into a compact cached summary vector `topo_vec`
3. Load `topo_vec` during training
4. Use it to mine positive pairs for supervised contrastive learning

Why it exists:

- avoids expensive per-batch PH computation
- keeps GPU training fast
- gives a structural prior without destabilizing segmentation

This is implemented in:

- [src/topology/persistence.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/topology/persistence.py)
- [scripts/preprocess_acdc.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/scripts/preprocess_acdc.py)
- [src/losses/contrastive_loss.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/losses/contrastive_loss.py)

---

## Experimental Topology Strategy

This branch now also includes a **riskier topology experiment**.

### Idea

Instead of using only a handcrafted summary vector, the repo now also caches
raw-ish topological information as fixed-size barcode tensors:

- `barcode_h0`
- `barcode_h1`
- plus valid point counts

These are represented in **birth-lifetime** coordinates.

Then a learnable encoder maps those barcode points into a topology embedding.

### Why

This follows the ideas from:

- differentiable barcode vectorization
- learning representations of persistence barcodes
- connectivity-optimized representation learning

The goal is to make topology:

- more learnable
- more task-adaptive
- less dependent on a single handcrafted `topo_vec`

### Current Experimental Loss

The experimental branch adds:

```text
L_total =
    L_EDL
  + w_dice * L_Dice
  + gamma * L_Con
  + lambda_topo * L_TopoAlign
```

Where:

- `L_TopoAlign` aligns:
  - image projection embedding
  - learnable barcode embedding

This is intentionally low-risk compared with full differentiable topology on
predicted masks.

### How It Is Implemented

Barcode cache:

- [src/topology/persistence.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/topology/persistence.py)
- [scripts/preprocess_acdc.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/scripts/preprocess_acdc.py)

Barcode loading:

- [src/data/acdc.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/data/acdc.py)

Learnable topology encoder:

- [src/models/topology_encoder.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/models/topology_encoder.py)

Alignment loss:

- [src/losses/topology_loss.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/losses/topology_loss.py)

Training integration:

- [src/models/topo_evidential_umamba.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/models/topo_evidential_umamba.py)
- [src/training/trainer.py](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/src/training/trainer.py)

---

## torchph vs Fallback

The experimental barcode encoder supports two backends:

1. **`torchph`**
   - if installed
   - uses the intended learnable barcode vectorization backend

2. **fallback SLayer-style encoder**
   - built into the repo
   - used when `torchph` is not available

So the branch remains runnable even before extra topology libraries are installed.

Config:

- [configs/config.yaml](/Users/xuanloc/Documents/NCKH/topo_evidential_umamba/configs/config.yaml)
  - `topology_experimental.backend`

Optional install:

```bash
pip install git+https://github.com/c-hofer/torchph.git
```

`TopologyLayer` is not yet used directly in the forward training path.

---

## Why The Repo Does Not Use Full Differentiable Topology Everywhere Yet

Because the current strategy is trying to protect:

- ACDC Dice
- OOD Dice
- training stability
- training speed

Full differentiable PH on predictions or images is more expressive, but also:

- heavier
- harder to optimize
- easier to destabilize segmentation

So the repo currently uses an **incremental topology strategy**:

1. stable cached `topo_vec`
2. learnable barcode encoder
3. mild topology alignment loss
4. only later, if needed:
   - prediction-level topology loss
   - TopologyLayer on soft masks
   - latent PH regularization

---

## Current Config Meaning

The most important config sections are:

- `preprocessing`
  - image-only harmonization
- `data.context_slices`
  - 2.5D input width
- `topology`
  - cached persistent-homology settings
- `loss`
  - stable segmentation + contrastive balance
- `topology_experimental`
  - new learnable barcode experiment

If `topology_experimental.enabled: true`, the branch expects the preprocessed
ACDC cache to include barcode fields.

That means you must rebuild the cache after enabling this branch:

```bash
python scripts/preprocess_acdc.py --config configs/config.yaml --overwrite
```

---

## Current Recommended Workflow

### 1. Rebuild cache

```bash
python scripts/preprocess_acdc.py --config configs/config.yaml --overwrite
```

### 2. Train

```bash
python train.py --config configs/config.yaml --device cuda
```

### 3. Evaluate

```bash
python evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best_model.pth \
  --device auto \
  --datasets acdc mnm1
```

---

## Current Risk Profile

### Stable pieces

- Phase 1 preprocessing
- Phase 2 2.5D input
- EDL + Dice
- cached `topo_vec` contrastive loss

### Experimental pieces

- cached barcode tensors
- learnable barcode encoder
- topology alignment loss
- `torchph` backend usage

So this branch should be treated as:

- **stable segmentation path**
- plus **experimental richer topology path**

---

## Not Yet Included

The following are **not** part of the current strategy yet:

- boundary-aware loss (Phase 3)
- TopologyLayer loss on predicted masks
- differentiable level-set topology regularization on segmentation outputs
- latent connectivity PH loss
- topology-guided postprocessing

Those are possible next steps, but they are not active now.

---

## Short Summary

The current strategy is:

- preprocess images to reduce domain gap
- use 2.5D slices for better anatomical context
- train segmentation with `EDL + Dice`
- keep cached topology-guided contrastive learning
- experimentally add a learnable barcode branch for richer topology

In one sentence:

> The repo currently follows a segmentation-first strategy, where topology is
> used as a controlled structural prior rather than the main optimization
> target.
