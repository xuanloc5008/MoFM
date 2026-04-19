# ACDC-Only Roadmap To Reach `ACDC > 0.90` And `M&Ms-1 > 0.85`

This roadmap is specific to the current repository layout and current baseline.
It assumes:

- Training uses **ACDC only**.
- Validation/model selection also uses **ACDC only**.
- **M&Ms-1 remains external validation** and is never used for tuning.
- Current architecture direction stays the same:
  `U-Mamba + EDL + Soft Dice + mild topology contrastive`.

## Current Baseline

Current observed baseline before the new preprocessing strategy:

- ACDC test mean Dice: `0.8711`
- ACDC ECE: `0.0024`
- ACDC AURRC: `0.9921`
- M&Ms-1 Canon Dice: `0.7184`
- M&Ms-1 Philips Dice: `0.8606`
- M&Ms-1 Siemens Dice: `0.8187`
- M&Ms-1 GE Dice: `0.8425`
- M&Ms-1 overall calibration remains strong:
  `ECE=0.0117`, `AURRC=0.9547`

Target:

- ACDC mean Dice `> 0.90`
- M&Ms-1 overall mean Dice `> 0.85`
- M&Ms-1 each vendor Dice `> 0.80`

## Guiding Principles

- Do not weaken the OOD claim by using M&Ms labels for training or model selection.
- Prefer **image-only harmonization** over any label-informed preprocessing at inference.
- Keep TDA, but make it operate on cleaner anatomy rather than raw scanner style.
- Improve OOD first through:
  preprocessing, input context, augmentation, and postprocessing.
- Keep the current loss balance conservative:
  segmentation-first, topology as auxiliary.

## Exact Build Order

### Phase 0: Freeze The Baseline

Goal:

- Preserve a clean baseline for comparison before new changes.

Actions:

- Keep the current checkpoint and evaluation outputs as the reference run.
- Save the current metrics table and vendor-wise M&Ms-1 results in the experiment log.

Files to keep as reference:

- `outputs/eval/acdc_test_summary.csv`
- `outputs/eval/mnm1_metrics.csv`
- `outputs/eval/mnm1_ece.csv`
- `outputs/eval/evaluate.log`

Success criterion:

- We can compare every next phase against the current baseline numbers above.

---

### Phase 1: Image Harmonization Preprocessing

Why first:

- This is the highest-value change for OOD transfer.
- It directly targets scanner/vendor differences before the model sees the data.
- It should help both ACDC and M&Ms-1.

What to add:

- Fixed **in-plane spacing resampling**.
- **Volume-level normalization** instead of purely slice-level normalization.
- Consistent image-only **cardiac-centered crop / crop-then-pad**.

Repository files to modify:

- `src/data/transforms.py`
- `src/data/acdc.py`
- `src/data/mnm.py`
- `scripts/preprocess_acdc.py`
- `configs/config.yaml`

Implementation direction:

1. Add config entries for:
   - `target_spacing_xy`
   - `use_volume_normalization`
   - `crop_mode`
   - `crop_margin`
2. Resample each slice/volume to common in-plane spacing before final crop.
3. Normalize per volume/phase instead of independent per slice.
4. Keep preprocessing image-only at inference time.

Important constraint:

- Do not use GT masks to crop at test time.

Expected effect:

- ACDC: modest gain
- M&Ms-1: meaningful gain, especially Canon / Siemens / GE

Decision gate:

- If M&Ms-1 Canon does not move upward at all, the harmonization is too weak or too ACDC-specific.

---

### Phase 2: Upgrade From 2D To 2.5D Input

Why second:

- The current network sees a single slice only.
- RV/Myo consistency often improves a lot with adjacent-slice context.
- 2.5D is lower risk than a full 3D rewrite.

What to add:

- Input as `3` or `5` adjacent slices stacked as channels.

Repository files to modify:

- `src/data/acdc.py`
- `src/data/mnm.py`
- `src/models/topo_evidential_umamba.py`
- `configs/config.yaml`
- possibly `scripts/preprocess_acdc.py` if the offline cache also stores 2.5D inputs

Implementation direction:

1. Add config entries:
   - `data.context_slices`
   - example values: `1`, `3`, `5`
2. For each center slice, build channel stack:
   - `[s-1, s, s+1]` or `[s-2, ..., s+2]`
3. Change `data.in_channels` accordingly.
4. Preserve the center-slice label as supervision target.

Expected effect:

- ACDC: moderate gain
- M&Ms-1: moderate-to-strong gain
- Often especially useful for RV

Decision gate:

- If RV does not improve but LV/Myo do, context is working but class imbalance/boundary quality still needs more help.

---

### Phase 3: Add Boundary-Aware Segmentation Loss

Why third:

- Once the image representation is cleaner and context is richer, the next bottleneck is often contour quality.
- RV and Myo usually benefit from overlap + boundary supervision.

What to add:

- Keep current `EDL + SoftDice + gamma * contrastive`
- Add one boundary-sensitive term:
  - boundary loss
  - surface loss
  - or focal Tversky

Repository files to modify:

- `src/losses/edl_loss.py`
- `src/training/trainer.py`
- `configs/config.yaml`

Implementation direction:

1. Introduce `loss.boundary_weight` or `loss.tversky_weight`.
2. Keep `gamma` small.
3. Do not replace EDL.
4. Combine as:
   `L_total = L_EDL + w_dice * L_dice + w_boundary * L_boundary + gamma * L_con`

Expected effect:

- ACDC: good chance to push closer to `0.90`
- M&Ms-1: improves class borders and reduces sloppy contours under shift

Decision gate:

- If ACDC improves but M&Ms-1 stalls, the remaining gap is more likely appearance harmonization or postprocessing.

---

### Phase 4: Anatomy-Aware Postprocessing

Why fourth:

- This usually gives some of the cheapest OOD gains.
- Canon-like failures often include small islands, fragmented RV, or implausible small blobs.

What to add:

- Largest connected component per class
- Tiny component removal
- Optional simple cardiac prior checks

Repository files to modify:

- create `src/evaluation/postprocess.py`
- `evaluate.py`
- optionally `src/training/trainer.py` for validation-time postprocessing
- `configs/config.yaml`

Implementation direction:

1. Add a reusable postprocessing function for predictions.
2. Apply it consistently during validation and evaluation.
3. Start simple:
   - largest component per foreground class
   - minimum component size threshold

Expected effect:

- Small ACDC gain
- Often meaningful M&Ms-1 gain
- Canon is the main vendor expected to benefit

Decision gate:

- If postprocessing improves only one vendor and hurts ACDC, make thresholds less aggressive.

---

### Phase 5: Stronger MRI Domain Randomization

Why fifth:

- After harmonization and better supervision, this is the main ACDC-only tool for closing vendor gaps.

What to strengthen:

- bias field
- blur
- noise
- Gibbs/ringing
- contrast/gamma
- resolution degradation then resize
- intensity drift

Repository files to modify:

- `src/data/transforms.py`
- `configs/config.yaml`

Implementation direction:

1. Keep the current augmentations.
2. Add a stronger config profile for OOD-focused training.
3. Prefer realistic MRI corruptions over generic random distortions.

Expected effect:

- Mainly M&Ms-1 gain
- should help Canon most if tuned well

Decision gate:

- If ACDC drops sharply while M&Ms-1 barely changes, augmentation is too strong or unrealistic.

---

### Phase 6: Fold-Based ACDC Ensembling

Why sixth:

- This is one of the best ways to push ACDC beyond `0.90`.
- It also usually improves OOD mean Dice.

What to add:

- 5-fold ACDC training and checkpoint ensembling

Repository files to modify:

- `train.py`
- `evaluate.py`
- possibly add:
  - `scripts/train_folds.py`
  - `scripts/evaluate_ensemble.py`
- `configs/config.yaml`

Implementation direction:

1. Train folds using only ACDC.
2. Select each fold checkpoint on ACDC val.
3. At evaluation, average logits or probabilities.

Expected effect:

- ACDC: strong chance to cross `0.90`
- M&Ms-1: moderate gain

Decision gate:

- If single-model performance is still too weak, fix upstream phases before relying on ensemble.

---

### Phase 7: Revisit Topology Branch After Harmonization

Why last:

- TDA should stay, but only after the image pipeline is cleaner.
- Otherwise topology descriptors can partly encode scanner noise rather than anatomy.

What to refine:

- keep `top-k` positives
- keep contrastive weight mild
- consider better topology descriptors after spacing harmonization

Repository files to modify:

- `src/topology/persistence.py`
- `src/losses/contrastive_loss.py`
- `src/training/trainer.py`
- `configs/config.yaml`

Implementation direction:

1. Recompute cached topology vectors after the new preprocessing pipeline is finalized.
2. Inspect whether topology similarity becomes more selective.
3. Only then consider modest contrastive refinements.

Expected effect:

- Better interpretability of TDA contribution
- Small-to-moderate gain, mostly as complementary regularization

Decision gate:

- If segmentation and OOD improve without changing TDA, leave TDA mild.
- Do not let contrastive dominate training again.

## What Not To Change First

- Do not increase `loss.gamma` again.
- Do not train on M&Ms labels.
- Do not make label-guided inference-time crops.
- Do not start with a full 3D rewrite.
- Do not judge TDA before harmonized preprocessing is in place.

## Expected Milestone Path

Reasonable milestone expectations:

- After Phase 1:
  - ACDC around `0.88`
  - M&Ms-1 Canon improves noticeably
- After Phase 2 + 3:
  - ACDC around `0.89`
  - M&Ms-1 overall enters the mid `0.84` range if the changes are healthy
- After Phase 4 + 5:
  - Canon has the best chance to move above `0.80`
  - M&Ms-1 overall has the best chance to exceed `0.85`
- After Phase 6:
  - ACDC has the best chance to exceed `0.90`

## Exact Implementation Sequence To Follow In This Repo

1. Build **Phase 1** first.
2. Rebuild ACDC offline cache and rerun baseline training/evaluation.
3. Build **Phase 2** next.
4. Then add **Phase 3**.
5. Then add **Phase 4**.
6. Then tune **Phase 5**.
7. Only after the single-model pipeline is strong, add **Phase 6**.
8. Revisit **Phase 7** last.

This order is intentional:

- harmonization before topology refinement
- context before stronger loss engineering
- postprocessing before ensembling
- ensembling only after the single model is already strong

## Immediate Next Build Step

The next change to implement first is:

- **Phase 1: fixed-spacing harmonized preprocessing**

That is the highest-payoff change for both:

- `ACDC > 0.90`
- `M&Ms-1 overall > 0.85`
- `Canon > 0.80`
