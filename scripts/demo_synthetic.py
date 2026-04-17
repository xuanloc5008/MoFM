"""
demo_synthetic.py
------------------
End-to-end demonstration on SYNTHETIC cardiac MRI data.

Run this to verify the entire pipeline works without needing
actual ACDC / M&Ms datasets:

  python scripts/demo_synthetic.py

Generates:
  outputs/demo/figures/  – all evaluation plots
  outputs/demo/eval/     – metric CSV files
"""
import os
import sys
import logging
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.topo_evidential_umamba import build_model
from src.losses.edl_loss          import EDLLoss, get_lambda_t
from src.losses.contrastive_loss  import PDSupervisedContrastiveLoss
from src.topology.persistence     import (
    compute_persistence_diagram, compute_topology_vector, sim_pd
)
from src.evaluation.metrics       import (
    compute_segmentation_metrics,
    expected_calibration_error,
    coverage_accuracy_curve,
    MetricAggregator,
)
from src.evaluation.clinical_metrics import (
    compute_cardiac_volumes, bland_altman_stats, ClinicalMetricAggregator
)
from src.visualization.plots import (
    plot_training_curves, plot_dice_violin, plot_uncertainty_map,
    plot_reliability_diagram, plot_coverage_accuracy,
    plot_domain_invariance, plot_ood_uncertainty_comparison,
    plot_persistence_diagram, plot_clinical_summary,
    plot_bland_altman,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "demo")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
EVAL_DIR   = os.path.join(OUTPUT_DIR, "eval")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generator
# ─────────────────────────────────────────────────────────────────────────────

def make_ellipse_mask(H, W, cx, cy, rx, ry, label):
    """Draw a filled ellipse on a label map."""
    mask = np.zeros((H, W), dtype=np.int64)
    for y in range(H):
        for x in range(W):
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                mask[y, x] = label
    return mask


def generate_cardiac_slice(H=128, W=128, noise_std=0.05):
    """
    Generate a synthetic cardiac MRI slice with 4 classes:
      0=BG, 1=RV, 2=Myocardium ring, 3=LV
    Returns: image (1,H,W) float32, label (H,W) int64
    """
    image = np.random.randn(H, W).astype(np.float32) * noise_std
    label = np.zeros((H, W), dtype=np.int64)

    # LV (bright, centre)
    lv_mask  = make_ellipse_mask(H, W, W//2, H//2, 20, 20, 3)
    # Myocardium (ring around LV)
    myo_mask = make_ellipse_mask(H, W, W//2, H//2, 30, 30, 2)
    myo_mask[lv_mask == 3] = 0
    # RV (to the left)
    rv_mask  = make_ellipse_mask(H, W, W//3, H//2, 18, 22, 1)
    rv_mask[lv_mask == 3] = 0
    rv_mask[myo_mask == 2] = 0

    # Set intensities
    image[lv_mask  == 3] += np.random.uniform(0.7, 0.9)
    image[myo_mask == 2] += np.random.uniform(0.4, 0.6)
    image[rv_mask  == 1] += np.random.uniform(0.5, 0.7)
    image = np.clip(image, 0, 1)

    # Compose label
    label[myo_mask == 2] = 2
    label[lv_mask  == 3] = 3
    label[rv_mask  == 1] = 1

    # Normalise image
    image = (image - image.mean()) / (image.std() + 1e-6)
    return image[np.newaxis].astype(np.float32), label


def generate_ood_slice(H=128, W=128):
    """Generate an out-of-distribution slice (no cardiac structure)."""
    image = np.random.randn(H, W).astype(np.float32) * 0.5
    # Add random blobs that look nothing like cardiac anatomy
    for _ in range(5):
        cx, cy = np.random.randint(20, H-20, 2)
        r = np.random.randint(5, 15)
        y, x = np.ogrid[:H, :W]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        image[mask] += np.random.uniform(-0.5, 0.5)
    image = (image - image.mean()) / (image.std() + 1e-6)
    label = np.zeros((H, W), dtype=np.int64)
    return image[np.newaxis].astype(np.float32), label


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=200, noise_std=0.05, H=128, W=128, seed=42):
        np.random.seed(seed)
        self.data = [generate_cardiac_slice(H, W, noise_std) for _ in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return {
            "image":      torch.tensor(image),
            "label":      torch.tensor(label),
            "topo_vec":   torch.tensor(compute_topology_vector(image), dtype=torch.float32),
            "patient_id": f"synth_{idx:04d}",
            "phase":      "ED" if idx % 2 == 0 else "ES",
            "slice_idx":  idx % 8,
            "spacing":    torch.tensor([1.5, 1.5]),
        }


class DomainShiftDataset(Dataset):
    """Simulated domain-shifted data (different noise / contrast)."""
    def __init__(self, n_samples=80, vendor="Siemens", H=128, W=128, seed=99):
        np.random.seed(seed)
        noise = {"Siemens": 0.05, "GE": 0.12, "Philips": 0.08, "Canon": 0.15}
        self.vendor = vendor
        self.data = [
            generate_cardiac_slice(H, W, noise.get(vendor, 0.10))
            for _ in range(n_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return {
            "image":      torch.tensor(image),
            "label":      torch.tensor(label),
            "topo_vec":   torch.tensor(compute_topology_vector(image), dtype=torch.float32),
            "patient_id": f"{self.vendor}_{idx:03d}",
            "phase":      "ED",
            "slice_idx":  idx % 8,
            "spacing":    torch.tensor([1.5, 1.5]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Minimal config for demo
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CFG = {
    "data":  {"in_channels": 1, "num_classes": 4, "class_names": ["BG","RV","Myo","LV"]},
    "model": {
        "feature_channels": [8, 16, 32, 64, 128],
        "ssm_d_state": 4,
        "ssm_expand_factor": 1,
        "projection_dim": 32,
        "dropout": 0.1,
    },
    "loss": {
        "gamma": 0.3,
        "temperature": 0.1,
        "lambda_t_schedule": {"start": 0.0, "end": 1.0, "warmup_epochs": 2},
    },
    "topology": {"topo_positive_threshold": 0.6},
    "training": {
        "epochs": 8,          # short demo – replace with 200 for full training
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "warmup_epochs": 1,
        "grad_clip": 1.0,
        "val_interval": 4,
    },
    "evaluation": {"ece_bins": 10},
}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (minimal, for demo)
# ─────────────────────────────────────────────────────────────────────────────

def demo_train(model, train_loader, device, cfg):
    import torch.nn as nn
    from torch.optim import AdamW

    edl_loss_fn = EDLLoss()
    con_loss_fn = PDSupervisedContrastiveLoss(
        temperature=cfg["loss"]["temperature"],
        positive_threshold=cfg["topology"]["topo_positive_threshold"],
    )
    optimizer = AdamW(model.parameters(),
                      lr=cfg["training"]["learning_rate"],
                      weight_decay=cfg["training"]["weight_decay"])

    epochs   = cfg["training"]["epochs"]
    gamma    = cfg["loss"]["gamma"]
    lam_sched = cfg["loss"]["lambda_t_schedule"]

    history = {"train_loss": [], "val_dice_mean": [None] * epochs, "lr": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        lambda_t = get_lambda_t(
            epoch, lam_sched["start"], lam_sched["end"], lam_sched["warmup_epochs"]
        )

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            topo_vec = batch["topo_vec"].to(device)

            out = model(images, return_projections=True)
            evidence    = out["evidence"]
            projections = out["projections"]

            l_edl, _ = edl_loss_fn(evidence, labels, lambda_t=lambda_t)
            l_con = con_loss_fn(projections, topo_vec)

            loss = l_edl + gamma * l_con
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item() * images.shape[0]
            n += images.shape[0]

        avg_loss = epoch_loss / n
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_loss)
        history["lr"].append(lr)

        logger.info(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  λ_t={lambda_t:.3f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Inference + metrics collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def demo_evaluate(model, loader, device, num_classes=4, tag=""):
    model.eval()
    metric_agg    = MetricAggregator(num_classes)
    dice_records  = []
    all_probs     = []
    all_labels    = []
    all_unc       = []
    all_correct   = []
    sample_tuples = []

    for batch in loader:
        images  = batch["image"].to(device)
        gt_segs = batch["label"].cpu().numpy()

        out      = model(images, return_projections=False)
        probs    = out["probs"].cpu().numpy()
        seg_maps = probs.argmax(axis=1)
        unc_maps = out["uncertainty"].cpu().numpy().squeeze(1)

        for i in range(len(images)):
            pid = batch["patient_id"][i]
            m   = compute_segmentation_metrics(seg_maps[i], gt_segs[i], num_classes)
            m["patient_id"] = pid
            metric_agg.update(m)
            dice_records.append({
                "dice_c1": m.get("dice_c1", 0),
                "dice_c2": m.get("dice_c2", 0),
                "dice_c3": m.get("dice_c3", 0),
            })

            # Calibration arrays (downsample for speed)
            step = 4
            flat_p = probs[i].transpose(1,2,0)[::step, ::step].reshape(-1, num_classes)
            flat_l = gt_segs[i][::step, ::step].reshape(-1)
            flat_u = unc_maps[i][::step, ::step].reshape(-1)
            flat_c = (seg_maps[i] == gt_segs[i])[::step, ::step].reshape(-1)
            all_probs.append(flat_p)
            all_labels.append(flat_l)
            all_unc.append(flat_u)
            all_correct.append(flat_c)

            if len(sample_tuples) < 4:
                img_np = images[i].cpu().numpy().squeeze()
                sample_tuples.append((img_np, seg_maps[i], gt_segs[i], unc_maps[i]))

    return {
        "summary":       metric_agg.summary(),
        "dice_records":  dice_records,
        "all_probs":     np.concatenate(all_probs),
        "all_labels":    np.concatenate(all_labels),
        "all_unc":       np.concatenate(all_unc),
        "all_correct":   np.concatenate(all_correct),
        "sample_tuples": sample_tuples,
        "metric_agg":    metric_agg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Clinical metrics on synthetic 3-D volumes
# ─────────────────────────────────────────────────────────────────────────────

def demo_clinical_metrics(model, device, n_patients=20, num_classes=4):
    """Generate synthetic 3-D volumes, run model, compute clinical indices."""
    spacing = (1.5, 1.5, 8.0)
    pred_list, gt_list = [], []

    for pid in range(n_patients):
        np.random.seed(pid)
        n_slices = 8

        ed_pred_slices, es_pred_slices = [], []
        ed_gt_slices,   es_gt_slices   = [], []
        slice_idxs = list(range(n_slices))

        for s in range(n_slices):
            for phase in ["ED", "ES"]:
                img, gt = generate_cardiac_slice(64, 64, noise_std=0.05)
                img_t   = torch.tensor(img).unsqueeze(0).to(device)
                out = model(img_t, return_projections=False)
                pred_seg = out["probs"].argmax(dim=1).squeeze().cpu().numpy()

                if phase == "ED":
                    ed_pred_slices.append(pred_seg)
                    ed_gt_slices.append(gt)
                else:
                    es_pred_slices.append(pred_seg)
                    es_gt_slices.append(gt)

        from src.evaluation.clinical_metrics import stack_slices_to_volume
        ed_pred_vol = stack_slices_to_volume(ed_pred_slices, slice_idxs)
        es_pred_vol = stack_slices_to_volume(es_pred_slices, slice_idxs)
        ed_gt_vol   = stack_slices_to_volume(ed_gt_slices,   slice_idxs)
        es_gt_vol   = stack_slices_to_volume(es_gt_slices,   slice_idxs)

        pred_m = compute_cardiac_volumes(ed_pred_vol, es_pred_vol, spacing)
        gt_m   = compute_cardiac_volumes(ed_gt_vol,   es_gt_vol,   spacing)
        pred_m["patient_id"] = f"synth_{pid:03d}"
        gt_m["patient_id"]   = f"synth_{pid:03d}"
        pred_list.append(pred_m)
        gt_list.append(gt_m)

    return pred_list, gt_list


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  Topo-Evidential U-Mamba — Synthetic Demo")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Build model ─────────────────────────────────────────────────────────
    model = build_model(DEMO_CFG)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = SyntheticDataset(n_samples=32,  noise_std=0.05, H=64, W=64, seed=42)
    val_ds   = SyntheticDataset(n_samples=16,  noise_std=0.05, H=64, W=64, seed=7)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

    # ── 1. Training ──────────────────────────────────────────────────────────
    logger.info("\n[1/7] Training (20 epochs on synthetic data)...")
    history = demo_train(model, train_dl, device, DEMO_CFG)
    plot_training_curves(history, os.path.join(FIG_DIR, "training_curves.png"))
    logger.info("✓ Training curves saved")

    # ── 2. In-distribution evaluation ───────────────────────────────────────
    logger.info("\n[2/7] Evaluating on synthetic test set...")
    model.eval()
    test_ds = SyntheticDataset(n_samples=20, noise_std=0.05, H=64, W=64, seed=123)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    test_res = demo_evaluate(model, test_dl, device, num_classes=4, tag="test")

    logger.info(f"  Test Dice mean: {test_res['summary'].get('dice_mean_mean', 0):.4f}")
    logger.info(f"  Test HD95 mean: {test_res['summary'].get('hd95_mean_mean', 0):.2f}")

    import pandas as pd
    pd.DataFrame(test_res["dice_records"]).to_csv(
        os.path.join(EVAL_DIR, "test_dice.csv"), index=False
    )
    plot_dice_violin(
        test_res["dice_records"],
        os.path.join(FIG_DIR, "test_dice_violin.png"),
        title="Synthetic Test — Dice per Class",
    )
    logger.info("✓ Dice violin plot saved")

    # ── 3. Uncertainty maps ──────────────────────────────────────────────────
    logger.info("\n[3/7] Saving uncertainty map examples...")
    for idx, (img, pred, gt, unc) in enumerate(test_res["sample_tuples"][:3]):
        plot_uncertainty_map(
            img, pred, gt, unc,
            os.path.join(FIG_DIR, f"uncertainty_sample_{idx}.png"),
            title=f"Sample {idx+1}: Prediction + Uncertainty",
        )
    logger.info("✓ Uncertainty maps saved")

    # ── 4. Calibration (ECE, reliability diagram) ────────────────────────────
    logger.info("\n[4/7] Computing calibration metrics...")
    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        test_res["all_probs"], test_res["all_labels"], n_bins=10,
    )
    logger.info(f"  ECE: {ece:.4f}")
    plot_reliability_diagram(
        bin_accs, bin_confs, bin_counts, ece,
        os.path.join(FIG_DIR, "reliability_diagram.png"),
    )

    covs, accs = coverage_accuracy_curve(
        test_res["all_unc"], test_res["all_correct"]
    )
    plot_coverage_accuracy(
        covs, accs,
        os.path.join(FIG_DIR, "coverage_accuracy_curve.png"),
        baseline_acc=test_res["all_correct"].mean(),
    )
    pd.DataFrame({"ece": [ece]}).to_csv(
        os.path.join(EVAL_DIR, "calibration.csv"), index=False
    )
    logger.info("✓ Calibration plots saved")

    # ── 5. Domain shift (vendor simulation) ──────────────────────────────────
    logger.info("\n[5/7] Evaluating domain-shift invariance...")
    vendors = ["Siemens", "GE", "Philips", "Canon"]
    vendor_dice_dict = {}
    vendor_rows = []

    for vendor in vendors:
        v_ds = DomainShiftDataset(n_samples=12, vendor=vendor, H=64, W=64, seed=77)
        v_dl = DataLoader(v_ds, batch_size=8, shuffle=False, num_workers=0)
        v_res = demo_evaluate(model, v_dl, device, num_classes=4)
        mean_dice = v_res["summary"].get("dice_mean_mean", 0.0)
        vendor_dice_dict[vendor] = [r["dice_c1"] for r in v_res["dice_records"]] + \
                                   [r["dice_c2"] for r in v_res["dice_records"]] + \
                                   [r["dice_c3"] for r in v_res["dice_records"]]
        vendor_rows.append({"vendor": vendor, "dice_mean": mean_dice})
        logger.info(f"  {vendor}: Dice = {mean_dice:.4f}")

    pd.DataFrame(vendor_rows).to_csv(
        os.path.join(EVAL_DIR, "domain_shift_dice.csv"), index=False
    )
    plot_domain_invariance(
        vendor_dice_dict,
        os.path.join(FIG_DIR, "domain_invariance.png"),
        title="Simulated Domain Shift — Dice per Scanner Vendor",
    )
    logger.info("✓ Domain invariance plot saved")

    # ── 6. OOD detection ─────────────────────────────────────────────────────
    logger.info("\n[6/7] OOD uncertainty analysis...")
    ood_slices = [generate_ood_slice(64, 64) for _ in range(20)]

    class OODDataset(torch.utils.data.Dataset):
        def __init__(self, slices):
            self.slices = slices
        def __len__(self): return len(self.slices)
        def __getitem__(self, idx):
            img, lbl = self.slices[idx]
            return {"image": torch.tensor(img), "label": torch.tensor(lbl)}

    ood_dl  = DataLoader(OODDataset(ood_slices), batch_size=8, shuffle=False)
    ood_unc_list = []
    with torch.no_grad():
        for batch in ood_dl:
            imgs = batch["image"].to(device)
            out  = model(imgs, return_projections=False)
            unc  = out["uncertainty"].cpu().numpy().squeeze(1)
            ood_unc_list.append(unc.reshape(-1)[::4])

    ood_unc    = np.concatenate(ood_unc_list)
    id_unc_sub = test_res["all_unc"][:len(ood_unc)]

    logger.info(f"  In-distribution uncertainty mean: {id_unc_sub.mean():.4f}")
    logger.info(f"  OOD uncertainty mean:             {ood_unc.mean():.4f}")

    plot_ood_uncertainty_comparison(
        in_dist_unc = id_unc_sub,
        ood_unc     = ood_unc,
        save_path   = os.path.join(FIG_DIR, "ood_uncertainty.png"),
    )
    pd.DataFrame({
        "id_unc_mean":  [float(id_unc_sub.mean())],
        "ood_unc_mean": [float(ood_unc.mean())],
    }).to_csv(os.path.join(EVAL_DIR, "ood_analysis.csv"), index=False)
    logger.info("✓ OOD uncertainty plot saved")

    # ── 7. Clinical metrics ───────────────────────────────────────────────────
    logger.info("\n[7/7] Computing clinical cardiac metrics...")
    pred_clinical, gt_clinical = demo_clinical_metrics(model, device, n_patients=6)

    plot_clinical_summary(
        pred_clinical, gt_clinical,
        os.path.join(FIG_DIR, "clinical_summary.png"),
    )
    pd.DataFrame(pred_clinical).to_csv(os.path.join(EVAL_DIR, "clinical_pred.csv"), index=False)
    pd.DataFrame(gt_clinical).to_csv(  os.path.join(EVAL_DIR, "clinical_gt.csv"),   index=False)

    # Bland-Altman per metric
    import pandas as pd2
    pred_df = pd.DataFrame(pred_clinical).set_index("patient_id")
    gt_df   = pd.DataFrame(gt_clinical).set_index("patient_id")
    common  = pred_df.index.intersection(gt_df.index)
    ba_rows = []
    for metric, unit in [("LV_EF_pct","%"),("LV_EDV_mL","mL"),
                         ("LV_ESV_mL","mL"),("Myo_mass_g","g")]:
        if metric not in pred_df.columns:
            continue
        p = pred_df.loc[common, metric].values
        g = gt_df.loc[common,   metric].values
        ba = bland_altman_stats(p, g)
        ba["metric"] = metric
        ba_rows.append(ba)
        logger.info(f"  {metric}: bias={ba['bias']:+.2f}, "
                    f"LoA=[{ba['loa_lower']:+.2f}, {ba['loa_upper']:+.2f}], "
                    f"r={ba['pearson_r']:.3f}")
        plot_bland_altman(p, g, metric, unit,
                          os.path.join(FIG_DIR, f"bland_altman_{metric}.png"))

    pd.DataFrame(ba_rows).to_csv(
        os.path.join(EVAL_DIR, "bland_altman.csv"), index=False
    )

    # ── Persistence diagram example ──────────────────────────────────────────
    sample_img = train_ds[0]["image"].numpy().squeeze()
    pds = compute_persistence_diagram(sample_img, max_dim=1)
    plot_persistence_diagram(
        pds,
        os.path.join(FIG_DIR, "persistence_diagram.png"),
        title="Persistence Diagram (Synthetic Cardiac Slice)",
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  DEMO COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Figures → {FIG_DIR}")
    logger.info(f"  Metrics → {EVAL_DIR}")
    logger.info(f"  Files generated:")
    for f in sorted(os.listdir(FIG_DIR)):
        logger.info(f"    figures/{f}")
    for f in sorted(os.listdir(EVAL_DIR)):
        logger.info(f"    eval/{f}")


if __name__ == "__main__":
    main()
