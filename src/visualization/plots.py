"""
Visualization Module
---------------------
All plots for the paper / report:
  1. Training / validation curves (loss, Dice, LR)
  2. Dice violin plots per class
  3. Uncertainty maps with segmentation overlay
  4. Reliability diagram (ECE)
  5. Coverage-Accuracy curve
  6. Bland-Altman plots for clinical metrics (EDV, ESV, EF)
  7. Persistence Diagram visualisation
  8. Domain invariance: per-vendor Dice comparison
  9. OOD uncertainty analysis
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Style
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi":     150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

CLASS_NAMES  = ["BG", "RV", "Myocardium", "LV"]
CLASS_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]   # for RV, Myo, LV (skip BG)
LABEL_COLORS = np.array([
    [0,   0,   0],    # BG - black
    [255, 100, 100],  # RV - red
    [100, 255, 100],  # Myo - green
    [100, 100, 255],  # LV - blue
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history:   Dict[str, List],
    save_path: str,
):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#E53935", label="Train Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss (L_Total)")
    ax.legend(); ax.grid(alpha=0.3)

    # Validation Dice (skip None values)
    ax = axes[1]
    val_epochs = [e for e, v in zip(epochs, history.get("val_dice_mean", [])) if v is not None]
    val_dice   = [v for v in history.get("val_dice_mean", []) if v is not None]
    if val_dice:
        ax.plot(val_epochs, val_dice, color="#1976D2", marker="o", markersize=3,
                label="Val Dice Mean")
        best_ep  = val_epochs[int(np.argmax(val_dice))]
        best_val = max(val_dice)
        ax.axvline(best_ep, color="red", linestyle="--", alpha=0.6,
                   label=f"Best: {best_val:.4f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice")
    ax.set_title("Validation Dice")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    # Learning Rate
    ax = axes[2]
    ax.semilogy(epochs, history.get("lr", [1e-4] * len(epochs)), color="#388E3C")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR (log scale)")
    ax.set_title("Learning Rate Schedule")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dice Violin Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_dice_violin(
    dice_records: List[Dict],   # list of {dice_c1, dice_c2, dice_c3, ...}
    save_path:    str,
    title:        str = "Dice Score Distribution per Class",
):
    df = pd.DataFrame(dice_records)
    cols = [c for c in ["dice_c1", "dice_c2", "dice_c3"] if c in df.columns]
    long = df[cols].melt(var_name="Class", value_name="Dice")
    long["Class"] = long["Class"].map(
        {"dice_c1": "RV", "dice_c2": "Myocardium", "dice_c3": "LV"}
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=long, x="Class", y="Dice",
        palette=CLASS_COLORS, inner="box", cut=0, ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("Dice Similarity Coefficient")
    ax.axhline(0.9, color="grey", linestyle="--", alpha=0.5, label="DSC=0.9")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate medians
    medians = long.groupby("Class")["Dice"].median()
    for i, cls in enumerate(["RV", "Myocardium", "LV"]):
        if cls in medians:
            ax.text(i, medians[cls] + 0.02, f"{medians[cls]:.3f}",
                    ha="center", fontsize=9, color="black", fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Uncertainty Map
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_map(
    image:       np.ndarray,      # (H, W) raw intensity
    pred_seg:    np.ndarray,      # (H, W) integer class map
    gt_seg:      np.ndarray,      # (H, W) integer class map
    uncertainty: np.ndarray,      # (H, W) uncertainty in [0, 1]
    save_path:   str,
    title:       str = "Uncertainty Analysis",
):
    def seg_to_rgb(seg):
        h, w = seg.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(4):
            rgb[seg == c] = LABEL_COLORS[c]
        return rgb

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")

    axes[1].imshow(seg_to_rgb(gt_seg))
    axes[1].set_title("Ground Truth")

    axes[2].imshow(seg_to_rgb(pred_seg))
    axes[2].set_title("Prediction")

    im = axes[3].imshow(uncertainty, cmap="hot", vmin=0, vmax=1)
    axes[3].set_title("Uncertainty Map u_p")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    # Add legend
    patches = [
        mpatches.Patch(color=LABEL_COLORS[i]/255, label=CLASS_NAMES[i])
        for i in range(1, 4)
    ]
    axes[1].legend(handles=patches, loc="lower right", fontsize=7)
    axes[2].legend(handles=patches, loc="lower right", fontsize=7)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reliability Diagram (ECE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    bin_accs:   np.ndarray,
    bin_confs:  np.ndarray,
    bin_counts: np.ndarray,
    ece:        float,
    save_path:  str,
):
    n_bins = len(bin_accs)
    x = np.arange(n_bins)
    width = 0.8

    fig, ax = plt.subplots(figsize=(7, 5))

    # Bars: accuracy per bin
    ax.bar(x, bin_accs,  width, color="#1976D2", alpha=0.8, label="Accuracy")
    ax.bar(x, bin_confs, width, color="#E53935", alpha=0.3, label="Confidence")

    # Perfect calibration line
    ax.plot([0, n_bins - 1], [0, 1], "--", color="black", alpha=0.5, label="Perfect")

    ax.set_xlabel("Confidence Bin")
    ax.set_ylabel("Accuracy / Confidence")
    ax.set_title(f"Reliability Diagram  (ECE = {ece:.4f})")
    ax.set_xticks(x[::3])
    ax.set_xticklabels([f"{i/n_bins:.2f}" for i in range(n_bins)][::3])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Coverage-Accuracy Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage_accuracy(
    coverages:   np.ndarray,
    accuracies:  np.ndarray,
    save_path:   str,
    baseline_acc: float = None,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages * 100, accuracies * 100, color="#1976D2", linewidth=2.0,
            label="Model (uncertainty-filtered)")
    if baseline_acc is not None:
        ax.axhline(baseline_acc * 100, color="grey", linestyle="--",
                   label=f"Baseline accuracy ({baseline_acc*100:.1f}%)")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Coverage-Accuracy Curve\n(reject high-uncertainty pixels → accuracy↑)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bland-Altman Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_bland_altman(
    pred_vals:  np.ndarray,
    gt_vals:    np.ndarray,
    metric_name: str,
    units:       str,
    save_path:  str,
):
    diff = pred_vals - gt_vals
    mean = (pred_vals + gt_vals) / 2.0
    bias = np.mean(diff)
    loa  = 1.96 * np.std(diff)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean, diff, alpha=0.7, color="#1976D2", s=30, label="Patient")
    ax.axhline(bias,        color="#E53935", linewidth=1.5, label=f"Bias: {bias:+.2f}")
    ax.axhline(bias + loa,  color="#FB8C00", linestyle="--", linewidth=1.5,
               label=f"+1.96 SD: {bias+loa:+.2f}")
    ax.axhline(bias - loa,  color="#FB8C00", linestyle="--", linewidth=1.5,
               label=f"−1.96 SD: {bias-loa:+.2f}")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(f"Mean ({units})")
    ax.set_ylabel(f"Predicted − GT ({units})")
    ax.set_title(f"Bland-Altman: {metric_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Persistence Diagram
# ─────────────────────────────────────────────────────────────────────────────

def plot_persistence_diagram(
    diagrams:   List,   # [PD_H0, PD_H1]  list of (birth, death) tuples
    save_path:  str,
    title:      str = "Persistence Diagram",
):
    colors = ["#1976D2", "#E53935", "#388E3C"]
    fig, ax = plt.subplots(figsize=(6, 6))

    max_val = 0.0
    for dim, pd in enumerate(diagrams[:2]):
        if len(pd) == 0:
            continue
        pts = np.array(pd)
        ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.7,
                   color=colors[dim], label=f"H{dim}", zorder=3)
        max_val = max(max_val, pts.max())

    max_val = max(max_val, 1.0)
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=0.8, label="Diagonal")
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Per-vendor Dice (Domain Invariance)
# ─────────────────────────────────────────────────────────────────────────────

def plot_domain_invariance(
    vendor_dice: Dict[str, List[float]],   # {vendor_name: [dice scores]}
    save_path:   str,
    title:       str = "Domain Invariance: Dice per Scanner Vendor",
):
    vendors = sorted(vendor_dice.keys())
    data = [vendor_dice[v] for v in vendors]

    fig, ax = plt.subplots(figsize=(max(6, len(vendors) * 1.5), 5))
    bp = ax.boxplot(data, labels=vendors, patch_artist=True, notch=False)

    palette = sns.color_palette("tab10", len(vendors))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, scores in enumerate(data):
        x = np.random.normal(i + 1, 0.07, size=len(scores))
        ax.scatter(x, scores, alpha=0.5, s=15, color="grey", zorder=3)

    ax.axhline(0.85, color="red", linestyle="--", alpha=0.5, label="DSC = 0.85")
    ax.set_ylabel("Mean Dice")
    ax.set_xlabel("Scanner Vendor")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. OOD Uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def plot_ood_uncertainty_comparison(
    in_dist_unc:  np.ndarray,   # uncertainty values for in-distribution samples
    ood_unc:      np.ndarray,   # uncertainty values for OOD samples
    save_path:    str,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(in_dist_unc, bins=50, alpha=0.7, color="#1976D2",
            label="In-Distribution", density=True)
    ax.hist(ood_unc, bins=50, alpha=0.7, color="#E53935",
            label="Out-of-Distribution", density=True)

    ax.axvline(np.mean(in_dist_unc),  color="#1976D2", linestyle="--",
               label=f"ID mean: {np.mean(in_dist_unc):.3f}")
    ax.axvline(np.mean(ood_unc), color="#E53935",  linestyle="--",
               label=f"OOD mean: {np.mean(ood_unc):.3f}")

    ax.set_xlabel("Pixel Uncertainty u_p = K/S_p")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution: In-Distribution vs OOD")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Master summary figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_clinical_summary(
    pred_metrics: List[Dict],
    gt_metrics:   List[Dict],
    save_path:    str,
):
    """
    4-panel figure: Scatter + Bland-Altman for LV_EF and LV_EDV.
    """
    pred_df = pd.DataFrame(pred_metrics).set_index("patient_id")
    gt_df   = pd.DataFrame(gt_metrics).set_index("patient_id")
    common  = pred_df.index.intersection(gt_df.index)
    pred_df = pred_df.loc[common]
    gt_df   = gt_df.loc[common]

    metrics_to_plot = [
        ("LV_EF_pct",  "%",   "LV Ejection Fraction"),
        ("LV_EDV_mL",  "mL",  "LV End-Diastolic Volume"),
        ("LV_ESV_mL",  "mL",  "LV End-Systolic Volume"),
        ("Myo_mass_g", "g",   "Myocardial Mass"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for col_idx, (metric, unit, label) in enumerate(metrics_to_plot):
        if metric not in pred_df.columns or metric not in gt_df.columns:
            continue
        p_vals = pred_df[metric].values
        g_vals = gt_df[metric].values

        # Scatter
        ax = axes[0, col_idx]
        ax.scatter(g_vals, p_vals, alpha=0.7, color="#1976D2", s=30)
        lim = [min(g_vals.min(), p_vals.min()) * 0.9,
               max(g_vals.max(), p_vals.max()) * 1.1]
        ax.plot(lim, lim, "k--", alpha=0.4)
        r = np.corrcoef(g_vals, p_vals)[0, 1]
        ax.set_xlabel(f"GT {label} ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(f"{label}\nr = {r:.3f}")
        ax.grid(alpha=0.3)

        # Bland-Altman
        ax = axes[1, col_idx]
        diff = p_vals - g_vals
        mean = (p_vals + g_vals) / 2
        bias = np.mean(diff)
        loa  = 1.96 * np.std(diff)
        ax.scatter(mean, diff, alpha=0.7, color="#E53935", s=30)
        ax.axhline(bias,       color="#E53935", linewidth=1.5)
        ax.axhline(bias + loa, color="#FB8C00", linestyle="--")
        ax.axhline(bias - loa, color="#FB8C00", linestyle="--")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(f"Mean ({unit})")
        ax.set_ylabel(f"Pred − GT ({unit})")
        ax.set_title(f"B-A: bias={bias:+.2f}")
        ax.grid(alpha=0.3)

    fig.suptitle("Clinical Metrics: Prediction vs Ground Truth", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
