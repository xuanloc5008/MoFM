"""
visualization/plots.py
All plots described in the paper:
  - Training curves
  - Segmentation overlays
  - Uncertainty maps (Vacuity, Dissonance)
  - Persistence diagrams
  - Betti violation distribution
  - Clinical index scatter (EF predicted vs reference)
  - Uncertainty propagation (σ(EF) vs |ΔEF|)
  - TTEC confusion matrix
  - Calibration curves (reliability diagram)
  - TU distribution (OOD detection)
  - Ablation bar charts
  - σ convergence (Phase 3)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# ─── colour palette consistent with IEEE TMI style ──────────────────────────
PALETTE = {
    "ours":   "#2563EB",    # blue
    "ablation": "#DC2626",  # red
    "baseline": "#6B7280",  # grey
    "mms":    "#059669",    # green
    "mms2":   "#D97706",    # amber
    "type1":  "#EF4444",
    "type2":  "#F59E0B",
    "type3":  "#10B981",
}

STRUCT_COLORS = {"LV": "#EF4444", "Myo": "#3B82F6", "RV": "#22C55E"}


def save_fig(fig, path: str, dpi: int = 150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {Path(path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict,      # {"train": [...], "val": [...]}
    seeds_histories: Optional[List[Dict]] = None,
    out_path: str = "outputs/training_curves.png",
):
    """Mean ± std over seeds, with phase boundaries."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics_to_plot = ["dice_avg", "hd95_avg", "ter"]
    titles = ["Dice (avg)", "HD95 (mm)", "TER (%)"]

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        if seeds_histories:
            vals = np.array([[e.get(metric, 0)
                              for e in h.get("val", [])]
                             for h in seeds_histories])
            x = np.arange(vals.shape[1])
            mu, std = vals.mean(0), vals.std(0)
            ax.plot(x, mu, color=PALETTE["ours"], lw=2)
            ax.fill_between(x, mu - std, mu + std,
                            alpha=0.25, color=PALETTE["ours"])
        # Phase boundaries
        p1 = 50; p2 = 100
        ax.axvline(p1, color="grey", ls="--", lw=0.8, alpha=0.6,
                   label="Phase 1→2")
        ax.axvline(p2, color="grey", ls=":",  lw=0.8, alpha=0.6,
                   label="Phase 2→3")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Training Curves (5 seeds)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Segmentation Overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_segmentation_overlay(
    image: np.ndarray,       # (H, W)  single slice
    label: np.ndarray,       # (H, W)  ground truth
    pred:  np.ndarray,       # (H, W)  prediction
    vacuity:  np.ndarray,    # (H, W)
    dissonance: np.ndarray,  # (H, W)
    out_path: str = "outputs/segmentation_overlay.png",
):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    cmap_seg = plt.cm.get_cmap("tab10", 4)

    axes[0].imshow(image, cmap="gray"); axes[0].set_title("Input MRI")
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(label, cmap=cmap_seg, alpha=0.5, vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(pred,  cmap=cmap_seg, alpha=0.5, vmin=0, vmax=3)
    axes[2].set_title("Prediction")
    im3 = axes[3].imshow(vacuity,    cmap="hot_r", vmin=0, vmax=1)
    axes[3].set_title("Vacuity (epistemic)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    im4 = axes[4].imshow(dissonance, cmap="YlOrRd", vmin=0, vmax=0.5)
    axes[4].set_title("Dissonance (aleatoric)")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    # Legend
    patches = [mpatches.Patch(color=cmap_seg(i), label=n)
               for i, n in enumerate(["BG", "LV", "Myo", "RV"])]
    axes[2].legend(handles=patches, loc="lower right", fontsize=7)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Persistence Diagram Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_persistence_diagram(
    pd_points: List[Tuple[float, float]],   # list of (birth, death) for dim 0
    pd_points_1: List[Tuple[float, float]], # for dim 1
    title: str = "Persistence Diagram",
    out_path: str = "outputs/persistence_diagram.png",
):
    fig, ax = plt.subplots(figsize=(5, 5))
    if pd_points:
        b0, d0 = zip(*pd_points)
        ax.scatter(b0, d0, c=PALETTE["ours"], s=30, label="H₀", zorder=3)
    if pd_points_1:
        b1, d1 = zip(*pd_points_1)
        ax.scatter(b1, d1, c=PALETTE["ablation"], s=30,
                   marker="^", label="H₁", zorder=3)
    lim = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1.0
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Diagonal")
    ax.set_xlabel("Birth"); ax.set_ylabel("Death")
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Clinical Index Scatter: EF predicted vs reference
# ─────────────────────────────────────────────────────────────────────────────

def plot_ef_scatter(
    ef_pred: np.ndarray,     # predicted EF (%)
    ef_ref:  np.ndarray,     # CMR reference EF (%)
    sigma_ef: np.ndarray,    # σ(EF) per sample
    method_name: str = "Ours (TTEC)",
    out_path: str = "outputs/ef_scatter.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bland-Altman
    mean  = (ef_pred + ef_ref) / 2
    diff  = ef_pred - ef_ref
    bias  = diff.mean()
    loa   = 1.96 * diff.std()

    axes[0].scatter(mean, diff, alpha=0.6, color=PALETTE["ours"], s=25,
                    label=method_name)
    axes[0].axhline(bias,     color="red",   ls="--", lw=1.5,
                    label=f"Bias={bias:.1f}%")
    axes[0].axhline(bias+loa, color="orange",ls=":",  lw=1.0,
                    label=f"±1.96σ={loa:.1f}%")
    axes[0].axhline(bias-loa, color="orange",ls=":",  lw=1.0)
    axes[0].axhline(5.0,  color="grey", ls=":", alpha=0.5)
    axes[0].axhline(-5.0, color="grey", ls=":", alpha=0.5,
                    label="5% inter-obs. threshold")
    axes[0].set_xlabel("Mean EF (%)")
    axes[0].set_ylabel("EF Pred - EF Ref (%)")
    axes[0].set_title("Bland-Altman: EF")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Scatter + σ error bars
    axes[1].errorbar(ef_ref, ef_pred, yerr=1.96*sigma_ef,
                     fmt="o", color=PALETTE["ours"], alpha=0.5,
                     ecolor="lightblue", elinewidth=0.8, capsize=2,
                     ms=4, label=f"{method_name} (95% CI)")
    lo, hi = ef_ref.min(), ef_ref.max()
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Identity")
    axes[1].set_xlabel("CMR Reference EF (%)")
    axes[1].set_ylabel("Predicted EF (%)")
    axes[1].set_title("EF: Predicted vs Reference")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Pearson r
    from scipy.stats import pearsonr
    r, p = pearsonr(ef_ref, ef_pred)
    axes[1].text(0.05, 0.92, f"r={r:.3f}, p={p:.3e}",
                 transform=axes[1].transAxes, fontsize=9)

    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Uncertainty Propagation: σ(EF) vs |ΔEF|
# ─────────────────────────────────────────────────────────────────────────────

def plot_ef_uncertainty(
    sigma_analytic: np.ndarray,
    sigma_mc:       np.ndarray,
    delta_ef:       np.ndarray,
    out_path: str = "outputs/ef_uncertainty.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Analytic vs MC agreement
    axes[0].scatter(sigma_mc, sigma_analytic, alpha=0.6,
                    color=PALETTE["ours"], s=25)
    lo = min(sigma_mc.min(), sigma_analytic.min())
    hi = max(sigma_mc.max(), sigma_analytic.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Identity")
    from scipy.stats import pearsonr
    r, _ = pearsonr(sigma_mc, sigma_analytic)
    axes[0].set_xlabel("σ(EF) MC (%)")
    axes[0].set_ylabel("σ(EF) Analytic (%)")
    axes[0].set_title(f"Analytic vs MC Propagation  (r={r:.3f})")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # σ(EF) vs |ΔEF|
    axes[1].scatter(sigma_analytic, delta_ef, alpha=0.6,
                    color=PALETTE["ours"], s=25)
    axes[1].set_xlabel("σ(EF) Analytic (%)")
    axes[1].set_ylabel("|ΔEF| (vs CMR reference, %)")
    axes[1].set_title("Uncertainty vs Actual Error")
    axes[1].grid(alpha=0.3)

    from scipy.stats import spearmanr
    rho, p = spearmanr(sigma_analytic, delta_ef)
    axes[1].text(0.05, 0.92, f"Spearman ρ={rho:.3f}",
                 transform=axes[1].transAxes, fontsize=9)

    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TU Distribution (OOD Detection)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tu_distribution(
    tu_train: np.ndarray,
    tu_test_in: np.ndarray,
    tu_test_ood: Dict[str, np.ndarray],   # {name: scores}
    out_path: str = "outputs/tu_distribution.png",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(tu_train,   ax=ax, label="Train (ID)", color="blue",
                fill=True, alpha=0.3)
    sns.kdeplot(tu_test_in, ax=ax, label="Test (ID)",  color="green",
                fill=True, alpha=0.3)
    colors_ood = ["red", "orange", "purple", "brown"]
    for (name, scores), col in zip(tu_test_ood.items(), colors_ood):
        if len(scores) > 2:
            sns.kdeplot(scores, ax=ax, label=f"OOD: {name}",
                        color=col, fill=False, lw=2)
    ax.set_xlabel("Topological Uncertainty (TU)")
    ax.set_ylabel("Density")
    ax.set_title("TU Distribution: In-Distribution vs OOD")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. TTEC Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_ttec_confusion_matrix(
    cm: np.ndarray,       # (3, 3)
    kappa: float,
    gttec_acc: float,
    hard_acc:  float,
    out_path: str = "outputs/ttec_confusion.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Type-I", "Type-II", "Type-III"]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title(
        f"TTEC Confusion Matrix\n"
        f"G_TTEC acc={gttec_acc:.1%}  Hard acc={hard_acc:.1%}  κ={kappa:.3f}"
    )

    # Per-class F1 comparison: G_TTEC vs hard thresholds
    # (placeholder data structure)
    axes[1].set_title("G_TTEC vs Hard Thresholds per Class")
    axes[1].text(0.5, 0.5,
                 "Per-class F1 bar chart\n(fill with actual results)",
                 ha="center", va="center", fontsize=11,
                 transform=axes[1].transAxes)
    axes[1].axis("off")

    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Calibration Reliability Diagram
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(
    probs_dict: Dict[str, np.ndarray],   # {method: (N, K) probs}
    labels: np.ndarray,
    struct_id: int = 2,
    struct_name: str = "Myo",
    out_path: str = "outputs/calibration.png",
):
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

    colors = [PALETTE["ours"], PALETTE["ablation"], PALETTE["baseline"]]
    y_bin = (labels == struct_id).astype(int)

    for (method, probs), col in zip(probs_dict.items(), colors):
        p_k = probs[:, struct_id]
        try:
            frac_pos, mean_pred = calibration_curve(y_bin, p_k, n_bins=10)
            ax.plot(mean_pred, frac_pos, "o-", color=col,
                    lw=2, ms=5, label=method)
        except Exception:
            pass

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration: {struct_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Ablation Study Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(
    results: Dict[str, Dict],   # {variant: {metric: val}}
    metric: str = "dice_Myo",
    out_path: str = "outputs/ablation.png",
):
    fig, ax = plt.subplots(figsize=(10, 4))
    variants = list(results.keys())
    vals = [results[v].get(metric, 0.0) for v in variants]
    errs = [results[v].get(f"{metric}_std", 0.0) for v in variants]

    colors = [PALETTE["ours"] if "Full" in v else
              PALETTE["ablation"] if "w/o" in v else
              PALETTE["baseline"]
              for v in variants]

    bars = ax.barh(variants, vals, xerr=errs, color=colors, alpha=0.85,
                   height=0.6, capsize=4)
    ax.axvline(vals[0], color="blue", ls="--", lw=1.0, alpha=0.5)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Ablation Study: {metric}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 10. σ Convergence (Phase 3 Homoscedastic)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sigma_convergence(
    sigma_history: np.ndarray,   # (n_epochs, M, 4)
    out_path: str = "outputs/sigma_convergence.png",
):
    n_epochs, M, n_loss = sigma_history.shape
    loss_names  = ["EDL", "Topo3D", "TCL", "Seg"]
    struct_names = ["LV", "Myo", "RV"]
    fig, axes = plt.subplots(M, n_loss, figsize=(16, 9))
    x = np.arange(n_epochs)

    for m in range(M):
        for l in range(n_loss):
            ax = axes[m][l]
            ax.plot(x, sigma_history[:, m, l],
                    color=list(STRUCT_COLORS.values())[m], lw=2)
            ax.set_title(f"{struct_names[m]} — σ({loss_names[l]})",
                         fontsize=9)
            ax.set_xlabel("Phase 3 Epoch", fontsize=8)
            ax.grid(alpha=0.3)

    fig.suptitle("Homoscedastic σ Convergence (Phase 3)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Statistical Summary Table (for paper)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_table(
    results: Dict[str, Dict],   # {method: {metric: val ± std}}
    metrics: List[str],
    out_path: str = "outputs/results_table.png",
):
    """Render results table as matplotlib figure for paper-quality output."""
    methods = list(results.keys())
    rows = []
    for m in methods:
        row = [results[m].get(met, 0.0) for met in metrics]
        rows.append(row)

    data = np.array(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(len(metrics) * 2 + 2, len(methods) * 0.5 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=[[f"{v:.3f}" for v in row] for row in rows],
        rowLabels=methods,
        colLabels=[m.replace("_", "\n") for m in metrics],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)

    # Highlight best per column
    for j in range(len(metrics)):
        best_row = int(np.argmax(data[:, j]) if "dice" in metrics[j]
                       else np.argmin(data[:, j]))
        tbl[best_row + 1, j].set_facecolor("#DBEAFE")
        tbl[best_row + 1, j].set_text_props(fontweight="bold")

    plt.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Clinical Statistics Summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_clinical_statistics(
    clinical_results: Dict[str, List[float]],
    # {"method": list of |ΔEF| values}
    out_path: str = "outputs/clinical_statistics.png",
):
    """Box plots of |ΔEF| and |ΔMass| per method."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df_ef = {k: v for k, v in clinical_results.items() if "ef" in k.lower()}
    df_mass = {k: v for k, v in clinical_results.items() if "mass" in k.lower()}

    for ax, df, title, ylabel in [
        (axes[0], df_ef,   "|ΔEF| per Method",   "|ΔEF| (%)"),
        (axes[1], df_mass, "|ΔMass| per Method", "|ΔMass| (g)"),
    ]:
        if not df:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        methods = list(df.keys())
        data    = [df[m] for m in methods]
        bp = ax.boxplot(data, labels=methods, patch_artist=True,
                        medianprops={"color": "black", "lw": 2})
        colors = list(PALETTE.values())[:len(methods)]
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        ax.axhline(5.0, color="red", ls="--", lw=1.0, alpha=0.6,
                   label="5% inter-observer threshold")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_fig(fig, out_path)
