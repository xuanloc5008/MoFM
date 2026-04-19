"""
analyze_ece.py
==============
Tính ECE (Expected Calibration Error) cho M&Ms-1.
Chạy inference trực tiếp từ checkpoint + dataset,
không cần file metrics CSV có sẵn.

Cách chạy:
    python analyze_ece.py

Output:
    outputs/eval/mnm1_ece.csv
    outputs/figures/mnm1_ece_full.png
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH — chỉnh các đường dẫn này
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path("/Users/xuanloc/Documents/topo_evidential_umamba")
CHECKPOINT    = PROJECT_ROOT / "outputs/checkpoints/best_model.pth"
MNM1_ROOT     = Path("/Volumes/Transcend/M&M1")
ACDC_ROOT     = Path("/Volumes/Transcend/ACDC-dataset/database")   # để so sánh (tuỳ chọn)

OUT_FIG  = PROJECT_ROOT / "outputs/figures";  OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_EVAL = PROJECT_ROOT / "outputs/eval";     OUT_EVAL.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4
BATCH_SIZE  = 8
ECE_BINS    = 15
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from torch.utils.data import DataLoader
from src.models.topo_evidential_umamba import build_model
from src.data.mnm        import collect_mnm1_slices, DomainShiftDataset
from src.data.acdc       import collect_acdc_slices, SliceDataset
from src.data.transforms import get_val_transforms


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load model
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    with open(PROJECT_ROOT / "configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg)
    ckpt  = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()
    print(f"Model loaded  — epoch {ckpt.get('epoch','?')}  "
          f"best_dice={ckpt.get('best_dice',0):.4f}")
    return model, cfg


def find_mnm1_metadata_csv(root: Path) -> Path | None:
    """Discover the visible root-level metadata CSV instead of hardcoding a filename."""
    csvs = sorted(path for path in root.glob("*.csv") if not path.name.startswith("."))
    return csvs[0] if csvs else None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Collect per-pixel probabilities
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_probs(model, dataloader, desc=""):
    """
    Chạy inference và thu thập:
      all_probs    : (N, K)   — flattened per-pixel probabilities
      all_labels   : (N,)     — flattened ground-truth labels
      all_unc      : (N,)     — per-pixel uncertainty u_p = K/S_p
      all_correct  : (N,)     — bool, pixel-level correctness
      all_conf_max : (N,)     — max confidence per pixel
    """
    all_probs, all_labels, all_unc, all_correct = [], [], [], []
    n_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        imgs = batch["image"].to(DEVICE)
        gts  = batch["label"].cpu().numpy()          # B H W

        out      = model(imgs, return_projections=False)
        probs    = out["probs"].cpu().numpy()         # B K H W
        seg_maps = probs.argmax(axis=1)               # B H W
        unc_maps = out["uncertainty"].cpu().numpy().squeeze(1)  # B H W

        for i in range(len(imgs)):
            # Downsample 4x per axis for speed (optional, remove if RAM allows)
            step = 2
            fp = probs[i].transpose(1,2,0)[::step,::step].reshape(-1, NUM_CLASSES)
            fl = gts[i][::step,::step].reshape(-1)
            fu = unc_maps[i][::step,::step].reshape(-1)
            fc = (seg_maps[i] == gts[i])[::step,::step].reshape(-1)

            all_probs.append(fp)
            all_labels.append(fl)
            all_unc.append(fu)
            all_correct.append(fc)

        if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
            print(f"  {desc}  [{batch_idx+1}/{n_batches}]", end="\r")

    print()
    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_unc),
        np.concatenate(all_correct).astype(bool),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: ECE computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ece(probs, labels, n_bins=15):
    """
    ECE = Σ_b (|B_b|/N) * |acc(B_b) - conf(B_b)|

    Returns:
        ece         : scalar
        bin_accs    : (n_bins,)
        bin_confs   : (n_bins,)
        bin_counts  : (n_bins,)
        bin_edges   : (n_bins+1,)
    """
    confidences = probs.max(axis=1)        # max probability per pixel
    predictions = probs.argmax(axis=1)     # predicted class

    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bin_accs   = np.zeros(n_bins)
    bin_confs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_accs[i]   = (predictions[mask] == labels[mask]).mean()
        bin_confs[i]  = confidences[mask].mean()
        bin_counts[i] = mask.sum()

    ece = float((np.abs(bin_accs - bin_confs) * bin_counts).sum() / bin_counts.sum())
    return ece, bin_accs, bin_confs, bin_counts, bin_edges


def compute_ece_per_class(probs, labels, n_bins=15):
    """ECE for each class separately (one-vs-rest)."""
    results = {}
    for c in range(1, NUM_CLASSES):           # skip background
        prob_c  = probs[:, c]                 # P(class=c)
        label_c = (labels == c).astype(int)   # binary: is this pixel class c?
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs  = np.zeros(n_bins)
        bin_confs = np.zeros(n_bins)
        bin_counts= np.zeros(n_bins)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i+1]
            mask = (prob_c >= lo) & (prob_c < hi)
            if mask.sum() == 0: continue
            bin_accs[i]   = label_c[mask].mean()
            bin_confs[i]  = prob_c[mask].mean()
            bin_counts[i] = mask.sum()
        ece_c = float((np.abs(bin_accs - bin_confs) * bin_counts).sum() / bin_counts.sum())
        results[c] = {"ece": ece_c, "bin_accs": bin_accs, "bin_confs": bin_confs,
                      "bin_counts": bin_counts}
    return results


def compute_aurrc(unc, correct):
    """Area Under Risk-Rejection Curve (higher = uncertainty correlates with errors)."""
    from sklearn.metrics import roc_auc_score
    errors = (~correct).astype(int)
    try:
        return float(roc_auc_score(errors, unc))
    except Exception:
        return 0.5


def coverage_accuracy_curve(unc, correct, n_steps=100):
    idx = np.argsort(unc)     # sort by uncertainty ascending
    covs, accs = [], []
    N = len(unc)
    for frac in np.linspace(0.05, 1.0, n_steps):
        n_keep = max(1, int(frac * N))
        kept   = idx[:n_keep]
        covs.append(frac)
        accs.append(correct[kept].mean())
    return np.array(covs), np.array(accs)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Plotting
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES  = {1: "RV", 2: "Myo", 3: "LV"}
STRUCT_COLOR = {1: "#2196F3", 2: "#4CAF50", 3: "#FF5722"}

def plot_ece_full(results: dict, out_path: Path):
    """
    5-panel figure:
      [0] Reliability diagram (overall)
      [1] Reliability diagram per class (3 lines)
      [2] Coverage-Accuracy curve
      [3] Uncertainty histogram: ID vs M&Ms-1
      [4] ECE bar chart per dataset/vendor
    """
    fig = plt.figure(figsize=(24, 5))
    gs  = fig.add_gridspec(1, 5, wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    dataset_label = results.get("label", "Dataset")
    ece    = results["ece"]
    n_bins = ECE_BINS

    # ── Panel 0: Overall Reliability Diagram ─────────────────────────────
    ax = axes[0]
    ba, bc, bn = results["bin_accs"], results["bin_confs"], results["bin_counts"]
    x = np.arange(n_bins)
    ax.bar(x, ba, 0.8, color="#1976D2", alpha=0.85, label="Accuracy")
    ax.bar(x, bc, 0.8, color="#E53935", alpha=0.30, label="Confidence")
    ax.plot([0, n_bins-1], [0, 1], "k--", alpha=0.4, lw=1, label="Perfect")
    ax.set_title(f"Reliability Diagram\nECE = {ece:.4f}", fontweight="bold")
    ax.set_xlabel("Confidence Bin"); ax.set_ylabel("Accuracy / Confidence")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xticks(x[::3])
    ax.set_xticklabels([f"{i/n_bins:.2f}" for i in range(n_bins)][::3])

    # ── Panel 1: Per-class Reliability Diagram ────────────────────────────
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Perfect")
    for c, info in results["per_class"].items():
        ba_c = info["bin_accs"]; bc_c = info["bin_confs"]
        mask = info["bin_counts"] > 0
        x_c  = bc_c[mask]
        ax.plot(x_c, ba_c[mask], "o-", color=STRUCT_COLOR[c], lw=2, markersize=5,
                label=f"{CLASS_NAMES[c]} (ECE={info['ece']:.4f})")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Per-Class Reliability\n(One-vs-Rest)", fontweight="bold")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 2: Coverage-Accuracy Curve ─────────────────────────────────
    ax = axes[2]
    covs, accs = results["covs"], results["accs"]
    baseline   = results["baseline_acc"]
    ax.plot(covs*100, accs*100, color="#1976D2", lw=2.0, label=dataset_label)
    ax.axhline(baseline*100, color="grey", ls="--", lw=1.5,
               label=f"Baseline ({baseline*100:.1f}%)")
    ax.fill_between(covs*100, baseline*100, accs*100,
                    where=accs > baseline, alpha=0.15, color="#1976D2")
    ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Coverage-Accuracy Curve\nAURRC = {results['aurrc']:.4f}", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 3: Uncertainty histogram comparison ─────────────────────────
    ax = axes[3]
    ax.hist(results["unc"], bins=60, alpha=0.75, color="#1976D2",
            density=True, label=dataset_label)
    if "unc_ref" in results:
        ax.hist(results["unc_ref"], bins=60, alpha=0.55, color="#E53935",
                density=True, label=results.get("ref_label","ACDC (ID)"))
    ax.axvline(results["unc"].mean(), color="#1976D2", ls="--", lw=1.5,
               label=f"Mean={results['unc'].mean():.3f}")
    ax.set_xlabel("Pixel Uncertainty $u_p = K/S_p$")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 4: ECE bar chart per vendor ─────────────────────────────────
    ax = axes[4]
    if "vendor_ece" in results and results["vendor_ece"]:
        vendors = list(results["vendor_ece"].keys())
        eces    = [results["vendor_ece"][v] for v in vendors]
        colors  = sns.color_palette("tab10", len(vendors))
        bars    = ax.bar(vendors, eces, color=colors, alpha=0.8, edgecolor="white", lw=1)
        for bar, val in zip(bars, eces):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(results["ece"], color="black", ls="--", lw=1.5,
                   label=f"Overall ECE={results['ece']:.4f}")
        ax.set_xlabel("Scanner Vendor"); ax.set_ylabel("ECE")
        ax.set_title("ECE per Vendor\n(lower = better calibrated)", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(vendors, rotation=12, ha="right")
    else:
        ax.text(0.5, 0.5, "Vendor ECE\nnot available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title("ECE per Vendor", fontweight="bold")

    fig.suptitle(f"ECE Analysis — {dataset_label}  "
                 f"(ECE={ece:.4f}  |  AURRC={results['aurrc']:.4f})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print(" ECE Analysis — M&Ms-1")
    print("="*60)
    print(f" Device: {DEVICE}\n")

    # ── Load model ────────────────────────────────────────────────────────
    model, cfg = load_model()
    preprocess_cfg = cfg.get("preprocessing", {})
    context_slices = int(cfg["data"].get("context_slices", cfg["data"].get("in_channels", 1)))
    val_tf = get_val_transforms(
        tuple(cfg["data"]["spatial_size"]),
        preprocess_cfg=preprocess_cfg,
    )
    loader_workers = 0 if DEVICE.type in {"cpu", "mps"} else 4

    # ── Load metadata for vendor mapping ─────────────────────────────────
    vendor_map = {}
    mnm1_meta_csv = find_mnm1_metadata_csv(MNM1_ROOT)
    if mnm1_meta_csv is not None and mnm1_meta_csv.exists():
        meta = pd.read_csv(mnm1_meta_csv)
        meta.columns = [c.strip() for c in meta.columns]
        meta["External code"] = meta["External code"].astype(str).str.strip()
        vendor_map = dict(zip(meta["External code"], meta["VendorName"]))
        print(f" Vendor map: {len(vendor_map)} subjects → "
              f"{sorted(set(vendor_map.values()))}")
    else:
        print(f" ⚠  Metadata CSV not found under: {MNM1_ROOT}")

    # ── M&Ms-1 inference ─────────────────────────────────────────────────
    print("\n[1/3] Running inference on M&Ms-1...")
    mnm1_slices = collect_mnm1_slices(
        str(MNM1_ROOT),
        splits=["Testing","Validation"],
        preprocess_cfg=preprocess_cfg,
    )
    mnm1_ds = DomainShiftDataset(
        mnm1_slices,
        transforms=val_tf,
        context_slices=context_slices,
    )
    mnm1_dl = DataLoader(
        mnm1_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=loader_workers,
    )
    print(f"  {len(mnm1_slices)} slices from "
          f"{len(set(s['patient_id'] for s in mnm1_slices))} patients")

    mnm1_probs, mnm1_labels, mnm1_unc, mnm1_correct = \
        collect_probs(model, mnm1_dl, desc="M&Ms-1")

    # ── (Optional) ACDC inference for comparison ─────────────────────────
    acdc_unc = None
    acdc_ece = None
    if ACDC_ROOT.exists():
        print("\n[2/3] Running inference on ACDC (for comparison)...")
        acdc_slices = collect_acdc_slices(
            str(ACDC_ROOT),
            split="testing",
            preprocess_cfg=preprocess_cfg,
        )
        acdc_ds = SliceDataset(
            acdc_slices,
            transforms=val_tf,
            context_slices=context_slices,
        )
        acdc_dl = DataLoader(
            acdc_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=loader_workers,
        )
        acdc_probs, acdc_labels, acdc_unc_arr, acdc_correct = \
            collect_probs(model, acdc_dl, desc="ACDC")
        acdc_ece, *_ = compute_ece(acdc_probs, acdc_labels, n_bins=ECE_BINS)
        acdc_unc     = acdc_unc_arr
        print(f"  ACDC ECE: {acdc_ece:.4f}")
    else:
        print("\n[2/3] ACDC root not found — skipping comparison")

    # ── Compute ECE ───────────────────────────────────────────────────────
    print("\n[3/3] Computing ECE metrics...")
    ece, ba, bc, bn, edges = compute_ece(mnm1_probs, mnm1_labels, n_bins=ECE_BINS)
    per_class = compute_ece_per_class(mnm1_probs, mnm1_labels, n_bins=ECE_BINS)
    aurrc     = compute_aurrc(mnm1_unc, mnm1_correct)
    covs, accs = coverage_accuracy_curve(mnm1_unc, mnm1_correct)

    print(f"\n  Overall ECE  : {ece:.4f}")
    if acdc_ece:
        print(f"  ACDC ECE     : {acdc_ece:.4f}  "
              f"(ΔCE = {ece-acdc_ece:+.4f})")
    print(f"  AURRC        : {aurrc:.4f}")
    print(f"  Mean unc     : {mnm1_unc.mean():.4f}")
    print(f"  Baseline acc : {mnm1_correct.mean()*100:.2f}%")
    for c, info in per_class.items():
        print(f"  ECE {CLASS_NAMES[c]:<4}     : {info['ece']:.4f}")

    # ── Per-vendor ECE ────────────────────────────────────────────────────
    vendor_ece = {}
    if vendor_map:
        print("\n  Per-vendor ECE:")
        pid_per_slice = [s["patient_id"] for s in mnm1_slices]
        # Rebuild full (un-downsampled) vendor array aligned to downsampled probs
        # We stored step=2 downsampling → each slice contributes (H/2)*(W/2) pixels
        # Reconstruct vendor label per pixel group
        H = W = cfg["data"]["spatial_size"][0]
        step = 2
        pix_per_slice = (H // step) * (W // step)

        vendor_labels = []
        for pid in pid_per_slice:
            v = vendor_map.get(pid, "Unknown")
            vendor_labels.extend([v] * pix_per_slice)

        vendor_arr = np.array(vendor_labels[:len(mnm1_probs)])
        for v in sorted(set(vendor_map.values())):
            mask = vendor_arr == v
            if mask.sum() < 100:
                continue
            ece_v, *_ = compute_ece(mnm1_probs[mask], mnm1_labels[mask], n_bins=ECE_BINS)
            vendor_ece[v] = ece_v
            print(f"    {v:<12}: ECE={ece_v:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    results = dict(
        label       = "M&Ms-1",
        ece         = ece,
        bin_accs    = ba,
        bin_confs   = bc,
        bin_counts  = bn,
        per_class   = per_class,
        aurrc       = aurrc,
        covs        = covs,
        accs        = accs,
        baseline_acc= float(mnm1_correct.mean()),
        unc         = mnm1_unc,
        vendor_ece  = vendor_ece,
    )
    if acdc_unc is not None:
        results["unc_ref"]   = acdc_unc
        results["ref_label"] = "ACDC (in-distribution)"

    out_png = OUT_FIG / "mnm1_ece_full.png"
    plot_ece_full(results, out_png)

    # ── Save CSV ──────────────────────────────────────────────────────────
    rows = [{
        "dataset":          "MnM1",
        "ece_overall":       ece,
        "ece_rv":            per_class[1]["ece"],
        "ece_myo":           per_class[2]["ece"],
        "ece_lv":            per_class[3]["ece"],
        "aurrc":             aurrc,
        "baseline_acc":      float(mnm1_correct.mean()),
        "mean_uncertainty":  float(mnm1_unc.mean()),
        "acdc_ece":          acdc_ece if acdc_ece else "N/A",
        "delta_ece":         f"{ece-acdc_ece:+.4f}" if acdc_ece else "N/A",
    }]
    for v, ece_v in vendor_ece.items():
        rows.append({"dataset": f"MnM1_{v}", "ece_overall": ece_v})

    out_csv = OUT_EVAL / "mnm1_ece.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"  ECE overall  : {ece:.4f}  {'✓ well calibrated' if ece<0.05 else '⚠ over-confident'}")
    print(f"  AURRC        : {aurrc:.4f}  {'✓ uncertainty useful' if aurrc>0.6 else '⚠ uncertainty not informative'}")
    if vendor_ece:
        worst_v = max(vendor_ece, key=vendor_ece.get)
        best_v  = min(vendor_ece, key=vendor_ece.get)
        print(f"  Best  vendor : {best_v} (ECE={vendor_ece[best_v]:.4f})")
        print(f"  Worst vendor : {worst_v} (ECE={vendor_ece[worst_v]:.4f})")
    print(f"\n  Figures → {OUT_FIG}/mnm1_ece_full.png")
    print(f"  CSV     → {OUT_EVAL}/mnm1_ece.csv")


if __name__ == "__main__":
    main()
