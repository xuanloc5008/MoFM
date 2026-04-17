"""
Evaluation Metrics
-------------------
Standard medical image segmentation metrics + uncertainty calibration metrics.

Segmentation:
  - Dice Similarity Coefficient (DSC)
  - Hausdorff Distance 95th percentile (HD95)
  - Average Surface Distance (ASD)
  - Jaccard / IoU

Calibration & Uncertainty:
  - Expected Calibration Error (ECE)
  - Reliability curve
  - Uncertainty-error correlation (AURC)
  - Coverage-Accuracy curve

All functions operate on numpy arrays for compatibility with monai metrics.
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import binary_erosion, distance_transform_edt, label as scipy_label
import warnings


# ─────────────────────────────────────────────────────────────────────────────
#  Segmentation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """DSC for binary masks."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = (pred & gt).sum()
    return (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def dice_per_class(
    pred:        np.ndarray,    # (H, W)  integer labels
    gt:          np.ndarray,    # (H, W)  integer labels
    num_classes: int = 4,
    ignore_bg:   bool = True,
) -> Dict[int, float]:
    """DSC per class, optionally ignoring background (class 0)."""
    scores = {}
    start = 1 if ignore_bg else 0
    for c in range(start, num_classes):
        scores[c] = dice_score(pred == c, gt == c)
    return scores


def hausdorff_distance_95(
    pred:    np.ndarray,    # binary
    gt:      np.ndarray,    # binary
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> float:
    """
    95th percentile Hausdorff Distance in physical units.
    Returns inf if either mask is empty.
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if not pred.any() or not gt.any():
        return float("inf")

    # Surface extraction via erosion
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf   = gt   ^ binary_erosion(gt)

    pred_pts = np.column_stack(np.where(pred_surf)) * np.array(spacing)
    gt_pts   = np.column_stack(np.where(gt_surf))   * np.array(spacing)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    # Directed distances (vectorised)
    d_p2g = _min_distances(pred_pts, gt_pts)
    d_g2p = _min_distances(gt_pts,   pred_pts)

    return float(np.percentile(np.concatenate([d_p2g, d_g2p]), 95))


def _min_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Min distance from each point in A to any point in B."""
    # Batched to avoid O(N*M) memory for large surfaces
    batch = 512
    mins = []
    for i in range(0, len(A), batch):
        diff = A[i:i+batch, np.newaxis, :] - B[np.newaxis, :, :]   # k x M x 2
        d    = np.linalg.norm(diff, axis=-1).min(axis=1)
        mins.append(d)
    return np.concatenate(mins)


def average_surface_distance(
    pred:    np.ndarray,
    gt:      np.ndarray,
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> float:
    """Mean symmetric surface distance."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if not pred.any() or not gt.any():
        return float("inf")

    pred_surf = pred ^ binary_erosion(pred)
    gt_surf   = gt   ^ binary_erosion(gt)

    pred_pts = np.column_stack(np.where(pred_surf)) * np.array(spacing)
    gt_pts   = np.column_stack(np.where(gt_surf))   * np.array(spacing)

    d1 = _min_distances(pred_pts, gt_pts).mean()
    d2 = _min_distances(gt_pts,   pred_pts).mean()
    return float((d1 + d2) / 2)


def compute_segmentation_metrics(
    pred:        np.ndarray,
    gt:          np.ndarray,
    num_classes: int = 4,
    spacing:     Tuple[float, float] = (1.0, 1.0),
) -> Dict:
    """
    Full segmentation evaluation for a single 2-D slice.
    Returns dict with per-class and mean metrics.
    """
    results = {}
    dice_scores = []
    hd95_scores = []
    asd_scores  = []

    for c in range(1, num_classes):   # skip background
        d   = dice_score(pred == c, gt == c)
        h95 = hausdorff_distance_95(pred == c, gt == c, spacing)
        asd = average_surface_distance(pred == c, gt == c, spacing)

        results[f"dice_c{c}"] = d
        results[f"hd95_c{c}"] = h95
        results[f"asd_c{c}"]  = asd
        dice_scores.append(d)
        if h95 != float("inf"):
            hd95_scores.append(h95)
        if asd != float("inf"):
            asd_scores.append(asd)

    results["dice_mean"] = float(np.mean(dice_scores)) if dice_scores else 0.0
    results["hd95_mean"] = float(np.mean(hd95_scores)) if hd95_scores else float("inf")
    results["asd_mean"]  = float(np.mean(asd_scores))  if asd_scores  else float("inf")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration Metrics
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    probs:   np.ndarray,    # (N, K) softmax probabilities or (N,) max_prob
    labels:  np.ndarray,    # (N,) integer labels
    n_bins:  int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    ECE: Expected Calibration Error.
    Uses equal-width confidence bins.

    Returns:
      ece         : scalar ECE
      bin_accs    : (n_bins,) accuracy per bin
      bin_confs   : (n_bins,) avg confidence per bin
      bin_counts  : (n_bins,) sample count per bin
    """
    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs > 0.5).astype(int)

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

    ece = (np.abs(bin_accs - bin_confs) * bin_counts).sum() / bin_counts.sum()
    return float(ece), bin_accs, bin_confs, bin_counts


def uncertainty_error_correlation(
    uncertainty: np.ndarray,   # (N,) per-pixel uncertainty u_p
    is_correct:  np.ndarray,   # (N,) bool: was the prediction correct?
) -> float:
    """
    AURRC: Area Under Risk-Rejection Curve.
    Measures how well uncertainty correlates with errors.
    A model with perfect calibration has AURRC close to 0.
    """
    from sklearn.metrics import roc_auc_score
    # Invert: high uncertainty → high chance of error
    errors = (~is_correct.astype(bool)).astype(int)
    try:
        auroc = roc_auc_score(errors, uncertainty)
    except Exception:
        auroc = 0.5
    return float(auroc)


def coverage_accuracy_curve(
    uncertainty: np.ndarray,    # (N,)
    is_correct:  np.ndarray,    # (N,) bool
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coverage-Accuracy curve: as we reject highest-uncertainty predictions,
    accuracy on remaining predictions should increase.

    Returns:
      coverages   : fraction of data retained
      accuracies  : accuracy on retained data
    """
    idx  = np.argsort(uncertainty)   # sort by uncertainty ascending
    coverages  = []
    accuracies = []
    N = len(uncertainty)

    for frac in np.linspace(0.1, 1.0, n_thresholds):
        n_keep = max(1, int(frac * N))
        kept   = idx[:n_keep]
        coverages.append(frac)
        accuracies.append(is_correct[kept].mean())

    return np.array(coverages), np.array(accuracies)


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate metrics over a dataset
# ─────────────────────────────────────────────────────────────────────────────

class MetricAggregator:
    """Accumulates per-sample metrics and computes statistics."""

    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        self.num_classes  = num_classes
        self.class_names  = class_names or [f"C{i}" for i in range(num_classes)]
        self._records: List[Dict] = []

    def update(self, metrics: Dict):
        self._records.append(metrics)

    def summary(self) -> Dict:
        if not self._records:
            return {}
        import pandas as pd
        df = pd.DataFrame(self._records)
        # Only process numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        result = {}
        for col in numeric_df.columns:
            finite = numeric_df[col].replace([float("inf"), -float("inf")], np.nan).dropna()
            if len(finite) > 0:
                result[f"{col}_mean"] = float(finite.mean())
                result[f"{col}_std"]  = float(finite.std())
                result[f"{col}_med"]  = float(finite.median())
        return result

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self._records)

    def reset(self):
        self._records = []
