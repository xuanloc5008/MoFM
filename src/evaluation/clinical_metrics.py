"""
Clinical Cardiac Metrics
--------------------------
Computes clinical indices from 3-D segmentation volumes:
  - End-Diastolic Volume (EDV) [mL]
  - End-Systolic Volume (ESV)  [mL]
  - Ejection Fraction (EF)     [%]
  - Myocardial Mass            [g]
  - Stroke Volume (SV)         [mL]

These are computed from the 3-D stacked 2-D predictions using the
Simpson's method of discs (standard cardiac MRI convention).

Myocardial density = 1.05 g/mL (standard value)
"""
import numpy as np
from typing import Dict, Tuple, List, Optional


MYOCARDIUM_DENSITY = 1.05   # g/mL


def volume_from_slices(
    mask_3d:  np.ndarray,        # (H, W, Z)  binary mask
    spacing:  Tuple[float, float, float],   # (sx, sy, sz) in mm
) -> float:
    """
    Compute volume in mL using the disc summation method (Simpson's rule).
    """
    sx, sy, sz = spacing
    voxel_vol_mm3 = sx * sy * sz
    voxel_vol_mL  = voxel_vol_mm3 / 1000.0
    return float(mask_3d.sum()) * voxel_vol_mL


def compute_cardiac_volumes(
    ed_seg:    np.ndarray,    # (H, W, Z) ED segmentation
    es_seg:    np.ndarray,    # (H, W, Z) ES segmentation
    spacing:   Tuple[float, float, float],
    lv_class:  int = 3,       # LV cavity class index
    rv_class:  int = 1,       # RV class index
    myo_class: int = 2,       # Myocardium class index
) -> Dict[str, float]:
    """
    Compute all clinical cardiac indices from ED and ES segmentation volumes.

    Returns dict with:
      LV_EDV_mL, LV_ESV_mL, LV_EF_pct, LV_SV_mL
      RV_EDV_mL, RV_ESV_mL, RV_EF_pct, RV_SV_mL
      Myo_mass_g
    """
    results = {}

    # ── LV ────────────────────────────────────────────────────────────────
    lv_edv = volume_from_slices(ed_seg == lv_class, spacing)
    lv_esv = volume_from_slices(es_seg == lv_class, spacing)
    lv_sv  = lv_edv - lv_esv
    lv_ef  = (lv_sv / lv_edv * 100) if lv_edv > 0 else 0.0

    results["LV_EDV_mL"] = lv_edv
    results["LV_ESV_mL"] = lv_esv
    results["LV_SV_mL"]  = lv_sv
    results["LV_EF_pct"] = lv_ef

    # ── RV ────────────────────────────────────────────────────────────────
    rv_edv = volume_from_slices(ed_seg == rv_class, spacing)
    rv_esv = volume_from_slices(es_seg == rv_class, spacing)
    rv_sv  = rv_edv - rv_esv
    rv_ef  = (rv_sv / rv_edv * 100) if rv_edv > 0 else 0.0

    results["RV_EDV_mL"] = rv_edv
    results["RV_ESV_mL"] = rv_esv
    results["RV_SV_mL"]  = rv_sv
    results["RV_EF_pct"] = rv_ef

    # ── Myocardial Mass ───────────────────────────────────────────────────
    myo_vol  = volume_from_slices(ed_seg == myo_class, spacing)
    myo_mass = myo_vol * MYOCARDIUM_DENSITY

    results["Myo_vol_mL"] = myo_vol
    results["Myo_mass_g"] = myo_mass

    return results


def clinical_error(
    pred_metrics: Dict[str, float],
    gt_metrics:   Dict[str, float],
) -> Dict[str, float]:
    """
    Absolute and relative errors between predicted and ground-truth clinical metrics.
    """
    errors = {}
    for key in gt_metrics:
        if key in pred_metrics:
            ae = abs(pred_metrics[key] - gt_metrics[key])
            re = ae / (abs(gt_metrics[key]) + 1e-6) * 100
            errors[f"{key}_AE"]  = ae
            errors[f"{key}_RE%"] = re
    return errors


def bland_altman_stats(
    pred_vals: np.ndarray,
    gt_vals:   np.ndarray,
) -> Dict[str, float]:
    """
    Bland-Altman analysis for agreement between predicted and GT clinical values.
    Returns mean bias and limits of agreement (±1.96 SD).
    """
    diff = pred_vals - gt_vals
    mean = (pred_vals + gt_vals) / 2

    bias = float(np.mean(diff))
    std  = float(np.std(diff))
    loa_upper = bias + 1.96 * std
    loa_lower = bias - 1.96 * std

    return {
        "bias":      bias,
        "std":       std,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "pearson_r": float(np.corrcoef(pred_vals, gt_vals)[0, 1])
            if len(pred_vals) > 1 else 0.0,
    }


def stack_slices_to_volume(
    slices:     List[np.ndarray],   # list of (H, W) 2-D segmentation maps
    slice_idxs: List[int],          # physical slice positions (for z-gap handling)
) -> np.ndarray:
    """
    Stack 2-D slices into a 3-D volume in correct order.
    Missing slices (gaps) are filled with zeros.
    """
    if not slices:
        return np.zeros((1, 1, 1), dtype=np.int64)

    H, W   = slices[0].shape
    z_min  = min(slice_idxs)
    z_max  = max(slice_idxs)
    Z      = z_max - z_min + 1

    volume = np.zeros((H, W, Z), dtype=np.int64)
    for slc, z in zip(slices, slice_idxs):
        volume[:, :, z - z_min] = slc

    return volume


class ClinicalMetricAggregator:
    """
    Per-patient aggregator for clinical metrics.
    Accumulates 2-D predictions, reconstructs 3-D volumes, and computes
    EDV, ESV, EF, mass for each patient.
    """

    def __init__(self, spacing: Tuple[float, float, float] = (1.5, 1.5, 8.0)):
        self.default_spacing = spacing
        self._patients: Dict[str, Dict] = {}

    def add_slice(
        self,
        patient_id: str,
        phase:      str,       # "ED" or "ES"
        slice_idx:  int,
        pred_seg:   np.ndarray,
        gt_seg:     Optional[np.ndarray],
        spacing:    Optional[Tuple] = None,
    ):
        sp = spacing or self.default_spacing
        if patient_id not in self._patients:
            self._patients[patient_id] = {
                "spacing": sp,
                "ED_pred": {}, "ED_gt": {},
                "ES_pred": {}, "ES_gt": {},
            }
        self._patients[patient_id][f"{phase}_pred"][slice_idx] = pred_seg
        if gt_seg is not None:
            self._patients[patient_id][f"{phase}_gt"][slice_idx] = gt_seg

    def compute_all(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Compute clinical metrics for all patients.
        Returns (pred_metrics_list, gt_metrics_list).
        """
        pred_list, gt_list = [], []

        for pid, data in self._patients.items():
            spacing = data["spacing"]

            def build_vol(d):
                if not d:
                    return None
                idxs  = sorted(d.keys())
                slices = [d[i] for i in idxs]
                return stack_slices_to_volume(slices, idxs)

            ed_pred = build_vol(data["ED_pred"])
            es_pred = build_vol(data["ES_pred"])
            ed_gt   = build_vol(data["ED_gt"])
            es_gt   = build_vol(data["ES_gt"])

            if ed_pred is not None and es_pred is not None:
                p_metrics = compute_cardiac_volumes(ed_pred, es_pred, spacing)
                p_metrics["patient_id"] = pid
                pred_list.append(p_metrics)

            if ed_gt is not None and es_gt is not None:
                g_metrics = compute_cardiac_volumes(ed_gt, es_gt, spacing)
                g_metrics["patient_id"] = pid
                gt_list.append(g_metrics)

        return pred_list, gt_list
