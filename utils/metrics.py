"""
Evaluation metrics: Dice, HD95, TER, ECE, clinical index propagation.
"""
import numpy as np
import torch
from typing import Dict, List, Optional
from scipy.ndimage import distance_transform_edt


def dice_score(pred: np.ndarray, target: np.ndarray, class_idx: int) -> float:
    p = (pred == class_idx)
    t = (target == class_idx)
    if p.sum() + t.sum() == 0:
        return 1.0
    return 2 * (p & t).sum() / (p.sum() + t.sum())


def hausdorff_95(pred: np.ndarray, target: np.ndarray, class_idx: int,
                 voxel_spacing: tuple = (1.0, 1.0)) -> float:
    p = (pred == class_idx).astype(np.uint8)
    t = (target == class_idx).astype(np.uint8)
    if p.sum() == 0 or t.sum() == 0:
        return float('inf')

    try:
        from medpy.metric.binary import hd95
        return hd95(p, t, voxelspacing=voxel_spacing)
    except ImportError:
        # Fallback
        dt_p = distance_transform_edt(1 - p)
        dt_t = distance_transform_edt(1 - t)
        d_pt = dt_t[p > 0]
        d_tp = dt_p[t > 0]
        if len(d_pt) == 0 or len(d_tp) == 0:
            return float('inf')
        return max(np.percentile(d_pt, 95), np.percentile(d_tp, 95))


def topology_error_rate(predictions: List[np.ndarray],
                        betti_targets: Dict = None) -> float:
    """TER: percentage of slices with at least one Betti violation."""
    from data.topology import compute_all_betti, check_betti_violations
    if betti_targets is None:
        betti_targets = {"LV": (1, 0), "Myo": (1, 1), "RV": (1, 0)}

    violations = 0
    for pred in predictions:
        betti = compute_all_betti(pred)
        if check_betti_violations(betti, betti_targets):
            violations += 1
    return violations / max(len(predictions), 1) * 100


def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                               n_bins: int = 15) -> float:
    """ECE: Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += mask.sum() / total * abs(avg_conf - avg_acc)
    return ece


# ---- Clinical Index Computation ----

def compute_ef(area_lv_ed: float, area_lv_es: float) -> float:
    """Ejection Fraction = (A_ED - A_ES) / A_ED * 100."""
    if area_lv_ed == 0:
        return 0.0
    return (area_lv_ed - area_lv_es) / area_lv_ed * 100


def compute_myocardial_mass(area_myo: float, slice_thickness: float = 10.0,
                            pixel_spacing: float = 1.0,
                            density: float = 1.05) -> float:
    """Myocardial mass in grams. density of myocardium ~ 1.05 g/ml."""
    volume_ml = area_myo * pixel_spacing ** 2 * slice_thickness / 1000
    return volume_ml * density


# ---- Uncertainty Propagation ----

class ClinicalUncertaintyPropagator:
    """Propagate pixel-level uncertainty to clinical indices.
    Method 1: Analytic (first-order Taylor expansion)
    Method 2: Monte Carlo sampling from Dirichlet posterior
    """

    def __init__(self, mc_samples: int = 200):
        self.N_mc = mc_samples

    def analytic_area_uncertainty(self, p_bar: np.ndarray, vacuity: np.ndarray,
                                 dissonance: np.ndarray,
                                 class_idx: int) -> float:
        """Eq. 11: sigma^2(A_s) from boundary pixel uncertainties."""
        pred = p_bar.argmax(axis=0)
        boundary = self._get_boundary(pred, class_idx)

        if boundary.sum() == 0:
            return 0.0

        p_s = p_bar[class_idx]

        # Epistemic component
        sigma2_epi = (vacuity[boundary] * (1 - p_s[boundary]) ** 2).sum()
        # Aleatoric component
        sigma2_ale = (dissonance[boundary] * p_s[boundary] * (1 - p_s[boundary])).sum()

        return sigma2_epi + sigma2_ale

    def analytic_ef_uncertainty(self, sigma2_a_ed: float, sigma2_a_es: float,
                                a_ed: float, a_es: float) -> float:
        """Eq. 12: sigma^2(EF) via error propagation."""
        if a_ed == 0:
            return 0.0
        dEF_dAed = a_es / (a_ed ** 2) * 100
        dEF_dAes = -1.0 / a_ed * 100
        return (dEF_dAed ** 2 * sigma2_a_ed + dEF_dAes ** 2 * sigma2_a_es)

    def mc_ef_uncertainty(self, alpha: np.ndarray,
                          class_idx_lv: int = 1) -> Dict[str, float]:
        """Monte Carlo sampling from Dirichlet posterior for EF uncertainty.
        alpha: (K, H, W) Dirichlet parameters.
        Assumes alpha contains both ED and ES frames.
        """
        # This is called per-frame; caller handles ED/ES separately
        K, H, W = alpha.shape
        areas = []

        for _ in range(self.N_mc):
            # Sample from Dirichlet at each pixel
            p_sample = np.zeros((K, H, W))
            for y in range(H):
                for x in range(W):
                    p_sample[:, y, x] = np.random.dirichlet(alpha[:, y, x])

            pred = p_sample.argmax(axis=0)
            area = (pred == class_idx_lv).sum()
            areas.append(area)

        return {
            "mean_area": np.mean(areas),
            "std_area": np.std(areas),
            "areas": areas,
        }

    def mc_ef_uncertainty_fast(self, alpha: np.ndarray,
                               class_idx_lv: int = 1,
                               n_samples: int = None) -> Dict:
        """Faster MC sampling — sample only at boundary pixels."""
        if n_samples is None:
            n_samples = self.N_mc

        K, H, W = alpha.shape
        pred_mode = (alpha / alpha.sum(axis=0, keepdims=True)).argmax(axis=0)
        boundary = self._get_boundary(pred_mode, class_idx_lv)

        base_area = (pred_mode == class_idx_lv).sum()
        boundary_coords = np.argwhere(boundary)

        if len(boundary_coords) == 0:
            return {"mean_area": float(base_area), "std_area": 0.0}

        area_samples = []
        for _ in range(n_samples):
            delta = 0
            for y, x in boundary_coords:
                p = np.random.dirichlet(alpha[:, y, x])
                sampled_class = np.argmax(p)
                current_class = pred_mode[y, x]
                if sampled_class == class_idx_lv and current_class != class_idx_lv:
                    delta += 1
                elif sampled_class != class_idx_lv and current_class == class_idx_lv:
                    delta -= 1
            area_samples.append(base_area + delta)

        return {
            "mean_area": np.mean(area_samples),
            "std_area": np.std(area_samples),
        }

    def _get_boundary(self, pred: np.ndarray, class_idx: int) -> np.ndarray:
        mask = (pred == class_idx).astype(np.float32)
        dx = np.abs(mask[:, 1:] - mask[:, :-1])
        dy = np.abs(mask[1:, :] - mask[:-1, :])
        boundary = np.zeros_like(mask, dtype=bool)
        boundary[:, 1:] |= dx > 0
        boundary[1:, :] |= dy > 0
        return boundary

    def compute_all(self, ensemble_output: Dict, frame_type: str,
                    pixel_spacing: float = 1.0) -> Dict:
        """Compute all clinical uncertainties for one frame."""
        p_bar = ensemble_output["p_bar"]  # (K, H, W)
        vacuity = ensemble_output["vacuity"]  # (H, W)
        dissonance = ensemble_output["dissonance"]
        alpha = ensemble_output.get("alpha")  # (K, H, W) if available

        pred = p_bar.argmax(axis=0)

        # Areas
        area_lv = (pred == 1).sum() * pixel_spacing ** 2
        area_myo = (pred == 2).sum() * pixel_spacing ** 2
        area_rv = (pred == 3).sum() * pixel_spacing ** 2

        # Analytic uncertainty
        sigma2_lv = self.analytic_area_uncertainty(p_bar, vacuity, dissonance, 1)
        sigma2_myo = self.analytic_area_uncertainty(p_bar, vacuity, dissonance, 2)

        result = {
            "frame": frame_type,
            "area_lv": area_lv,
            "area_myo": area_myo,
            "area_rv": area_rv,
            "sigma2_lv_analytic": sigma2_lv,
            "sigma2_myo_analytic": sigma2_myo,
        }

        # MC uncertainty (if alpha available)
        if alpha is not None:
            mc_lv = self.mc_ef_uncertainty_fast(alpha, class_idx_lv=1)
            result["sigma_lv_mc"] = mc_lv["std_area"] * pixel_spacing ** 2
            result["mean_area_lv_mc"] = mc_lv["mean_area"] * pixel_spacing ** 2

        return result
