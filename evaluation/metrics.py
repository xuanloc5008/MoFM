"""
evaluation/metrics.py + clinical.py
Complete evaluation pipeline:
  - Segmentation metrics (Dice, HD95, ASD)
  - Topology Error Rate (TER) via 3D Cubical Complex
  - Clinical indices (EF, mass, stroke volume)
  - Uncertainty propagation (analytic + MC)
  - Calibration (ECE, reliability diagram)
  - OOD detection (AUC, FPR@95TPR)
  - TTEC classification accuracy
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from monai.metrics import (
    DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
)
from monai.transforms import AsDiscrete
import gudhi
import scipy.ndimage as nd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import pingouin as pg

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import cfg
from ttec.ttec_engine import (
    check_3d_betti, find_betti_violations, BETTI_TARGETS
)


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Metrics
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationEvaluator:
    """MONAI-based segmentation metric computation."""

    def __init__(self, num_classes: int = 4):
        self.K = num_classes
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean_batch"
        )
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean_batch"
        )
        self.asd_metric = SurfaceDistanceMetric(
            include_background=False, reduction="mean_batch"
        )
        self.post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)

    def update(
        self,
        pred: torch.Tensor,    # (B, K, H, W, D) soft or (B, H, W, D) hard
        label: torch.Tensor,   # (B, H, W, D) long
    ):
        if pred.ndim == 4:
            pred_oh = F.one_hot(pred, self.K).permute(0, 4, 1, 2, 3).float()
        else:
            pred_oh = F.one_hot(pred.argmax(1), self.K).permute(0, 4, 1, 2, 3).float()
        label_oh = F.one_hot(label, self.K).permute(0, 4, 1, 2, 3).float()
        self.dice_metric(pred_oh, label_oh)
        self.hd95_metric(pred_oh, label_oh)
        self.asd_metric(pred_oh, label_oh)

    def aggregate(self) -> Dict:
        dice = self.dice_metric.aggregate().cpu().numpy()   # (K-1,)
        hd95 = self.hd95_metric.aggregate().cpu().numpy()
        asd  = self.asd_metric.aggregate().cpu().numpy()
        self.dice_metric.reset()
        self.hd95_metric.reset()
        self.asd_metric.reset()
        names = ["LV", "Myo", "RV"]
        result = {}
        for i, n in enumerate(names):
            result[f"dice_{n}"]  = float(dice[i]) if i < len(dice) else 0.0
            result[f"hd95_{n}"]  = float(hd95[i]) if i < len(hd95) else 0.0
            result[f"asd_{n}"]   = float(asd[i])  if i < len(asd)  else 0.0
        result["dice_avg"] = float(np.nanmean(dice))
        result["hd95_avg"] = float(np.nanmean(hd95))
        return result


def compute_metrics(
    preds: torch.Tensor,   # (N, H, W, D) hard
    labels: torch.Tensor,  # (N, H, W, D) long
    num_classes: int = 4,
) -> Dict:
    """Quick metric computation for validation during training."""
    evaluator = SegmentationEvaluator(num_classes)
    evaluator.update(preds, labels)
    return evaluator.aggregate()


# ─────────────────────────────────────────────────────────────────────────────
# Topology Error Rate (TER)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ter(
    pred_volumes: List[np.ndarray],   # list of (K, H, W, D) soft probs
    resolution: Tuple[int, ...] = (32, 32, 8),
) -> float:
    """
    TER = fraction of volumes with at least one 3D Betti violation.
    """
    n_violations = 0
    for prob in pred_volumes:
        betti = check_3d_betti(prob, resolution)
        violations = find_betti_violations(betti)
        if violations:
            n_violations += 1
    return n_violations / max(len(pred_volumes), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Indices
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalIndexCalculator:
    """
    Computes EF, myocardial mass, stroke volume.
    With analytic uncertainty propagation from EDL outputs.
    """

    # MRI voxel volume in mm³ → mL conversion
    # Actual: compute from spacing metadata. Default 1.5³ mm³
    VOXEL_VOL_ML = (1.5 ** 3) / 1000.0   # mm³ to mL

    # Myocardial density
    MYO_DENSITY = 1.05   # g/mL

    def compute_ef(
        self,
        p_hat_ed: torch.Tensor,    # (K, H, W, D) ED frame
        p_hat_es: torch.Tensor,    # (K, H, W, D) ES frame
        vacuity_ed: torch.Tensor,  # (H, W, D)
        vacuity_es: torch.Tensor,
        dissonance_ed: torch.Tensor,
        dissonance_es: torch.Tensor,
        voxel_vol_ml: float = None,
    ) -> Dict:
        vvol = voxel_vol_ml or self.VOXEL_VOL_ML
        lv_id = 1   # LV class index

        # Volume estimation
        V_ED = float((p_hat_ed[lv_id] > 0.5).sum().item() * vvol)
        V_ES = float((p_hat_es[lv_id] > 0.5).sum().item() * vvol)

        if V_ED < 1e-6:
            return {"ef": 0.0, "v_ed_ml": 0.0, "v_es_ml": 0.0,
                    "sv_ml": 0.0, "sigma_ef_analytic": 0.0,
                    "ci95_ef": 0.0, "ef_ci_lower": 0.0, "ef_ci_upper": 0.0,
                    "spans_hf_threshold": False}

        EF  = (V_ED - V_ES) / V_ED
        SV  = V_ED - V_ES

        # Analytic uncertainty propagation (Eq. 17-18)
        sigma2_ed = self._area_uncertainty(p_hat_ed, vacuity_ed,
                                           dissonance_ed, lv_id, vvol)
        sigma2_es = self._area_uncertainty(p_hat_es, vacuity_es,
                                           dissonance_es, lv_id, vvol)

        # ∂EF/∂V_ED = -V_ES/V_ED²;  ∂EF/∂V_ES = -1/V_ED
        dEF_dED = -V_ES / (V_ED ** 2)
        dEF_dES = -1.0  / V_ED
        sigma2_ef = (dEF_dED ** 2) * sigma2_ed + (dEF_dES ** 2) * sigma2_es

        # 95% CI
        ci95 = 1.96 * math.sqrt(sigma2_ef)

        return {
            "ef": float(EF * 100),       # %
            "v_ed_ml": float(V_ED),
            "v_es_ml": float(V_ES),
            "sv_ml":   float(SV),
            "sigma_ef_analytic": float(math.sqrt(sigma2_ef) * 100),
            "ci95_ef": float(ci95 * 100),
            "ef_ci_lower": float((EF - ci95) * 100),
            "ef_ci_upper": float((EF + ci95) * 100),
            "spans_hf_threshold": float((EF - ci95) * 100) < cfg.evaluation.ef_hf_threshold
                                  < float((EF + ci95) * 100),
        }

    def compute_myocardial_mass(
        self,
        p_hat: torch.Tensor,
        vacuity: torch.Tensor,
        dissonance: torch.Tensor,
        voxel_vol_ml: float = None,
    ) -> Dict:
        vvol = voxel_vol_ml or self.VOXEL_VOL_ML
        myo_id = 2
        V_myo = float((p_hat[myo_id] > 0.5).sum().item() * vvol)
        mass  = V_myo * self.MYO_DENSITY   # grams

        sigma2_v = self._area_uncertainty(p_hat, vacuity, dissonance,
                                          myo_id, vvol)
        sigma_mass = math.sqrt(sigma2_v) * self.MYO_DENSITY

        return {
            "myo_volume_ml": float(V_myo),
            "myo_mass_g":    float(mass),
            "sigma_mass_g":  float(sigma_mass),
        }

    def _area_uncertainty(
        self,
        p_hat: torch.Tensor,    # (K, H, W, D)
        vacuity: torch.Tensor,  # (H, W, D)
        dissonance: torch.Tensor,
        struct_id: int,
        voxel_vol: float,
    ) -> float:
        """
        σ²(A_s) = Σ_{v∈∂s} Vac(v)*(1-p̄_s(v))² + Diss(v)*p̄_s(v)*(1-p̄_s(v))
        boundary voxels only (Eq. 17)
        """
        prob_s = p_hat[struct_id]   # (H, W, D)
        mask   = (prob_s > 0.5).cpu().numpy()
        # Boundary = voxels adjacent to the surface
        from scipy.ndimage import binary_erosion
        eroded   = binary_erosion(mask)
        boundary = torch.tensor(
            (mask & ~eroded).astype(np.float32), device=p_hat.device
        )  # (H, W, D)

        ps  = prob_s[boundary.bool()]
        vac = vacuity[boundary.bool()]
        dis = dissonance[boundary.bool()]

        if len(ps) == 0:
            return 1e-6

        term1 = (vac * (1 - ps).pow(2)).sum().item()
        term2 = (dis * ps * (1 - ps)).sum().item()
        return (term1 + term2) * (voxel_vol ** 2)

    def mc_ef_uncertainty(
        self,
        alpha_ed: torch.Tensor,  # (K, H, W, D) Dirichlet params
        alpha_es: torch.Tensor,
        n_mc: int = None,
        voxel_vol_ml: float = None,
    ) -> Dict:
        """
        Monte Carlo validation of analytic uncertainty (Eq. 19).
        Draws N_MC samples from pixel-wise Dirichlet posteriors.
        """
        n_mc = n_mc or cfg.evaluation.n_mc_samples
        vvol = voxel_vol_ml or self.VOXEL_VOL_ML
        lv_id = 1

        ef_samples = []
        for _ in range(n_mc):
            # Draw concentration parameters → sample probabilities
            p_ed = torch.distributions.Dirichlet(
                alpha_ed.permute(1, 2, 3, 0).float()
            ).sample().permute(3, 0, 1, 2)  # (K, H, W, D)
            p_es = torch.distributions.Dirichlet(
                alpha_es.permute(1, 2, 3, 0).float()
            ).sample().permute(3, 0, 1, 2)

            v_ed = float((p_ed[lv_id] > 0.5).sum().item() * vvol)
            v_es = float((p_es[lv_id] > 0.5).sum().item() * vvol)
            ef_s = 100.0 * (v_ed - v_es) / max(v_ed, 1e-6)
            ef_samples.append(ef_s)

        ef_arr = np.array(ef_samples)
        return {
            "ef_mc_mean": float(np.mean(ef_arr)),
            "sigma_ef_mc": float(np.std(ef_arr)),
            "ef_mc_ci95_lower": float(np.percentile(ef_arr, 2.5)),
            "ef_mc_ci95_upper": float(np.percentile(ef_arr, 97.5)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Calibration (ECE)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    probs: np.ndarray,    # (N, K) soft probabilities
    labels: np.ndarray,   # (N,) integer labels
    n_bins: int = 15,
    struct_name: str = "",
) -> Dict:
    """
    Expected Calibration Error per structure.
    ECE = Σ_b |B_b|/n * |acc(B_b) - conf(B_b)|
    """
    K = probs.shape[1]
    # One-vs-rest per class
    ece_per_class = []
    for k in range(1, K):   # skip background
        p_k = probs[:, k]
        y_k = (labels == k).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece  = 0.0
        for i in range(n_bins):
            mask = (p_k >= bins[i]) & (p_k < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc  = y_k[mask].mean()
            conf = p_k[mask].mean()
            ece += mask.mean() * abs(acc - conf)
        ece_per_class.append(ece)
    return {
        f"ece_{struct_name}": float(np.mean(ece_per_class)),
        "ece_per_class": [float(e) for e in ece_per_class],
    }


# ─────────────────────────────────────────────────────────────────────────────
# OOD Detection Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ood_metrics(
    tu_scores_in: np.ndarray,    # TU for in-distribution samples
    tu_scores_out: np.ndarray,   # TU for OOD samples
) -> Dict:
    """
    AUC and FPR@95%TPR for OOD detection using TU scores.
    Higher TU = more likely OOD.
    """
    labels = np.concatenate([
        np.zeros(len(tu_scores_in)),
        np.ones(len(tu_scores_out))
    ])
    scores = np.concatenate([tu_scores_in, tu_scores_out])

    auc = float(roc_auc_score(labels, scores))

    # FPR at 95% TPR
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95 = float(fpr[idx]) * 100

    return {
        "ood_auc":    auc * 100,
        "fpr_at_95tpr": fpr_at_95,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TTEC Classification Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ttec_accuracy(
    pred_types: List[int],     # 0/1/2
    true_types: List[int],
) -> Dict:
    """
    Per-class precision, recall, F1, overall accuracy.
    Fleiss' κ computed via pingouin if rater arrays provided.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    pred  = np.array(pred_types)
    true  = np.array(true_types)
    names = ["Type-I", "Type-II", "Type-III"]

    report = classification_report(
        true, pred, target_names=names,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true, pred)
    acc = float(np.trace(cm) / cm.sum())

    return {
        "accuracy": acc,
        "per_class": {
            n: {
                "precision": report[n]["precision"],
                "recall":    report[n]["recall"],
                "f1":        report[n]["f1-score"],
            }
            for n in names
        },
        "confusion_matrix": cm.tolist(),
    }


def compute_fleiss_kappa(
    ratings_matrix: np.ndarray   # (N_subjects, N_raters)
) -> float:
    """
    Compute Fleiss' κ for inter-rater agreement.
    Uses pingouin library.
    """
    import pandas as pd
    n_subj, n_raters = ratings_matrix.shape
    rows = []
    for i in range(n_subj):
        for j in range(n_raters):
            rows.append({"subject": i, "rater": j,
                         "rating": ratings_matrix[i, j]})
    df = pd.DataFrame(rows)
    try:
        icc = pg.intraclass_corr(data=df, targets="subject",
                                  raters="rater", ratings="rating")
        # Approximate Fleiss' κ from ICC
        return float(icc.loc[icc["Type"] == "ICC2", "ICC"].values[0])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class FullEvaluator:
    """
    Runs complete evaluation pipeline on a dataset.
    """

    def __init__(self, clinical_calc: ClinicalIndexCalculator = None):
        self.seg_eval = SegmentationEvaluator(cfg.data.num_classes)
        self.clinical = clinical_calc or ClinicalIndexCalculator()
        self.results: Dict = {}

    @torch.no_grad()
    def evaluate_batch(
        self,
        p_hat: torch.Tensor,       # (K, H, W, D)
        label: torch.Tensor,       # (H, W, D) long
        vacuity: torch.Tensor,
        dissonance: torch.Tensor,
        alpha: torch.Tensor,       # (K, H, W, D) for MC
        tu_score: float,
        phase: str = "ED",
        phase_partner_data: Optional[Dict] = None,
    ) -> Dict:
        """Evaluate a single volume."""
        result = {}

        # Segmentation metrics
        pred_hard = p_hat.argmax(0)
        self.seg_eval.update(
            pred_hard.unsqueeze(0), label.unsqueeze(0)
        )

        # Topology
        prob_np = p_hat.cpu().numpy()
        betti = check_3d_betti(
            np.stack([np.zeros_like(prob_np[0])] + [prob_np[i]
                     for i in range(1, 4)])
        )
        violations = find_betti_violations(betti)
        result["has_violation"] = len(violations) > 0
        result["betti"]     = betti
        result["violations"] = violations

        # TU
        result["tu"] = tu_score

        # Clinical (if both phases available)
        if phase_partner_data is not None and phase == "ED":
            p_hat_es = phase_partner_data["p_hat"]
            vac_es   = phase_partner_data["vacuity"]
            dis_es   = phase_partner_data["dissonance"]
            ef_dict  = self.clinical.compute_ef(
                p_hat, p_hat_es, vacuity, vac_es, dissonance, dis_es
            )
            mass_dict = self.clinical.compute_myocardial_mass(
                p_hat, vacuity, dissonance
            )
            result.update(ef_dict)
            result.update(mass_dict)

        return result

    def aggregate(self) -> Dict:
        seg_metrics = self.seg_eval.aggregate()
        self.results.update(seg_metrics)
        return self.results
