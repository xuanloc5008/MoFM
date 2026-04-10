"""
Inference Pipeline: Phases 3-4.
Phase 3: TTEC classification of topological violations.
Phase 4: Topo-A* correction for Type-III errors.
Also computes clinical uncertainty propagation.
"""
import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from config import Config
from ttec.classifier import TTECClassifier, ErrorType
from ttec.topo_astar import TopoAStar
from utils.metrics import (
    dice_score, hausdorff_95, topology_error_rate,
    compute_ef, ClinicalUncertaintyPropagator
)


class InferencePipeline:
    """Full inference: Experts → Gating → TTEC → Topo-A* → Clinical Propagation."""

    def __init__(self, experts: list, gating, config: Config,
                 training_embeddings: np.ndarray = None):
        self.experts = experts
        self.gating = gating
        self.config = config
        self.device = next(experts[0].parameters()).device

        # TTEC
        self.ttec = TTECClassifier(config.ttec, training_embeddings)

        # Topo-A*
        self.topo_astar = TopoAStar(config.topo_astar)

        # Clinical uncertainty
        self.clinical_propagator = ClinicalUncertaintyPropagator(config.inference.mc_samples)

        # Set eval mode
        for e in self.experts:
            e.eval()
        if self.gating is not None:
            self.gating.eval()

    @torch.no_grad()
    def predict_single(self, image: torch.Tensor, z_norm: float = 0.5,
                       phase: str = "ED") -> Dict:
        """Full inference for a single 2D slice.

        Args:
            image: (1, 1, H, W) tensor
            z_norm: normalised slice position [0, 1]
            phase: "ED" or "ES"

        Returns:
            Dict with prediction, uncertainties, TTEC result, clinical metrics.
        """
        image = image.to(self.device)

        # Phase 1: Expert predictions
        expert_outputs = [e(image) for e in self.experts]

        # Phase 2: Gating ensemble
        weights = self.gating(image, expert_outputs)
        ensemble = self.gating.ensemble_predict(weights, expert_outputs)

        prediction = ensemble["prediction"][0].cpu().numpy()
        p_bar = ensemble["p_bar"][0].cpu().numpy()
        vacuity = ensemble["vacuity"][0].cpu().numpy()
        dissonance = ensemble["dissonance"][0].cpu().numpy()
        u_sys = ensemble["u_sys_epistemic"][0].cpu().numpy()
        embedding = expert_outputs[0]["embedding"][0].cpu().numpy()  # Use first expert's embedding

        # Topo maps
        topo_maps = {}
        for name in ["LV", "Myo", "RV"]:
            topo_maps[name] = ensemble["p_bar"][0, {"LV": 1, "Myo": 2, "RV": 3}[name]].cpu().numpy()

        # Phase 3: TTEC
        ensemble_for_ttec = {
            "u_sys_epistemic_mean": u_sys.mean(),
            "embedding": embedding,
            "topo_maps": topo_maps,
        }
        ttec_result = self.ttec.classify(prediction, ensemble_for_ttec, z_norm, phase)

        # Phase 4: Topo-A* (only for Type III)
        corrected = prediction.copy()
        corrections_applied = []

        if ttec_result["type"] == ErrorType.TYPE_III:
            type3_violations = {
                s: v for s, v in ttec_result.get("violations", {}).items()
                if v["type"] == ErrorType.TYPE_III
            }
            if type3_violations:
                corrected = self.topo_astar.correct(
                    prediction, topo_maps, vacuity,
                    type3_violations, self.config.ttec.betti_targets
                )
                corrections_applied = list(type3_violations.keys())

        # Alpha for MC sampling
        alpha = expert_outputs[0]["alpha"][0].cpu().numpy() if "alpha" in expert_outputs[0] else None

        return {
            "prediction": prediction,
            "corrected": corrected,
            "p_bar": p_bar,
            "vacuity": vacuity,
            "dissonance": dissonance,
            "u_sys_epistemic": u_sys,
            "ttec_result": ttec_result,
            "corrections_applied": corrections_applied,
            "weights": weights[0].cpu().numpy(),
            "alpha": alpha,
            "z_norm": z_norm,
            "phase": phase,
        }

    def predict_patient(self, slices: List[Dict]) -> Dict:
        """Full inference for a patient (multiple slices, ED+ES).

        Args:
            slices: list of dicts with 'image', 'z_norm', 'phase', 'mask' (optional)

        Returns:
            Dict with per-slice results, clinical indices, and uncertainties.
        """
        results = []
        for s in slices:
            image = s["image"].unsqueeze(0) if s["image"].dim() == 3 else s["image"]
            result = self.predict_single(image, s["z_norm"], s["phase"])
            result["gt_mask"] = s.get("mask", None)
            results.append(result)

        # Inter-slice consistency check
        ttec_results = [r["ttec_result"] for r in results]
        ttec_results = self.ttec.inter_slice_consistency_check(ttec_results)
        for i, r in enumerate(results):
            r["ttec_result"] = ttec_results[i]

        # Clinical indices
        clinical = self._compute_clinical_indices(results)

        return {
            "slices": results,
            "clinical": clinical,
        }

    def _compute_clinical_indices(self, results: List[Dict]) -> Dict:
        """Compute EF, mass, and their uncertainties from per-slice results."""
        ed_slices = [r for r in results if r["phase"] == "ED"]
        es_slices = [r for r in results if r["phase"] == "ES"]

        # Sum areas across slices (using corrected predictions)
        area_lv_ed = sum((r["corrected"] == 1).sum() for r in ed_slices)
        area_lv_es = sum((r["corrected"] == 1).sum() for r in es_slices)
        area_myo_ed = sum((r["corrected"] == 2).sum() for r in ed_slices)

        ef = compute_ef(area_lv_ed, area_lv_es)

        # Analytic uncertainty
        sigma2_ed = sum(
            self.clinical_propagator.analytic_area_uncertainty(
                r["p_bar"], r["vacuity"], r["dissonance"], 1)
            for r in ed_slices
        )
        sigma2_es = sum(
            self.clinical_propagator.analytic_area_uncertainty(
                r["p_bar"], r["vacuity"], r["dissonance"], 1)
            for r in es_slices
        )
        sigma2_ef = self.clinical_propagator.analytic_ef_uncertainty(
            sigma2_ed, sigma2_es, area_lv_ed, area_lv_es)
        sigma_ef_analytic = np.sqrt(max(sigma2_ef, 0))

        # Flag if CI spans diagnostic threshold
        ef_lower = ef - 1.96 * sigma_ef_analytic
        ef_upper = ef + 1.96 * sigma_ef_analytic
        threshold = self.config.inference.ef_threshold
        spans_threshold = ef_lower < threshold < ef_upper

        return {
            "ef": ef,
            "area_lv_ed": area_lv_ed,
            "area_lv_es": area_lv_es,
            "area_myo_ed": area_myo_ed,
            "sigma_ef_analytic": sigma_ef_analytic,
            "ef_ci_95": (ef_lower, ef_upper),
            "spans_diagnostic_threshold": spans_threshold,
            "n_type1": sum(1 for r in results if r["ttec_result"]["type"] == ErrorType.TYPE_I),
            "n_type2": sum(1 for r in results if r["ttec_result"]["type"] == ErrorType.TYPE_II),
            "n_type3": sum(1 for r in results if r["ttec_result"]["type"] == ErrorType.TYPE_III),
            "corrections_applied": sum(len(r["corrections_applied"]) for r in results),
        }

    def evaluate_dataset(self, dataloader, verbose: bool = True) -> Dict:
        """Evaluate on full dataset. Report Dice, HD95, TER, clinical metrics."""
        all_dice = {s: [] for s in ["LV", "Myo", "RV"]}
        all_hd95 = {s: [] for s in ["LV", "Myo", "RV"]}
        all_preds = []
        all_preds_corrected = []
        struct_map = {"LV": 1, "Myo": 2, "RV": 3}

        for batch in tqdm(dataloader, desc="Evaluating", disable=not verbose):
            images = batch["image"]
            masks = batch["mask"].numpy()
            z_norms = batch["z_norm"].numpy()
            phases = batch["phase"]

            for b in range(images.shape[0]):
                result = self.predict_single(
                    images[b:b+1], z_norms[b], phases[b])

                pred = result["corrected"]
                gt = masks[b]
                all_preds.append(result["prediction"])
                all_preds_corrected.append(pred)

                for name, idx in struct_map.items():
                    all_dice[name].append(dice_score(pred, gt, idx))
                    hd = hausdorff_95(pred, gt, idx)
                    if hd != float('inf'):
                        all_hd95[name].append(hd)

        # TER
        ter_before = topology_error_rate(all_preds)
        ter_after = topology_error_rate(all_preds_corrected)

        results = {
            "dice": {s: np.mean(v) for s, v in all_dice.items()},
            "dice_std": {s: np.std(v) for s, v in all_dice.items()},
            "hd95": {s: np.mean(v) if v else float('inf') for s, v in all_hd95.items()},
            "ter_before_correction": ter_before,
            "ter_after_correction": ter_after,
            "n_samples": len(all_preds),
        }

        if verbose:
            print("\n" + "=" * 50)
            print("Evaluation Results")
            print("=" * 50)
            for s in ["LV", "Myo", "RV"]:
                print(f"  {s}: Dice={results['dice'][s]:.4f}±{results['dice_std'][s]:.4f} "
                      f"HD95={results['hd95'][s]:.2f}mm")
            print(f"  TER (before): {ter_before:.1f}%")
            print(f"  TER (after):  {ter_after:.1f}%")

        return results
