"""
Tripartite Topology-Error Classification (TTEC).
3 signals: DSS, PLR, SPCS. No TTSS (would require full temporal sequence).
Includes inter-slice consistency check.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from sklearn.neighbors import NearestNeighbors

from data.topology import (
    compute_all_betti, check_betti_violations,
    compute_persistence_diagram, compute_plr
)


class ErrorType(Enum):
    NONE = "none"
    TYPE_I = "distribution_induced"
    TYPE_II = "acquisition_inherent"
    TYPE_III = "genuine_model_error"


class SPCSTable:
    """Slice-Position Context Score lookup table.
    Built from training annotations: P(inherent | z_norm, phase, structure).
    """

    def __init__(self):
        self.table = {}
        self._build_default()

    def _build_default(self):
        """Default SPCS based on known cardiac anatomy."""
        # Myo ring is open at basal (z<0.2) and apical (z>0.85)
        for phase in ["ED", "ES"]:
            for z in np.arange(0, 1.01, 0.05):
                z_key = round(z, 2)
                # Myo: high SPCS at basal/apical
                if z_key < 0.20 or z_key > 0.85:
                    self.table[("Myo", z_key, phase)] = 0.85
                elif z_key < 0.30 or z_key > 0.75:
                    self.table[("Myo", z_key, phase)] = 0.4
                else:
                    self.table[("Myo", z_key, phase)] = 0.05
                # LV/RV: low SPCS everywhere (should always be connected)
                self.table[("LV", z_key, phase)] = 0.05
                self.table[("RV", z_key, phase)] = 0.1

    def build_from_annotations(self, annotations: List[Dict]):
        """Build SPCS from training data annotations.
        Each annotation: {structure, z_norm, phase, is_inherent}
        """
        from collections import defaultdict
        counts = defaultdict(lambda: [0, 0])  # [inherent_count, total_count]
        for ann in annotations:
            key = (ann["structure"],
                   round(ann["z_norm"] * 20) / 20,  # Quantise to 0.05
                   ann["phase"])
            counts[key][1] += 1
            if ann["is_inherent"]:
                counts[key][0] += 1

        for key, (inh, total) in counts.items():
            self.table[key] = inh / max(total, 1)

    def query(self, structure: str, z_norm: float, phase: str) -> float:
        z_key = round(z_norm * 20) / 20
        return self.table.get((structure, z_key, phase), 0.1)


class TTECClassifier:
    """Tripartite Topology-Error Classification engine."""

    def __init__(self, config, training_embeddings: np.ndarray = None):
        self.tau_dss = config.tau_dss
        self.tau_plr = config.tau_plr
        self.tau_spcs = config.tau_spcs
        self.alpha_dss = config.alpha_dss
        self.knn_k = config.knn_k
        self.betti_targets = config.betti_targets

        self.spcs_table = SPCSTable()
        self.knn = None

        if training_embeddings is not None:
            self.fit_knn(training_embeddings)

    def fit_knn(self, embeddings: np.ndarray):
        """Fit k-NN on training set embeddings for OOD distance."""
        self.knn = NearestNeighbors(n_neighbors=self.knn_k, metric='cosine')
        self.knn.fit(embeddings)

    def calibrate_thresholds(self, val_violations: List[Dict]):
        """Calibrate thresholds on a validation set of labelled violations.
        Each entry: {signals: {dss, plr, spcs}, label: ErrorType}
        """
        # Simple: find thresholds that maximise accuracy
        best_acc = 0
        best_thresholds = (self.tau_dss, self.tau_plr, self.tau_spcs)

        for tau_dss in np.arange(0.3, 0.9, 0.05):
            for tau_plr in np.arange(0.1, 0.6, 0.05):
                for tau_spcs in np.arange(0.3, 0.9, 0.05):
                    correct = 0
                    for v in val_violations:
                        pred = self._classify_with_thresholds(
                            v["signals"], tau_dss, tau_plr, tau_spcs)
                        if pred == v["label"]:
                            correct += 1
                    acc = correct / max(len(val_violations), 1)
                    if acc > best_acc:
                        best_acc = acc
                        best_thresholds = (tau_dss, tau_plr, tau_spcs)

        self.tau_dss, self.tau_plr, self.tau_spcs = best_thresholds
        return best_acc, best_thresholds

    # ---- Signal Computation ----

    def compute_dss(self, u_sys_epistemic_mean: float,
                    embedding: np.ndarray) -> float:
        """Signal 1: Distribution Shift Score."""
        d_ood = 0.0
        if self.knn is not None:
            distances, _ = self.knn.kneighbors(embedding.reshape(1, -1))
            d_ood = distances.mean()

        return self.alpha_dss * u_sys_epistemic_mean + (1 - self.alpha_dss) * d_ood

    def compute_plr(self, pred_prob_map: np.ndarray,
                    structure: str) -> float:
        """Signal 2: Persistence Lifetime Ratio."""
        pd = compute_persistence_diagram(None, pred_prob_map)
        if not pd:
            return 0.0
        return compute_plr(pd)

    def compute_spcs(self, z_norm: float, phase: str,
                     structure: str) -> float:
        """Signal 3: Slice-Position Context Score."""
        return self.spcs_table.query(structure, z_norm, phase)

    # ---- Classification ----

    def classify(self, prediction: np.ndarray, ensemble_output: Dict,
                 z_norm: float, phase: str) -> Dict:
        """Classify all topological violations in a prediction.

        Args:
            prediction: (H, W) integer segmentation
            ensemble_output: dict with u_sys_epistemic, embedding, topo_maps
            z_norm: normalised slice position [0, 1]
            phase: "ED" or "ES"

        Returns:
            Dict with violations and their classifications.
        """
        betti = compute_all_betti(prediction)
        violations = check_betti_violations(betti, self.betti_targets)

        if not violations:
            return {"type": ErrorType.NONE, "violations": {}}

        # Signal 1: DSS (image-level)
        u_sys_mean = ensemble_output.get("u_sys_epistemic_mean", 0.0)
        embedding = ensemble_output.get("embedding", np.zeros(128))
        dss = self.compute_dss(u_sys_mean, embedding)

        # Type I check
        if dss > self.tau_dss:
            return {
                "type": ErrorType.TYPE_I,
                "violations": violations,
                "signals": {"dss": dss},
                "action": "flag_for_human_review",
            }

        # Per-violation classification
        results = {}
        worst_type = ErrorType.NONE

        for structure, (actual, target) in violations.items():
            prob_map = ensemble_output.get("topo_maps", {}).get(structure, None)
            prob_np = prob_map if isinstance(prob_map, np.ndarray) else np.zeros((256, 256))

            plr = self.compute_plr(prob_np, structure)
            spcs = self.compute_spcs(z_norm, phase, structure)

            if plr < self.tau_plr or spcs > self.tau_spcs:
                etype = ErrorType.TYPE_II
                action = "accept_inherent"
            else:
                etype = ErrorType.TYPE_III
                action = "activate_topo_astar"

            results[structure] = {
                "type": etype,
                "actual_betti": actual,
                "target_betti": target,
                "signals": {"dss": dss, "plr": plr, "spcs": spcs},
                "action": action,
            }

            # Track worst type
            severity = {ErrorType.NONE: 0, ErrorType.TYPE_II: 1, ErrorType.TYPE_III: 2}
            if severity.get(etype, 0) > severity.get(worst_type, 0):
                worst_type = etype

        return {
            "type": worst_type,
            "violations": results,
            "signals": {"dss": dss},
        }

    def _classify_with_thresholds(self, signals, tau_dss, tau_plr, tau_spcs):
        if signals["dss"] > tau_dss:
            return ErrorType.TYPE_I
        if signals.get("plr", 1.0) < tau_plr or signals.get("spcs", 0.0) > tau_spcs:
            return ErrorType.TYPE_II
        return ErrorType.TYPE_III

    # ---- Inter-Slice Consistency ----

    def inter_slice_consistency_check(
            self, slice_results: List[Dict]) -> List[Dict]:
        """Check consistency of TTEC classifications across the short-axis stack.
        Flag slices where classification is inconsistent with neighbours.

        Args:
            slice_results: list of TTEC results ordered by z_norm
        Returns:
            Updated list with 'consistency_flag' added
        """
        n = len(slice_results)
        for i in range(n):
            slice_results[i]["consistency_flag"] = False

            if slice_results[i]["type"] == ErrorType.NONE:
                continue

            # Check neighbours
            for structure, viol in slice_results[i].get("violations", {}).items():
                vtype = viol["type"]

                # Get neighbour types for same structure
                neighbor_types = []
                for j in [i - 1, i + 1]:
                    if 0 <= j < n:
                        nv = slice_results[j].get("violations", {})
                        if structure in nv:
                            neighbor_types.append(nv[structure]["type"])

                # Flag if this slice disagrees with both neighbours
                if (len(neighbor_types) == 2
                        and neighbor_types[0] == neighbor_types[1]
                        and neighbor_types[0] != vtype):
                    slice_results[i]["consistency_flag"] = True

        return slice_results
