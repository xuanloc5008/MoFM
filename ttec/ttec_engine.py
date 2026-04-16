"""
ttec/ttec_engine.py
Tripartite Topology-Error Classification with Learned Soft Gating.
  - Signal computation: TU, PLR, TTSS
  - G_TTEC: 2-layer MLP with learnable temperature
  - Topo-A* correction for Type-III violations
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import heapq
import gudhi

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Persistence Diagram utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_pd_from_hidden(h: torch.Tensor) -> torch.Tensor:
    """
    Approximate 1D persistence diagram from hidden state vector.
    PD ≈ sorted(|h|) in descending order.
    h: (d,) tensor
    """
    return h.abs().sort(descending=True).values


def wasserstein_1d(pd1: torch.Tensor, pd2: torch.Tensor) -> float:
    """W_2 distance between two 1D diagrams of possibly different length."""
    N = min(len(pd1), len(pd2))
    if N == 0:
        return 0.0
    diff = pd1[:N] - pd2[:N]
    return (diff.pow(2).mean()).sqrt().item()


def compute_frechet_mean(pds: List[torch.Tensor]) -> torch.Tensor:
    """Fréchet mean of a list of 1D persistence diagrams."""
    if not pds:
        return torch.zeros(64)
    N = min(len(p) for p in pds)
    stacked = torch.stack([p[:N] for p in pds], dim=0)
    return stacked.mean(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Topological Uncertainty (TU) — global OOD signal
# ─────────────────────────────────────────────────────────────────────────────

class TopologicalUncertaintyScorer:
    """
    Computes TU(x, F) = (1/L) Σ_ℓ W₂(PD_ℓ(x,F), FM^train_ℓ)
    from Mamba hidden states.
    Maintains per-layer Fréchet means from training set.
    """
    def __init__(self):
        self.frechet_means: Dict[int, torch.Tensor] = {}
        self._buffers: Dict[int, List[torch.Tensor]] = {}

    def update_from_batch(
        self,
        hidden_states: List[torch.Tensor],   # L × (B, d)
        predicted_class: Optional[int] = None,
    ):
        """Accumulate training hidden states per layer."""
        for layer_idx, h in enumerate(hidden_states):
            if layer_idx not in self._buffers:
                self._buffers[layer_idx] = []
            # Store mean hidden per sample in batch
            for b in range(h.shape[0]):
                pd = compute_pd_from_hidden(h[b].detach().cpu())
                self._buffers[layer_idx].append(pd)

    def finalise(self):
        """Compute Fréchet means from accumulated buffers."""
        for layer_idx, pds in self._buffers.items():
            self.frechet_means[layer_idx] = compute_frechet_mean(pds)
        self._buffers.clear()

    def compute_tu(
        self, hidden_states: List[torch.Tensor]
    ) -> float:
        """Compute TU for a single sample's hidden states."""
        if not self.frechet_means:
            return 0.0
        L = len(hidden_states)
        total = 0.0
        count = 0
        for layer_idx, h in enumerate(hidden_states):
            if layer_idx not in self.frechet_means:
                continue
            pd_test = compute_pd_from_hidden(h[0].detach().cpu())
            fm = self.frechet_means[layer_idx]
            total += wasserstein_1d(pd_test, fm)
            count += 1
        return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# PLR — Persistence Lifetime Ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_plr(
    prob_volume: np.ndarray,
    struct_id: int,
    resolution: Tuple[int, ...] = (32, 32, 8),
) -> float:
    """
    PLR(f) = ℓ_violating / median(ℓ_stable)
    Computed from the 3D persistence diagram of the probability map.
    """
    try:
        import skimage.transform as skT
        vol = prob_volume[struct_id]   # (H,W,D)
        if vol.shape != resolution:
            zoom = tuple(r / s for r, s in zip(resolution, vol.shape))
            import scipy.ndimage as nd
            vol = nd.zoom(vol, zoom, order=1)

        inv = 1.0 - vol
        cc = gudhi.CubicalComplex(
            dimensions=list(inv.shape),
            top_dimensional_cells=inv.flatten().tolist()
        )
        cc.compute_persistence()
        pd = cc.persistence()

        # Separate violating (dim=1, loops) from stable (dim=0, components)
        lifetimes_stable    = [abs(d - b) for (dim, (b, d)) in pd
                               if dim == 0 and d != float("inf")]
        lifetimes_violating = [abs(d - b) for (dim, (b, d)) in pd
                               if dim == 1]

        if not lifetimes_stable or not lifetimes_violating:
            return 1.0
        med_stable = np.median(lifetimes_stable)
        if med_stable < 1e-8:
            return 1.0
        return float(np.mean(lifetimes_violating) / med_stable)

    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TTSS — Temporal Trajectory Smoothness Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_ttss(
    betti_current: Tuple[int, int, int],
    betti_prev: Tuple[int, int, int],
    u_sys: float,
) -> float:
    """
    TTSS(t) = ||β(t) - β(t-1)||₁ × (1 - ū_e^sys(t))
    High TTSS = abrupt confident jump = likely genuine error.
    """
    delta = sum(abs(c - p) for c, p in zip(betti_current, betti_prev))
    return float(delta * (1.0 - u_sys))


# ─────────────────────────────────────────────────────────────────────────────
# G_TTEC: Learned Soft Gating Network
# ─────────────────────────────────────────────────────────────────────────────

class GTTECGating(nn.Module):
    """
    2-layer MLP with learnable temperature.
    Input:  f_s = [TU, PLR, TTSS, u_sys, Vac_s, Diss_s, β_viol] ∈ R^7
    Output: q_s = softmax(MLP(f_s)/τ) ∈ Δ²
    """
    FEATURE_DIM = 7
    NUM_TYPES    = 3

    def __init__(self, hidden_dim: int = None):
        super().__init__()
        h = hidden_dim or cfg.ttec.gate_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, self.NUM_TYPES),
        )
        # Learnable temperature
        self.log_tau = nn.Parameter(
            torch.log(torch.tensor(cfg.ttec.gate_temperature))
        )
        # Action-policy thresholds (clinically motivated, NOT learned)
        self.theta_1 = cfg.ttec.theta_type1
        self.theta_2 = cfg.ttec.theta_type2

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(0.01, 10.0)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f: (N, 7) → q: (N, 3)"""
        logits = self.net(f)
        return F.softmax(logits / self.tau, dim=-1)

    def classify(self, f: torch.Tensor) -> List[int]:
        """
        Apply action policy (Eq. action_policy in paper).
        Returns list of type indices: 0=Type-I, 1=Type-II, 2=Type-III
        """
        q = self.forward(f)   # (N, 3)
        types = []
        for qi in q:
            if qi[0].item() > self.theta_1:
                types.append(0)   # Type I
            elif qi[1].item() > self.theta_2:
                types.append(1)   # Type II
            else:
                types.append(2)   # Type III
        return types

    def build_feature_vector(
        self,
        tu: float,
        plr: float,
        ttss: float,
        u_sys: float,
        mean_vac: float,
        mean_diss: float,
        beta_viol_dim: int,   # which Betti dimension is violated
    ) -> torch.Tensor:
        """Assemble and normalise feature vector f_s ∈ R^7."""
        f = torch.tensor([
            tu, plr, ttss, u_sys,
            mean_vac, mean_diss,
            float(beta_viol_dim) / 2.0,   # normalise to [0,1]
        ], dtype=torch.float32)
        return f.unsqueeze(0)   # (1, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Betti Number Checker for 3D volumes
# ─────────────────────────────────────────────────────────────────────────────

def check_3d_betti(
    prob_volume: np.ndarray,   # (K, H, W, D)
    resolution: Tuple[int, ...] = (32, 32, 8),
) -> Dict[str, Tuple[int, int, int]]:
    """
    Compute 3D Betti numbers for each structure.
    Returns dict: struct_name → (β0, β1, β2)
    """
    struct_map = {1: "LV", 2: "Myo", 3: "RV"}
    result = {}
    for sid, sname in struct_map.items():
        vol = prob_volume[sid]
        import scipy.ndimage as nd
        if vol.shape != resolution:
            zoom = tuple(r / s for r, s in zip(resolution, vol.shape))
            vol = nd.zoom(vol, zoom, order=1)
        inv = 1.0 - vol
        try:
            cc = gudhi.CubicalComplex(
                dimensions=list(inv.shape),
                top_dimensional_cells=inv.flatten().tolist()
            )
            cc.compute_persistence()
            betti = cc.betti_numbers()
            b0 = betti[0] if len(betti) > 0 else 0
            b1 = betti[1] if len(betti) > 1 else 0
            b2 = betti[2] if len(betti) > 2 else 0
        except Exception:
            b0, b1, b2 = 1, 0, 0
        result[sname] = (b0, b1, b2)
    return result


BETTI_TARGETS = {
    "LV":  (1, 0, 0),
    "Myo": (1, 1, 0),
    "RV":  (1, 0, 0),
}


def find_betti_violations(
    betti_dict: Dict[str, Tuple[int, int, int]]
) -> Dict[str, List[int]]:
    """
    Returns dict: struct_name → list of violated Betti dimensions
    e.g. {"Myo": [1]} means β₁ is wrong for Myo
    """
    violations = {}
    for s, betti in betti_dict.items():
        target = BETTI_TARGETS.get(s)
        if target is None:
            continue
        viol_dims = [k for k in range(3) if betti[k] != target[k]]
        if viol_dims:
            violations[s] = viol_dims
    return violations


# ─────────────────────────────────────────────────────────────────────────────
# Topo-A* Correction for Type-III violations
# ─────────────────────────────────────────────────────────────────────────────

class TopoAStarCorrector:
    """
    Gradient-free Topo-A* correction for Type-III violations.
    Operates on probability maps without modifying model weights.
    """

    def __init__(
        self,
        lambda_topo: float = None,
        vac_thresh:  float = None,
        prob_thresh: float = None,
    ):
        self.lambda_topo = lambda_topo or cfg.ttec.astar_lambda_topo
        self.vac_thresh  = vac_thresh  or cfg.ttec.anchor_vac_thresh
        self.prob_thresh = prob_thresh or cfg.ttec.anchor_prob_thresh

    def correct(
        self,
        p_hat: torch.Tensor,     # (K, H, W, D) soft probabilities
        vacuity: torch.Tensor,   # (H, W, D)
        structure: str,          # "LV", "Myo", "RV"
    ) -> torch.Tensor:
        """
        Attempt to correct a Type-III topological violation.
        Returns corrected p_hat (same shape) or original if correction fails.
        """
        struct_id = {"LV": 1, "Myo": 2, "RV": 3}[structure]
        target = BETTI_TARGETS[structure]

        # Compute anchors: high-confidence, low-vacuity voxels
        anchors = (
            (vacuity < self.vac_thresh) &
            (p_hat[struct_id] > self.prob_thresh)
        )

        # Structure-specific correction strategy
        if structure == "Myo":
            corrected = self._bridge_ruptured_ring(
                p_hat, struct_id, anchors
            )
        elif structure == "LV":
            corrected = self._flood_fill_solid(
                p_hat, struct_id, anchors
            )
        else:   # RV
            corrected = self._reconnect_fragments(
                p_hat, struct_id, anchors
            )

        # Verify correction achieved target topology
        prob_full = corrected.cpu().numpy()   # (K, H, W, D)
        betti_after = check_3d_betti(prob_full)
        if betti_after.get(structure, (0, 0, 0)) == target:
            return corrected
        else:
            # Correction failed: return original with flag
            return p_hat   # caller should add low-confidence flag

    def _bridge_ruptured_ring(
        self,
        p_hat: torch.Tensor,
        struct_id: int,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bridge gaps in a ruptured Myo ring by boosting probability
        along shortest path between disconnected anchor regions.
        """
        import scipy.ndimage as nd

        corrected = p_hat.clone()
        prob_np   = p_hat[struct_id].cpu().numpy()
        anchor_np = anchors.cpu().numpy()

        # Label connected components of anchors
        labeled, n_comp = nd.label(anchor_np)
        if n_comp < 2:
            return corrected

        # Find centroids of two largest components
        comp_sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
        top2 = sorted(range(n_comp), key=lambda i: -comp_sizes[i])[:2]
        c1 = np.array(nd.center_of_mass(anchor_np, labeled, top2[0] + 1),
                      dtype=int)
        c2 = np.array(nd.center_of_mass(anchor_np, labeled, top2[1] + 1),
                      dtype=int)

        # Draw a line between centroids and boost probability
        steps = max(abs(c2 - c1))
        if steps == 0:
            return corrected
        for t in np.linspace(0, 1, steps * 2):
            pt = (c1 + t * (c2 - c1)).astype(int)
            pt = np.clip(pt, 0, np.array(prob_np.shape) - 1)
            corrected[struct_id, pt[0], pt[1], pt[2]] = max(
                corrected[struct_id, pt[0], pt[1], pt[2]].item(), 0.8
            )
        return corrected

    def _flood_fill_solid(
        self,
        p_hat: torch.Tensor,
        struct_id: int,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Morphological fill to make LV a solid disc."""
        import scipy.ndimage as nd
        corrected = p_hat.clone()
        prob_np   = (p_hat[struct_id] > 0.5).cpu().numpy()
        # Fill holes
        filled = nd.binary_fill_holes(prob_np)
        diff = torch.tensor(filled.astype(np.float32), device=p_hat.device)
        corrected[struct_id] = torch.maximum(corrected[struct_id], diff * 0.6)
        return corrected

    def _reconnect_fragments(
        self,
        p_hat: torch.Tensor,
        struct_id: int,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Reconnect RV fragments via bridge (same as Myo)."""
        return self._bridge_ruptured_ring(p_hat, struct_id, anchors)


# ─────────────────────────────────────────────────────────────────────────────
# Full TTEC Engine
# ─────────────────────────────────────────────────────────────────────────────

class TTECEngine:
    """
    Full TTEC inference pipeline.
    Combines TU, PLR, TTSS → G_TTEC → Topo-A* correction.
    """

    def __init__(
        self,
        gating: GTTECGating,
        tu_scorer: TopologicalUncertaintyScorer,
        corrector: TopoAStarCorrector,
    ):
        self.gating    = gating
        self.tu_scorer = tu_scorer
        self.corrector = corrector
        self._prev_betti: Optional[Dict] = None

    @torch.no_grad()
    def run(
        self,
        p_hat: torch.Tensor,         # (K, H, W, D)
        vacuity: torch.Tensor,       # (H, W, D)
        dissonance: torch.Tensor,    # (H, W, D)
        hidden_states: List[torch.Tensor],
        u_sys: float,
        phase: str = "ED",           # cardiac phase
    ) -> Dict:
        """
        Full TTEC pipeline for one volume.
        Returns:
          - corrected_p_hat
          - violation_types: Dict[struct → "I"/"II"/"III"/"None"]
          - ttec_log: detailed signals per violation
        """
        prob_np = p_hat.cpu().numpy()

        # 1. Compute Betti violations
        betti = check_3d_betti(
            np.stack([np.zeros_like(prob_np[0])] + [prob_np[i]
                     for i in range(1, 4)], axis=0),
        )
        violations = find_betti_violations(betti)

        if not violations:
            return {
                "corrected_p_hat": p_hat,
                "violation_types": {},
                "ttec_log": {},
                "betti": betti,
            }

        # 2. Global TU
        tu = self.tu_scorer.compute_tu(hidden_states)

        # 3. Per-violation classification
        violation_types = {}
        ttec_log = {}
        corrected = p_hat.clone()

        for struct_name, viol_dims in violations.items():
            struct_id = {"LV": 1, "Myo": 2, "RV": 3}[struct_name]
            plr   = compute_plr(prob_np, struct_id)
            ttss  = compute_ttss(
                betti[struct_name],
                BETTI_TARGETS[struct_name] if self._prev_betti is None
                else self._prev_betti.get(struct_name, BETTI_TARGETS[struct_name]),
                u_sys,
            )
            # Region stats
            mask = vacuity[prob_np[struct_id] > 0.3]
            mean_vac  = mask.mean().item() if len(mask) > 0 else 0.5
            d_mask    = dissonance[prob_np[struct_id] > 0.3]
            mean_diss = d_mask.mean().item() if len(d_mask) > 0 else 0.5

            # Build feature vector
            f = self.gating.build_feature_vector(
                tu, plr, ttss, u_sys,
                mean_vac, mean_diss,
                viol_dims[0],   # primary violated Betti dim
            )
            violation_type_idx = self.gating.classify(f.to(
                next(self.gating.parameters()).device
                if list(self.gating.parameters()) else torch.device("cpu")
            ))[0]
            type_str = ["I", "II", "III"][violation_type_idx]
            violation_types[struct_name] = type_str

            ttec_log[struct_name] = {
                "tu": tu, "plr": plr, "ttss": ttss,
                "u_sys": u_sys, "vac": mean_vac, "diss": mean_diss,
                "type": type_str,
                "betti_found": betti[struct_name],
                "betti_target": BETTI_TARGETS[struct_name],
            }

            # 4. Type-III: activate Topo-A*
            if type_str == "III":
                corrected = self.corrector.correct(
                    corrected, vacuity, struct_name
                )

        self._prev_betti = betti
        return {
            "corrected_p_hat": corrected,
            "violation_types": violation_types,
            "ttec_log": ttec_log,
            "betti": betti,
            "tu": tu,
        }
