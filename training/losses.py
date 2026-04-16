"""
training/losses.py
All loss functions for the DUT-Mamba-MoE framework.
  - EDL loss (MSE Bayes risk + KL regularisation)
  - 3D Betti loss via GUDHI
  - Topological Contrastive Learning (TCL)
  - Routing loss
  - Bayesian homoscedastic weighting
  - G_TTEC training loss
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

import numpy as np
import gudhi

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# EDL Loss (Eq. 5 from Sensoy et al. + KL regularisation)
# ─────────────────────────────────────────────────────────────────────────────

class EDLLoss(nn.Module):
    """
    Bayes risk w.r.t. L2/MSE loss + KL divergence regularisation.
    L_i = Σ_j [(y_ij - p̂_ij)² + p̂_ij(1-p̂_ij)/(S+1)]
          + λ_t * KL[Dir(α̃_i) || Dir(1)]
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        alpha: torch.Tensor,    # (B, K, H, W, D)
        y_one_hot: torch.Tensor,  # (B, K, H, W, D) float
        epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        S = alpha.sum(dim=1, keepdim=True)
        p_hat = alpha / S

        # MSE prediction error
        err = (y_one_hot - p_hat).pow(2)

        # Variance term
        var = p_hat * (1 - p_hat) / (S + 1)

        mse_loss = (err + var).sum(dim=1).mean()

        # KL regularisation: anneal coefficient λ_t
        lam_t = min(1.0, epoch / max(cfg.training.kl_anneal_epochs, 1))
        # Remove non-misleading evidence: α̃ = y + (1-y)*α
        alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha
        kl = self._kl_dirichlet(alpha_tilde)

        return mse_loss + lam_t * kl

    @staticmethod
    def _kl_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
        """
        KL[Dir(α) || Dir(1)]
        Eq. from Sensoy et al. (2018)
        """
        K = alpha.shape[1]
        S = alpha.sum(dim=1, keepdim=True)
        # log B(1) / B(α)
        log_B_ratio = (
            torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        )
        # Σ (α_k - 1) * [ψ(α_k) - ψ(S)]
        psi_term = (
            (alpha - 1) *
            (torch.digamma(alpha) - torch.digamma(S))
        ).sum(dim=1, keepdim=True)
        kl = (log_B_ratio + psi_term).mean()
        return kl.clamp(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3D Betti Loss via GUDHI Cubical Complex
# ─────────────────────────────────────────────────────────────────────────────

class Betti3DLoss(nn.Module):
    """
    L_Topo3D = Σ_{k∈{0,1,2}} |β_k(p̂^(m)) - β_k^target|
    Computed on downsampled prediction for efficiency.
    Uses GUDHI CubicalComplex (not differentiable directly;
    we use it as a structured penalty on the MSE loss).
    For differentiable PH, TorchPH can be substituted.
    """

    # 3D Betti targets from paper
    TARGETS = {
        "LV":  (1, 0, 0),
        "Myo": (1, 1, 0),
        "RV":  (1, 0, 0),
        "BG":  (1, 0, 0),
    }
    STRUCT_IDS = {"BG": 0, "LV": 1, "Myo": 2, "RV": 3}

    def __init__(self, resolution: Tuple[int, ...] = (32, 32, 8)):
        super().__init__()
        self.resolution = resolution

    def forward(
        self,
        p_hat: torch.Tensor,   # (B, K, H, W, D) soft probabilities
        structure: str,        # "LV", "Myo", "RV"
    ) -> torch.Tensor:
        struct_id = self.STRUCT_IDS[structure]
        target = self.TARGETS[structure]

        # Downsample for efficiency
        p_down = F.interpolate(
            p_hat, size=self.resolution,
            mode="trilinear", align_corners=False,
        )
        p_struct = p_down[:, struct_id]   # (B, H, W, D)

        total_loss = torch.tensor(0.0, device=p_hat.device,
                                  requires_grad=True)
        B = p_struct.shape[0]
        for b in range(B):
            scalar_field = p_struct[b].detach().cpu().numpy()
            betti = self._compute_3d_betti(scalar_field)
            betti_error = sum(
                abs(betti[k] - target[k]) for k in range(3)
            )
            if betti_error > 0:
                # Gradient proxy: penalise low-confidence predictions
                # in regions that likely cause the Betti violation
                confidence = p_struct[b].max()
                total_loss = total_loss + (1.0 - confidence) * betti_error

        return total_loss / max(B, 1)

    @staticmethod
    def _compute_3d_betti(
        scalar_field: np.ndarray
    ) -> Tuple[int, int, int]:
        """Compute (β0, β1, β2) via GUDHI 3D Cubical Complex."""
        try:
            # Superlevel: invert
            inv_field = 1.0 - scalar_field
            cc = gudhi.CubicalComplex(
                dimensions=list(scalar_field.shape),
                top_dimensional_cells=inv_field.flatten().tolist()
            )
            cc.compute_persistence()
            betti = cc.betti_numbers()
            b0 = betti[0] if len(betti) > 0 else 0
            b1 = betti[1] if len(betti) > 1 else 0
            b2 = betti[2] if len(betti) > 2 else 0
            return b0, b1, b2
        except Exception:
            return 1, 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# Topological Contrastive Learning (TCL) in latent space
# ─────────────────────────────────────────────────────────────────────────────

class TopologicalContrastiveLoss(nn.Module):
    """
    L_TCL = W2(PD(h), PD(h'))
    Wasserstein distance between persistence diagrams of
    hidden states from original and augmented volumes.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: List[torch.Tensor],         # (B, d)  per layer
        hidden_states_aug: List[torch.Tensor],     # (B, d)  per layer
    ) -> torch.Tensor:
        total = torch.tensor(0.0,
                             device=hidden_states[0].device,
                             requires_grad=True)
        for h, h_aug in zip(hidden_states, hidden_states_aug):
            pd1 = self._hidden_to_pd(h)   # (B, N_pts)
            pd2 = self._hidden_to_pd(h_aug)
            w2  = self._wasserstein_1d(pd1, pd2)
            total = total + w2
        return total / max(len(hidden_states), 1)

    @staticmethod
    def _hidden_to_pd(h: torch.Tensor) -> torch.Tensor:
        """
        Approximate 1D persistence diagram from hidden state vector.
        Sort the values: PD ≈ sorted(|h|) (superlevel filtration).
        """
        pd = h.abs().sort(dim=-1, descending=True).values
        return pd

    @staticmethod
    def _wasserstein_1d(pd1: torch.Tensor, pd2: torch.Tensor) -> torch.Tensor:
        """
        1D Wasserstein-2 distance between sorted diagrams.
        Dist²(μ, ν) = (1/N) Σ |w_i - w'_i|²
        """
        N = min(pd1.shape[-1], pd2.shape[-1])
        diff = pd1[..., :N] - pd2[..., :N]
        return (diff.pow(2).mean(dim=-1)).sqrt().mean()


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Loss: Dice + Boundary CE + Hausdorff
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        from monai.losses import DiceLoss, HausdorffDTLoss
        self.dice = DiceLoss(
            include_background=False, to_onehot_y=True,
            softmax=True, reduction="mean"
        )
        self.ce = nn.CrossEntropyLoss()
        try:
            self.hd = HausdorffDTLoss(
                include_background=False, to_onehot_y=True,
                softmax=True, reduction="mean"
            )
            self.use_hd = True
        except Exception:
            self.use_hd = False
        self.w_dice = 1.0
        self.w_ce   = 1.0
        self.w_hd   = 0.5

    def forward(
        self,
        logits: torch.Tensor,     # (B, K, H, W, D)
        label: torch.Tensor,      # (B, H, W, D) long
    ) -> torch.Tensor:
        label_4d = label.unsqueeze(1)   # (B, 1, H, W, D)
        loss = (self.w_dice * self.dice(logits, label_4d) +
                self.w_ce   * self.ce(logits, label))
        if self.use_hd:
            try:
                loss = loss + self.w_hd * self.hd(logits, label_4d)
            except Exception:
                pass
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Routing Loss
# ─────────────────────────────────────────────────────────────────────────────

class RoutingLoss(nn.Module):
    """
    L_route = (1/N) Σ_v Σ_m w_m(v) [Vac_m(v) + L_Topo_m(v)]
    Penalises routing to uncertain or topologically unreliable experts.
    """
    def forward(
        self,
        weights: torch.Tensor,           # (B, M, H, W, D)
        vacuities: List[torch.Tensor],   # M × (B, H, W, D)
        topo_losses: List[float],        # scalar per expert
    ) -> torch.Tensor:
        M = weights.shape[1]
        total = torch.tensor(0.0, device=weights.device, requires_grad=True)
        for m in range(M):
            vac_m = vacuities[m]             # (B, H, W, D)
            topo_m = topo_losses[m]
            penalty = vac_m + topo_m         # broadcast scalar
            total = total + (weights[:, m] * penalty).mean()
        return total / M


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Homoscedastic Loss (Eq. 13)
# ─────────────────────────────────────────────────────────────────────────────

class HomoscedasticLoss(nn.Module):
    """
    L_expert = Σ_i [1/(2σ_i²) * L_i + log(σ_i)]
    log_sigma2: (4,) learnable, indices [EDL, Topo3D, TCL, Seg]
    """
    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        log_sigma2: torch.Tensor,   # (4,)
    ) -> torch.Tensor:
        keys = ["edl", "topo", "tcl", "seg"]
        total = torch.tensor(0.0,
                             device=log_sigma2.device,
                             requires_grad=True)
        for i, k in enumerate(keys):
            if k in losses:
                precision = torch.exp(-log_sigma2[i])   # 1/σ²
                total = total + (
                    0.5 * precision * losses[k] +
                    0.5 * log_sigma2[i]
                )
        return total


# ─────────────────────────────────────────────────────────────────────────────
# G_TTEC Training Loss (Learned soft gating)
# ─────────────────────────────────────────────────────────────────────────────

class GTTECLoss(nn.Module):
    """
    L_gate = -Σ_s c^T * y_s ⊙ log(q_s)
    Weighted cross-entropy reflecting clinical cost asymmetry.
    """
    def __init__(self):
        super().__init__()
        c = torch.tensor([
            cfg.ttec.cost_type1,
            cfg.ttec.cost_type2,
            cfg.ttec.cost_type3,
        ])
        self.register_buffer("class_weights", c)

    def forward(
        self,
        q: torch.Tensor,          # (N, 3) soft predictions
        y: torch.Tensor,          # (N,) long class labels {0,1,2}
    ) -> torch.Tensor:
        log_q = (q + 1e-8).log()
        y_oh = F.one_hot(y, num_classes=3).float()   # (N, 3)
        weighted = self.class_weights * y_oh * log_q  # (N, 3)
        return -weighted.sum(dim=-1).mean()
