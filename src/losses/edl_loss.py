"""
Evidential Deep Learning (EDL) Loss — Pixel-Level
---------------------------------------------------
Based on: Sensoy et al. "Evidential Deep Learning to Quantify
          Classification Uncertainty" (NeurIPS 2018)

For each pixel p with K classes:
  - Network outputs evidence  e_p ≥ 0  (K-vector, after ReLU)
  - Dirichlet params:   α_p,k = e_p,k + 1
  - Strength:           S_p = Σ α_p,k
  - Expected prob:      p̂_p,k = α_p,k / S_p
  - Uncertainty:        u_p = K / S_p

L_EDL(p) = L_MSE(p) + L_Var(p) + λ_t * L_KL(p)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def evidence_to_dirichlet(
    evidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert raw evidence (B, K, H, W) to Dirichlet parameters.

    Returns:
        alpha : (B, K, H, W)  α_k = e_k + 1
        S     : (B, 1, H, W)  S = Σ_k α_k
        probs : (B, K, H, W)  p̂_k = α_k / S  (expected probabilities)
    """
    alpha = evidence + 1.0
    S     = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    return alpha, S, probs


def uncertainty_map(evidence: torch.Tensor) -> torch.Tensor:
    """
    u_p = K / S_p  ∈ (0, 1].
    evidence: (B, K, H, W)
    returns: (B, 1, H, W)
    """
    K = evidence.shape[1]
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    return K / S


def mse_loss_edl(
    probs:  torch.Tensor,   # (B, K, H, W) expected probabilities
    target: torch.Tensor,   # (B, H, W)   integer class labels
    alpha:  torch.Tensor,   # (B, K, H, W)
    S:      torch.Tensor,   # (B, 1, H, W)
) -> torch.Tensor:
    """
    L_MSE + L_Var for EDL.
    L_MSE(p) = Σ_k (y_k - p̂_k)^2
    L_Var(p) = Σ_k p̂_k(1 - p̂_k) / (S + 1)
    """
    K = probs.shape[1]
    # One-hot encode target: (B, K, H, W)
    y = F.one_hot(target, num_classes=K).permute(0, 3, 1, 2).float()

    # MSE term
    mse = ((y - probs) ** 2).sum(dim=1)    # (B, H, W)

    # Variance term — encourages high total evidence on correct pixels
    # S: (B, 1, H, W)  →  keep for broadcast with (B, K, H, W)
    var = (probs * (1 - probs) / (S + 1)).sum(dim=1)   # (B, H, W)

    return (mse + var).mean()


def kl_divergence_edl(
    alpha:  torch.Tensor,   # (B, K, H, W)
    target: torch.Tensor,   # (B, H, W)
) -> torch.Tensor:
    """
    KL[Dir(α̃) || Dir(1,...,1)]
    α̃_k = y_k + (1 - y_k) * α_k   (zero evidence on wrong classes)

    Penalises evidence on incorrect classes after a wrong prediction.
    """
    K = alpha.shape[1]
    y = F.one_hot(target, num_classes=K).permute(0, 3, 1, 2).float()

    # Adjusted alpha: zero out evidence on wrong classes
    alpha_hat = y + (1 - y) * alpha                   # (B, K, H, W)
    S_hat     = alpha_hat.sum(dim=1, keepdim=True)    # (B, 1, H, W)

    # KL(Dir(α̃) || Dir(1,...,1))
    # = lgamma(S_hat) - Σ lgamma(α̃_k) - lgamma(K)
    #   + Σ (α̃_k - 1) * [digamma(α̃_k) - digamma(S_hat)]
    kl = (
        torch.lgamma(S_hat.squeeze(1))
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.lgamma(alpha_hat).sum(dim=1)
        + ((alpha_hat - 1) * (
            torch.digamma(alpha_hat)
            - torch.digamma(S_hat)
        )).sum(dim=1)
    )   # (B, H, W)

    return kl.mean()


class EDLLoss(nn.Module):
    """
    Full pixel-wise EDL loss.

    L_EDL = L_MSE_Var + λ_t * L_KL

    λ_t is annealed from 0 → 1 over training to allow the network
    to first learn to predict correctly before being penalised for
    generating evidence on wrong classes.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        evidence: torch.Tensor,   # (B, K, H, W)  raw ReLU evidence
        target:   torch.Tensor,   # (B, H, W)     integer labels
        lambda_t: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns total loss and component dict for logging.
        """
        alpha, S, probs = evidence_to_dirichlet(evidence)

        l_mse_var = mse_loss_edl(probs, target, alpha, S)
        l_kl      = kl_divergence_edl(alpha, target)

        loss = l_mse_var + lambda_t * l_kl

        return loss, {
            "edl_mse_var": l_mse_var.item(),
            "edl_kl":      l_kl.item(),
            "edl_total":   loss.item(),
        }


def get_lambda_t(epoch: int, start: float, end: float, warmup_epochs: int) -> float:
    """
    Linear annealing schedule for λ_t.
    0 → warmup_epochs: λ_t = start + (end - start) * epoch / warmup_epochs
    After warmup:      λ_t = end
    """
    if epoch >= warmup_epochs:
        return end
    return start + (end - start) * epoch / warmup_epochs
