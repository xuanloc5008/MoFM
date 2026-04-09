"""
Loss functions: EDL, Aleatoric Attention, PD-Contrastive, Topology,
Structure losses with uncertainty weighting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


# ---- Structure Losses ----

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                class_idx: int = None) -> torch.Tensor:
        """pred: (B, K, H, W) probabilities. target: (B, H, W) integer labels."""
        if class_idx is not None:
            pred_c = pred[:, class_idx]
            target_c = (target == class_idx).float()
            intersection = (pred_c * target_c).sum(dim=(-2, -1))
            union = pred_c.sum(dim=(-2, -1)) + target_c.sum(dim=(-2, -1))
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()

        # Multi-class
        loss = 0.0
        K = pred.shape[1]
        for k in range(K):
            loss += self.forward(pred, target, k)
        return loss / K


class BoundaryAwareCE(nn.Module):
    """Cross-entropy with higher weight at structure boundaries."""

    def __init__(self, a: float = 5.0, sigma: float = 3.0):
        super().__init__()
        self.a = a
        self.sigma = sigma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (B, K, H, W). target: (B, H, W)."""
        # Compute boundary distance map
        boundary_weight = self._compute_boundary_weights(target)
        ce = F.cross_entropy(logits, target, reduction='none')  # (B, H, W)
        weighted_ce = ce * boundary_weight
        return weighted_ce.mean()

    def _compute_boundary_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary-aware pixel weights."""
        B, H, W = target.shape
        weights = torch.ones_like(target, dtype=torch.float32)

        # Detect boundaries via gradient of label map
        target_f = target.float().unsqueeze(1)  # (B, 1, H, W)
        dx = torch.abs(target_f[:, :, :, 1:] - target_f[:, :, :, :-1])
        dy = torch.abs(target_f[:, :, 1:, :] - target_f[:, :, :-1, :])

        boundary = torch.zeros(B, 1, H, W, device=target.device)
        boundary[:, :, :, 1:] += dx
        boundary[:, :, 1:, :] += dy
        boundary = (boundary > 0).float().squeeze(1)

        # Distance transform approximation via Gaussian blur
        from torch.nn.functional import avg_pool2d
        k = int(self.sigma * 3) * 2 + 1
        padded = F.pad(boundary.unsqueeze(1), [k // 2] * 4, mode='reflect')
        blurred = F.avg_pool2d(padded, k, stride=1).squeeze(1)

        weights = 1.0 + self.a * blurred
        return weights


class HausdorffLoss(nn.Module):
    """Differentiable approximation of Hausdorff distance loss."""

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                class_idx: int = 1) -> torch.Tensor:
        pred_c = pred[:, class_idx]
        target_c = (target == class_idx).float()

        # Distance transform approximation
        pred_border = self._border(pred_c)
        target_border = self._border(target_c)

        # Hausdorff via max of directional distances
        d_pt = self._directed_hausdorff(pred_border, target_border)
        d_tp = self._directed_hausdorff(target_border, pred_border)
        return (d_pt + d_tp) / 2

    def _border(self, x: torch.Tensor) -> torch.Tensor:
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                 dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
        border = F.conv2d(x.unsqueeze(1), laplacian, padding=1).abs().squeeze(1)
        return border

    def _directed_hausdorff(self, src, tgt):
        # Soft approximation
        src_flat = src.reshape(src.shape[0], -1)
        tgt_flat = tgt.reshape(tgt.shape[0], -1)
        return ((src_flat - tgt_flat) ** 2).mean()


def structure_loss(pred_prob, logits, target, class_idx):
    """Combined Dice + boundary CE + Hausdorff for one structure."""
    dice = DiceLoss()(pred_prob, target, class_idx)
    bce = BoundaryAwareCE()(logits, target)
    hd = HausdorffLoss()(pred_prob, target, class_idx)
    return dice + 0.5 * bce + 0.1 * hd


# ---- EDL Loss ----

class EDLLoss(nn.Module):
    """Sum-of-Squares EDL loss with KL regularisation (Sensoy et al. 2018)."""

    def __init__(self, num_classes: int = 4, kl_weight: float = 0.01):
        super().__init__()
        self.K = num_classes
        self.kl_weight = kl_weight

    def forward(self, alpha: torch.Tensor, target: torch.Tensor,
                epoch: int = 0, anneal_epochs: int = 10) -> torch.Tensor:
        """
        alpha: (B, K, H, W) Dirichlet parameters.
        target: (B, H, W) integer labels.
        """
        B, K, H, W = alpha.shape
        S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        p_hat = alpha / S

        # One-hot target
        y = F.one_hot(target.long(), K).permute(0, 3, 1, 2).float()  # (B, K, H, W)

        # Sum of squares loss
        err = (y - p_hat) ** 2
        var = p_hat * (1 - p_hat) / (S + 1)
        sse = (err + var).sum(dim=1).mean()

        # KL divergence regularisation
        annealing = min(1.0, epoch / max(anneal_epochs, 1))
        alpha_tilde = y + (1 - y) * alpha  # Zero out correct-class evidence
        kl = self._kl_dirichlet(alpha_tilde)
        kl_loss = annealing * self.kl_weight * kl.mean()

        return sse + kl_loss

    def _kl_dirichlet(self, alpha: torch.Tensor) -> torch.Tensor:
        """KL[Dir(alpha) || Dir(1)]"""
        K = alpha.shape[1]
        ones = torch.ones_like(alpha)
        S_alpha = alpha.sum(dim=1, keepdim=True)
        S_ones = K * torch.ones(alpha.shape[0], 1, *alpha.shape[2:], device=alpha.device)

        kl = (torch.lgamma(S_alpha) - torch.lgamma(S_ones).sum(dim=1, keepdim=True)
              - torch.lgamma(alpha).sum(dim=1, keepdim=True)
              + torch.lgamma(ones).sum(dim=1, keepdim=True)
              + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(
                    dim=1, keepdim=True))
        return kl.squeeze(1).mean(dim=(-2, -1))


# ---- Aleatoric Attention Loss ----

class AleatoricAttentionLoss(nn.Module):
    """Attenuates CE at high-dissonance pixels (genuine boundary ambiguity)."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                dissonance: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction='none')  # (B, H, W)
        # exp(Diss) reduces CE penalty at ambiguous pixels
        attenuation = torch.exp(dissonance)
        # Diss term regularises against spurious dissonance
        loss = (attenuation * ce + dissonance).mean()
        return loss


# ---- PD-Guided Contrastive Loss ----

class PDContrastiveLoss(nn.Module):
    """PD-guided Supervised Contrastive Loss.
    Positive pairs: same topology (SimPD > threshold), possibly different scanners.
    """

    def __init__(self, temperature: float = 0.1, threshold: float = 0.8):
        super().__init__()
        self.tau = temperature
        self.threshold = threshold

    def forward(self, embeddings: torch.Tensor,
                pd_similarities: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D) L2-normalised.
        pd_similarities: (B, B) pairwise SimPD values.
        """
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Cosine similarity matrix
        sim = torch.matmul(embeddings, embeddings.T) / self.tau  # (B, B)

        # Positive mask: SimPD > threshold
        pos_mask = (pd_similarities > self.threshold).float()
        # Remove self-similarity
        pos_mask.fill_diagonal_(0)

        # Check if any positives exist
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        # Log-softmax over all non-self entries
        logits_mask = torch.ones(B, B, device=embeddings.device)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of positives
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
        return loss[valid].mean()


# ---- Topology Loss ----

class TopologyLoss(nn.Module):
    """Wasserstein distance between predicted and ground-truth PDs.
    Uses gudhi for PD computation (non-differentiable; gradient via straight-through
    estimator on the probability map).
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_prob_map: torch.Tensor,
                gt_pd: list, structure: str = "Myo") -> torch.Tensor:
        """
        pred_prob_map: (B, H, W) probability map for one structure.
        gt_pd: list of ground-truth PDs (one per batch element).
        Returns: mean Wasserstein distance.
        """
        from data.topology import compute_persistence_diagram, wasserstein_distance_pd

        B = pred_prob_map.shape[0]
        total_loss = torch.tensor(0.0, device=pred_prob_map.device, requires_grad=True)

        for b in range(B):
            prob_np = pred_prob_map[b].detach().cpu().numpy()
            pred_pd = compute_persistence_diagram(None, prob_np)

            gt = gt_pd[b] if b < len(gt_pd) else []
            w_dist = wasserstein_distance_pd(pred_pd, gt, p=1)

            # Straight-through: use Wasserstein as a scalar multiplier
            # on the mean of the prob map to enable gradient flow
            prob_mean = pred_prob_map[b].mean()
            total_loss = total_loss + w_dist * prob_mean

        return total_loss / max(B, 1)


# ---- Combined Expert Loss ----

class ExpertLoss(nn.Module):
    """Total loss for one Expert, with uncertainty weighting."""

    def __init__(self, loss_config, edl_config, num_classes: int = 4):
        super().__init__()
        self.config = loss_config
        self.edl_config = edl_config
        self.K = num_classes

        self.edl_loss = EDLLoss(num_classes)
        self.aleatoric_loss = AleatoricAttentionLoss()
        self.pd_contrastive = PDContrastiveLoss(
            loss_config.contrastive_tau, loss_config.contrastive_threshold)
        self.topo_loss = TopologyLoss()

        self.structure_class_map = {"LV": 1, "Myo": 2, "RV": 3}
        self.topo_weights = {
            "LV": loss_config.gamma_topo_lv,
            "Myo": loss_config.gamma_topo_myo,
            "RV": loss_config.gamma_topo_rv,
        }

    def forward(self, outputs: Dict, batch: Dict, epoch: int = 0,
                gt_pds: Optional[Dict] = None,
                pd_similarities: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        outputs: from SAMExpert.forward()
        batch: from dataloader
        """
        target = batch["mask"]
        logits = outputs["logits"]
        alpha = outputs["alpha"]
        p_hat = outputs["p_hat"]
        dissonance = outputs["dissonance"]
        embedding = outputs["embedding"]
        topo_maps = outputs["topo_maps"]
        log_sigma2 = outputs["log_sigma2"]

        losses = {}

        # 1. Structure losses with uncertainty weighting (Eq. 8 in paper)
        L_struct_uw = torch.tensor(0.0, device=logits.device)
        for name, class_idx in self.structure_class_map.items():
            L_s = structure_loss(p_hat, logits, target, class_idx)
            sigma2 = torch.exp(log_sigma2[name])
            # 1/(2*sigma^2) * L_s + log(sigma)
            weighted = 0.5 / sigma2 * L_s + 0.5 * log_sigma2[name]
            L_struct_uw = L_struct_uw + weighted
            losses[f"struct_{name}"] = L_s.detach()
        losses["struct_uw"] = L_struct_uw

        # 2. EDL loss
        L_edl = self.edl_loss(alpha, target, epoch, self.edl_config.kl_anneal_epochs)
        losses["edl"] = L_edl

        # 3. Aleatoric attention loss
        L_al = self.aleatoric_loss(logits, target, dissonance)
        losses["aleatoric"] = L_al

        # 4. PD-guided contrastive loss
        L_pd = torch.tensor(0.0, device=logits.device)
        if pd_similarities is not None:
            L_pd = self.pd_contrastive(embedding, pd_similarities)
        losses["pd_contrastive"] = L_pd

        # 5. Topology loss per structure
        L_topo = torch.tensor(0.0, device=logits.device)
        if gt_pds is not None:
            for name in ["LV", "Myo", "RV"]:
                if name in gt_pds and name in topo_maps:
                    lt = self.topo_loss(topo_maps[name], gt_pds[name], name)
                    L_topo = L_topo + self.topo_weights[name] * lt
                    losses[f"topo_{name}"] = lt.detach()
        losses["topo"] = L_topo

        # Total
        total = (L_struct_uw
                 + self.config.lambda_edl * L_edl
                 + self.config.lambda_aleatoric * L_al
                 + self.config.lambda_pd_contrastive * L_pd
                 + L_topo)
        losses["total"] = total

        return losses
