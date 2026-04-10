"""
Topology-Penalised Gating Network: Mini-UNet with CBAM.
Routes pixels to the most topologically faithful expert.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attn(x).unsqueeze(-1).unsqueeze(-1)
        x = x * ca
        # Spatial attention
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        return x * sa


class MiniUNet(nn.Module):
    """Lightweight 5-layer UNet for gating."""

    def __init__(self, in_channels: int, out_channels: int, base_ch: int = 32,
                 cbam_reduction: int = 8):
        super().__init__()

        self.enc1 = self._block(in_channels, base_ch)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.bottleneck = nn.Sequential(
            self._block(base_ch * 2, base_ch * 4),
            CBAM(base_ch * 4, cbam_reduction),
        )
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = self._block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = self._block(base_ch * 2, base_ch)
        self.final = nn.Conv2d(base_ch, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class GatingNetwork(nn.Module):
    """Topology-Penalised Gating Network.

    Input: image + per-expert predictions + per-expert Vacuity + per-expert Dissonance
    Output: pixel-wise expert weights (B, M, H, W)
    """

    def __init__(self, num_experts: int = 3, num_classes: int = 4,
                 base_channels: int = 32, cbam_reduction: int = 8):
        super().__init__()
        self.M = num_experts
        self.K = num_classes

        # Input channels: 1 (image) + M * (K + 1 + 1) = 1 + M*(K+2)
        # per expert: K probs + 1 vacuity + 1 dissonance
        in_ch = 1 + num_experts * (num_classes + 2)

        self.net = MiniUNet(in_ch, num_experts, base_channels, cbam_reduction)

    def forward(self, image: torch.Tensor,
                expert_outputs: List[Dict]) -> torch.Tensor:
        """
        image: (B, 1, H, W)
        expert_outputs: list of M dicts from SAMExpert.forward()
        Returns: weights (B, M, H, W), softmax-normalised over M
        """
        parts = [image]
        for eo in expert_outputs:
            parts.append(eo["p_hat"])           # (B, K, H, W)
            parts.append(eo["vacuity"].unsqueeze(1))   # (B, 1, H, W)
            parts.append(eo["dissonance"].unsqueeze(1)) # (B, 1, H, W)

        x = torch.cat(parts, dim=1)
        logits = self.net(x)  # (B, M, H, W)
        weights = F.softmax(logits, dim=1)
        return weights

    def ensemble_predict(self, weights: torch.Tensor,
                         expert_outputs: List[Dict]) -> Dict:
        """Compute gating-weighted ensemble prediction.

        Returns dict with ensemble p_bar, system-level epistemic uncertainty, etc.
        """
        M = len(expert_outputs)
        B, _, H, W = weights.shape
        K = expert_outputs[0]["p_hat"].shape[1]

        # Weighted average probabilities: p_bar = sum_m w_m * p_hat_m
        p_bar = torch.zeros(B, K, H, W, device=weights.device)
        for m in range(M):
            w_m = weights[:, m:m+1]  # (B, 1, H, W)
            p_bar += w_m * expert_outputs[m]["p_hat"]

        # System-level epistemic: MI = H(p_bar) - sum_m w_m * H(p_m)
        H_bar = -torch.sum(p_bar * torch.log(p_bar + 1e-8), dim=1)  # (B, H, W)
        H_experts = torch.zeros(B, H, W, device=weights.device)
        for m in range(M):
            H_m = -torch.sum(expert_outputs[m]["p_hat"] * torch.log(
                expert_outputs[m]["p_hat"] + 1e-8), dim=1)
            H_experts += weights[:, m] * H_m

        u_sys_epistemic = H_bar - H_experts  # (B, H, W)

        # Ensemble Vacuity/Dissonance (weighted average)
        vac_ensemble = torch.zeros(B, H, W, device=weights.device)
        diss_ensemble = torch.zeros(B, H, W, device=weights.device)
        for m in range(M):
            vac_ensemble += weights[:, m] * expert_outputs[m]["vacuity"]
            diss_ensemble += weights[:, m] * expert_outputs[m]["dissonance"]

        return {
            "p_bar": p_bar,
            "prediction": p_bar.argmax(dim=1),
            "u_sys_epistemic": u_sys_epistemic,
            "vacuity": vac_ensemble,
            "dissonance": diss_ensemble,
            "weights": weights,
        }


class GatingLoss(nn.Module):
    """Gating training loss: segmentation accuracy + routing penalty."""

    def __init__(self, route_penalty_weight: float = 0.1):
        super().__init__()
        self.gamma = route_penalty_weight
        self.dice_loss = DiceLoss()

    def forward(self, ensemble: Dict, expert_outputs: List[Dict],
                weights: torch.Tensor, target: torch.Tensor) -> Dict:
        from models.losses import DiceLoss
        dice = DiceLoss()

        p_bar = ensemble["p_bar"]

        # Segmentation loss on ensemble
        L_dice = dice(p_bar, target)
        L_ce = F.cross_entropy(torch.log(p_bar + 1e-8), target)

        # Routing penalty: penalise routing to uncertain/topo-unreliable experts
        M = len(expert_outputs)
        L_route = torch.tensor(0.0, device=weights.device)
        for m in range(M):
            w_m = weights[:, m]  # (B, H, W)
            vac_m = expert_outputs[m]["vacuity"]
            L_route += (w_m * vac_m).mean()

        L_route /= M

        total = L_dice + L_ce + self.gamma * L_route

        return {
            "total": total,
            "dice": L_dice.detach(),
            "ce": L_ce.detach(),
            "route_penalty": L_route.detach(),
        }
