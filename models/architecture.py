"""
models/architecture.py  –  DUT-Mamba-MoE
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import cfg


# ── SSM-LoRA adapter ──────────────────────────────────────────────────────────
class SSMLoRALinear(nn.Module):
    """W_adapt = W_0 + B*A  (additive LoRA, standard formulation)."""
    def __init__(self, in_f, out_f, r=6, bias=False):
        super().__init__()
        self.base = nn.Linear(in_f, out_f, bias=bias)
        self.A    = nn.Linear(in_f, r, bias=False)
        self.B    = nn.Linear(r, out_f, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x))


# ── VSS Block ─────────────────────────────────────────────────────────────────
class VSSBlock(nn.Module):
    """Simplified 3-D Visual State Space block with SSM-LoRA."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, lora_r=6):
        super().__init__()
        self.d_inner = d_inner = int(d_model * expand)
        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = SSMLoRALinear(d_model, d_inner * 2, r=lora_r)
        self.conv3d   = nn.Conv3d(d_inner, d_inner,
                                   kernel_size=d_conv,
                                   padding=d_conv//2 if d_conv % 2 == 1 else d_conv//2 - 0,
                                   groups=d_inner,
                                   padding_mode="zeros")
        self.A_log    = nn.Parameter(
            torch.log(torch.arange(1, d_state+1, dtype=torch.float32)
                      .unsqueeze(0).expand(d_inner, -1).clone()))
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.x_proj   = nn.Linear(d_inner, d_state*2+1, bias=False)
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)
        self.out_proj  = SSMLoRALinear(d_inner, d_model, r=lora_r)

    # ── selective scan ────────────────────────────────────────────
    def _scan(self, x):
        B, L, D = x.shape
        N = self.A_log.shape[1]
        xz = self.x_proj(x)
        dt_r, B_s, C_s = xz.split([1, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt_r))           # (B,L,D)
        A  = -torch.exp(self.A_log)                   # (D,N)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) *
                       dt.unsqueeze(-1))               # (B,L,D,N)
        dB = dt.unsqueeze(-1) * B_s.unsqueeze(2)      # (B,L,D,N)
        h  = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i]*h + dB[:, i]*x[:, i].unsqueeze(-1)  # (B,D,N)
            # C_s[:,i] is (B,N);  h is (B,D,N)  → broadcast over D
            y = (h * C_s[:, i].unsqueeze(1)).sum(-1)           # (B,D)
            ys.append(y)
        return torch.stack(ys, 1) + x * self.D        # (B,L,D)

    # ── reorder helpers ───────────────────────────────────────────
    def _reorder(self, x, d, H, W, D):
        B, L, C = x.shape
        if d == "hw_fwd": return x
        if d == "hw_bwd": return x.flip(1)
        r = x.reshape(B,H,W,D,C).permute(0,3,1,2,4).reshape(B,L,C)
        return r if d == "d_fwd" else r.flip(1)

    def _reorder_back(self, x, d, H, W, D):
        B, L, C = x.shape
        if d == "hw_fwd": return x
        if d == "hw_bwd": return x.flip(1)
        base = x if d == "d_fwd" else x.flip(1)
        return base.reshape(B,D,H,W,C).permute(0,2,3,1,4).reshape(B,L,C)

    def forward(self, x):
        B, C, H, W, Dz = x.shape
        res  = x
        xn   = self.norm(x.permute(0,2,3,4,1)).reshape(B,-1,C)
        xz   = self.in_proj(xn)
        xi, z = xz.chunk(2, dim=-1)
        xc   = xi.reshape(B,H,W,Dz,self.d_inner).permute(0,4,1,2,3)
        xc   = F.silu(self.conv3d(xc))
        # Trim to exact input spatial size (conv with even kernel adds 1 pad)
        xc   = xc[:, :, :H, :W, :Dz]
        aH, aW, aDz = H, W, Dz
        L    = H * W * Dz
        xc_flat = xc.permute(0,2,3,4,1).reshape(B, L, self.d_inner)
        z_   = z   # already (B, L, d_inner) with same L
        hs = [self._reorder_back(
                  self._scan(self._reorder(xc_flat, d, aH, aW, aDz)),
                  d, aH, aW, aDz)
              for d in ["hw_fwd","hw_bwd","d_fwd","d_bwd"]]
        y    = sum(hs)/4 * F.silu(z_)
        out  = self.out_proj(y).reshape(B, aH, aW, aDz, C).permute(0,4,1,2,3)
        return out + res, y.mean(1)


# ── Hierarchical Encoder ──────────────────────────────────────────────────────
class VSSEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        chs = cfg.model.encoder_channels
        self.downs  = nn.ModuleList()
        self.stages = nn.ModuleList()
        prev = in_channels
        for ch in chs:
            self.downs.append(nn.Conv3d(prev, ch, 3, stride=2, padding=1))
            self.stages.append(VSSBlock(ch,
                d_state=cfg.model.ssm_d_state,
                d_conv =cfg.model.ssm_d_conv,
                expand =cfg.model.ssm_expand,
                lora_r =cfg.model.lora_r_init))
            prev = ch

    def forward(self, x):
        feats, hiddens = [], []
        for down, stage in zip(self.downs, self.stages):
            x = F.relu(down(x))
            x, h = stage(x)
            feats.append(x); hiddens.append(h)
        return feats, hiddens


# ── Unified EDL Decoder (single branch) ──────────────────────────────────────
class UnifiedEDLDecoder(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        K  = num_classes
        chs = list(reversed(cfg.model.encoder_channels))   # deep→shallow
        self.ups   = nn.ModuleList()
        self.skips = nn.ModuleList()
        prev = chs[0]
        for ch in chs[1:]:
            self.ups.append(nn.Sequential(
                nn.ConvTranspose3d(prev, ch, 2, stride=2),
                nn.GroupNorm(8, ch), nn.GELU(),
                nn.Conv3d(ch, ch, 3, padding=1),
                nn.GroupNorm(8, ch), nn.GELU()))
            self.skips.append(nn.Conv3d(ch*2, ch, 1))
            prev = ch
        self.final_up = nn.ConvTranspose3d(prev, prev, 2, stride=2)
        self.head     = nn.Conv3d(prev, K, 1)
        delta         = torch.zeros(K)
        delta[2] = cfg.model.delta_myo
        delta[1] = delta[3] = cfg.model.delta_lv_rv
        self.register_buffer("prior_delta", delta)

    def forward(self, feats):
        enc = list(reversed(feats))
        x   = enc[0]
        for i, (up, skip) in enumerate(zip(self.ups, self.skips)):
            x = up(x)
            s = enc[i+1]
            if x.shape != s.shape:
                x = F.interpolate(x, s.shape[2:], mode="trilinear", align_corners=False)
            x = skip(torch.cat([x, s], 1))
        x = self.final_up(x)
        z = self.head(x)
        tgt = cfg.data.spatial_size
        if list(z.shape[2:]) != list(tgt):
            z = F.interpolate(z, tgt, mode="trilinear", align_corners=False)
        e     = F.softplus(z)
        delta = self.prior_delta.view(1,-1,1,1,1)
        alpha = e + 1.0 + delta
        S     = alpha.sum(1, keepdim=True)
        p_hat = alpha / S
        K     = alpha.shape[1]
        vac   = (K / S).squeeze(1)
        b     = e / S
        diss  = sum(b[:,i]*b[:,j]/(b[:,i]+b[:,j]+1e-8)
                    for i in range(K) for j in range(K) if i!=j)
        return dict(logits=z, evidence=e, alpha=alpha,
                    S=S.squeeze(1), p_hat=p_hat,
                    vacuity=vac, dissonance=diss)


# ── Single Expert ─────────────────────────────────────────────────────────────
class MambaEDLExpert(nn.Module):
    def __init__(self, structure, num_classes=4):
        super().__init__()
        self.structure = structure
        self.encoder   = VSSEncoder(in_channels=1)
        self.decoder   = UnifiedEDLDecoder(num_classes)

    def forward(self, x):
        feats, hiddens = self.encoder(x)
        return self.decoder(feats), hiddens


# ── Routing Network ───────────────────────────────────────────────────────────
class MambaRoutingNetwork(nn.Module):
    def __init__(self, num_experts=3, num_classes=4):
        super().__init__()
        M, K = num_experts, num_classes
        in_ch = 1 + M*K + 2*M
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 64, 3, padding=1), nn.GroupNorm(8,64), nn.GELU(),
            nn.Conv3d(64, 32, 3, padding=1),    nn.GroupNorm(8,32), nn.GELU(),
            nn.Conv3d(32, M, 1))

    def forward(self, image, expert_outputs):
        parts = [image]
        for o in expert_outputs:
            parts += [o["p_hat"], o["vacuity"].unsqueeze(1),
                      o["dissonance"].unsqueeze(1)]
        return F.softmax(self.net(torch.cat(parts, 1)), dim=1)


# ── Full DUT-Mamba-MoE ────────────────────────────────────────────────────────
class DUTMambaMoE(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        K = num_classes
        M = cfg.model.num_experts
        self.num_classes = K
        self.num_experts  = M
        self.structures   = ["LV", "Myo", "RV"]
        self.experts      = nn.ModuleList(
            [MambaEDLExpert(s, K) for s in self.structures])
        self.routing      = MambaRoutingNetwork(M, K)
        self.log_sigma2   = nn.Parameter(torch.zeros(M, 4))
        self._init_sigma()

    def _init_sigma(self):
        v = math.log(cfg.training.sigma_init_secondary)
        self.log_sigma2.data.fill_(v)
        for m in range(self.num_experts):
            self.log_sigma2.data[m, 3] = math.log(cfg.training.sigma_init_primary)

    def forward(self, x, phase=2):
        expert_outs, hidden_lists = [], []
        for exp in self.experts:
            o, h = exp(x)
            expert_outs.append(o); hidden_lists.append(h)
        if phase == 1:
            return dict(expert_outputs=expert_outs, hidden_states=hidden_lists)
        weights = self.routing(x, expert_outs)
        M = self.num_experts
        p_bar   = sum(weights[:,m:m+1]*expert_outs[m]["p_hat"] for m in range(M))
        H_bar   = -(p_bar*(p_bar+1e-8).log()).sum(1)
        H_exp   = torch.stack([
            -(expert_outs[m]["p_hat"]*(expert_outs[m]["p_hat"]+1e-8).log()).sum(1)
            for m in range(M)], 1)
        u_sys   = H_bar - (weights*H_exp).sum(1)
        return dict(
            p_bar=p_bar,
            routing_weights=weights,
            expert_outputs=expert_outs,
            hidden_states=hidden_lists,
            u_sys=u_sys,
            vacuity   = sum(weights[:,m].squeeze(1) if weights[:,m].ndim>4
                            else weights[:,m]*expert_outs[m]["vacuity"]
                            for m in range(M)),
            dissonance= sum(weights[:,m]*expert_outs[m]["dissonance"]
                            for m in range(M)),
        )

    def get_sigma(self):
        return torch.exp(self.log_sigma2 / 2.0)
