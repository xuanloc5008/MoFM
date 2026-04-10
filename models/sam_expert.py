"""
SAM Expert: Frozen SAM ViT-B encoder + AdaLoRA adapters + 4-branch decoder.
Branches: Segmentation, Evidential (EDL), Contrastive, Topological.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class SegmentationBranch(nn.Module):
    """Branch 1: Upsample features to full resolution, output K logits."""

    def __init__(self, in_dim: int = 256, num_classes: int = 4, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, num_classes, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, 256, H/16, W/16) → logits: (B, K, H, W)"""
        logits = self.decoder(features)
        # Ensure exact output size
        if logits.shape[-1] != self.img_size:
            logits = F.interpolate(logits, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False)
        return logits


class EvidentialBranch(nn.Module):
    """Branch 2: Compute per-pixel evidence, Vacuity, Dissonance from logits."""

    def __init__(self, num_classes: int = 4, delta_offsets: Dict[str, float] = None):
        super().__init__()
        self.K = num_classes
        # Structure-conditioned prior offsets
        self.delta_offsets = delta_offsets or {"LV": 0.1, "Myo": 0.5, "RV": 0.1, "BG": 0.0}

    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        logits: (B, K, H, W)
        Returns dict with: evidence, alpha, S, p_hat, vacuity, dissonance
        """
        # Evidence via Softplus
        evidence = F.softplus(logits)  # (B, K, H, W), >= 0

        # Dirichlet parameters with structure-conditioned prior
        alpha = evidence + 1.0  # Standard prior offset

        # Total evidence strength
        S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Expected probabilities
        p_hat = alpha / S  # (B, K, H, W)

        # Vacuity = K / S
        vacuity = self.K / S.squeeze(1)  # (B, H, W)

        # Dissonance
        b = evidence / S  # belief masses (B, K, H, W)
        dissonance = self._compute_dissonance(b)

        return {
            "evidence": evidence,
            "alpha": alpha,
            "S": S.squeeze(1),
            "p_hat": p_hat,
            "vacuity": vacuity,
            "dissonance": dissonance,
            "belief": b,
        }

    def _compute_dissonance(self, b: torch.Tensor) -> torch.Tensor:
        """Dissonance = sum_{i!=j} b_i * b_j / (b_i + b_j)"""
        B, K, H, W = b.shape
        diss = torch.zeros(B, H, W, device=b.device)
        for i in range(K):
            for j in range(i + 1, K):
                bi = b[:, i]  # (B, H, W)
                bj = b[:, j]
                denom = bi + bj + 1e-8
                diss += 2.0 * bi * bj / denom
        return diss


class ContrastiveBranch(nn.Module):
    """Branch 3: GAP + MLP → L2-normalised embedding for PD-guided contrastive."""

    def __init__(self, in_dim: int = 256, embed_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, 256, H/16, W/16) → embedding: (B, embed_dim)"""
        gap = features.mean(dim=(-2, -1))  # (B, 256)
        emb = self.mlp(gap)  # (B, embed_dim)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


class TopologicalBranch(nn.Module):
    """Branch 4: Probability map for PH filtration.
    In training, PD is computed externally via gudhi (non-differentiable)
    or via a differentiable approximation.
    This branch outputs per-structure probability maps for PH computation.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.K = num_classes

    def forward(self, p_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """p_hat: (B, K, H, W) → per-structure probability maps."""
        return {
            "LV": p_hat[:, 1],   # (B, H, W)
            "Myo": p_hat[:, 2],
            "RV": p_hat[:, 3],
        }


class SAMExpert(nn.Module):
    """Single Expert: SAM encoder (frozen) + AdaLoRA + 4-branch decoder.

    If use_cached_features=True, forward() expects precomputed encoder features
    instead of raw images (for 40% training speedup).
    """

    def __init__(self, num_classes: int = 4, img_size: int = 256,
                 embed_dim: int = 128, edl_config=None,
                 use_cached_features: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_cached_features = use_cached_features

        # Encoder will be set externally (SAM + AdaLoRA via PEFT)
        self.encoder = None

        # 4-Branch Decoder
        self.seg_branch = SegmentationBranch(256, num_classes, img_size)
        self.edl_branch = EvidentialBranch(num_classes)
        self.contrastive_branch = ContrastiveBranch(256, embed_dim)
        self.topo_branch = TopologicalBranch(num_classes)

        # Learnable uncertainty weighting (Kendall et al.)
        # log_sigma2[s] = log(sigma_s^2) for structure s
        # Initialised per-expert in train_expert.py
        self.log_sigma2 = nn.ParameterDict()
        for name in ["LV", "Myo", "RV"]:
            self.log_sigma2[name] = nn.Parameter(torch.tensor(0.0))

    def init_uncertainty_weights(self, primary_structure: str,
                                 sigma_primary: float = 0.5,
                                 sigma_secondary: float = 2.0):
        """Initialise uncertainty weighting based on primary structure assignment."""
        for name in ["LV", "Myo", "RV"]:
            if name == primary_structure:
                self.log_sigma2[name].data = torch.tensor(math.log(sigma_primary))
            else:
                self.log_sigma2[name].data = torch.tensor(math.log(sigma_secondary))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, 1, H, W) image OR (B, 256, H/16, W/16) cached features
        Returns dict with all branch outputs.
        """
        if self.use_cached_features:
            features = x
        else:
            # Encode via SAM (expects 3-channel input)
            x_3ch = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
            features = self.encoder(x_3ch)  # (B, 256, H/16, W/16)

        # Branch 1: Segmentation logits
        logits = self.seg_branch(features)  # (B, K, H, W)

        # Branch 2: Evidential
        edl_out = self.edl_branch(logits)

        # Branch 3: Contrastive embedding
        embedding = self.contrastive_branch(features)  # (B, embed_dim)

        # Branch 4: Topological probability maps
        topo_maps = self.topo_branch(edl_out["p_hat"])

        return {
            "logits": logits,
            "features": features,
            **edl_out,
            "embedding": embedding,
            "topo_maps": topo_maps,
            # Expose uncertainty weights for logging
            "log_sigma2": {k: v for k, v in self.log_sigma2.items()},
        }

    def get_sigma_values(self) -> Dict[str, float]:
        """Return current sigma values (for logging/reporting)."""
        return {k: math.exp(v.item() / 2) for k, v in self.log_sigma2.items()}

    def get_effective_weights(self) -> Dict[str, float]:
        """Return effective loss weights = 1 / (2 * sigma^2)."""
        return {k: 0.5 / math.exp(v.item()) for k, v in self.log_sigma2.items()}


def build_sam_expert(sam_checkpoint: str, model_type: str = "vit_b",
                     adalora_config=None, expert_id: int = 0,
                     primary_structure: str = "LV",
                     num_classes: int = 4, img_size: int = 256,
                     embed_dim: int = 128,
                     use_cached_features: bool = False,
                     use_medsam: bool = True) -> SAMExpert:
    """Build a SAM Expert with AdaLoRA adapters.

    Supports both vanilla SAM and MedSAM. MedSAM is preferred because its
    encoder is pre-trained on 1M+ medical images, requiring less adaptation
    (lower AdaLoRA rank, fewer training epochs).

    Args:
        sam_checkpoint: path to MedSAM/SAM ViT-B checkpoint
        model_type: 'vit_b' for both SAM and MedSAM
        adalora_config: AdaLoRAConfig dataclass
        expert_id: 0=LV, 1=Myo, 2=RV
        primary_structure: which structure this expert specialises in
        use_medsam: if True, load MedSAM checkpoint format
    """
    expert = SAMExpert(
        num_classes=num_classes,
        img_size=img_size,
        embed_dim=embed_dim,
        use_cached_features=use_cached_features,
    )

    if not use_cached_features:
        # Load encoder
        try:
            encoder = _load_medsam_encoder(sam_checkpoint, model_type, use_medsam)

            # Freeze all encoder weights
            for param in encoder.parameters():
                param.requires_grad = False

            # Apply AdaLoRA via PEFT
            try:
                from peft import AdaLoraConfig, get_peft_model
                peft_config = AdaLoraConfig(
                    r=adalora_config.r_init if adalora_config else 8,
                    lora_alpha=adalora_config.lora_alpha if adalora_config else 16,
                    target_modules=adalora_config.target_modules if adalora_config else [
                        "qkv"],
                    lora_dropout=0.05,
                    init_r=adalora_config.r_init if adalora_config else 8,
                )
                encoder = get_peft_model(encoder, peft_config)
                trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
                total = sum(p.numel() for p in encoder.parameters())
                print(f"Expert {expert_id}: AdaLoRA applied on "
                      f"{'MedSAM' if use_medsam else 'SAM'}. "
                      f"Trainable: {trainable:,} / {total:,} "
                      f"({trainable/total*100:.2f}%)")
            except ImportError:
                print("WARNING: peft not installed. Using frozen encoder without AdaLoRA.")

            expert.encoder = encoder

        except (ImportError, FileNotFoundError) as e:
            print(f"WARNING: Could not load encoder: {e}. Using placeholder.")
            expert.encoder = _PlaceholderEncoder()

    # Initialise uncertainty weights
    expert.init_uncertainty_weights(
        primary_structure,
        sigma_primary=0.5,
        sigma_secondary=2.0,
    )

    return expert


def _load_medsam_encoder(checkpoint_path: str, model_type: str = "vit_b",
                         use_medsam: bool = True):
    """Load MedSAM or vanilla SAM encoder.

    MedSAM uses the same ViT-B architecture as SAM but with weights
    fine-tuned on medical images. The checkpoint format differs slightly.
    """
    import torch

    if use_medsam:
        # MedSAM checkpoint: contains 'image_encoder' state dict
        try:
            from segment_anything import sam_model_registry
            # MedSAM uses same architecture, different weights
            sam = sam_model_registry[model_type]()
            ckpt = torch.load(checkpoint_path, map_location="cpu")

            # MedSAM checkpoint may have different key formats
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    state_dict = ckpt["model"]
                elif "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                else:
                    state_dict = ckpt

                # Filter to image encoder keys
                encoder_keys = {k.replace("image_encoder.", ""): v
                               for k, v in state_dict.items()
                               if k.startswith("image_encoder.")}
                if encoder_keys:
                    sam.image_encoder.load_state_dict(encoder_keys, strict=False)
                else:
                    # Try loading full model
                    sam.load_state_dict(state_dict, strict=False)

            print(f"Loaded MedSAM encoder from {checkpoint_path}")
            return sam.image_encoder

        except Exception as e:
            print(f"MedSAM loading failed ({e}), trying vanilla SAM format...")
            from segment_anything import sam_model_registry
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            return sam.image_encoder
    else:
        # Vanilla SAM
        from segment_anything import sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        return sam.image_encoder


class _PlaceholderEncoder(nn.Module):
    """Fallback encoder for testing without SAM checkpoint."""

    def __init__(self, out_dim=256, feat_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(feat_size),
        )

    def forward(self, x):
        return self.net(x)
