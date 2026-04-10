"""
MOFMTDA-TTA Configuration
All hyperparameters in one place. No heuristic structure-specific weights.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    data_root: str = "./dataset/acdcdata/ACDC/database/training"
    mms_root: str = "./data/MMs"
    mms2_root: str = "./data/MMs2"
    img_size: Tuple[int, int] = (256, 256)
    num_classes: int = 4  # BG, LV, Myo, RV
    train_split: float = 0.7
    val_split: float = 0.1
    test_split: float = 0.2
    num_workers: int = 4


@dataclass
class AugConfig:
    """Shared augmentation pool — no per-expert assignment."""
    elastic_sigma_range: Tuple[float, float] = (10.0, 30.0)
    elastic_alpha_range: Tuple[float, float] = (50.0, 200.0)
    tps_num_control_points: List[int] = field(default_factory=lambda: [8, 16])
    rotation_range: float = 20.0
    intensity_scale_range: Tuple[float, float] = (0.8, 1.2)
    gaussian_noise_std: float = 0.05
    gaussian_blur_sigma: Tuple[float, float] = (0.5, 2.0)
    tpa_enabled: bool = True  # Topology-Preserving Augmentation filter


@dataclass
class SAMConfig:
    """MedSAM: SAM ViT-B pre-trained on 1M+ medical image-mask pairs."""
    model_type: str = "vit_b"
    checkpoint: str = "./checkpoints/medsam_vit_b.pth"
    freeze_encoder: bool = True
    feature_dim: int = 256
    feature_size: int = 16  # H/16, W/16
    use_medsam: bool = True  # Use MedSAM instead of vanilla SAM


@dataclass
class AdaLoRAConfig:
    """AdaLoRA — uniform init, adaptive pruning. No heuristic ranks.
    r_init=8 (reduced from 16) because MedSAM encoder already has
    medical-domain features, requiring less adaptation."""
    r_init: int = 8  # Lower than vanilla SAM (16) due to MedSAM pretraining
    lora_alpha: int = 16
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"]
    )
    # AdaLoRA-specific
    orth_reg_weight: float = 0.5
    rank_pattern: Optional[dict] = None  # Learned during training
    total_step: int = 7000  # Fewer steps due to MedSAM pretraining
    warmup_step: int = 300
    final_rank: int = 2  # Minimum rank after pruning


@dataclass
class EDLConfig:
    """Anatomy-specialised Evidential Deep Learning."""
    # Structure-conditioned Dirichlet prior offsets
    delta_myo: float = 0.5   # Higher uncertainty for ring topology
    delta_lv: float = 0.1
    delta_rv: float = 0.1
    # KL annealing
    kl_anneal_epochs: int = 10


@dataclass
class UncertaintyWeightingConfig:
    """Homoscedastic uncertainty weighting (Kendall et al.)."""
    # Initialisation: primary structure gets low sigma (high weight)
    sigma_primary_init: float = 0.5
    sigma_secondary_init: float = 2.0


@dataclass
class LossConfig:
    """Loss weights. Only topo weights are manually set (from Betti complexity)."""
    lambda_edl: float = 0.5  # Reduced from 1.0 so segmentation dominates early
    lambda_aleatoric: float = 0.1
    lambda_pd_contrastive: float = 0.05
    # Topology loss weights — derived from Betti complexity ordering
    gamma_topo_myo: float = 0.1   # beta1=1, most complex
    gamma_topo_rv: float = 0.05   # beta0=1, crescent
    gamma_topo_lv: float = 0.01   # beta0=1, simple disc
    # Contrastive
    contrastive_tau: float = 0.1
    contrastive_threshold: float = 0.8
    contrastive_embed_dim: int = 128
    # Specialisation
    # No per-expert focus — all experts are general-purpose.
    # Specialisation emerges from gating (Phase 2), not from training.
    # Each expert is trained with a different seed for diversity.
    expert_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


@dataclass
class GatingConfig:
    num_experts: int = 3
    gating_channels: int = 32
    cbam_reduction: int = 8
    route_penalty_weight: float = 0.1


@dataclass
class TTECConfig:
    """Tripartite Topology-Error Classification — 3 signals (no TTSS)."""
    # Thresholds — calibrated on validation set
    tau_dss: float = 0.7
    tau_plr: float = 0.3
    tau_spcs: float = 0.6
    # DSS composition
    alpha_dss: float = 0.5
    # OOD k-NN
    knn_k: int = 10
    # Betti targets (mid-ventricular)
    betti_targets: dict = field(default_factory=lambda: {
        "LV": (1, 0),   # beta0=1, beta1=0
        "Myo": (1, 1),  # beta0=1, beta1=1
        "RV": (1, 0),   # beta0=1, beta1=0
    })


@dataclass
class TopoAStarConfig:
    tau_anchor_vacuity: float = 0.1
    anchor_confidence: float = 0.95
    lambda_topo: float = 10.0
    repulsion_cutoff: float = 5.0
    path_sigma: float = 2.0
    path_rho: float = 0.8
    max_iterations: int = 10000


@dataclass
class TrainConfig:
    # Phase 1: Expert training (fewer epochs than vanilla SAM due to MedSAM pretraining)
    expert_epochs: int = 70
    expert_lr: float = 1e-4
    expert_batch_size: int = 4
    # Phase 2: Gating training
    gating_epochs: int = 50
    gating_lr: float = 5e-5
    gating_batch_size: int = 4
    # General
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    num_seeds: int = 5
    # Cache
    cache_encoder_features: bool = True
    cache_dir: str = "./cache"
    # Checkpoints
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class InferenceConfig:
    mc_samples: int = 200  # For MC uncertainty propagation
    ef_threshold: float = 35.0  # HF diagnostic threshold (%)
    flag_if_ci_spans_threshold: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    adalora: AdaLoRAConfig = field(default_factory=AdaLoRAConfig)
    edl: EDLConfig = field(default_factory=EDLConfig)
    uw: UncertaintyWeightingConfig = field(default_factory=UncertaintyWeightingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    ttec: TTECConfig = field(default_factory=TTECConfig)
    topo_astar: TopoAStarConfig = field(default_factory=TopoAStarConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # No expert-to-structure assignment.
    # All experts are general-purpose, trained with different seeds.
    # Specialisation emerges from gating routing.
