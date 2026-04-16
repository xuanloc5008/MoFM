"""
configs/config.py
All hyperparameters for DUT-Mamba-MoE — single source of truth.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    # ── Spatial ──────────────────────────────────────────────────────────────
    spatial_size:    Tuple[int, ...] = (256, 256, 16)  # H × W × D
    num_classes:     int             = 4               # BG / LV / Myo / RV
    label_map:       dict = field(default_factory=lambda: {
        0: "Background", 1: "LV", 2: "Myo", 3: "RV"
    })
    # ── Splits ───────────────────────────────────────────────────────────────
    acdc_train_ratio: float = 0.70
    acdc_val_ratio:   float = 0.10
    acdc_test_ratio:  float = 0.20
    # ── Normalisation ────────────────────────────────────────────────────────
    intensity_percentile: Tuple[float, float] = (0.5, 99.5)
    # ── Augmentation ─────────────────────────────────────────────────────────
    elastic_sigma:    Tuple[float, float] = (10.0, 30.0)
    elastic_alpha:    Tuple[float, float] = (50.0, 200.0)
    rotation_range:   float = 20.0      # degrees
    noise_std:        float = 0.05
    blur_sigma:       float = 1.0
    intensity_scale:  Tuple[float, float] = (0.8, 1.2)
    # ── Topology-Preserving Augmentation ─────────────────────────────────────
    tpa_max_attempts: int = 20


@dataclass
class ModelConfig:
    # ── VSS Encoder ──────────────────────────────────────────────────────────
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
    ssm_d_state:  int   = 16
    ssm_d_conv:   int   = 4
    ssm_expand:   int   = 2
    # ── SSM-LoRA ─────────────────────────────────────────────────────────────
    lora_r_init:  int   = 6
    lora_alpha:   float = 16.0
    # ── EDL priors ───────────────────────────────────────────────────────────
    evidence_activation: str = "softplus"
    delta_myo:    float = 0.5   # higher initial uncertainty for Myo ring
    delta_lv_rv:  float = 0.1
    # ── MoE ──────────────────────────────────────────────────────────────────
    num_experts:    int = 3     # E1=LV, E2=Myo, E3=RV
    routing_hidden: int = 32
    # ── Contrastive branch ───────────────────────────────────────────────────
    contrastive_dim:  int   = 128
    contrastive_temp: float = 0.07
    pd_sim_threshold: float = 0.7   # θ+ for topology-positive pairs


@dataclass
class TrainingConfig:
    # ── Curriculum phases ────────────────────────────────────────────────────
    phase1_epochs: int = 50    # Burn-in: seg + EDL only
    phase2_epochs: int = 50    # Routing + topology awakening
    phase3_epochs: int = 30    # Bayesian auto-balancing (σ unfrozen)
    # ── Optimisers ───────────────────────────────────────────────────────────
    lr_expert:    float = 8e-5
    lr_routing:   float = 4e-5
    lr_gate:      float = 1e-4
    weight_decay: float = 1e-4
    # ── Homoscedastic σ init ─────────────────────────────────────────────────
    sigma_init_primary:   float = 0.5   # primary structure
    sigma_init_secondary: float = 2.0   # secondary structures
    # ── Loss weights (only manually fixed ones) ───────────────────────────────
    gamma_topo_myo: float = 0.1
    gamma_topo_rv:  float = 0.05
    gamma_topo_lv:  float = 0.01
    lambda_al:      float = 0.1
    lambda_pd:      float = 0.1
    # ── KL annealing ─────────────────────────────────────────────────────────
    kl_anneal_epochs: int = 10
    # ── DataLoader ───────────────────────────────────────────────────────────
    batch_size:  int = 2
    num_workers: int = 4
    # ── Multi-seed ───────────────────────────────────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])


@dataclass
class TopologyConfig:
    # ── 3D Betti targets (paper Eqs. 4-6) ────────────────────────────────────
    betti_lv_rv: Tuple[int, int, int] = (1, 0, 0)  # solid cavity
    betti_myo:   Tuple[int, int, int] = (1, 1, 0)  # hollow torus
    # ── PH computation ───────────────────────────────────────────────────────
    cubical_resolution: Tuple[int, int, int] = (64, 64, 8)  # downsampled for speed
    # ── TU / Fréchet mean ────────────────────────────────────────────────────
    tu_num_layers: int   = 4     # VSS layers used for TU
    fm_buffer_size: int  = 100   # N_buf for adaptive FM (future work)
    fm_beta_fast:   float = 0.05


@dataclass
class TTECConfig:
    # ── Learned gating G_TTEC ────────────────────────────────────────────────
    gate_hidden_dim: int   = 32
    gate_temperature: float = 1.0  # learnable, init value
    # ── Action-policy thresholds (clinically motivated, NOT calibrated) ───────
    theta_type1: float = 0.50
    theta_type2: float = 0.60
    # ── Class weights for G_TTEC loss (clinical cost asymmetry) ──────────────
    cost_type1:  float = 1.0
    cost_type2:  float = 2.5   # false acceptance most harmful
    cost_type3:  float = 1.5
    # ── Topo-A* correction ───────────────────────────────────────────────────
    astar_lambda_topo: float = 10.0
    anchor_vac_thresh: float = 0.1
    anchor_prob_thresh: float = 0.95


@dataclass
class EvaluationConfig:
    # ── Clinical thresholds ──────────────────────────────────────────────────
    ef_hf_threshold:     float = 35.0   # % — heart failure
    ef_normal_lower:     float = 55.0   # % — lower bound normal
    mass_tolerance_pct:  float = 10.0   # %
    # ── Monte Carlo uncertainty ──────────────────────────────────────────────
    n_mc_samples: int = 200
    # ── Statistics ───────────────────────────────────────────────────────────
    alpha:       float = 0.05
    n_bootstrap: int   = 1000
    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_dpi: int = 150


@dataclass
class Config:
    data:       DataConfig       = field(default_factory=DataConfig)
    model:      ModelConfig      = field(default_factory=ModelConfig)
    training:   TrainingConfig   = field(default_factory=TrainingConfig)
    topology:   TopologyConfig   = field(default_factory=TopologyConfig)
    ttec:       TTECConfig       = field(default_factory=TTECConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root:      str = "/data/cardiac"
    output_dir:     str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    device:         str = "cuda"


cfg = Config()
