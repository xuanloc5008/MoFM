"""
train.py
---------
Entry point for training Topo-Evidential U-Mamba on ACDC.

Usage:
  python train.py --config configs/config.yaml

After training, the best model checkpoint is saved to:
  outputs/checkpoints/best_model.pth

TensorBoard logs are written to:
  outputs/logs/
"""
import argparse
import copy
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.data.acdc        import (
    PreprocessedSliceDataset,
    SliceDataset,
    acdc_train_val_split,
)
from src.data.transforms  import (
    get_train_transforms,
    get_train_transforms_for_preprocessed,
    get_val_transforms,
    get_val_transforms_for_preprocessed,
)
from src.models.topo_evidential_umamba import build_model
from src.runtime          import (
    configure_torch_runtime,
    dataloader_kwargs,
    resolve_device,
    resolve_runtime_settings,
)
from src.training.trainer import Trainer


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "train.log")),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Train Topo-Evidential U-Mamba")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Load config ─────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(os.path.join(output_dir, "logs"))
    logger = logging.getLogger(__name__)

    device = resolve_device(args.device, gpu_index=args.gpu)
    runtime_settings = resolve_runtime_settings(cfg, device, gpu_index=args.gpu)
    configure_torch_runtime(device, runtime_settings)
    seed_everything(cfg.get("seed", 42))

    logger.info(f"Device: {device} ({runtime_settings['device_name']})")
    logger.info(
        "Runtime profile: "
        f"{' -> '.join(runtime_settings['profile_names'])} | "
        f"train_bs={runtime_settings['train_batch_size']} "
        f"val_bs={runtime_settings['val_batch_size']} "
        f"workers={runtime_settings['num_workers']} "
        f"amp={'on' if runtime_settings['use_amp'] else 'off'}"
    )

    # ── Data ────────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    preprocess_cfg = cfg.get("preprocessing", {})
    spatial_size = tuple(data_cfg["spatial_size"])   # (H, W)
    context_slices = int(data_cfg.get("context_slices", data_cfg.get("in_channels", 1)))

    use_preprocessed = bool(data_cfg.get("use_preprocessed_acdc", False))
    preprocessed_root = cfg["paths"].get(
        "preprocessed_acdc_root",
        os.path.join(output_dir, "preprocessed", "acdc"),
    )
    contrastive_enabled = float(cfg.get("loss", {}).get("gamma", 0.0)) > 0.0

    if contrastive_enabled and not use_preprocessed:
        raise RuntimeError(
            "Contrastive topology training now expects cached 'topo_vec' features. "
            "Run scripts/preprocess_acdc.py, then set data.use_preprocessed_acdc=true."
        )

    if use_preprocessed:
        logger.info(f"Loading preprocessed ACDC dataset from {preprocessed_root}…")
        meta_path = os.path.join(preprocessed_root, "meta.json")
        if preprocess_cfg.get("enabled", False):
            if not os.path.exists(meta_path):
                raise RuntimeError(
                    "Phase-1 preprocessing is enabled but preprocessed ACDC cache is "
                    "missing meta.json. Rebuild with scripts/preprocess_acdc.py."
                )
            with open(meta_path, "r", encoding="utf-8") as f:
                cache_meta = json.load(f)
            if int(cache_meta.get("preprocessing_version", 0)) < 1:
                raise RuntimeError(
                    "Preprocessed ACDC cache was built before Phase-1 harmonization. "
                    "Rebuild with scripts/preprocess_acdc.py --overwrite."
                )
        train_transforms = get_train_transforms_for_preprocessed(cfg.get("augmentation", {}))
        val_transforms = get_val_transforms_for_preprocessed()
        train_dataset = PreprocessedSliceDataset(
            os.path.join(preprocessed_root, "train"),
            transforms=train_transforms,
            context_slices=context_slices,
        )
        val_dataset = PreprocessedSliceDataset(
            os.path.join(preprocessed_root, "val"),
            transforms=val_transforms,
            context_slices=context_slices,
        )
    else:
        logger.info("Loading ACDC dataset…")
        train_slices, val_slices = acdc_train_val_split(
            acdc_root  = cfg["paths"]["acdc_root"],
            val_ratio  = data_cfg.get("acdc_val_ratio", 0.20),
            seed       = cfg.get("seed", 42),
            preprocess_cfg = preprocess_cfg,
        )

        train_transforms = get_train_transforms(
            spatial_size,
            cfg.get("augmentation", {}),
            preprocess_cfg=preprocess_cfg,
        )
        val_transforms   = get_val_transforms(spatial_size, preprocess_cfg=preprocess_cfg)

        train_dataset = SliceDataset(
            train_slices,
            transforms=train_transforms,
            context_slices=context_slices,
        )
        val_dataset   = SliceDataset(
            val_slices,
            transforms=val_transforms,
            context_slices=context_slices,
        )

    train_cfg = copy.deepcopy(cfg["training"])
    train_cfg["batch_size"] = runtime_settings["train_batch_size"]
    cfg["training"] = train_cfg
    cfg["runtime_active"] = runtime_settings

    if device.type == "mps":
        logger.info("Apple Silicon MPS backend enabled.")
    if runtime_settings["num_workers"] != runtime_settings["requested_num_workers"]:
        logger.warning(
            f"Overriding num_workers={runtime_settings['requested_num_workers']} "
            f"-> {runtime_settings['num_workers']} "
            f"for {device.type} on macOS to avoid DataLoader spawn stalls."
        )
    loader_common = dataloader_kwargs(runtime_settings)
    train_loader = DataLoader(
        train_dataset,
        batch_size  = runtime_settings["train_batch_size"],
        shuffle     = True,
        drop_last   = True,
        **loader_common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = runtime_settings["val_batch_size"],
        shuffle     = False,
        **loader_common,
    )

    logger.info(
        f"Dataset ready: "
        f"train={len(train_dataset)} slices, "
        f"val={len(val_dataset)} slices"
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(cfg)
    logger.info(f"Model: {model.__class__.__name__}  "
                f"({model.count_parameters():,} parameters)")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        device       = device,
        runtime_cfg  = runtime_settings,
        output_dir   = output_dir,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    history = trainer.train()

    # ── Save training curves ────────────────────────────────────────────────
    from src.visualization.plots import plot_training_curves
    fig_dir = cfg["paths"].get("figure_dir", os.path.join(output_dir, "figures"))
    os.makedirs(fig_dir, exist_ok=True)
    plot_training_curves(history, os.path.join(fig_dir, "training_curves.png"))

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
