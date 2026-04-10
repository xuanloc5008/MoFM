"""
MOFMTDA-TTA: Main entry point.

Usage:
    python run.py --phase 1                    # Train experts only
    python run.py --phase 2                    # Train gating (requires phase 1 checkpoints)
    python run.py --phase 3                    # Inference + TTEC + Topo-A*
    python run.py --phase all                  # Full pipeline
    python run.py --phase all --seeds 5        # Full pipeline, 5 seeds
    python run.py --phase eval                 # Evaluate on test set
    python run.py --phase eval --dataset mms   # Evaluate on M&Ms (cross-dataset)
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from config import Config
from data.dataset import ACDCDataset, collate_fn
from train_expert import train_all_experts, train_one_expert
from train_gating import train_gating
from inference import InferencePipeline
from models.sam_expert import build_expert
from models.gating import GatingNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="MOFMTDA-TTA Pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["1", "2", "3", "all", "eval"],
                        help="Which phase to run")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of random seeds (for reproducibility reporting)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Starting random seed")
    parser.add_argument("--dataset", type=str, default="acdc",
                        choices=["acdc", "mms", "mms2"],
                        help="Dataset for evaluation")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root path")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs_expert", type=int, default=None)
    parser.add_argument("--epochs_gating", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def build_config(args) -> Config:
    """Build config with CLI overrides."""
    config = Config()
    if args.data_root:
        config.data.data_root = args.data_root
    if args.checkpoint_dir:
        config.train.checkpoint_dir = args.checkpoint_dir
    if args.batch_size:
        config.train.expert_batch_size = args.batch_size
        config.train.gating_batch_size = args.batch_size
    if args.epochs_expert:
        config.train.expert_epochs = args.epochs_expert
    if args.epochs_gating:
        config.train.gating_epochs = args.epochs_gating
    return config


def load_experts(config: Config, seed: int, device: torch.device) -> list:
    """Load trained UNet experts from checkpoints."""
    experts = []
    ckpt_dir = Path(config.train.checkpoint_dir) / f"seed_{seed}"

    for expert_id in range(config.gating.num_experts):
        expert = build_expert(
            num_classes=config.data.num_classes,
            img_size=config.data.img_size[0],
            encoder_name=config.backbone.encoder_name,
            embed_dim=config.loss.contrastive_embed_dim,
        ).to(device)

        ckpt_path = ckpt_dir / f"expert_{expert_id}_best.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            expert.load_state_dict(ckpt["state_dict"])
            print(f"Loaded Expert {expert_id} (dice={ckpt['val_dice']:.4f})")
        else:
            print(f"WARNING: {ckpt_path} not found")

        experts.append(expert)
    return experts


def load_gating(config: Config, seed: int, device: torch.device) -> GatingNetwork:
    """Load trained gating network."""
    gating = GatingNetwork(
        num_experts=config.gating.num_experts,
        num_classes=config.data.num_classes,
        base_channels=config.gating.gating_channels,
    ).to(device)

    ckpt_path = Path(config.train.checkpoint_dir) / f"seed_{seed}" / "gating_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        gating.load_state_dict(ckpt["state_dict"])
        print(f"Loaded Gating from {ckpt_path} (val_dice={ckpt['val_dice']:.4f})")
    else:
        print(f"WARNING: Gating checkpoint not found: {ckpt_path}")

    return gating


def collect_training_embeddings(experts, train_loader, device):
    """Collect embeddings from training set for TTEC k-NN."""
    embeddings = []
    experts[0].eval()
    with torch.no_grad():
        for batch in train_loader:
            images = batch["image"].to(device)
            out = experts[0](images)
            embeddings.append(out["embedding"].cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def run_phase1(config, seed):
    """Phase 1: Train all experts."""
    print(f"\n{'#'*60}")
    print(f"# PHASE 1: Expert Training (seed={seed})")
    print(f"{'#'*60}")
    return train_all_experts(config, seed)


def run_phase2(config, seed, experts=None):
    """Phase 2: Train gating network."""
    print(f"\n{'#'*60}")
    print(f"# PHASE 2: Gating Training (seed={seed})")
    print(f"{'#'*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if experts is None:
        experts = load_experts(config, seed, device)

    gating, history = train_gating(experts, config, seed)
    return gating, history


def run_phase3_eval(config, seed, dataset_name="acdc"):
    """Phase 3+4: Inference + Evaluation."""
    print(f"\n{'#'*60}")
    print(f"# PHASE 3-4: Inference + Evaluation (seed={seed}, dataset={dataset_name})")
    print(f"{'#'*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    experts = load_experts(config, seed, device)
    gating = load_gating(config, seed, device)

    # Collect training embeddings for TTEC k-NN
    train_ds = ACDCDataset(config.train.cache_dir, split="train",
                           augment=False, use_cached_features=False)
    train_loader = DataLoader(train_ds, batch_size=8, collate_fn=collate_fn)
    train_embeddings = collect_training_embeddings(experts, train_loader, device)

    # Test dataset
    if dataset_name == "acdc":
        test_ds = ACDCDataset(config.train.cache_dir, split="test",
                              augment=False, use_cached_features=False)
    elif dataset_name == "mms":
        test_ds = ACDCDataset(config.data.mms_root, split="test",
                              augment=False, use_cached_features=False)
    else:
        test_ds = ACDCDataset(config.data.mms2_root, split="test",
                              augment=False, use_cached_features=False)

    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)

    # Inference pipeline
    pipeline = InferencePipeline(experts, gating, config, train_embeddings)
    results = pipeline.evaluate_dataset(test_loader)

    # Save results
    out_dir = Path(config.train.checkpoint_dir) / f"seed_{seed}" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"eval_{dataset_name}.json", "w") as f:
        json.dump({k: v if not isinstance(v, np.floating) else float(v)
                   for k, v in results.items()}, f, indent=2, default=str)

    return results


def main():
    args = parse_args()
    config = build_config(args)

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    seeds = list(range(args.seed, args.seed + args.seeds))
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        if args.phase in ["1", "all"]:
            experts, exp_history = run_phase1(config, seed)
        else:
            experts = None

        if args.phase in ["2", "all"]:
            gating, gat_history = run_phase2(config, seed, experts)

        if args.phase in ["3", "all", "eval"]:
            results = run_phase3_eval(config, seed, args.dataset)
            all_results.append(results)

    # Aggregate results across seeds
    if all_results and len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS ({len(seeds)} seeds)")
        print(f"{'='*60}")
        for struct in ["LV", "Myo", "RV"]:
            dices = [r["dice"][struct] for r in all_results]
            print(f"  {struct}: Dice = {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        ters = [r["ter_after_correction"] for r in all_results]
        print(f"  TER = {np.mean(ters):.1f} ± {np.std(ters):.1f}%")


if __name__ == "__main__":
    main()
